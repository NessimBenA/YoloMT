"""The real-time runtime of the engines: the 30 FPS per-frame schedule on clips, and N paced streams on one GPU.

    python deploy/run.py timing --engines <dir> [--clips karate.mp4] [--loops 0 1 2 4 8 16] [--out report.json]
    python deploy/run.py serve --engines <dir> --streams 3 --loops 16 [--aligned] --out <dir>

`<dir>` is what `deploy/engine.py` built. The paper's claim: on one T4 the engine keeps the p99 time per frame at
12.66-12.69 ms at every L, within 16 ms per frame at 30 FPS, in 997 MiB, with a caption lag of 12.6-15.8 frames, and
three streams fit on one T4.

Per frame f of a clip at 30 FPS, back to back (a stream paces them on the clock):
- every 30th frame is a tick n. The tick draws its lookback L from `--loops`, cycled over the ticks (L is at most n,
  the seconds the clip has played). The vision engine writes frame t's 64 tokens; the prompt at L gets them and, oldest
  first, the pooled 16 tokens of ticks n - L .. n - 1 from a ring buffer; then the pooler (torch, float32) writes frame
  t's pooled tokens into ring slot n mod 16. No frame is ever encoded twice.
- from the next frame, one prefill chunk per frame (at most 115 tokens: L = 0, 1, 2 one chunk, 3-8 two, 9-14 three,
  15-16 four). The last chunk writes the first token and the decode states.
- on the following frames, 4 greedy decode steps per frame until <|im_end|> or 80 tokens. A caption still open at the
  next tick is abandoned and counted.
So no frame carries more than one of vision, one prefill chunk and the decode steps. Decode is two execution contexts
of the writer engine that alternate between two buffer sets, so the steps run back to back with no host work.

Time per frame is the CUDA events' time from after the frame's host-to-device copy to the end of its GPU work, p99
over the frames after the first second, taken per clip at each L; the table shows the worst clip's. Caption lag is the
frames from the tick to the frame at whose end the finished caption shows, averaged over every caption at that L. GPU
time per caption adds the median times of one tick's vision with the prompt fill and the pooler, the chunks of the
prefill and one decode step per caption token, each timed alone (`micro`). Memory is what the device holds, from
nvidia-smi, after loading and at its largest; the paper's 997 MiB is the engine alone on the card, so on a shared
device read the process's own figure beside it.

Clips: the three public clips of raushan-testing-hf/videos-test, decoded at their own 30 FPS and squashed to
256 x 256 (bicubic). A clip fills the L its ticks reach: only the long one reaches every L.

`serve` runs the streams as N processes that each own an engine copy, start together and play Cooking_cake from
frame 300 i, paced by the clock for 60 + L s. Stream i's arrivals are shifted by i (30 + 1) / (30 N) s so that the
streams' ticks and frames spread out; `--aligned` starts them together. A stream passes if its p99 time per frame
from the CUDA events stays within 16 ms and no caption is abandoned.

TensorRT contexts keep raw addresses: every tensor bound to a context stays referenced by the `Writer`. An engine
whose bound tensor was freed read reused memory and wrote wrong captions.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from functools import partial
from typing import Any, Callable, Literal

import av
import numpy as np
import tensorrt as trt
import torch
from huggingface_hub import hf_hub_download
from numpy.typing import NDArray
from PIL import Image
from transformers import AutoTokenizer

from yolomt.graphs import DECODE, EXTEND, Runtime, load_runtime
from yolomt.model import BASE, BASE_REVISION, POOLED, SIDE, TOKENS, Pooler

FPS = 30
# the deployed schedule: decode steps per frame, and frames from a tick to its first prefill chunk
STEPS = 4
OFFSET = 1
BUDGET_MS = 16.0
VIDEOS = "raushan-testing-hf/videos-test"
VIDEOS_REVISION = "4cba700bd771f44d72b549253da025c32e944d42"
CLIPS = ("Big_Buck_Bunny_720_10s_10MB.mp4", "karate.mp4", "Cooking_cake.mp4")
STREAM_CLIP = "Cooking_cake.mp4"
# stream i starts STREAM_SKIP i frames into the clip and plays STREAM_SECONDS + L seconds
STREAM_SKIP = 300
STREAM_SECONDS = 60
# decode steps between two reads of the history when `caption` writes a caption to its end
EVAL_STEPS = 8
TORCH = {trt.float16: torch.float16, trt.float32: torch.float32, trt.uint8: torch.uint8, trt.int64: torch.int64}


@dataclass(frozen=True)
class Buffers:
    """One decode buffer set: the current token (1,), its embedding (1, 1, d), its position (1,), and the conv windows,
    then K and V caches, in the engine's input order."""

    token: torch.Tensor
    embeds: torch.Tensor
    position: torch.Tensor
    states: list[torch.Tensor]


@dataclass(frozen=True)
class Writer:
    """The loaded engines and every tensor their contexts are bound to. frame (1, 3, 256, 256) uint8, tokens
    (1, 64, d) FP16 and siglip (1, 768) float32 are the vision engine's; prompts[L]: (1, 79 + 18 L, d) FP16, whose
    slices `runtime.prompts[L].chunks` the extend context reads; ring (16, 16, d): the pooled tokens of the last 16
    ticks, tick n in slot n mod 16; history (written,): the tokens written so far; hidden (1, d): an output every call
    writes and nothing reads; zero_position and zero_conv (by name): the first chunk's inputs; trt_runtime: it must
    outlive the engines."""

    runtime: Runtime
    trt_runtime: Any
    vision_engine: Any
    vision: Any
    engine: Any
    contexts: list[Any]
    frame: torch.Tensor
    tokens: torch.Tensor
    siglip: torch.Tensor
    prompts: list[torch.Tensor]
    ring: torch.Tensor
    history: torch.Tensor
    sets: tuple[Buffers, Buffers]
    pooler: Pooler
    state_names: list[str]
    hidden: torch.Tensor
    zero_position: torch.Tensor
    zero_conv: dict[str, torch.Tensor]
    tokenizer: Any


@dataclass(frozen=True)
class Progress:
    """The first `count` tokens written (1 after the prefill); ids: those before <|im_end|>; done: <|im_end|> is among
    them or the caption reached its limit."""

    count: int
    done: bool
    ids: list[int]


@dataclass(frozen=True)
class Frame:
    """loops: the L of the frame's tick; kind: its vision tick, a prefill chunk, or neither; events_ms: CUDA events'
    time; host_ms: wall time; late_ms: how long after its arrival the frame started (paced runs, else 0)."""

    loops: int
    kind: Literal["tick", "prefill", "other"]
    events_ms: float
    host_ms: float
    late_ms: float


@dataclass(frozen=True)
class Caption:
    """loops: its tick's L; lag: frames from the tick to the frame at whose end it shows; tokens: before <|im_end|>."""

    loops: int
    lag: int
    tokens: int
    text: str


@dataclass(frozen=True)
class Streamed:
    """A clip's frames and completed captions; abandoned[L]: captions at L still open at the next tick."""

    frames: list[Frame]
    captions: list[Caption]
    abandoned: dict[int, int]


@dataclass(frozen=True)
class Stat:
    p50_ms: float
    p99_ms: float
    max_ms: float


@dataclass(frozen=True)
class Memory:
    """process_mib: the process's GPU memory from nvidia-smi; device_used_mib: what the whole device holds."""

    process_mib: float
    device_used_mib: float


@dataclass(frozen=True)
class Micro:
    """Each call timed alone between its own CUDA events: vision (a tick's vision, prompt fill and pooler at L = 16);
    chunks[L][i]: chunk i of the prefill at L; step: one decode step."""

    vision: Stat
    chunks: list[list[Stat]]
    step: Stat


@dataclass(frozen=True)
class Row:
    """One L over the clips that reach it with a frame after the first second: the worst clip's time per frame (all
    frames, then the vision ticks, the prefill chunks' frames and the others); the completed captions' mean lag in
    frames and mean length over those clips; GPU seconds per caption (None when no caption completed)."""

    loops: int
    prompt: int
    chunks: list[int]
    worst_clip: str
    frame: Stat
    tick: Stat
    prefill: Stat
    other: Stat
    captions: int
    abandoned: int
    lag_frames: float | None
    tokens: float | None
    gpu_seconds: float | None


@dataclass(frozen=True)
class Report:
    tensorrt: str
    gpu: str
    head_rows: int
    t_max: int
    written: int
    after_load: Memory
    peak: Memory
    rows: list[Row]
    micro: Micro


@dataclass(frozen=True)
class Stream:
    """One stream of `serve` at its steady L (the frames whose tick drew L = loops)."""

    index: int
    phase_s: float
    first_frame: int
    loops: int
    p99_events_ms: float
    p99_host_ms: float
    p99_late_ms: float
    captions: int
    abandoned: int
    lag_frames: float | None
    device_mib: float


def frames_of(path: Path, limit: int | None) -> list[NDArray[np.uint8]]:
    """The first `limit` frames (all when None) of a video, each (256, 256, 3) uint8."""
    container = av.open(str(path))
    stream = container.streams.video[0]
    frames = [np.array(frame.to_image().resize((SIDE, SIDE), Image.Resampling.BICUBIC))
              for frame in itertools.islice(container.decode(stream), limit)]
    container.close()
    return frames


def clip(name: str, *, limit: int | None = None) -> list[NDArray[np.uint8]]:
    return frames_of(Path(hf_hub_download(VIDEOS, name, repo_type="dataset", revision=VIDEOS_REVISION)), limit)


def own_mib() -> Memory:
    rows = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                          check=True, capture_output=True, text=True).stdout.strip().splitlines()
    mine = float(sum(int(m) for p, m in (r.split(",") for r in rows if r.strip()) if int(p) == os.getpid()))
    free, total = torch.cuda.mem_get_info()
    return Memory(process_mib=mine, device_used_mib=(total - free) / 2**20)


def context_for(engine: Any, profile: int) -> Any:
    context = engine.create_execution_context()
    assert context.set_optimization_profile_async(profile, torch.cuda.current_stream().cuda_stream)
    torch.cuda.current_stream().synchronize()
    return context


def bind(engine: Any, context: Any, tensors: dict[str, torch.Tensor]) -> None:
    """Fix a context's tensors by name; the caller keeps them alive and in place."""
    for name, tensor in tensors.items():
        assert tensor.dtype == TORCH[engine.get_tensor_dtype(name)] and tensor.is_contiguous(), name
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            assert context.set_input_shape(name, tuple(tensor.shape)), name
        else:
            assert tuple(context.get_tensor_shape(name)) == tuple(tensor.shape), (name, context.get_tensor_shape(name))
        assert context.set_tensor_address(name, tensor.data_ptr()), name


def call(engine: Any, context: Any, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """One run of a context on new output tensors, the inputs cast to the engine's dtypes."""
    held = {name: tensor.to(TORCH[engine.get_tensor_dtype(name)]).contiguous() for name, tensor in inputs.items()}
    for name, tensor in held.items():
        assert context.set_input_shape(name, tuple(tensor.shape)), name
    names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    out = {name: torch.empty(tuple(context.get_tensor_shape(name)), dtype=TORCH[engine.get_tensor_dtype(name)],
                             device="cuda")
           for name in names if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT}
    for name, tensor in (held | out).items():
        assert context.set_tensor_address(name, tensor.data_ptr()), name
    assert context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.current_stream().synchronize()
    return out


def outputs_of(into: Buffers, hidden: torch.Tensor, state_names: list[str]) -> dict[str, torch.Tensor]:
    """A writer call's outputs bound to buffer set `into` and the shared hidden state."""
    return {"next_token": into.token, "next_embeds": into.embeds, "next_position": into.position, "hidden": hidden,
            **{f"{name}_out": tensor for name, tensor in zip(state_names, into.states)}}


def load(folder: Path) -> Writer:
    """The engines and runtime data of `folder`, warmed up with three captions of a blank frame at every L: without
    it the first clip's worst frame was far slower than the later clips'."""
    runtime = load_runtime(folder / "runtime.pt")
    k_max = len(runtime.prompts) - 1
    assert k_max >= 1, "the ring needs a lookback"
    trt_runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
    vision_engine = trt_runtime.deserialize_cuda_engine((folder / "vision.engine").read_bytes())
    engine = trt_runtime.deserialize_cuda_engine((folder / "writer.engine").read_bytes())
    assert engine.num_optimization_profiles == 3, engine.num_optimization_profiles
    contexts = [context_for(engine, profile) for profile in range(3)]
    names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    state_names = [n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT
                   and n not in ("embeds", "position")]

    def zeros(name: str) -> torch.Tensor:
        return torch.zeros(tuple(engine.get_tensor_shape(name)), dtype=TORCH[engine.get_tensor_dtype(name)],
                           device="cuda")

    prompts = [p.embeds.cuda()[None].contiguous() for p in runtime.prompts]
    width = prompts[0].shape[2]
    sets = (Buffers(torch.zeros(1, dtype=torch.int64, device="cuda"),
                    torch.zeros(1, 1, width, dtype=torch.float16, device="cuda"),
                    torch.zeros(1, dtype=torch.int64, device="cuda"), [zeros(n) for n in state_names]),
            Buffers(torch.zeros(1, dtype=torch.int64, device="cuda"),
                    torch.zeros(1, 1, width, dtype=torch.float16, device="cuda"),
                    torch.zeros(1, dtype=torch.int64, device="cuda"), [zeros(n) for n in state_names]))
    assert all(t.shape[2] == runtime.t_max for n, t in zip(state_names, sets[0].states) if not n.startswith("conv"))
    hidden = torch.zeros(1, width, dtype=torch.float16, device="cuda")
    for c in DECODE:
        bind(engine, contexts[c], {"embeds": sets[c].embeds, "position": sets[c].position,
                                   **dict(zip(state_names, sets[c].states)),
                                   **outputs_of(sets[1 - c], hidden, state_names)})
    vision = vision_engine.create_execution_context()
    frame = torch.zeros(1, 3, SIDE, SIDE, dtype=torch.uint8, device="cuda")
    tokens = torch.zeros(1, 64, width, dtype=torch.float16, device="cuda")
    siglip = torch.zeros(tuple(vision_engine.get_tensor_shape("siglip")), dtype=torch.float32, device="cuda")
    bind(vision_engine, vision, {"frame": frame, "tokens": tokens, "siglip": siglip})
    pooler = Pooler(width).cuda().requires_grad_(False)
    pooler.load_state_dict({"residual.weight": runtime.pooler_weight, "residual.bias": runtime.pooler_bias})
    w = Writer(runtime=runtime, trt_runtime=trt_runtime, vision_engine=vision_engine, vision=vision, engine=engine,
               contexts=contexts, frame=frame, tokens=tokens, siglip=siglip, prompts=prompts,
               ring=torch.zeros(k_max, POOLED, width, dtype=torch.float16, device="cuda"),
               history=torch.zeros(runtime.written, dtype=torch.int64, device="cuda"), sets=sets, pooler=pooler,
               state_names=state_names, hidden=hidden, zero_position=torch.zeros(1, dtype=torch.int64, device="cuda"),
               zero_conv={n: zeros(n) for n in state_names if n.startswith("conv")},
               tokenizer=AutoTokenizer.from_pretrained(BASE, revision=BASE_REVISION))
    for loops in range(k_max + 1):
        for _ in range(3):
            see(w, torch.zeros_like(frame))
            fill(w, k_max, loops)
            remember(w, k_max)
            prefill(w, loops)
            decode(w, 0, runtime.written - 1)
    torch.cuda.synchronize()
    return w


def see(w: Writer, frame: torch.Tensor) -> None:
    """Frame t, (1, 3, 256, 256) uint8, through the vision engine into the staging tokens."""
    w.frame.copy_(frame)
    assert w.vision.execute_async_v3(torch.cuda.current_stream().cuda_stream)


def fill(w: Writer, tick: int, loops: int) -> None:
    """The prompt at `loops` for the clip's tick-th tick: frame t's 64 staged tokens and, oldest first, the pooled
    tokens of ticks tick - loops .. tick - 1 from the ring. Runs before `remember` overwrites slot tick mod 16."""
    assert 0 <= loops <= min(tick, len(w.ring))
    prompt, slots = w.prompts[loops], w.runtime.prompts[loops]
    prompt[:, slots.current:slots.current + TOKENS].copy_(w.tokens)
    for j, at in enumerate(slots.pooled):
        prompt[0, at:at + POOLED].copy_(w.ring[(tick - loops + j) % len(w.ring)])


@torch.no_grad()
def remember(w: Writer, tick: int) -> None:
    """The pooler, in float32, on the staged frame's tokens into ring slot tick mod 16."""
    w.ring[tick % len(w.ring)] = w.pooler(w.tokens.float())[0].half()


def extend(w: Writer, loops: int, i: int) -> None:
    """Chunk i of the prompt at `loops`, bound now on the extend context. Chunk i of n reads buffer set (n - i) mod 2
    and writes the other, so the last chunk writes set 0, where decode starts. Chunk 0 reads position 0, zero conv
    windows and the set's K/V caches, whose columns past the position the mask hides."""
    chunks = w.runtime.prompts[loops].chunks
    n, start = len(chunks), sum(chunks[:i])
    source, into = w.sets[(n - i) % 2], w.sets[(n - 1 - i) % 2]
    states = {name: w.zero_conv[name] if i == 0 and name in w.zero_conv else tensor
              for name, tensor in zip(w.state_names, source.states)}
    context = w.contexts[EXTEND]
    bind(w.engine, context, {"embeds": w.prompts[loops][:, start:start + chunks[i]],
                             "position": w.zero_position if i == 0 else source.position, **states,
                             **outputs_of(into, w.hidden, w.state_names)})
    assert context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    if i == n - 1:
        w.history[0:1].copy_(w.sets[0].token)


def prefill(w: Writer, loops: int) -> None:
    for i in range(len(w.runtime.prompts[loops].chunks)):
        extend(w, loops, i)


def decode(w: Writer, done: int, count: int) -> None:
    """Decode steps done .. done + count - 1; step j reads set j mod 2 and writes the other, token j + 1."""
    stream = torch.cuda.current_stream().cuda_stream
    for j in range(done, done + count):
        assert w.contexts[DECODE[j % 2]].execute_async_v3(stream)
        w.history[j + 1:j + 2].copy_(w.sets[1 - j % 2].token)


def progress(w: Writer, count: int) -> Progress:
    """After a synchronisation: the first `count` tokens of the history, cut at <|im_end|>."""
    ids = w.history[:count].tolist()
    if w.runtime.im_end in ids:
        return Progress(count=count, done=True, ids=ids[:ids.index(w.runtime.im_end)])
    return Progress(count=count, done=count >= w.runtime.written, ids=ids)


def text(w: Writer, ids: list[int]) -> str:
    decoded = w.tokenizer.decode(ids, skip_special_tokens=True)
    assert isinstance(decoded, str)
    return decoded.strip()


def caption(w: Writer, past: list[torch.Tensor], now: torch.Tensor, *, loops: int) -> Progress:
    """The caption of frame `now` at lookback `loops` written to its end. past: the frames t-1 s, t-2 s, ...; those of
    t-loops s .. t-1 s go through vision and the pooler into the ring, then frame t, every chunk back to back and the
    decode steps in groups of 8."""
    assert loops <= len(past)
    for n, frame in enumerate(reversed(past[:loops])):
        see(w, frame)
        remember(w, n)
    see(w, now)
    fill(w, loops, loops)
    prefill(w, loops)
    state = progress(w, 1)
    while not state.done:
        steps = min(EVAL_STEPS, w.runtime.written - state.count)
        decode(w, state.count - 1, steps)
        state = progress(w, state.count + steps)
    return state


def stream_clip(w: Writer, frames: list[NDArray[np.uint8]], *, schedule: list[int], pace: float | None) -> Streamed:
    """The frames of a clip through the schedule of the module docstring. Tick n draws L = min(schedule[n mod
    len(schedule)], n). pace: the wall-clock time (time.time) frame 0 arrives, frame f arriving f / 30 s later and
    never before; None runs the frames back to back."""
    log: list[Frame] = []
    captions: list[Caption] = []
    abandoned: dict[int, int] = {}
    writing, count, loops, tick_frame, due = False, 0, 0, 0, 0
    for f, image in enumerate(frames):
        late = 0.0
        if pace is not None:
            arrival = pace + f / FPS
            late = max(0.0, time.time() - arrival) * 1e3
            time.sleep(max(0.0, arrival - time.time()))
        started = time.perf_counter()
        frame = torch.from_numpy(image).permute(2, 0, 1)[None].pin_memory().cuda(non_blocking=True)
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
        tick = f % FPS == 0
        if tick:
            abandoned[loops] = abandoned.get(loops, 0) + int(writing)
            writing = False
            n = f // FPS
            loops = min(schedule[n % len(schedule)], n)
            see(w, frame)
            fill(w, n, loops)
            remember(w, n)
            tick_frame, due = f, f + OFFSET
        chunks = w.runtime.prompts[loops].chunks
        chunk = 0 <= f - due < len(chunks)
        if chunk:
            extend(w, loops, f - due)
            if f - due == len(chunks) - 1:
                count, writing = 1, True
        elif writing:
            steps = min(STEPS, w.runtime.written - count)
            decode(w, count - 1, steps)
            count += steps
        end.record()
        end.synchronize()
        if writing:
            now = progress(w, count)
            if now.done:
                writing = False
                captions.append(Caption(loops, f - tick_frame, len(now.ids), text(w, now.ids)))
        kind: Literal["tick", "prefill", "other"] = "tick" if tick else "prefill" if chunk else "other"
        log.append(Frame(loops, kind, begin.elapsed_time(end), (time.perf_counter() - started) * 1e3, late))
    abandoned[loops] = abandoned.get(loops, 0) + int(writing)
    return Streamed(log, captions, abandoned)


def stat(values: list[float]) -> Stat:
    a = np.asarray(values)
    return Stat(p50_ms=float(np.percentile(a, 50)), p99_ms=float(np.percentile(a, 99)), max_ms=float(a.max()))


def timed(work: Callable[[], object]) -> float:
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    begin.record()
    work()
    end.record()
    end.synchronize()
    return float(begin.elapsed_time(end))


def micro(w: Writer, past: list[torch.Tensor], now: torch.Tensor, *, repeats: int) -> Micro:
    """On one real window of 16 past frames: a tick's vision, prompt fill and pooler at L = 16; every chunk of the
    prefill at every L, each repeated on the states of its prefill; the 79 decode steps after the prefill at L = 16,
    whatever the tokens."""
    k_max = len(w.ring)

    def tick() -> None:
        see(w, now)
        fill(w, k_max, k_max)
        remember(w, k_max)

    caption(w, past, now, loops=k_max)
    vision = stat([timed(tick) for _ in range(repeats)])
    caption(w, past, now, loops=k_max)
    for loops in range(k_max + 1):
        fill(w, k_max, loops)
    chunks = []
    for loops in range(k_max + 1):
        prefill(w, loops)
        chunks.append([stat([timed(partial(extend, w, loops, i)) for _ in range(repeats)])
                       for i in range(len(w.runtime.prompts[loops].chunks))])
    steps: list[float] = []
    for _ in range(max(1, repeats // 10)):
        prefill(w, k_max)
        steps += [timed(partial(decode, w, j, 1)) for j in range(w.runtime.written - 1)]
    return Micro(vision=vision, chunks=chunks, step=stat(steps))


def row_of(w: Writer, loops: int, clips: dict[str, Streamed], measured: Micro) -> Row:
    steady = {name: [f for f in c.frames[FPS:] if f.loops == loops] for name, c in clips.items()}
    steady = {name: frames for name, frames in steady.items() if frames}
    worst = max(steady, key=lambda name: stat([f.events_ms for f in steady[name]]).p99_ms)
    frames = steady[worst]

    def of(*kinds: str) -> Stat:
        return stat([f.events_ms for f in frames if f.kind in kinds])

    done = [c for name in steady for c in clips[name].captions if c.loops == loops]
    dropped = sum(clips[name].abandoned.get(loops, 0) for name in steady)
    tokens = float(np.mean([c.tokens for c in done])) if done else None
    gpu = None if tokens is None else (measured.vision.p50_ms + sum(s.p50_ms for s in measured.chunks[loops])
                                       + measured.step.p50_ms * tokens) / 1e3
    return Row(loops=loops, prompt=len(w.runtime.prompts[loops].embeds), chunks=w.runtime.prompts[loops].chunks,
               worst_clip=worst, frame=of("tick", "prefill", "other"), tick=of("tick"), prefill=of("prefill"),
               other=of("other"), captions=len(done), abandoned=dropped,
               lag_frames=float(np.mean([c.lag for c in done])) if done else None, tokens=tokens, gpu_seconds=gpu)


def table(report: Report) -> str:
    lines = [f"{'L':>3} {'prompt':>6} {'chunks':>16} {'p99 ms':>7} {'lag frames':>10} {'GPU s':>6} {'done':>5} "
             f"{'abandoned':>9}  worst clip"]
    for r in report.rows:
        lag = "-" if r.lag_frames is None else f"{r.lag_frames:.1f}"
        gpu = "-" if r.gpu_seconds is None else f"{r.gpu_seconds:.3f}"
        lines.append(f"{r.loops:>3} {r.prompt:>6} {str(r.chunks):>16} {r.frame.p99_ms:>7.2f} {lag:>10} {gpu:>6} "
                     f"{r.captions:>5} {r.abandoned:>9}  {r.worst_clip}")
    lines.append(f"memory: {report.after_load.device_used_mib:.0f} MiB after loading, "
                 f"{report.peak.device_used_mib:.0f} MiB at most (this process {report.peak.process_mib:.0f}); "
                 f"head rows {report.head_rows}, T_MAX {report.t_max}")
    return "\n".join(lines)


def timing(folder: Path, *, clips: list[str], loops: list[int] | None) -> Report:
    w = load(folder)
    k_max = len(w.ring)
    schedule = loops if loops else list(range(k_max + 1))
    assert all(0 <= k <= k_max for k in schedule), schedule
    decoded = {name: clip(name) for name in clips}
    after_load = own_mib()
    memory = [after_load]
    results = {}
    for name, frames in decoded.items():
        results[name] = stream_clip(w, frames, schedule=schedule, pace=None)
        memory.append(own_mib())
    source = next(frames for frames in decoded.values() if len(frames) > k_max * FPS)
    gpu_frames = [torch.from_numpy(source[s * FPS]).permute(2, 0, 1)[None].cuda() for s in range(k_max + 1)]
    measured = micro(w, gpu_frames[-2::-1], gpu_frames[-1], repeats=100)
    memory.append(own_mib())
    reached = sorted({f.loops for r in results.values() for f in r.frames[FPS:]})
    return Report(
        tensorrt=trt.__version__, gpu=torch.cuda.get_device_name(0), head_rows=w.runtime.head_rows,
        t_max=w.runtime.t_max, written=w.runtime.written, after_load=after_load,
        peak=Memory(max(m.process_mib for m in memory), max(m.device_used_mib for m in memory)),
        rows=[row_of(w, k, results, measured) for k in reached], micro=measured)


def rendezvous(directory: Path, *, index: int, count: int) -> float:
    """Marks stream `index` ready in `directory` (new for each group) and waits for the other count - 1; returns the
    common start time, 2 s after the last was ready."""
    (directory / f"ready_{index}").write_text("")
    waited = time.time()
    while len(list(directory.glob("ready_*"))) < count:
        assert time.time() - waited < 900, f"streams ready in {directory}: {sorted(directory.glob('ready_*'))}"
        time.sleep(0.05)
    return max(p.stat().st_mtime for p in directory.glob("ready_*")) + 2.0


def play(folder: Path, *, index: int, streams: int, loops: int, aligned: bool, barrier: Path) -> Stream:
    """Stream `index` of `streams` processes: 60 + loops s of Cooking_cake from frame 300 index at L = loops."""
    w = load(folder)
    seconds = STREAM_SECONDS + loops
    frames = clip(STREAM_CLIP, limit=seconds * FPS + (streams - 1) * STREAM_SKIP)
    start = rendezvous(barrier, index=index, count=streams)
    phase = 0.0 if aligned else index * (FPS + 1) / (streams * FPS)
    first = index * STREAM_SKIP
    mine = frames[first:first + seconds * FPS]
    assert len(mine) == seconds * FPS, (len(mine), seconds, index)
    played = stream_clip(w, mine, schedule=[loops], pace=start + phase)
    steady = [f for f in played.frames[FPS:] if f.loops == loops]
    done = [c.lag for c in played.captions if c.loops == loops]
    return Stream(index=index, phase_s=phase, first_frame=first, loops=loops,
                  p99_events_ms=stat([f.events_ms for f in steady]).p99_ms,
                  p99_host_ms=stat([f.host_ms for f in steady]).p99_ms,
                  p99_late_ms=stat([f.late_ms for f in steady]).p99_ms, captions=len(done),
                  abandoned=played.abandoned.get(loops, 0), lag_frames=float(np.mean(done)) if done else None,
                  device_mib=own_mib().device_used_mib)


def serve(folder: Path, *, streams: int, loops: int, aligned: bool, out: Path) -> bool:
    """Starts the `streams` processes, each running `stream`, and reports whether every one passes."""
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as barrier:
        children = [subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "stream", "--engines", str(folder),
                                      "--streams", str(streams), "--index", str(i), "--loops", str(loops),
                                      "--barrier", barrier, "--out", str(out / f"stream_{i}.json"),
                                      *(["--aligned"] if aligned else [])]) for i in range(streams)]
        codes = [child.wait() for child in children]
    assert codes == [0] * streams, codes
    played = [Stream(**json.loads((out / f"stream_{i}.json").read_text())) for i in range(streams)]
    passed = all(s.p99_events_ms <= BUDGET_MS and s.abandoned == 0 for s in played)
    (out / "streams.json").write_text(json.dumps({"passed": passed, "streams": [asdict(s) for s in played]}, indent=2))
    for s in played:
        lag = "-" if s.lag_frames is None else f"{s.lag_frames:.1f}"
        print(f"stream {s.index}: p99 {s.p99_events_ms:.2f} ms (events), {s.p99_host_ms:.2f} ms (host), late p99 "
              f"{s.p99_late_ms:.2f} ms, lag {lag} frames, {s.captions} captions, {s.abandoned} abandoned, "
              f"{s.device_mib:.0f} MiB on the device")
    print(f"{streams} streams at L = {loops}, {'aligned' if aligned else 'staggered'}: "
          f"{'every stream within' if passed else 'a stream beyond'} {BUDGET_MS:.0f} ms per frame with no abandoned "
          "caption")
    return passed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("timing", "serve", "stream"):
        command = commands.add_parser(name)
        command.add_argument("--engines", type=Path, required=True, help="the folder deploy/engine.py built")
        command.add_argument("--out", type=Path, required=name != "timing",
                             help="timing: the report's JSON; serve: a folder; stream: its JSON")
        if name == "timing":
            command.add_argument("--clips", nargs="+", choices=CLIPS, default=list(CLIPS))
            command.add_argument("--loops", type=int, nargs="+", help="the L the ticks cycle through, default all")
        else:
            command.add_argument("--streams", type=int, required=True)
            command.add_argument("--loops", type=int, required=True)
            command.add_argument("--aligned", action="store_true", help="every stream's frames arrive together")
        if name == "stream":
            command.add_argument("--index", type=int, required=True)
            command.add_argument("--barrier", type=Path, required=True)
    args = parser.parse_args()
    match args.command:
        case "timing":
            report = timing(args.engines, clips=args.clips, loops=args.loops)
            print(table(report))
            if args.out is not None:
                args.out.write_text(json.dumps(asdict(report), indent=2))
        case "serve":
            serve(args.engines, streams=args.streams, loops=args.loops, aligned=args.aligned, out=args.out)
        case _:
            played = play(args.engines, index=args.index, streams=args.streams, loops=args.loops,
                          aligned=args.aligned, barrier=args.barrier)
            args.out.write_text(json.dumps(asdict(played), indent=2))


if __name__ == "__main__":
    main()
