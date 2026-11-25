"""Build the real-time engines from the published weights and check them against float32 PyTorch.

    pip install -e .[deploy]
    python deploy/engine.py --weights Nessim/YoloMT --out <dir>

`--weights` is a model folder or a Hugging Face model repo. Two graphs are exported to ONNX (opset 17, the TorchScript
exporter, eager attention in the SigLIP layers) from a student whose weights are cast to FP16, LayerNorms in float32
(`yolomt.graphs`): vision.onnx (one frame -> 64 tokens) and writer.onnx (T tokens of the LM from explicit states, T up
to 115, the greedy next token over the head's kept rows). Each becomes a strongly typed FP16 TensorRT engine of batch 1
(`vision.engine`; `writer.engine` with three optimization profiles: two for T = 1, one for T = 1 .. 115 tuned at 115),
and `runtime.pt` holds what `deploy/run.py` needs: the prompt rows at every lookback L = 0 .. 16, their chunks, the
pooler and the constants. `engine.json` records the checks below with the versions and the ONNX digests.

The head keeps the 16,575 token ids of `data/vocab_trim.json` instead of the 65,536 rows of the tied embedding: the
most frequent tokens covering 99.9% of the 46,874,376 token occurrences of the 942,645 captions the 16 s SFT stage
trained on, plus the 256 byte tokens and the 127 special tokens (sha256 08e72cd4...). The file is the one the paper's
engine used, unchanged; rebuilding it needs the teacher's labels. A caption token outside it cannot be written.

Checks, each asserted, on `--windows` windows of Cooking_cake at seconds 16, 20, 24, ... (every L = 0 .. 16 on each):
- vision: the engine's tokens against float32 PyTorch's on every frame the windows use (minimum row cosine, at
  least 0.9: one token of one frame may sit far from float32 in FP16 while the frame's relative error is 0.02);
- writer: every prefill chunk and the decode step after it, the engine given the float32 states, against float32
  PyTorch: the next token of the last chunk and of the decode step agree on every window; the hidden state and every
  state row keep a cosine of at least 0.999. The next token of a chunk before the last is never read and only reported;
- captions: the whole runtime (`deploy/run.py`) against float32 `write` (full head, no previous caption), the share of
  captions that are equal, reported. FP16 may flip a token where two are close, so the paper's 84-94% is not 100%.
This check is not optional on a new GPU: some TensorRT versions build a writer whose T > 1 profiles are wrong on some
GPUs while its T = 1 profiles pass.

Versions of the paper's build: TensorRT 10.16.1.11, onnx 1.23.0, torch 2.11, transformers 5.16.1. With torch 2.10 and
transformers 5.16.1 the ONNX files of Nessim/YoloMT have the sha256 digests
1e303f0b85c02dd5... (vision.onnx) and 17d1e18aded521ba... (writer.onnx), those of the paper's engine graphs.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tensorrt as trt
import torch
import torch.nn.functional as F
import transformers
from run import FPS, STREAM_CLIP, Writer, call, caption, clip, context_for, load, text

from yolomt.graphs import (DECODE, EXTEND, Call, Chunk, Vision, decode_call, engine_precision, export_vision,
                           export_writer, io_names, prefill_calls, runtime_of, save_runtime)
from yolomt.model import (Full, Student, embed, folder_of, load_folder, prompt, read_config, see, see_past, visual,
                          write)

WORKSPACE = 3 << 30
# a vision engine without float32 LayerNorms reaches 0.67; the paper's build 0.997 on its frames
VISION_COSINE = 0.9
STATE_COSINE = 0.999


@dataclass(frozen=True)
class Match:
    """The engine's outputs of one call against float32 PyTorch's: cos, the smallest row cosine of the hidden state
    and the new states; rel, the largest relative error of them; token: the next token agrees."""

    token: bool
    cos: float
    rel: float


@dataclass(frozen=True)
class Parity:
    """One L over the windows: the share of windows whose last prefill chunk's next token agrees, of all chunks' next
    tokens that agree and of decode steps' that agree; the smallest cosine and largest relative error of any call; the
    share of captions equal to float32 PyTorch's."""

    loops: int
    last_chunk: float
    chunks: float
    decode: float
    cos: float
    rel: float
    captions: float


@dataclass(frozen=True)
class Built:
    versions: dict[str, str]
    sha256: dict[str, str]
    vision_cos: float
    vision_rel: float
    parity: list[Parity]


def build(onnx: Path, path: Path, *, width: int, profiles: list[tuple[int, int, int]]) -> None:
    """A strongly typed engine of an ONNX graph. profiles: the (low, opt, high) of the T of the input embeds
    (1, T, width) of each optimization profile; none for the vision graph, whose shapes are static."""
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, logger)
    assert parser.parse_from_file(str(onnx)), [str(parser.get_error(i)) for i in range(parser.num_errors)]
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, WORKSPACE)
    for low, opt, high in profiles:
        profile = builder.create_optimization_profile()
        profile.set_shape("embeds", (1, low, width), (1, opt, width), (1, high, width))
        config.add_optimization_profile(profile)
    serialized = builder.build_serialized_network(network, config)
    assert serialized is not None, f"TensorRT could not build {onnx.name}"
    path.write_bytes(bytes(serialized))


def compare(got: dict[str, torch.Tensor], want: Call, outputs: list[str]) -> Match:
    """`got`: the engine's outputs by name; outputs: the writer's names, hidden state and states from the fourth."""
    cos, rel = [], []
    for name, wanted in zip(outputs[3:], want.outputs[3:]):
        g, w = got[name].float(), wanted.float()
        if name.startswith("conv"):
            g, w = g.transpose(-1, -2), w.transpose(-1, -2)
        g, w = g.flatten(0, -2), w.flatten(0, -2)
        # a cache column the prompt has not reached is zero in both
        live = w.norm(dim=-1) > 0
        cos.append(float(F.cosine_similarity(g[live], w[live], dim=-1).min()))
        rel.append(float((g - w).norm() / w.norm()))
    return Match(token=int(got["next_token"][0]) == int(want.outputs[0][0]), cos=min(cos), rel=max(rel))


def vision_parity(student: Student, engine: Any, frames: torch.Tensor) -> tuple[float, float]:
    """frames (S, 3, 256, 256) uint8: the engine's tokens against float32 PyTorch's, the smallest row cosine and the
    largest relative error over the frames."""
    context = engine.create_execution_context()
    cos, rel = [], []
    for frame, want in zip(frames, see(student, frames)):
        got = call(engine, context, {"frame": frame[None]})["tokens"][0].float()
        cos.append(float(F.cosine_similarity(got, want, dim=-1).min()))
        rel.append(float((got - want).norm() / want.norm()))
    return min(cos), max(rel)


def writer_parity(w: Writer, student: Student, chunk: Chunk, contexts: list[Any], frames: torch.Tensor,
                  seconds: list[int], *, loops: int) -> Parity:
    """Lookback `loops` on the windows whose frame t is frames[t] for t in `seconds`, the past frames being the
    seconds before it: the writer engine against float32 PyTorch's chunk by chunk, then the whole runtime's captions
    against float32 `write`."""
    inputs, outputs = io_names(chunk)
    device = frames.device
    k_max = len(w.ring)
    pasts = torch.stack([frames[t - k_max:t].flip(0) for t in seconds])
    tokens_past = torch.cat([see_past(student, pasts[i:i + 1]) for i in range(len(seconds))])
    tokens_t = see(student, frames[seconds])
    ids = torch.tensor([prompt(student, loops=loops)], device=device)
    chunks = w.runtime.prompts[loops].chunks
    last, every, decoded = [], [], []
    for i in range(len(seconds)):
        embeds = embed(student, ids, visual(student, tokens_t[i:i + 1], tokens_past[i:i + 1], loops=loops))
        calls = prefill_calls(chunk, embeds, chunks=chunks)
        steps = [(EXTEND, c) for c in calls] + [(DECODE[0], decode_call(chunk, calls[-1]))]
        matches = [compare(call(w.engine, contexts[profile], dict(zip(inputs, c.inputs))), c, outputs)
                   for profile, c in steps]
        last.append(matches[-2])
        every += matches[:-1]
        decoded.append(matches[-1])
    wanted = write(student, frames[seconds], pasts, loops=loops, max_new=w.runtime.written)
    got = [text(w, caption(w, list(pasts[i][:, None]), frames[t][None], loops=loops).ids)
           for i, t in enumerate(seconds)]
    return Parity(loops=loops, last_chunk=float(np.mean([m.token for m in last])),
                  chunks=float(np.mean([m.token for m in every])), decode=float(np.mean([m.token for m in decoded])),
                  cos=min(m.cos for m in [*every, *decoded]), rel=max(m.rel for m in [*every, *decoded]),
                  captions=float(np.mean([a == b for a, b in zip(wanted, got)])))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", default="Nessim/YoloMT", help="a model folder or a Hugging Face model repo")
    parser.add_argument("--vocab", type=Path, default=Path("data/vocab_trim.json"))
    parser.add_argument("--windows", type=int, default=8, help="windows of the checks, 4 s apart from second 16")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    args.out.mkdir(parents=True, exist_ok=True)
    folder = folder_of(args.weights)
    config = read_config(folder)
    assert isinstance(config.tuning, Full), "the engines carry the LM of a fully tuned student"
    kept: list[int] = json.loads(args.vocab.read_text())["ids"]
    student = load_folder(folder, dtype=torch.float32, attention="eager").eval()
    assert student.layout.im_end in kept, "the kept head could never close a caption"
    runtime = runtime_of(student, k_max=config.k_max, written=config.max_new, kept=kept)
    save_runtime(runtime, args.out / "runtime.pt")
    half = copy.deepcopy(student)
    engine_precision(half)
    export_vision(Vision(half), args.out / "vision.onnx")
    export_writer(Chunk(half.lm, kept=kept, t_max=runtime.t_max), args.out / "writer.onnx",
                  tokens=len(runtime.prompts[1].embeds))
    del half
    width = int(student.embed.weight.shape[1])
    longest = max(max(p.chunks) for p in runtime.prompts)
    assert EXTEND == len(DECODE)
    build(args.out / "vision.onnx", args.out / "vision.engine", width=width, profiles=[])
    build(args.out / "writer.onnx", args.out / "writer.engine", width=width,
          profiles=[*[(1, 1, 1)] * len(DECODE), (1, longest, longest)])

    student.cuda()
    chunk = Chunk(student.lm, kept=kept, t_max=runtime.t_max)
    seconds = [config.k_max + 4 * i for i in range(args.windows)]
    source = clip(STREAM_CLIP, limit=(seconds[-1] + 1) * FPS)
    frames = torch.from_numpy(np.stack([source[s * FPS] for s in range(seconds[-1] + 1)])).permute(0, 3, 1, 2)
    frames = frames.contiguous().cuda()
    w = load(args.out)
    vision_cos, vision_rel = vision_parity(student, w.vision_engine, frames)
    assert vision_cos >= VISION_COSINE, f"vision tokens: minimum row cosine {vision_cos}"
    contexts = [context_for(w.engine, profile) for profile in range(3)]
    rows = []
    for loops in range(config.k_max + 1):
        row = writer_parity(w, student, chunk, contexts, frames, seconds, loops=loops)
        assert row.last_chunk == 1.0 and row.decode == 1.0 and row.cos >= STATE_COSINE, row
        print(f"L = {loops:2d}: next token agrees at the last chunk {row.last_chunk:.3f}, any chunk "
              f"{row.chunks:.3f}, decode {row.decode:.3f}; state cosine {row.cos:.5f}, relative error {row.rel:.4f}; "
              f"captions equal to float32 {row.captions:.3f}", flush=True)
        rows.append(row)
    versions = {"tensorrt": trt.__version__, "torch": torch.__version__, "transformers": transformers.__version__,
                "onnx": importlib.metadata.version("onnx")}
    built = Built(versions=versions, sha256={n: sha256(args.out / f"{n}.onnx") for n in ("vision", "writer")},
                  vision_cos=vision_cos, vision_rel=vision_rel, parity=rows)
    (args.out / "engine.json").write_text(json.dumps(asdict(built), indent=2))
    print(f"vision tokens: minimum row cosine {vision_cos:.5f}, relative error {vision_rel:.4f}; "
          f"{built.sha256}; {versions}")


if __name__ == "__main__":
    main()
