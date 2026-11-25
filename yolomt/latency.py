"""Eager FP16 latency of one captioner on one GPU, batch 1: the paper's timing table.

A caption is timed on the device from the frames to the last token (vision encoder, projector, prefill and greedy
decode), with torch.cuda.synchronize before the clock starts and before it stops. Turning the frames into model
inputs is reported apart: for YoloMT the copy of the new frame to the device, for a baseline its processor and the
copy of the inputs.

Windows: the 1 fps, 256 x 256 frames of two public clips, as `yolomt.video` prepares them; every second t >= 16 of both clips is a
window, listed in a shuffle seeded 0 that every model and lookback L shares. A window at L holds the seconds
t - L .. t. For each L and each length, `warmup` windows are run untimed, then the next `count` are timed:
    natural   the caption stops at the end token or after 80 new tokens,
    fixed-50  exactly 50 new tokens: the end token is suppressed until then, so all models decode the same count.
The paper's counts are 5 warm-up, 100 natural and 25 fixed-50 captions, and its time is the median of fixed-50.
A baseline is asked INSTRUCTION; YoloMT reads its own prompt (`yolomt.model.prompt`). The process memory is
nvidia-smi's after each setting; the paper's is the natural-length one at L = 16.

YoloMT reads one frame at L = 0. At L > 0 the L pooled blocks of the past seconds are already cached, as in a stream:
a caption sends one new frame through the vision encoder and the pooler and prefills the prompt. A baseline reads
all L + 1 frames anew for each caption: LFM2.5-VL as L + 1 images, SmolVLM2, InternVL3.5 and Qwen3-VL as one image at
L = 0 and one video above.
"""

from __future__ import annotations

import gc
import os
import random
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, assert_never

import numpy as np
import torch
import transformers
from huggingface_hub import hf_hub_download
from numpy.typing import NDArray
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.video_utils import VideoMetadata

from yolomt.model import POOLED, SIDE, Student, decoded, embed, folder_of, load_folder, prompt, see
from yolomt.video import decode, duration, rgb

CLIPS_REPO = "raushan-testing-hf/videos-test"
CLIPS_REVISION = "4cba700bd771f44d72b549253da025c32e944d42"
CLIPS = ("Cooking_cake", "karate")
HORIZON = 16
SEED = 0
MAX_NEW = 80
FORCED = 50
INSTRUCTION = (
    "As a live narrator would, describe what is happening now: the main people, animals or objects, what they are "
    "doing, and where. Use one to three plain sentences of 30 to 50 words. State only what is visible. Do not "
    "mention the video, the frames or the camera."
)

Frames = NDArray[np.uint8]


@dataclass(frozen=True)
class Lfm:
    """L + 1 images; each 256 x 256 frame is 64 visual tokens."""


@dataclass(frozen=True)
class Smol:
    """One image at L = 0, one video above."""


@dataclass(frozen=True)
class Intern:
    """One tile at L = 0, one 448 x 448 video above."""


@dataclass(frozen=True)
class Qwen:
    """One image at L = 0, one video above."""


Family = Lfm | Smol | Intern | Qwen


@dataclass(frozen=True)
class Baseline:
    repo: str
    revision: str
    family: Family


@dataclass(frozen=True)
class Yolo:
    """YoloMT from a model folder or a Hugging Face model repo."""

    weights: str


Subject = Yolo | Baseline

BASELINES = {
    "lfm2.5-vl-450m": Baseline("LiquidAI/LFM2.5-VL-450M", "fc6221ca597f3315e4f82fc2df606783267b34ba", Lfm()),
    "lfm2.5-vl-1.6b": Baseline("LiquidAI/LFM2.5-VL-1.6B", "919fde3d022e3f90a4716006f993938ee8c2eb97", Lfm()),
    "smolvlm2-500m": Baseline("HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
                              "7b375e1b73b11138ff12fe22c8f2822d8fe03467", Smol()),
    "smolvlm2-2.2b": Baseline("HuggingFaceTB/SmolVLM2-2.2B-Instruct", "482adb537c021c86670beed01cd58990d01e72e4",
                              Smol()),
    "internvl3.5-1b": Baseline("OpenGVLab/InternVL3_5-1B-HF", "9191dbccf312b537016f041b25d61c72e7c5c9f3", Intern()),
    "internvl3.5-2b": Baseline("OpenGVLab/InternVL3_5-2B-HF", "3f301ffcf3dcbb47893afae6650ea3e78d96fb6d", Intern()),
    "qwen3-vl-2b": Baseline("Qwen/Qwen3-VL-2B-Instruct", "89644892e4d85e24eaac8bacfd4f463576704203", Qwen()),
    "qwen3-vl-4b": Baseline("Qwen/Qwen3-VL-4B-Instruct", "ebb281ec70b05090aa6165b016eac8ec08e71b17", Qwen()),
}


@dataclass(frozen=True)
class Window:
    """clip: one of CLIPS; t: the second of the newest frame."""

    clip: str
    t: int


@dataclass(frozen=True)
class Length:
    """New tokens of a caption: it stops at the end token when min_new is 0, else it has exactly min_new = max_new."""

    name: str
    min_new: int
    max_new: int
    count: int


@dataclass(frozen=True)
class Caption:
    """ms: the timed span; preprocess_ms: the span reported apart; new_tokens: generated, the end token included."""

    ms: float
    preprocess_ms: float
    prompt_tokens: int
    new_tokens: int
    text: str


@dataclass(frozen=True)
class Summary:
    n: int
    median: float
    p99: float


@dataclass(frozen=True)
class Setting:
    loops: int
    length: str
    prompt_tokens: int
    latency_ms: Summary
    preprocess_ms: Summary
    process_mib: float
    captions: list[Caption]


@dataclass(frozen=True)
class Result:
    source: str
    parameters: int
    after_load_mib: float
    settings: list[Setting]


CaptionAt = Callable[[Frames, int, Length], Caption]


def clip_frames(name: str) -> Frames:
    """(S, 256, 256, 3) uint8: the frame of each whole second 0 .. S-1 that the clip reaches."""
    path = hf_hub_download(CLIPS_REPO, f"{name}.mp4", repo_type="dataset", revision=CLIPS_REVISION)
    data = Path(path).read_bytes()
    student = decode(data, seconds=set(range(int(duration(data)) + 1)), grid=set()).student
    assert sorted(student) == list(range(len(student))), f"{name}: a second without a frame"
    return np.stack([rgb(student[s]) for s in range(len(student))])


def windows_of(clips: dict[str, Frames], *, need: int) -> list[Window]:
    every = [Window(name, t) for name in sorted(clips) for t in range(HORIZON, len(clips[name]))]
    random.Random(SEED).shuffle(every)
    assert len(every) >= need, (len(every), need)
    return every[:need]


def frames_of(clips: dict[str, Frames], window: Window, *, loops: int) -> Frames:
    """(L + 1, 256, 256, 3) the seconds t - L .. t, oldest first."""
    frames = clips[window.clip][window.t - loops:window.t + 1]
    assert frames.shape == (loops + 1, SIDE, SIDE, 3), frames.shape
    return frames


def clock() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


def process_mib() -> float:
    """This process's GPU memory as nvidia-smi lists it."""
    rows = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                          check=True, capture_output=True, text=True).stdout.strip().splitlines()
    mine = [int(mib) for pid, mib in (row.split(",") for row in rows) if int(pid) == os.getpid()]
    assert len(mine) == 1, "this process is not in nvidia-smi's list"
    return float(mine[0])


def summary(values: list[float]) -> Summary:
    return Summary(n=len(values), median=float(np.percentile(values, 50)), p99=float(np.percentile(values, 99)))


def to_device(frames: Frames) -> torch.Tensor:
    """(N, 256, 256, 3) uint8 host frames -> (N, 3, 256, 256) uint8 on the device."""
    out: torch.Tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous().cuda()
    return out


@torch.no_grad()
def yolomt_caption(student: Student, frames: Frames, *, loops: int, length: Length) -> Caption:
    """frames: (L + 1, 256, 256, 3) uint8, oldest first. The L past frames are pooled before the clock: they stand
    for the blocks the stream cached."""
    ids = torch.tensor([prompt(student, loops=loops)], device="cuda")
    cached = student.pooler(see(student, to_device(frames[:-1]))).reshape(1, loops * POOLED, -1) if loops else None
    start = clock()
    entering = to_device(frames[-1:])
    preprocess_ms = (clock() - start) * 1e3
    start = clock()
    tokens_t = see(student, entering)
    if cached is None:
        slots = tokens_t
    else:
        # the new frame's block joins the cache for the later captions
        student.pooler(tokens_t)
        slots = torch.cat([cached, tokens_t], dim=1)
    result = student.lm(inputs_embeds=embed(student, ids, slots), use_cache=True)
    cache, h = result.past_key_values, result.last_hidden_state[:, -1]
    produced: list[int] = []
    while True:
        logits = h @ student.embed.weight.T
        if len(produced) < length.min_new:
            logits[:, student.layout.im_end] = -torch.inf
        token = logits.argmax(-1)
        produced.append(int(token))
        if produced[-1] == student.layout.im_end or len(produced) == length.max_new:
            break
        step = student.lm(inputs_embeds=student.embed(token[:, None]), past_key_values=cache, use_cache=True)
        cache, h = step.past_key_values, step.last_hidden_state[:, -1]
    ms = (clock() - start) * 1e3
    text = decoded(student, [t for t in produced if t != student.layout.im_end], special=False)
    return Caption(ms, preprocess_ms, ids.shape[1], len(produced), text)


def chat_text(processor: Any, kinds: list[str]) -> str:
    content = [{"type": kind} for kind in kinds] + [{"type": "text", "text": INSTRUCTION}]
    text = processor.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True,
                                         tokenize=False)
    assert isinstance(text, str)
    return text


def inputs_of(processor: Any, family: Family, frames: Frames) -> Any:
    """(L + 1, 256, 256, 3) uint8 -> the model's inputs for one caption, on the host."""
    count = frames.shape[0]
    metadata = VideoMetadata(total_num_frames=count, fps=1.0, duration=float(count), frames_indices=list(range(count)))
    match family:
        case Lfm():
            return processor(text=[chat_text(processor, ["image"] * count)], images=list(map(Image.fromarray, frames)),
                             return_tensors="pt")
        case Smol():
            if count == 1:
                return processor(text=[chat_text(processor, ["image"])], images=[[Image.fromarray(frames[0])]],
                                 do_image_splitting=False, return_tensors="pt")
            return processor(text=[chat_text(processor, ["video"])], videos=[[frames]], video_metadata=[[metadata]],
                             do_sample_frames=False, return_tensors="pt")
        case Intern():
            if count == 1:
                return processor(text=[chat_text(processor, ["image"])], images=[Image.fromarray(frames[0])],
                                 crop_to_patches=False, return_tensors="pt")
            return processor(text=[chat_text(processor, ["video"])], videos=[frames], video_metadata=[metadata],
                             do_sample_frames=False, size={"height": 448, "width": 448}, return_tensors="pt")
        case Qwen():
            if count == 1:
                return processor(text=[chat_text(processor, ["image"])], images=[Image.fromarray(frames[0])],
                                 return_tensors="pt")
            return processor(text=[chat_text(processor, ["video"])], videos=[frames], video_metadata=[metadata],
                             do_sample_frames=False, return_tensors="pt")
        case _:
            assert_never(family)


def processor_kwargs(family: Family) -> dict[str, Any]:
    match family:
        case Lfm():
            return {"do_image_splitting": False, "max_image_tokens": 64}
        case Smol() | Intern() | Qwen():
            return {}
        case _:
            assert_never(family)


@torch.no_grad()
def baseline_caption(model: Any, processor: Any, family: Family, frames: Frames, *, length: Length) -> Caption:
    start = clock()
    inputs = inputs_of(processor, family, frames).to(device="cuda", dtype=torch.float16)
    preprocess_ms = (clock() - start) * 1e3
    start = clock()
    ids = model.generate(**inputs, do_sample=False, min_new_tokens=length.min_new or None,
                         max_new_tokens=length.max_new)
    ms = (clock() - start) * 1e3
    prompt_tokens = inputs["input_ids"].shape[1]
    new = ids[0, prompt_tokens:].tolist()
    return Caption(ms, preprocess_ms, prompt_tokens, len(new),
                   processor.tokenizer.decode(new, skip_special_tokens=True).strip())


def run_setting(caption_at: CaptionAt, clips: dict[str, Frames], windows: list[Window], *, loops: int, length: Length,
                warmup: int) -> Setting:
    gc.collect()
    torch.cuda.empty_cache()
    for window in windows[:warmup]:
        caption_at(frames_of(clips, window, loops=loops), loops, length)
    captions = [caption_at(frames_of(clips, window, loops=loops), loops, length)
                for window in windows[warmup:warmup + length.count]]
    prompt_tokens = {c.prompt_tokens for c in captions}
    assert len(prompt_tokens) == 1, prompt_tokens
    assert length.min_new == 0 or {c.new_tokens for c in captions} == {length.min_new}
    return Setting(loops=loops, length=length.name, prompt_tokens=prompt_tokens.pop(),
                   latency_ms=summary([c.ms for c in captions]),
                   preprocess_ms=summary([c.preprocess_ms for c in captions]), process_mib=process_mib(),
                   captions=captions)


def measure(subject: Subject, clips: dict[str, Frames], *, loops: list[int], warmup: int,
            lengths: list[Length]) -> Result:
    """Every L of `loops` at every length, in order, on the GPU the process sees."""
    assert all(0 <= n <= HORIZON for n in loops), loops
    windows = windows_of(clips, need=warmup + max(length.count for length in lengths))
    caption_at: CaptionAt
    model: torch.nn.Module
    match subject:
        case Yolo(weights=weights):
            folder = folder_of(weights)
            student = load_folder(folder, dtype=torch.float16).cuda().eval()
            model, source = student, str(folder)
            caption_at = lambda frames, n, length: yolomt_caption(student, frames, loops=n, length=length)
        case Baseline(repo=repo, revision=revision, family=family):
            load_processor: Any = AutoProcessor.from_pretrained
            load_model: Any = AutoModelForImageTextToText.from_pretrained
            processor = load_processor(repo, revision=revision, **processor_kwargs(family))
            baseline = load_model(repo, revision=revision, dtype=torch.float16).cuda().eval()
            model, source = baseline, f"{repo}@{revision}"
            caption_at = lambda frames, n, length: baseline_caption(baseline, processor, family, frames, length=length)
        case _:
            assert_never(subject)
    parameters = sum(p.numel() for p in model.parameters())
    after_load_mib = process_mib()
    settings = [run_setting(caption_at, clips, windows, loops=n, length=length, warmup=warmup)
                for n in loops for length in lengths]
    return Result(source, parameters, after_load_mib, settings)


def provenance() -> dict[str, Any]:
    """The code and library versions that produced a measurement."""
    root = Path(__file__).resolve().parent.parent

    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True).stdout.strip()

    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain")),
            "torch": torch.__version__, "cuda": torch.version.cuda, "transformers": transformers.__version__,
            "gpu": torch.cuda.get_device_name()}
