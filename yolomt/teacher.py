"""The teacher, Qwen3.8-27B-FP8 on vLLM: what it sees of a window and what it is asked.

- 2 s window (the labels of stage 1 and 2): covers [t-2, t]. The teacher sees the grid frames from t-5.5 s to t+1 s
  as far as the video has them, at 2 fps, and describes [t-2, t] in 15 to 30 words.
- 16 s window (the labels of stage 3 and 4): covers [t0, t], t0 = max(0, t-16). The teacher sees exactly the
  student's frames of t0 ... t as a 1 fps clip, with no look-ahead, and narrates the whole span, ending with now, in
  30 to 50 words. Qwen merges frames in temporal pairs, so the clip has an odd count, the oldest frame doubled when
  needed, and frame t stands alone in the last pair.

A caption is kept when the teacher ends it, it has at most 36 (2 s) or 60 (16 s) words and at most 48 or 80 student
tokens; a window whose greedy caption is not kept is dropped.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, assert_never

import numpy as np
from numpy.typing import NDArray

from yolomt.video import FPS, SIDE, Decoded, rgb

TEACHER = "Qwen/Qwen3.8-27B-FP8"
TEACHER_REVISION = "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a"
SHORT_PROMPT = (
    "This video clip is {duration:.1f} seconds long. Describe what is happening from {a:.1f} to {b:.1f} "
    "seconds, as a live narrator would at the {b:.1f} second mark: the main people, animals or objects, "
    "what they are doing, and where. Use one or two plain sentences of 15 to 30 words. State only what "
    "is visible. Do not mention the video, the clip, the camera or timestamps."
)
HORIZON_PROMPT = (
    "This video clip is {duration:.1f} seconds long and its last moment is now. As a live narrator would at the "
    "{duration:.1f} second mark, describe what happened over the whole clip, in the order it happened, ending with "
    "what is happening now: the main people, animals or objects, what they did, and where. Use one to three plain "
    "sentences of 30 to 50 words. State only what is visible. Do not mention the video, the clip, the camera or "
    "timestamps."
)
# the 2 s teacher's view, in seconds from the window's end t
SHORT_VIEW = (-5.5, 1.0)
HORIZON = 16


@dataclass(frozen=True)
class Short:
    """2 s windows."""


@dataclass(frozen=True)
class Horizon:
    """16 s windows."""


Task = Short | Horizon


@dataclass(frozen=True)
class Limits:
    """A kept caption's most words and student tokens; the teacher may write 2 tokens per allowed word."""

    words: int
    tokens: int


def limits(task: Task) -> Limits:
    match task:
        case Short():
            return Limits(words=36, tokens=48)
        case Horizon():
            return Limits(words=60, tokens=80)
        case _:
            assert_never(task)


def start(task: Task, t: int) -> int:
    """t0 of the window ending at t: the first second it covers."""
    match task:
        case Short():
            return t - 2
        case Horizon():
            return max(0, t - HORIZON)
        case _:
            assert_never(task)


def grid_span(task: Task, t: int) -> range:
    """The grid indices the teacher's view of the window ending at t asks for."""
    match task:
        case Short():
            return range(max(0, round((t + SHORT_VIEW[0]) * FPS)), round((t + SHORT_VIEW[1]) * FPS) + 1)
        case Horizon():
            return range(0)
        case _:
            assert_never(task)


def horizon_request(frames: Mapping[int, NDArray[np.uint8]], t: int, *, lookback: int, template: str) -> dict[str, Any]:
    """vLLM's input for the 16 s view of seconds t - lookback .. t, `frames[s]` being (H, W, 3) uint8: the chat
    `template` with PROMPT_TEXT replaced, the view and its timing (frames_indices place each frame in the clip) and the
    pixel budget, which keeps the view's resolution. A lookback of 0 is frame t twice, a second apart: a clip is at
    least one temporal pair."""
    t0 = t - lookback
    if lookback == 0:
        shown, stamps, span = [t, t], [0, 1], 1
    else:
        shown = list(range(t0, t + 1))
        if len(shown) % 2 == 0:
            shown.insert(0, t0)
        stamps, span = [s - t0 for s in shown], lookback
    text = HORIZON_PROMPT.format(duration=float(span))
    metadata = {"fps": 1.0, "duration": float(span + 1), "total_num_frames": len(shown), "frames_indices": stamps}
    # Qwen's resize bounds frames x height x width of the whole clip: the longest view keeps 256 x 256
    pixels = {"min_pixels": 64 * 32 * 32, "max_pixels": (HORIZON + 3) * SIDE * SIDE}
    return {"prompt": template.replace("PROMPT_TEXT", text),
            "multi_modal_data": {"video": (np.stack([frames[s] for s in shown]),
                                           {**metadata, "video_backend": "pyav", "do_sample_frames": False})},
            "mm_processor_kwargs": pixels}


def request(task: Task, decoded: Decoded, t: int, template: str) -> dict[str, Any]:
    """vLLM's input for the window ending at t: the chat `template` with PROMPT_TEXT replaced, the view and its
    timing (frames_indices place each frame in the clip) and the pixel budget, which keeps the view's resolution."""
    match task:
        case Short():
            span = grid_span(task, t)
            last = min(decoded.last, span[-1])
            frames = np.stack([decoded.grid[k] for k in range(span[0], last + 1)])
            offset = span[0] / FPS
            text = SHORT_PROMPT.format(duration=len(frames) / FPS, a=t - 2 - offset, b=t - offset)
            metadata = {"fps": FPS, "duration": len(frames) / FPS, "total_num_frames": len(frames),
                        "frames_indices": list(range(len(frames)))}
            pixels = {"min_pixels": 64 * 32 * 32, "max_pixels": 160 * 32 * 32}
        case Horizon():
            t0 = start(task, t)
            return horizon_request({s: rgb(decoded.student[s]) for s in range(t0, t + 1)}, t, lookback=t - t0,
                                   template=template)
        case _:
            assert_never(task)
    return {"prompt": template.replace("PROMPT_TEXT", text),
            "multi_modal_data": {"video": (frames, {**metadata, "video_backend": "pyav", "do_sample_frames": False})},
            "mm_processor_kwargs": pixels}


def kept(text: str, tokenizer: Any, task: Task) -> bool:
    """Whether a caption the teacher ended passes the length filters."""
    bound = limits(task)
    words = text.split()
    return 0 < len(words) <= bound.words and len(tokenizer(text, add_special_tokens=False).input_ids) <= bound.tokens
