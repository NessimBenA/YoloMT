"""Caption then answer: the benchmark protocol every describer is scored by.

A describer captions a clip as a live stream, blind to the question; a text-only answerer that never sees a frame
answers each multiple-choice question from the captions alone. Only the captions differ between describers.

Benchmarks (built by `yolomt.benchmarks`)
    OVO-Bench Real-Time Visual Perception: 837 questions of 6 tasks on 237 videos.
    MVBench: 19 of its 20 tasks, 3,800 questions (Fine-grained Pose needs videos fetched by hand after registration).
    StreamingBench Real-Time Visual Understanding: 2,498 of its 2,500 questions on 500 videos (two questions fall after
    a corrupt packet of their video).
    Video-MME short (under 2 min), without subtitles: 900 questions of 12 task types on 300 videos.
    Options keep the benchmark's order; the answer is an index into them.

Caption requests (integer seconds of the video's clock; K, the paper's lookback L, is the number of past seconds)
    OVO-Bench and StreamingBench: one caption at t = floor(query time), clamped to the last decodable second, from
    seconds t - K .. t with K = min(t, 16).
    MVBench: the clip is the integer seconds in [start, end) of the question's bounds, else the whole video; first ..
    last are its seconds with a decoded frame. One caption at last with K = min(last - first, 16), and one at
    first + 16 with K = 16 when the clip is longer than 17 s.
    Video-MME: the whole video, tiled in 17 s windows: captions at first + 16, first + 33, ... (K = 16) while inside the
    clip, and one at last with K = min(last - first, 16) unless a tile ends there.
    The answerer sees a question's captions in time order, each stamped with its second from the clip's first second
    (`Question.origin`; 0 for OVO-Bench and StreamingBench). A K = 0 run caps every request at K = 0.

Answerer
    Mistral-Small-24B-Instruct-2501 at a pinned revision, bf16, vLLM, temperature 0, seed 0, at most 8 new tokens, the
    messages of `messages` (an explicit system message). The reply must be one option letter, bare, followed by "." or
    ")", or in parentheses (`parse`); anything else counts wrong. The floor is the same question with no captions.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

BENCHMARKS = ("ovo", "mvbench", "streaming", "videomme")
HORIZON = 16
TILE = HORIZON + 1
ANSWERER = "mistralai/Mistral-Small-24B-Instruct-2501"
ANSWERER_REVISION = "9527884be6e5616bdd54de542f9ae13384489724"
ANSWER_TOKENS = 8
ANSWERER_CONTEXT = 4096
LETTERS = "ABCDEFGHIJ"
SYSTEM = ("You answer multiple-choice questions about a video that you cannot see. You may be given timestamped text "
          "descriptions of it, written by a vision system. Reply with the letter of the correct option and nothing "
          "else.")


@dataclass(frozen=True)
class Request:
    """A caption to write: at second t of `video` from the frames of seconds t - loops .. t."""

    video: str
    t: int
    loops: int


@dataclass(frozen=True)
class Question:
    """origin: the second the captions' stamps count from; answer: an index into options; captions: the requests whose
    captions the answerer is shown, in time order."""

    key: str
    benchmark: str
    task: str
    video: str
    origin: int
    text: str
    options: tuple[str, ...]
    answer: int
    captions: tuple[Request, ...]


@dataclass(frozen=True)
class Answer:
    key: str
    text: str
    choice: int | None
    correct: bool


def query_request(video: str, query: float, *, last: int) -> Request:
    """The one request of an OVO-Bench or StreamingBench question asked at second `query` of a video whose last
    decodable second is `last`."""
    t = min(int(np.floor(query)), last)
    return Request(video, t, min(t, HORIZON))


def clip_requests(video: str, *, first: int, last: int) -> tuple[Request, ...]:
    """The requests of an MVBench clip of seconds first .. last."""
    assert 0 <= first <= last
    final = Request(video, last, min(last - first, HORIZON))
    if last - first > HORIZON:
        return (Request(video, first + HORIZON, HORIZON), final)
    return (final,)


def tiled_requests(video: str, *, first: int, last: int) -> tuple[Request, ...]:
    """The requests of a Video-MME clip of seconds first .. last."""
    assert 0 <= first <= last
    tiles = tuple(Request(video, t, HORIZON) for t in range(first + HORIZON, last + 1, TILE))
    if tiles and tiles[-1].t == last:
        return tiles
    return (*tiles, Request(video, last, min(last - first, HORIZON)))


def clock_seconds(text: str) -> int:
    """A StreamingBench time stamp, HH:MM:SS or MM:SS, in seconds."""
    parts = [int(part) for part in text.split(":")]
    assert len(parts) in (2, 3) and all(part >= 0 for part in parts), f"not a time stamp: {text!r}"
    hours, minutes, seconds = parts if len(parts) == 3 else (0, *parts)
    return 3600 * hours + 60 * minutes + seconds


def frame_names(request: Request, *, loops: int | None = None) -> tuple[str, ...]:
    """The frames t - K .. t of a request, oldest first, K = request.loops capped at `loops`."""
    k = request.loops if loops is None else min(loops, request.loops)
    return tuple(f"{request.video}/{s}.jpg" for s in range(request.t - k, request.t + 1))


def messages(question: Question, captions: tuple[str, ...] | None) -> list[dict[str, str]]:
    """The answerer's chat; captions[i] is the caption written for question.captions[i], None the floor."""
    options = "\n".join(f"{LETTERS[i]}. {option}" for i, option in enumerate(question.options))
    if captions is None:
        context = "No description of the video is available."
        tail = "Answer with the letter of the most likely option only."
    else:
        assert len(captions) == len(question.captions)
        stamped = "\n".join(f"[at {c.t - question.origin} s] {text}" for c, text in zip(question.captions, captions))
        context = ("Descriptions of the video, in time order. Each covers the seconds leading up to its timestamp "
                   f"(at most {HORIZON} s before it):\n{stamped}")
        tail = "Answer with the letter of the correct option only."
    user = f"{context}\n\nQuestion: {question.text}\nOptions:\n{options}\n{tail}"
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def parse(text: str, *, options: int) -> int | None:
    """The option index of a reply that is one letter, else None."""
    found = re.fullmatch(r"([A-J])[.)]?|\(([A-J])\)", text.strip())
    if found is None:
        return None
    index = LETTERS.index(found[1] or found[2])
    return index if index < options else None


def grade(question: Question, text: str) -> Answer:
    choice = parse(text, options=len(question.options))
    return Answer(question.key, text, choice, choice == question.answer)


def question_of(row: dict[str, Any]) -> Question:
    return Question(row["key"], row["benchmark"], row["task"], row["video"], row["origin"], row["text"],
                    tuple(row["options"]), row["answer"],
                    tuple(Request(c["video"], c["t"], c["loops"]) for c in row["captions"]))


def read_questions(path: Path) -> list[Question]:
    """A benchmark's `questions.jsonl`, in file order."""
    questions = [question_of(json.loads(line)) for line in path.read_text().splitlines() if line.strip()]
    assert len({q.key for q in questions}) == len(questions), f"{path}: a key twice"
    assert all(0 <= q.answer < len(q.options) <= len(LETTERS) for q in questions), f"{path}: an answer outside options"
    return questions


def read_requests(path: Path) -> list[Request]:
    """A benchmark's `requests.jsonl`: every caption to write, one per (video, t)."""
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    requests = [Request(r["video"], r["t"], r["loops"]) for r in rows]
    assert len({(r.video, r.t) for r in requests}) == len(requests), f"{path}: a (video, t) twice"
    return requests


def check() -> None:
    """Hand-computed values on tiny fixed inputs; runs at every launch."""
    assert parse("B", options=4) == 1 and parse(" (C). ", options=4) is None and parse("(C)", options=4) == 2
    assert parse("C)", options=4) == 2 and parse("(C", options=4) is None and parse("D.", options=4) == 3
    assert parse("E", options=4) is None and parse("The answer is B", options=4) is None
    assert parse("", options=4) is None and parse("b", options=4) is None and parse("A\nB", options=4) is None
    assert query_request("v", 88.43, last=500) == Request("v", 88, 16)
    assert query_request("v", 0.03, last=500) == Request("v", 0, 0)
    assert query_request("v", 90.0, last=60) == Request("v", 60, 16)
    assert clip_requests("v", first=3, last=19) == (Request("v", 19, 16),)
    assert clip_requests("v", first=3, last=20) == (Request("v", 19, 16), Request("v", 20, 16))
    assert tiled_requests("v", first=0, last=25) == (Request("v", 16, 16), Request("v", 25, 16))
    assert tiled_requests("v", first=0, last=33) == (Request("v", 16, 16), Request("v", 33, 16))
    assert tiled_requests("v", first=2, last=40) == (Request("v", 18, 16), Request("v", 35, 16), Request("v", 40, 16))
    assert all(tiled_requests("v", first=f, last=f + n) == clip_requests("v", first=f, last=f + n)
               for f in (0, 5) for n in range(33))
    assert clock_seconds("00:04:00") == 240 and clock_seconds("08:01") == 481 and clock_seconds("01:02:03") == 3723
    assert frame_names(Request("v", 5, 2)) == ("v/3.jpg", "v/4.jpg", "v/5.jpg")
    assert frame_names(Request("v", 5, 2), loops=0) == ("v/5.jpg",)
    question = Question("k", "x", "a", "v", 3, "Q?", ("one", "two"), 1, (Request("v", 7, 4),))
    assert messages(question, ("text",))[1]["content"].splitlines()[1] == "[at 4 s] text"
    assert messages(question, None)[1]["content"].startswith("No description of the video is available.")
    assert grade(question, "B") == Answer("k", "B", 1, True) and grade(question, "C") == Answer("k", "C", None, False)
