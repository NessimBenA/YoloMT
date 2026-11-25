"""Labelled windows and their frames.

A label folder `<root>/<source>/<archive>/` holds, per chunk of videos, `records_XXXXX.jsonl` (one window per line:
`video_id`, `run`, `t0`, `t1`, `split`, the teacher's greedy caption `canonical` and its sampled captions `samples`),
`frames_XXXXX.tar` (the 256 x 256 JPEGs `<video_id>/<second>.jpg` of every second the windows cover) and, once
`scripts/derive.py` has run, `derived_XXXXX.jsonl.gz` (the records plus the claims and questions they are scored on).

A window ending at second t covers [t0, t]. The past frames it can read are the seconds t-1, t-2, ... that the windows
of its run cover without a gap, nearest first: up to 16 for a 16 s window, and for a 2 s window the earlier seconds
of its run of consecutive windows.
"""

from __future__ import annotations

import gzip
import io
import json
import os
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


@dataclass(frozen=True)
class Question:
    q: str
    options: tuple[str, ...]
    answer: int


@dataclass(frozen=True)
class Target:
    """What a caption of a window is scored against: the claims most teacher samples make (consensus), every claim
    any of them makes (union) and multiple-choice questions on the consensus."""

    consensus: list[str]
    union: list[str]
    mcq: list[Question]


@dataclass(frozen=True)
class Window:
    """key: `<video_id>@<t1>`; target: the derived target, None before derivation or when it failed; eligible: a
    target with a consensus and at least one question, which RL needs."""

    key: str
    video_id: str
    run: str
    t0: int
    t1: int
    split: str
    canonical: str
    samples: list[str]
    target: Target | None
    eligible: bool


@dataclass(frozen=True)
class Location:
    path: str
    offset: int
    size: int


@dataclass(frozen=True)
class FrameStore:
    """Every frame of the tars under some folders, read with one pread each."""

    locations: dict[str, Location]


def lines(path: Path) -> list[str]:
    text = gzip.decompress(path.read_bytes()).decode() if path.suffix == ".gz" else path.read_text()
    return [line for line in text.splitlines() if line.strip()]


def window_of(r: dict[str, Any]) -> Window:
    derived = bool(r.get("derived", False))
    target = (Target(r["consensus_facts"], r["union_facts"], [Question(q["q"], tuple(q["options"]), q["answer"])
                                                               for q in r["mcq"]]) if derived else None)
    return Window(key=f"{r['video_id']}@{r['t1']}", video_id=r["video_id"], run=r["run"], t0=int(r["t0"]),
                  t1=int(r["t1"]), split=r["split"], canonical=r["canonical"], samples=list(r["samples"]),
                  target=target, eligible=derived and bool(r["rl_eligible"]))


def listed(path: Path) -> dict[tuple[str, int], str]:
    """The windows of a window list (`data/windows/*.tsv.gz`): each (video id, end second t) with its split."""
    out: dict[tuple[str, int], str] = {}
    for line in lines(path)[1:]:
        _, video_id, split, runs = line.split("\t")
        out |= {(video_id, int(t)): split for run in runs.split(";") for t in run.split(",")}
    return out


def windows(dirs: list[Path], splits: set[str], *, derived: bool) -> list[Window]:
    """The windows of `splits` under the label roots `dirs`, from the records files or, with `derived`, from the
    derived files."""
    stem = "derived_" if derived else "records_"
    files = sorted(p for d in dirs for p in d.rglob(f"{stem}*.jsonl*") if p.name.endswith((".jsonl", ".jsonl.gz")))
    return [window_of(r) for p in files for r in map(json.loads, lines(p)) if r["split"] in splits]


def frame_name(video_id: str, second: int) -> str:
    return f"{video_id}/{second}.jpg"


def past_frames(items: list[Window], *, k_max: int) -> list[tuple[str, ...]]:
    """Per window, the frames t-1 s, t-2 s, ... its run covers without a gap, at most `k_max`, nearest first."""
    covered: dict[str, set[int]] = {}
    for w in items:
        covered.setdefault(w.run, set()).update(range(w.t0, w.t1 + 1))
    out = []
    for w in items:
        seconds = covered[w.run]
        depth = 0
        while depth < k_max and w.t1 - depth - 1 in seconds:
            depth += 1
        out.append(tuple(frame_name(w.video_id, w.t1 - k) for k in range(1, depth + 1)))
    return out


def frame_store(dirs: list[Path]) -> FrameStore:
    locations: dict[str, Location] = {}
    for d in dirs:
        for path in sorted(d.rglob("frames_*.tar")):
            with tarfile.open(path, "r:") as tar:
                locations |= {m.name: Location(str(path), m.offset_data, m.size) for m in tar if m.isfile()}
    return FrameStore(locations)


def image(store: FrameStore, name: str) -> torch.Tensor:
    """(3, 256, 256) uint8."""
    where = store.locations[name]
    descriptor = os.open(where.path, os.O_RDONLY)
    data = os.pread(descriptor, where.size, where.offset)
    os.close(descriptor)
    assert len(data) == where.size
    array = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
    return torch.from_numpy(array.copy()).permute(2, 0, 1)


def assert_stored(store: FrameStore, names: list[str]) -> None:
    """A frame missing from the store would otherwise stop a run only when its window comes up."""
    missing = [n for n in names if n not in store.locations]
    assert not missing, f"{len(missing)} frames are not in the frame store, e.g. {missing[:3]}"


class Lookback(Dataset[dict[str, Any]]):
    """Frame t of each window, its first loops[i] past frames (t-1 s first) and its greedy caption."""

    def __init__(self, items: list[Window], pasts: list[tuple[str, ...]], store: FrameStore, *,
                 loops: list[int]) -> None:
        assert len(items) == len(pasts) == len(loops)
        assert all(k <= len(p) for p, k in zip(pasts, loops)), "a window reads more past frames than it has"
        self.items = items
        self.pasts = pasts
        self.store = store
        self.loops = loops

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item, names = self.items[index], self.pasts[index][:self.loops[index]]
        frame_t = image(self.store, frame_name(item.video_id, item.t1))
        past = (torch.stack([image(self.store, n) for n in names]) if names
                else frame_t.new_zeros(0, *frame_t.shape))
        return {"index": index, "frame_t": frame_t, "pasts": past, "caption": item.canonical}


def collate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """frame_t: (B, 3, 256, 256); pasts: (B, L, 3, 256, 256), one lookback L per batch."""
    assert len({r["pasts"].shape[0] for r in rows}) == 1, "a batch mixes lookbacks"
    return {"index": torch.tensor([r["index"] for r in rows]), "frame_t": torch.stack([r["frame_t"] for r in rows]),
            "pasts": torch.stack([r["pasts"] for r in rows]), "caption": [r["caption"] for r in rows]}
