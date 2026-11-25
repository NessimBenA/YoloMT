"""The held-out fact-F1 of the paper: a describer's captions of the windows of the dev, test and OpenVid sets
(`data/windows/{dev,test,openvid}.tsv.gz`), scored as a share of what the teacher scores on its own.

Scored windows: the listed windows that are derived (`scripts/derive.py`) and have a ceiling.

Ceiling: the teacher scored like a describer, its first sampled caption against what its other captions agree on. With
n sampled captions 0 .. n-1 (the greedy one is n), the consensus is the claims that at least ceil(5/8 (n-1)) of the
captions 1 .. n-1 make, the union the claims that any caption but 0 makes, the greedy one included. A window whose
consensus is empty or that has no question has no ceiling.

Rows: a caption is judged by `yolomt.facts.score` against its window's target (the consensus and union over all the
teacher's captions for a describer, the leave-one-out target for the teacher): recall, precision and whether it has
at most 60 words. Fact-F1 = 2 recall precision / (recall + precision); under the 60-word rule an empty or longer
caption scores 0, in the `free` variant it keeps the F1 of its judged recall and precision.

Share = 100 sum_w f1(describer, w) / sum_w f1(teacher, w) over the scored windows. An interval resamples the windows
10,000 times (`numpy.random.default_rng(0)`), takes the ratio of sums of each resample and reports the 2.5th and 97.5th
percentiles; a difference of two shares is resampled on the same windows.

Events: the share of captions with at least one of `EVENTS` as a word, over every caption written.

Controls: the past frames a describer reads instead of its window's own, at lookback K. `Repeated`: frame t K times.
`Foreign`: the first K past frames of a partner window of another video, drawn with `random.Random(seed)` over the
window list. `Shuffled`: its own K past frames in an order drawn with `random.Random(f"{seed}/{video}@{t}/{K}")`,
never the identity.
"""

from __future__ import annotations

import gzip
import json
import math
import random
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, assert_never

import numpy as np
from numpy.typing import NDArray

from yolomt.data import Question, Target, Window, frame_name, lines, past_frames, window_of
from yolomt.facts import CONSENSUS_FRACTION, Run, score
from yolomt.teacher import HORIZON, Horizon, limits

WORDS = limits(Horizon()).words
SEED = 0
DRAWS = 10_000
EVENTS = ("then", "after", "first", "continues", "before", "next", "finally", "later", "begins", "afterwards")

Column = NDArray[np.float64]
Draws = NDArray[np.int64]


@dataclass(frozen=True)
class Claim:
    """text: what the claim states; makers: the captions of the window that make it (0 .. n-1 the samples, n the
    greedy one)."""

    text: str
    makers: tuple[int, ...]


@dataclass(frozen=True)
class Ceiling:
    caption: str
    target: Target


@dataclass(frozen=True)
class Held:
    """A scored window: the target a describer's captions are scored against and the teacher's own ceiling."""

    window: Window
    target: Target
    ceiling: Ceiling


@dataclass(frozen=True)
class Row:
    """A judged caption of the window `key` written at lookback `loops` (None for the teacher's)."""

    key: str
    loops: int | None
    recall: float
    precision: float
    within: bool


@dataclass(frozen=True)
class Judging:
    """One judging run of a set: the teacher's rows and each describer's rows by lookback, all in the same key order."""

    teacher: list[Row]
    describers: dict[str, dict[int, list[Row]]]


@dataclass(frozen=True)
class Ratio:
    """100 sum(top) / sum(bottom) over windows; top, bottom: (windows,)."""

    top: Column
    bottom: Column


@dataclass(frozen=True)
class Interval:
    estimate: float
    low: float
    high: float


@dataclass(frozen=True)
class Own:
    """The window's own past."""


@dataclass(frozen=True)
class Repeated:
    pass


@dataclass(frozen=True)
class Foreign:
    seed: int


@dataclass(frozen=True)
class Shuffled:
    seed: int


Control = Own | Repeated | Foreign | Shuffled

CONTROLS: dict[str, Control] = {"plain": Own(), "repeat": Repeated(), "foreign": Foreign(SEED),
                                "shuffled": Shuffled(SEED)}


def ceiling_of(samples: list[str], claims: list[Claim], *, mcq: list[Question]) -> Ceiling | None:
    need = math.ceil(CONSENSUS_FRACTION * (len(samples) - 1))
    consensus, union = [], []
    for claim in claims:
        others = [m for m in claim.makers if m != 0]
        if others:
            union.append(claim.text)
        if len([m for m in others if m < len(samples)]) >= need:
            consensus.append(claim.text)
    if not consensus or not mcq:
        return None
    return Ceiling(samples[0], Target(consensus, union, mcq))


def listed(path: Path) -> set[tuple[str, int]]:
    """The (video id, end second) of every window of a window list."""
    rows = (line.split("\t") for line in gzip.decompress(path.read_bytes()).decode().splitlines()[1:])
    return {(video_id, int(t)) for _, video_id, _, runs in rows for run in runs.split(";") for t in run.split(",")}


def held_out(dirs: list[Path], *, wanted: set[tuple[str, int]]) -> list[Held]:
    """The windows of the derived files under `dirs` that are in `wanted` and have a ceiling, by key."""
    found = []
    for path in sorted(p for d in dirs for p in d.rglob("derived_*.jsonl.gz")):
        for r in map(json.loads, lines(path)):
            if not r["derived"] or (r["video_id"], int(r["t1"])) not in wanted:
                continue
            window = window_of(r)
            assert window.target is not None
            claims = [Claim(c["claim"], tuple(c["makers"])) for c in r["claims"]]
            ceiling = ceiling_of(window.samples, claims, mcq=window.target.mcq)
            if ceiling is not None:
                found.append(Held(window, window.target, ceiling))
    assert len({h.window.key for h in found}) == len(found), "a window is derived twice"
    return sorted(found, key=lambda h: h.window.key)


def within(caption: str) -> bool:
    return bool(caption.strip()) and len(caption.split()) <= WORDS


def judged(run: Run, held: list[Held], captions: list[str], targets: list[Target], *, loops: int | None) -> list[Row]:
    scores = score(run, captions, targets)
    return [Row(h.window.key, loops, s.recall, s.precision, within(c))
            for h, c, s in zip(held, captions, scores, strict=True)]


def teacher_rows(run: Run, held: list[Held]) -> list[Row]:
    return judged(run, held, [h.ceiling.caption for h in held], [h.ceiling.target for h in held], loops=None)


def caption_rows(run: Run, held: list[Held], captions: dict[str, str], *, loops: int) -> list[Row]:
    missing = [h.window.key for h in held if h.window.key not in captions]
    assert not missing, f"{len(missing)} scored windows have no caption at lookback {loops}, e.g. {missing[:3]}"
    return judged(run, held, [captions[h.window.key] for h in held], [h.target for h in held], loops=loops)


def read_captions(path: Path) -> dict[int, dict[str, str]]:
    """`{"key", "loops", "caption"}` lines -> the caption of each window key, per lookback."""
    out: dict[int, dict[str, str]] = defaultdict(dict)
    for row in map(json.loads, lines(path)):
        assert row["key"] not in out[row["loops"]], f"{path}: {row['key']} twice at lookback {row['loops']}"
        out[row["loops"]][row["key"]] = row["caption"]
    return out


def write_rows(path: Path, rows: list[Row]) -> None:
    """Writes a new file: an existing one is never overwritten."""
    with path.open("x") as handle:
        handle.writelines(json.dumps(asdict(r)) + "\n" for r in rows)


def read_judging(folder: Path) -> Judging:
    """`teacher.jsonl` and one `<describer>.jsonl` per describer, as `scripts/heldout_judge.py` writes them."""
    teacher = [Row(**json.loads(line)) for line in lines(folder / "teacher.jsonl")]
    describers: dict[str, dict[int, list[Row]]] = {}
    for path in sorted(folder.glob("*.jsonl")):
        if path.stem == "teacher":
            continue
        by_loops: dict[int, list[Row]] = defaultdict(list)
        for row in (Row(**json.loads(line)) for line in lines(path)):
            assert row.loops is not None
            by_loops[row.loops].append(row)
        assert all([r.key for r in rows] == [r.key for r in teacher] for rows in by_loops.values()), \
            f"{path}: other windows than the teacher's"
        describers[path.stem] = dict(by_loops)
    return Judging(teacher, describers)


def f1(row: Row, *, free: bool) -> float:
    total = row.recall + row.precision
    return 2 * row.recall * row.precision / total if total > 0 and (free or row.within) else 0.0


def column(rows: list[Row], *, free: bool) -> Column:
    return np.array([f1(r, free=free) for r in rows])


def percent(ratio: Ratio) -> float:
    return float(100 * ratio.top.sum() / ratio.bottom.sum())


def resample(rng: np.random.Generator, windows: int, *, count: int = DRAWS) -> Draws:
    """(count, windows): per resample, the windows drawn with replacement."""
    return rng.integers(0, windows, size=(count, windows))


def difference(a: Ratio, b: Ratio, draws: Draws) -> Interval:
    """a minus b in points, with the interval of the paired resamples `draws`."""
    assert a.top.shape == a.bottom.shape == b.top.shape == b.bottom.shape == (draws.shape[1],)
    drawn = 100 * (a.top[draws].sum(1) / a.bottom[draws].sum(1) - b.top[draws].sum(1) / b.bottom[draws].sum(1))
    low, high = np.percentile(drawn, [2.5, 97.5])
    return Interval(percent(a) - percent(b), float(low), float(high))


def has_event(caption: str) -> bool:
    return any(e in re.findall(r"[a-z0-9']+", caption.lower()) for e in EVENTS)


def events(captions: dict[str, str], keys: list[str]) -> Ratio:
    """The share of the captions of `keys` with an event word."""
    return Ratio(np.array([has_event(captions[k]) for k in keys], dtype=float), np.ones(len(keys)))


def partners(windows: list[Window], rng: random.Random) -> list[int]:
    """Per window, the index of a window of another video."""
    assert len({w.video_id for w in windows}) > 1, "a foreign past needs two videos"
    out = []
    for w in windows:
        partner = rng.randrange(len(windows))
        while windows[partner].video_id == w.video_id:
            partner = rng.randrange(len(windows))
        out.append(partner)
    return out


def not_identity(order: list[int], rng: random.Random) -> list[int]:
    assert len(order) >= 2, "one frame has no other order"
    drawn = list(order)
    while drawn == order:
        rng.shuffle(drawn)
    return drawn


def control_pasts(control: Control, windows: list[Window], *, loops: int) -> list[tuple[str, ...]]:
    """Per window, the `loops` frames a describer reads as its past, as `yolomt.data.Lookback` takes them."""
    own = past_frames(windows, k_max=HORIZON)
    assert all(len(p) >= loops for p in own), "a window has fewer past frames than the lookback"
    match control:
        case Own():
            return [p[:loops] for p in own]
        case Repeated():
            return [(frame_name(w.video_id, w.t1),) * loops for w in windows]
        case Foreign(seed=seed):
            return [own[p][:loops] for p in partners(windows, random.Random(seed))]
        case Shuffled(seed=seed):
            # the paper's seed spells the end second as a float
            orders = [not_identity(list(range(loops)), random.Random(f"{seed}/{w.video_id}@{float(w.t1)}/{loops}"))
                      for w in windows]
            return [tuple(p[j] for j in order) for p, order in zip(own, orders, strict=True)]
        case _:
            assert_never(control)


def check() -> None:
    """Hand-computed values on tiny fixed inputs; runs at every launch."""
    claims = [Claim("A", (0, 1, 2, 3, 4)), Claim("B", (0, 1, 7)), Claim("C", (0,)), Claim("D", (7,)),
              Claim("E", (2, 3, 5, 6))]
    samples = [f"s{i}" for i in range(7)]
    question = [Question("q", ("w", "x", "y", "z"), 0)]
    # 7 samples need ceil(5/8 6) = 4 of the others: A and E; the union leaves out C, made by caption 0 alone
    expected = Ceiling("s0", Target(["A", "E"], ["A", "B", "D", "E"], question))
    assert ceiling_of(samples, claims, mcq=question) == expected
    assert ceiling_of(samples, claims, mcq=[]) is None and ceiling_of(samples, claims[1:4], mcq=question) is None
    assert within("a " * 60) and not within("a " * 61) and not within("  ")
    rows = [Row("k", 16, 0.5, 1.0, False), Row("k", 16, 0.5, 0.5, True), Row("k", 16, 0.0, 0.0, False)]
    assert column(rows, free=False).tolist() == [0.0, 0.5, 0.0]
    assert abs(column(rows, free=True)[0] - 2 / 3) < 1e-12 and column(rows, free=True)[1:].tolist() == [0.5, 0.0]
    assert percent(Ratio(np.array([0.5, 1.0]), np.array([1.0, 1.0]))) == 75.0
    # two windows: a resample holds (1, 1) with probability 1/4, (0, 0) with 1/4 and one of each with 1/2
    draws = resample(np.random.default_rng(SEED), 2, count=1000)
    assert difference(Ratio(np.array([1.0, 0.0]), np.ones(2)), Ratio(np.zeros(2), np.ones(2)), draws) == \
        Interval(50.0, 0.0, 100.0)
    same = resample(np.random.default_rng(SEED), 4, count=100)
    assert difference(Ratio(np.ones(4), np.full(4, 2.0)), Ratio(np.ones(4), np.full(4, 4.0)), same) == \
        Interval(25.0, 25.0, 25.0)
    assert has_event("Then he left.") and has_event("Before dawn, a man walks.") and has_event("A next-door shop.")
    assert not has_event("A man is thereafter alone.") and not has_event("")
    assert events({"a": "First, a dog.", "b": "A dog."}, ["a", "b"]).top.tolist() == [1.0, 0.0]
    assert not_identity([0, 1], random.Random(0)) == [1, 0]
    assert sorted(not_identity([0, 1, 2], random.Random(0))) == [0, 1, 2]
    items = [Window(f"{v}@{t}", v, v, t - 16, t, "eval", "", [], None, False) for v in "ab" for t in (20, 21)]
    assert control_pasts(Repeated(), items, loops=2)[0] == ("a/20.jpg", "a/20.jpg")
    assert control_pasts(Own(), items, loops=2)[1] == ("a/20.jpg", "a/19.jpg")
    assert control_pasts(Shuffled(SEED), items, loops=2)[1] == ("a/19.jpg", "a/20.jpg")
    foreign = control_pasts(Foreign(SEED), items, loops=2)
    assert all(p[0][0] != w.video_id for p, w in zip(foreign, items, strict=True))
    assert foreign[0][0] in ("b/19.jpg", "b/20.jpg")
