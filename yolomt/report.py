"""The benchmark report: what the paper tabulates from a folder of answers.

Reads `<benchmarks>/<benchmark>/questions.jsonl` and `<answers>/<benchmark>/<name>.jsonl` for every answer set (one file
per set on each of the four benchmarks; each stored reply is graded again with `bench.grade` and must equal the stored
answer), and, to count words, `<captions>/<benchmark>/<name>.jsonl`. For every set:

- the accuracy on each benchmark and the unweighted mean of the four, in points, with percentile intervals,
- the accuracy within each task of each benchmark, with intervals,
- the mean words per caption on each benchmark and their mean (not for the floor).

Against the reference set: the paired difference reference minus every other set on each benchmark and on the mean of
the four. For every describer answered at both lookbacks, `<describer>-k0` and `<describer>-kprotocol` (L = 0 and the
protocol L): its gain from L = 0 to the protocol L, and the reference describer's gain minus its gain. Intervals are
those of `yolomt.stats`.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from yolomt import bench, stats
from yolomt.answer import FLOOR, read_captions, read_rows
from yolomt.stats import Across, Interval

PROTOCOL = "-kprotocol"
NO_PAST = "-k0"


@dataclass(frozen=True)
class Scores:
    """One answer set: accuracy; the accuracy within each task, by benchmark; words per caption (None without
    captions)."""

    accuracy: Across[Interval]
    tasks: dict[str, dict[str, Interval]]
    words: Across[float] | None


@dataclass(frozen=True)
class Report:
    """over[name]: reference minus set `name`; gains[describer]: protocol minus L = 0; over_gains[describer]: the
    reference describer's gain minus the describer's."""

    reference: str
    sets: dict[str, Scores]
    over: dict[str, Across[Interval]]
    gains: dict[str, Across[Interval]]
    over_gains: dict[str, Across[Interval]]


def correct_of(questions: list[bench.Question], path: Path) -> NDArray[np.float64]:
    """1.0 per correct answer in question order; the stored replies are graded again and must equal the stored rows."""
    rows = read_rows(path)
    graded = [bench.grade(q, row["text"]) for q, row in zip(questions, rows, strict=True)]
    assert [asdict(a) for a in graded] == rows, f"{path}: not the answers to the benchmark's questions, in order"
    return np.array([float(a.correct) for a in graded])


def describers(names: Sequence[str]) -> list[str]:
    """The describers with a set at both lookbacks."""
    return [n.removesuffix(PROTOCOL) for n in names
            if n.endswith(PROTOCOL) and n.removesuffix(PROTOCOL) + NO_PAST in names]


def scores(name: str, correct: stats.Vectors, sample: dict[str, stats.Draws], captions: Path | None) -> Scores:
    tasks = {b: stats.per_task(correct[b], sample[b]) for b in bench.BENCHMARKS}
    if captions is None or name == FLOOR:
        return Scores(stats.contrast(correct, sample), tasks, None)
    each = {b: stats.mean_words(read_captions(captions / b / f"{name}.jsonl").values()) for b in bench.BENCHMARKS}
    return Scores(stats.contrast(correct, sample), tasks, Across(each, float(np.mean(list(each.values())))))


def report(*, benchmarks: Path, answers: Path, captions: Path | None, reference: str) -> Report:
    """`captions` is None to leave the words out."""
    check()
    questions = {b: bench.read_questions(benchmarks / b / "questions.jsonl") for b in bench.BENCHMARKS}
    names = sorted(p.stem for p in (answers / bench.BENCHMARKS[0]).glob("*.jsonl"))
    assert all(sorted(p.stem for p in (answers / b).glob("*.jsonl")) == names for b in bench.BENCHMARKS), \
        "the benchmarks have different answer sets"
    assert reference in names, f"{reference} is not an answer set: {names}"
    sample = {b: stats.draws([q.task for q in questions[b]]) for b in bench.BENCHMARKS}
    correct = {name: {b: correct_of(questions[b], answers / b / f"{name}.jsonl") for b in bench.BENCHMARKS}
               for name in names}
    gain = {d: stats.difference(correct[d + PROTOCOL], correct[d + NO_PAST]) for d in describers(names)}
    own = reference.removesuffix(PROTOCOL)
    compared = [d for d in gain if reference.endswith(PROTOCOL) and own in gain and d != own]
    return Report(
        reference,
        {name: scores(name, correct[name], sample, captions) for name in names},
        {name: stats.contrast(stats.difference(correct[reference], correct[name]), sample)
         for name in names if name != reference},
        {d: stats.contrast(g, sample) for d, g in gain.items()},
        {d: stats.contrast(stats.difference(gain[own], gain[d]), sample) for d in compared})


def cell(x: Interval) -> str:
    return f"{x.estimate:+.1f} [{x.low:+.1f}, {x.high:+.1f}]"


def table(r: Report) -> str:
    """The accuracy of every set, best first, and its difference from the reference; then the gains."""
    columns = "".join(f"{b:>10}" for b in (*bench.BENCHMARKS, "mean"))
    lines = [f"{'set':<42}{columns}{'words':>8}  reference minus set"]
    for name, s in sorted(r.sets.items(), key=lambda item: -item[1].accuracy.mean.estimate):
        accuracy = [*(s.accuracy.benchmarks[b].estimate for b in bench.BENCHMARKS), s.accuracy.mean.estimate]
        words = "" if s.words is None else f"{s.words.mean:.1f}"
        over = "" if name == r.reference else cell(r.over[name].mean)
        lines.append(f"{name:<42}" + "".join(f"{a:>10.1f}" for a in accuracy) + f"{words:>8}  {over}")
    if r.gains:
        lines += ["", f"{'describer':<42}{'gain from L = 0':>24}  reference's gain minus its gain"]
        lines += [f"{d:<42}{cell(g.mean):>24}  {cell(r.over_gains[d].mean) if d in r.over_gains else ''}"
                  for d, g in r.gains.items()]
    return "\n".join(lines)


def write(r: Report, path: Path) -> None:
    path.write_text(json.dumps(asdict(r), indent=1))


def check() -> None:
    """Hand-computed values on tiny fixed inputs; runs at every launch."""
    bench.check()
    stats.check()
    assert describers(["a-k0", "a-kprotocol", "b-kprotocol", "floor", "c-k0", "d-k4", "d-kprotocol"]) == ["a"]
    assert cell(Interval(-1.6955, -3.049, -0.376)) == "-1.7 [-3.0, -0.4]"
