"""The paper's statistics: percentile bootstrap intervals over questions, always paired.

Every interval resamples a benchmark's questions with replacement 10,000 times from numpy's `default_rng(0)`: first one
pooled draw over all questions, then one draw per task within the task's questions, tasks in sorted order. The pooled
draw depends only on the number of questions, so one `Draws` per benchmark serves every comparison on it. A statistic
is the mean of a per-question vector in points (100 times the share of questions): the correctness of one caption set,
the difference of two sets' correctness, or the difference of two such differences. A comparison resamples the
difference vector itself, so both sets see the same resampled questions. The interval is the 2.5th to the 97.5th
percentile of the resampled means, the estimate is the mean itself.

The statistic across benchmarks is the unweighted mean of the benchmarks: draw b of every benchmark is averaged with
draw b of the others, the interval is the percentile of that average and the estimate the mean of the benchmarks'
estimates (not a mean pooled over questions).

A describer's gain from L = 0 to the protocol L is the difference of its correctness at the two lookbacks; the
difference of two describers' gains is the difference of their gain vectors.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

BOOTSTRAPS = 10_000
SEED = 0
Vectors = dict[str, NDArray[np.float64]]


@dataclass(frozen=True)
class Interval:
    estimate: float
    low: float
    high: float


@dataclass(frozen=True)
class Across[T]:
    """A statistic of each benchmark and of their unweighted mean."""

    benchmarks: dict[str, T]
    mean: T


@dataclass(frozen=True)
class Draws:
    """Bootstrap resamples of one benchmark's questions, as indices into its question list: `pooled` (B, n) over all n
    questions, `by_task[task]` (B, m) over the m questions of `members[task]`."""

    pooled: NDArray[np.int64]
    members: dict[str, NDArray[np.int64]]
    by_task: dict[str, NDArray[np.int64]]


def draws(tasks: Sequence[str]) -> Draws:
    """tasks[i] is the task of question i."""
    rng = np.random.default_rng(SEED)
    names = np.array(tasks)
    members = {task: np.flatnonzero(names == task) for task in sorted(set(tasks))}
    pooled = rng.integers(0, len(tasks), size=(BOOTSTRAPS, len(tasks)))
    by_task = {task: ids[rng.integers(0, len(ids), size=(BOOTSTRAPS, len(ids)))] for task, ids in members.items()}
    return Draws(pooled, members, by_task)


def points(v: NDArray[np.float64]) -> float:
    return 100 * float(v.mean())


def resampled(v: NDArray[np.float64], index: NDArray[np.int64]) -> NDArray[np.float64]:
    """The mean of v in points over each row of index, shape (B,)."""
    return 100 * v[index].mean(axis=1)


def interval(estimate: float, resamples: NDArray[np.float64]) -> Interval:
    low, high = np.percentile(resamples, [2.5, 97.5])
    return Interval(estimate, float(low), float(high))


def pooled(v: NDArray[np.float64], sample: Draws) -> Interval:
    """The interval of a per-question vector over all questions of a benchmark."""
    return interval(points(v), resampled(v, sample.pooled))


def per_task(v: NDArray[np.float64], sample: Draws) -> dict[str, Interval]:
    """The interval of a per-question vector within each task, tasks in sorted order."""
    return {task: interval(points(v[ids]), resampled(v, sample.by_task[task])) for task, ids in sample.members.items()}


def contrast(vectors: Vectors, sample: dict[str, Draws]) -> Across[Interval]:
    """The intervals of per-question vectors, one per benchmark in the benchmarks' order, and of their unweighted
    mean."""
    assert vectors.keys() == sample.keys()
    resamples = {b: resampled(v, sample[b].pooled) for b, v in vectors.items()}
    each = {b: interval(points(v), resamples[b]) for b, v in vectors.items()}
    low, high = np.percentile(np.mean(list(resamples.values()), axis=0), [2.5, 97.5])
    mean = Interval(float(np.mean([i.estimate for i in each.values()])), float(low), float(high))
    return Across(each, mean)


def difference(a: Vectors, b: Vectors) -> Vectors:
    """a - b question by question on every benchmark: a paired difference of two sets' correctness, a gain when they
    are one describer at two lookbacks, a difference of gains when they are two describers' gains."""
    assert a.keys() == b.keys() and all(a[k].shape == b[k].shape for k in a)
    return {k: a[k] - b[k] for k in a}


def mean_words(captions: Iterable[str]) -> float:
    return float(np.mean([len(caption.split()) for caption in captions]))


def check() -> None:
    """Hand-computed values on tiny fixed inputs; runs at every launch."""
    one = np.array([1.0, 1.0, 0.0, 0.0])
    assert points(one) == 50.0 and mean_words(["a b", "c", "d e f  g\nh"]) == 8 / 3 and mean_words([""]) == 0.0
    sample = draws(["a", "a", "b", "b"])
    assert sample.pooled.shape == (BOOTSTRAPS, 4) and list(sample.members) == ["a", "b"]
    assert (sample.pooled == np.random.default_rng(SEED).integers(0, 4, size=(BOOTSTRAPS, 4))).all()
    assert all(np.isin(sample.by_task[t], sample.members[t]).all() for t in "ab")
    assert pooled(one, sample).estimate == 50.0 and pooled(one, sample).low < 50.0 < pooled(one, sample).high
    assert per_task(one, sample) == {"a": Interval(100.0, 100.0, 100.0), "b": Interval(0.0, 0.0, 0.0)}
    assert pooled(np.zeros(4), sample) == Interval(0.0, 0.0, 0.0)

    def vector(*x: float) -> NDArray[np.float64]:
        return np.array(x, dtype=np.float64)

    # x: a goes 1 0 0 0 -> 1 1 1 0 (+50), b goes 1 1 0 0 -> 1 0 0 0 (-25), the gains differ by 0 2 1 0 (+75);
    # y: a goes 1 1 0 0 -> 1 0 0 0 (-25), b goes 1 1 0 0 -> 1 1 1 0 (+25), the gains differ by 0 -1 -1 0 (-50)
    a_high = {"x": vector(1, 1, 1, 0), "y": vector(1, 0, 0, 0)}
    a_low = {"x": vector(1, 0, 0, 0), "y": vector(1, 1, 0, 0)}
    b_high = {"x": vector(1, 0, 0, 0), "y": vector(1, 1, 1, 0)}
    b_low = {"x": vector(1, 1, 0, 0), "y": vector(1, 1, 0, 0)}
    each = {b: draws(["question"] * 4) for b in "xy"}
    gain_a, gain_b = contrast(difference(a_high, a_low), each), contrast(difference(b_high, b_low), each)
    both = contrast(difference(difference(a_high, a_low), difference(b_high, b_low)), each)
    assert [gain_a.benchmarks[b].estimate for b in "xy"] == [50.0, -25.0] and gain_a.mean.estimate == 12.5
    assert [gain_b.benchmarks[b].estimate for b in "xy"] == [-25.0, 25.0] and gain_b.mean.estimate == 0.0
    assert [both.benchmarks[b].estimate for b in "xy"] == [75.0, -50.0] and both.mean.estimate == 12.5
    assert all(i.low <= i.estimate <= i.high for i in (gain_a.mean, gain_b.mean, both.mean, *both.benchmarks.values()))
    # a difference of gains is the difference of the sets a_high + b_low and a_low + b_high
    left = {b: a_high[b] + b_low[b] for b in "xy"}
    right = {b: a_low[b] + b_high[b] for b in "xy"}
    assert contrast(difference(left, right), each) == both
    # every resample agrees when the questions do: a sure +100 on x, 0 on y
    sure = contrast(difference({"x": np.ones(4), "y": np.zeros(4)}, {"x": np.zeros(4), "y": np.zeros(4)}), each)
    assert sure.benchmarks == {"x": Interval(100.0, 100.0, 100.0), "y": Interval(0.0, 0.0, 0.0)}
    assert sure.mean == Interval(50.0, 50.0, 50.0)
