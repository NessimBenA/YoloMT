"""The learning-rate schedule and the exit rule every training stage shares.

Learning rate: linear warmup over the first 3% of the T planned steps, then a cosine to zero at T,

    lr(s) = peak (s + 1) / W                               s < W = max(1, round(0.03 T))
    lr(s) = peak (1 + cos(pi (s - W) / (T - W))) / 2       otherwise.

Exit rule: an evaluation that improves on the best value before it by under 0.5% (relative) is a strike, a larger
gain clears the strikes, and a stage stops at `patience` strikes in a row.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class ExitRule:
    """best: the best value so far, None before the first evaluation; last: the latest value."""

    higher_is_better: bool
    patience: int
    best: float | None = None
    last: float | None = None
    strikes: int = 0


def learning_rate(step: int, total: int, peak: float) -> float:
    ramp = max(1, round(0.03 * total))
    if step < ramp:
        return peak * (step + 1) / ramp
    progress = min(1.0, (step - ramp) / max(1, total - ramp))
    return peak * 0.5 * (1 + math.cos(math.pi * progress))


def evaluated(rule: ExitRule, value: float) -> ExitRule:
    """The rule after one evaluation of `value`."""
    if rule.best is None:
        return replace(rule, best=value, last=value)
    gain = value - rule.best if rule.higher_is_better else rule.best - value
    strikes = 0 if gain >= 0.005 * abs(rule.best) else rule.strikes + 1
    better = value > rule.best if rule.higher_is_better else value < rule.best
    return replace(rule, best=value if better else rule.best, last=value, strikes=strikes)


def stopped(rule: ExitRule) -> bool:
    return rule.strikes >= rule.patience


def improved(rule: ExitRule) -> bool:
    """Whether the latest value is the best one: the weights to keep."""
    return rule.last is not None and rule.last == rule.best
