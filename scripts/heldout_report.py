"""The held-out tables of the paper from judged rows: shares of the teacher's ceiling, differences, events, controls.

    python scripts/heldout_report.py --judged <judged> [<judged> ...] --captions <captions>/*.jsonl \\
        --reference yolomt [--controls]

`--judged` folders are judging runs of one set (`scripts/heldout_judge.py`); a describer's share is taken against the
teacher of its own run, and the runs must score the same windows. Printed, per describer and lookback L:
- the share of the ceiling with the 60-word rule, without it (`free`), and the share of captions over the 60 words;
- the gain from L = 0 to L = 16, and the difference to `--reference` at each L, in points of the ceiling with the
  95% interval of 10,000 resamples of the windows, paired (`yolomt.heldout`);
- from the `--captions` files, the share of captions with an event word at each L, over every caption in the file,
  and its gain from L = 0 to 16;
- with `--controls`, the shares of the describers plain, repeat, foreign and shuffled (`scripts/controls.py`) and
  their differences to plain at L = 0 and L = 16.
Every interval draws from one generator seeded with 0, in the order of this output.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from yolomt import heldout
from yolomt.heldout import Draws, Interval, Judging, Ratio


def ratios(judgings: list[Judging], *, free: bool) -> dict[str, dict[int, Ratio]]:
    out = {name: {loops: Ratio(heldout.column(rows, free=free), heldout.column(j.teacher, free=False))
                  for loops, rows in by_loops.items()}
           for j in judgings for name, by_loops in j.describers.items()}
    assert len(out) == sum(len(j.describers) for j in judgings), "two runs judge a describer of the same name"
    return out


def table(title: str, values: dict[str, dict[int, float]], lookbacks: list[int]) -> None:
    print(f"\n{title}\n{'describer':<24}" + "".join(f"{'L=' + str(k):>8}" for k in lookbacks))
    for name, by_loops in values.items():
        print(f"{name:<24}" + "".join(f"{by_loops[k]:>8.1f}" if k in by_loops else f"{'-':>8}" for k in lookbacks))


def show(label: str, i: Interval) -> None:
    print(f"{label}: {i.estimate:+.1f} [{i.low:+.1f}, {i.high:+.1f}]")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--judged", type=Path, nargs="+", required=True, help="judging runs of one set")
    parser.add_argument("--captions", type=Path, nargs="*", default=[], help="caption files, for the event words")
    parser.add_argument("--reference", help="a describer the others are compared with")
    parser.add_argument("--controls", action="store_true", help="the controls table")
    args = parser.parse_args()
    heldout.check()
    judgings = [heldout.read_judging(folder) for folder in args.judged]
    keys = [r.key for r in judgings[0].teacher]
    assert all([r.key for r in j.teacher] == keys for j in judgings), "the runs score other windows"
    capped, free = ratios(judgings, free=False), ratios(judgings, free=True)
    lookbacks = sorted({k for by_loops in capped.values() for k in by_loops})
    rng = np.random.default_rng(heldout.SEED)
    draws = heldout.resample(rng, len(keys))
    print(f"{len(keys)} windows of {len({k.rsplit('@', 1)[0] for k in keys})} videos; teacher ceiling fact-F1 "
          + ", ".join(f"{heldout.column(j.teacher, free=False).mean():.3f} ({folder.name})"
                      for j, folder in zip(judgings, args.judged, strict=True)))
    table("share of the ceiling (%), 60-word rule", {n: {k: heldout.percent(r) for k, r in by.items()}
                                                     for n, by in capped.items()}, lookbacks)
    table("share of the ceiling (%), free", {n: {k: heldout.percent(r) for k, r in by.items()}
                                             for n, by in free.items()}, lookbacks)
    table("captions over the 60 words (%)",
          {n: {k: 100 * float(np.mean([not r.within for r in rows])) for k, rows in by.items()}
           for j in judgings for n, by in j.describers.items()}, lookbacks)
    print()
    for name, by_loops in capped.items():
        if 0 in by_loops and 16 in by_loops:
            show(f"{name} gain L=0 to 16", heldout.difference(by_loops[16], by_loops[0], draws))
    if args.reference is not None:
        for name, by_loops in capped.items():
            for k in sorted(set(by_loops) & set(capped[args.reference])):
                if name != args.reference:
                    show(f"{args.reference} - {name} at L={k}",
                         heldout.difference(capped[args.reference][k], by_loops[k], draws))
    for path in args.captions:
        written = heldout.read_captions(path)
        by_event = {k: heldout.events(c, sorted(c)) for k, c in written.items()}
        print(f"\n{path.stem} captions with an event word (%): "
              + ", ".join(f"L={k} {heldout.percent(e):.1f} of {len(written[k])}" for k, e in sorted(by_event.items())))
        if 0 in by_event and 16 in by_event:
            assert sorted(written[0]) == sorted(written[16]), f"{path}: other windows at L=0 and L=16"
            show("  gain L=0 to 16", heldout.difference(by_event[16], by_event[0],
                                                         heldout.resample(rng, len(written[16]))))
    if args.controls:
        controls({name: capped[name] for name in heldout.CONTROLS}, draws)


def controls(arms: dict[str, dict[int, Ratio]], draws: Draws) -> None:
    plain = arms["plain"]
    print("\ncontrols at L=16, share of the ceiling (%): " + ", ".join(f"{n} {heldout.percent(a[16]):.1f}"
                                                                      for n, a in arms.items()))
    show("plain L=16 - plain L=0", heldout.difference(plain[16], plain[0], draws))
    for name, arm in arms.items():
        if name != "plain":
            show(f"{name} - plain L=0", heldout.difference(arm[16], plain[0], draws))
            show(f"{name} - plain L=16", heldout.difference(arm[16], plain[16], draws))


if __name__ == "__main__":
    main()
