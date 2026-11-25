"""The benchmark report of a folder of answers: accuracies, differences from a reference set, gains from L = 0.

    python scripts/bench_report.py --bench <bench> --answers <answers> --captions <captions> \\
        --reference yolomt-kprotocol --out report.json

For every answer set `<answers>/<benchmark>/<name>.jsonl`: the accuracy on each benchmark and the mean of the four,
the accuracy within each task, the mean words per caption (with `--captions`), and the paired difference of the
`--reference` set minus the set. For every describer with the sets `<describer>-k0` and `<describer>-kprotocol`: its
gain from L = 0 to the protocol L and the reference describer's gain minus it. All intervals are 95% percentile
bootstraps over questions (`yolomt.stats`). Writes the full report to `--out` and prints the table.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from yolomt.report import report, table, write


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bench", type=Path, required=True, help="the benchmark folders with questions.jsonl")
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--captions", type=Path, help="the caption sets, to count their words")
    parser.add_argument("--reference", required=True, help="the answer set every other is compared with")
    parser.add_argument("--out", type=Path, required=True, help="the report, JSON")
    args = parser.parse_args()
    result = report(benchmarks=args.bench, answers=args.answers, captions=args.captions, reference=args.reference)
    write(result, args.out)
    print(table(result))


if __name__ == "__main__":
    main()
