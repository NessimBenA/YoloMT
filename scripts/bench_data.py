"""Rebuild a benchmark subset of `yolomt.bench`: its questions, caption requests and frames.

    python scripts/bench_data.py --benchmark videomme --out <bench>/videomme --scratch <downloads>

`--benchmark` is ovo, mvbench, streaming or videomme. The annotation files and videos are downloaded from the revisions
of `data/bench.json` into `--scratch` (archives are deleted once decoded; room for the largest archives is needed:
OVO-Bench 46 GB, MVBench 17 GB, StreamingBench three 9 GB zips at a time, Video-MME none), `--workers` processes decode,
and `--out` receives questions.jsonl, requests.jsonl, frames_XXXXX.tar, frames.sha256 and manifest.json
(`yolomt.benchmarks`). The command fails when a file differs from its digest in `data/bench.json`. Video-MME needs
pyarrow (`pip install -e .[bench]`). CPU only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from yolomt.bench import BENCHMARKS, check
from yolomt.benchmarks import build


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark", choices=BENCHMARKS, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, default=Path("downloads"), help="folder for the downloaded archives")
    parser.add_argument("--workers", type=int, default=8, help="decoding processes")
    parser.add_argument("--pins", type=Path, default=Path("data/bench.json"))
    args = parser.parse_args()
    check()
    build(args.benchmark, pins=args.pins, out=args.out, scratch=args.scratch, workers=args.workers)


if __name__ == "__main__":
    main()
