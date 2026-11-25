"""Caption with a describer of the benchmark table: YoloMT, an open model, or the teacher.

    python scripts/describe.py --model yolomt --weights Nessim/YoloMT --bench <bench>/ovo --loops protocol \
        --out <captions>/ovo/yolomt-kprotocol.jsonl
    python scripts/describe.py --model lfm25-vl-450m --bench <bench>/ovo --loops 0 --out <captions>/ovo/lfm-k0.jsonl
    python scripts/describe.py --model yolomt --windows data/windows/test.tsv.gz --labels <labels>/test --loops 0 16 \
        --out <captions>/test.jsonl

With `--bench`, the folder (`requests.jsonl` and the frames tars of a benchmark) is captioned request by request,
at the request's own lookback (`--loops protocol`) or at a fixed K capped at it, and one JSON line per request is
written in the order of requests.jsonl: video, t, loops (the K used) and the caption. With `--windows`, the windows
of the list that the label roots (`scripts/label.py`) hold are captioned at each K of `--loops`, one line per window
and K, keyed by the window. A baseline's line also holds finish, prompt_tokens, model and wording
(`yolomt.describers`). The resolved settings, the git revision and the library versions go to `<out>.run.json`;
neither file is ever overwritten. One GPU process; `--memory` is the share of its memory vLLM may use.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from yolomt.describers import DESCRIBERS, Student, bench_rows, provenance, window_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=["yolomt", *DESCRIBERS], required=True)
    parser.add_argument("--weights", default="Nessim/YoloMT", help="YoloMT's model folder or Hugging Face model repo")
    where = parser.add_mutually_exclusive_group(required=True)
    where.add_argument("--bench", type=Path, help="a benchmark folder")
    where.add_argument("--windows", type=Path, help="a window list of data/windows")
    parser.add_argument("--labels", type=Path, nargs="+", help="the label roots of --windows")
    parser.add_argument("--loops", nargs="+", required=True,
                        help="--bench: protocol, or one K; --windows: the lookbacks K")
    parser.add_argument("--memory", type=float, default=0.9, help="the share of the GPU's memory vLLM may use")
    parser.add_argument("--out", type=Path, required=True, help="JSONL file to write")
    args = parser.parse_args()
    assert not args.out.exists(), f"{args.out} exists: captions are never overwritten"
    model = Student(args.weights) if args.model == "yolomt" else DESCRIBERS[args.model]
    if args.bench is not None:
        assert len(args.loops) == 1, "--bench takes one --loops"
        loops = None if args.loops[0] == "protocol" else int(args.loops[0])
        rows = bench_rows(args.model, model, args.bench, loops=loops, memory=args.memory)
    else:
        assert args.labels, "--windows needs --labels"
        rows = window_rows(args.model, model, args.windows, args.labels, loops=[int(k) for k in args.loops],
                           memory=args.memory)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    settings = {**vars(args), "describer": asdict(model)}
    with args.out.with_suffix(".run.json").open("x") as handle:
        handle.write(json.dumps({"settings": settings, **provenance()}, indent=2, default=str) + "\n")
    with args.out.open("x") as handle:
        handle.write("".join(json.dumps(row) + "\n" for row in rows))


if __name__ == "__main__":
    main()
