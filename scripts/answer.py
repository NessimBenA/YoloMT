"""Answer the benchmarks' questions from caption sets with the text-only answerer (Mistral-Small-24B-Instruct-2501).

    python scripts/answer.py --bench <bench> --captions <captions> --names yolomt-kprotocol lfm25-vl-450m-kprotocol \\
        --floor --out <answers> --tensor-parallel 2 --memory 0.9

`<bench>/<benchmark>/questions.jsonl` holds the questions and `<captions>/<benchmark>/<name>.jsonl` each caption set,
for the benchmarks ovo, mvbench, streaming and videomme. One vLLM engine answers every (benchmark, set) and writes
`<out>/<benchmark>/<name>.jsonl`, never over an existing file; `--floor` answers with no captions, as `floor`.
`<out>/run-answers-*.json` records the answerer's revision, the vLLM version, the settings and one prompt.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from yolomt.answer import answer_sets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bench", type=Path, required=True, help="the benchmark folders with questions.jsonl")
    parser.add_argument("--captions", type=Path, required=True)
    parser.add_argument("--names", nargs="*", default=[], help="the caption sets to answer, by file stem")
    parser.add_argument("--floor", action="store_true", help="also answer with no captions")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tensor-parallel", type=int, default=1, help="GPUs the engine shards the answerer over")
    parser.add_argument("--memory", type=float, default=0.9, help="the fraction of each GPU the engine may use")
    args = parser.parse_args()
    answer_sets(benchmarks=args.bench, captions=args.captions, names=args.names, floor=args.floor, out=args.out,
                tensor_parallel=args.tensor_parallel, memory=args.memory)


if __name__ == "__main__":
    main()
