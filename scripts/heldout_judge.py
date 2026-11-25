"""Judge the teacher's ceiling and the captions of a held-out set with phi-4.

    python scripts/heldout_judge.py --labels <labels>/test --windows data/windows/test.tsv.gz \\
        --captions <captions>/yolomt.jsonl <captions>/other.jsonl --out <judged>

The scored windows are the listed windows that are derived and have a ceiling (`yolomt.heldout`). A caption file holds
`{"key", "loops", "caption"}` lines; every scored window needs a caption at each of its lookbacks. phi-4 is served by
`vllm serve` on `--device`, greedy and constrained to JSON, and scores the teacher first, then each file's captions
lookback by lookback: `--out`/teacher.jsonl and `--out`/<file name>.jsonl receive one `yolomt.heldout.Row` per window
and lookback, read by `scripts/heldout_report.py`. A run judges the teacher again, so a share is always taken
against the teacher of its own run; `--out` must not exist. Needs vLLM and a GPU with room for the 14B judge.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from yolomt import heldout
from yolomt.facts import Ask, Reply
from yolomt.judge import HELD_OUT, Plan, ask, launch, stop


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", type=Path, nargs="+", required=True, help="derived label roots of the set")
    parser.add_argument("--windows", type=Path, required=True, help="the set's window list of data/windows")
    parser.add_argument("--captions", type=Path, nargs="*", default=[], help="caption files, one per describer")
    parser.add_argument("--out", type=Path, required=True, help="a new folder for the rows")
    parser.add_argument("--device", default="0", help="the CUDA device that serves the judge")
    parser.add_argument("--memory", type=float, default=0.98, help="the share of the device the judge may take")
    parser.add_argument("--port", type=int, default=8775)
    args = parser.parse_args()
    heldout.check()
    wanted = heldout.listed(args.windows)
    held = heldout.held_out(args.labels, wanted=wanted)
    captions = {path.stem: heldout.read_captions(path) for path in args.captions}
    assert len(captions) == len(args.captions), "two caption files share a name"
    print(f"{len(held)} scored windows of {len({h.window.video_id for h in held})} videos, {len(wanted)} listed",
          flush=True)
    args.out.mkdir()
    plan = Plan(HELD_OUT, port=args.port, max_model_len=4096, memory=args.memory, max_num_seqs=16,
                max_num_batched_tokens=2048, template={}, threads=64)
    server = launch(plan, device=args.device, log=str(args.out / "judge.log"))

    def run(asks: list[Ask]) -> list[Reply]:
        return ask(server, asks)

    heldout.write_rows(args.out / "teacher.jsonl", heldout.teacher_rows(run, held))
    for name, by_loops in captions.items():
        rows = [row for loops in sorted(by_loops)
                for row in heldout.caption_rows(run, held, by_loops[loops], loops=loops)]
        heldout.write_rows(args.out / f"{name}.jsonl", rows)
        print(f"{name}: {len(rows)} rows at lookbacks {sorted(by_loops)}", flush=True)
    stop(server)


if __name__ == "__main__":
    main()
