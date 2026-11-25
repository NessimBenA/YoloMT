"""Time a captioner's eager FP16 latency on one GPU at lookbacks L (the paper's timing table).

    python scripts/latency.py --model yolomt --weights Nessim/YoloMT --out runs/latency/yolomt.json
    python scripts/latency.py --model lfm2.5-vl-450m --out runs/latency/lfm2.5-vl-450m.json

The models are YoloMT and the transformers baselines of the table. Mobile-VideoGPT and FastVLM run their own
repositories' code under older library versions and are not here. Windows, prompt, lengths and what is timed are in
`yolomt.latency`. Run on a GPU with room for the model, alone on it; CUDA_VISIBLE_DEVICES picks the card. The JSON
holds every timed caption, its text and the summary of each (L, length) setting.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from yolomt.latency import BASELINES, CLIPS, FORCED, MAX_NEW, Length, Yolo, clip_frames, measure, provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=["yolomt", *BASELINES], required=True)
    parser.add_argument("--weights", default="Nessim/YoloMT", help="YoloMT: a model folder or a Hugging Face repo")
    parser.add_argument("--loops", type=int, nargs="+", default=[0, 16], help="the lookbacks L, 0 to 16")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--natural", type=int, default=100, help="timed natural-length captions per setting")
    parser.add_argument("--fixed", type=int, default=25, help="timed fixed-50 captions per setting")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    subject = Yolo(args.weights) if args.model == "yolomt" else BASELINES[args.model]
    lengths = [Length("natural", 0, MAX_NEW, args.natural), Length(f"fixed-{FORCED}", FORCED, FORCED, args.fixed)]
    clips = {name: clip_frames(name) for name in CLIPS}
    result = measure(subject, clips, loops=args.loops, warmup=args.warmup, lengths=lengths)
    record = {"model": args.model, "arguments": {k: str(v) for k, v in vars(args).items()}, **provenance(),
              **asdict(result)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
