"""Captions of a held-out set read from each control past, for the controls table of `scripts/heldout_report.py`.

    python scripts/controls.py --weights Nessim/YoloMT --labels <labels>/dev --windows data/windows/dev.tsv.gz \\
        --out <captions>

Each scored window of the set (`yolomt.heldout`) is captioned at the lookback L = 16 with greedy decoding, as
`scripts/caption.py` does, from frame t and one of four pasts: `plain`, its own (also at L = 0), `repeat`, frame t
sixteen times, `foreign`, the past of a window of another video, `shuffled`, its own past in another order
(`yolomt.heldout.control_pasts`). `--out`/<control>.jsonl receives one {"key", "loops", "caption"} line per window and
lookback; `--out` must not exist. Needs a GPU.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import assert_never

import torch
from torch.utils.data import DataLoader

from yolomt import heldout
from yolomt.data import Lookback, assert_stored, collate, frame_name, frame_store
from yolomt.heldout import Control, Foreign, Own, Repeated, Shuffled
from yolomt.model import folder_of, load_folder, read_config, write
from yolomt.teacher import HORIZON


def lookbacks(control: Control) -> tuple[int, ...]:
    match control:
        case Own():
            return (0, HORIZON)
        case Repeated() | Foreign() | Shuffled():
            return (HORIZON,)
        case _:
            assert_never(control)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", required=True, help="a model folder or a Hugging Face model repo")
    parser.add_argument("--labels", type=Path, nargs="+", required=True, help="derived label roots of the set")
    parser.add_argument("--windows", type=Path, required=True, help="the set's window list of data/windows")
    parser.add_argument("--out", type=Path, required=True, help="a new folder for the caption files")
    parser.add_argument("--batch", type=int, default=32, help="windows per forward pass")
    parser.add_argument("--device", type=torch.device, default=torch.device("cuda"))
    args = parser.parse_args()
    heldout.check()
    windows = [h.window for h in heldout.held_out(args.labels, wanted=heldout.listed(args.windows))]
    store = frame_store(args.labels)
    folder = folder_of(args.weights)
    config = read_config(folder)
    student = load_folder(folder, dtype=torch.bfloat16).to(args.device).eval()
    args.out.mkdir()
    for name, control in heldout.CONTROLS.items():
        with (args.out / f"{name}.jsonl").open("x") as handle:
            for loops in lookbacks(control):
                pasts = heldout.control_pasts(control, windows, loops=loops)
                assert_stored(store, [n for p in pasts for n in p] + [frame_name(w.video_id, w.t1) for w in windows])
                data = DataLoader(Lookback(windows, pasts, store, loops=[loops] * len(windows)),
                                  batch_size=args.batch, collate_fn=collate)
                captions: list[str] = []
                for batch in data:
                    with torch.autocast(args.device.type, dtype=torch.bfloat16):
                        captions += write(student, batch["frame_t"].to(args.device), batch["pasts"].to(args.device),
                                          loops=loops, max_new=config.max_new)
                handle.writelines(json.dumps({"key": w.key, "loops": loops, "caption": c}) + "\n"
                                  for w, c in zip(windows, captions, strict=True))
                print(f"{name} L={loops}: {len(captions)} captions", flush=True)


if __name__ == "__main__":
    main()
