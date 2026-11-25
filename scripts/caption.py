"""Caption a video every second with YoloMT.

    python scripts/caption.py --weights Nessim/YoloMT --video clip.mp4 --lookback 16 --out clip.jsonl

At second t the student reads frame t and the min(L, t) frames before it, L = `--lookback` (0 to 16), and writes a
caption of the last 16 s with greedy decoding. Frames are prepared as for training (`yolomt.video`). This reference
encodes the past frames of every window again; a streaming deployment caches each frame's 16 pooled tokens instead.
`--weights` is a model folder or a Hugging Face model repo. Writes one {"second", "caption"} JSON line per second.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from yolomt.model import load_folder, read_config, folder_of, write
from yolomt.video import decode, duration, rgb


def frames_of(data: bytes) -> torch.Tensor:
    """An encoded video -> (S, 3, 256, 256) uint8, the frame of each whole second 0 .. S-1, S = floor(duration)."""
    seconds = int(duration(data))
    decoded = decode(data, seconds=set(range(seconds)), grid=set())
    assert set(decoded.student) == set(range(seconds)), "the video ends before its last whole second"
    array = np.stack([rgb(decoded.student[s]) for s in range(seconds)])
    return torch.from_numpy(array).permute(0, 3, 1, 2).contiguous()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", required=True, help="a model folder or a Hugging Face model repo")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--lookback", type=int, required=True, help="past frames each caption reads, 0 to 16")
    parser.add_argument("--batch", type=int, default=8, help="windows per forward pass")
    parser.add_argument("--device", type=torch.device, default=torch.device("cuda"))
    parser.add_argument("--out", type=Path, help="JSONL file; stdout when absent")
    args = parser.parse_args()
    folder = folder_of(args.weights)
    config = read_config(folder)
    assert 0 <= args.lookback <= config.k_max, f"the weights were trained for lookbacks 0 to {config.k_max}"
    student = load_folder(folder, dtype=torch.bfloat16).to(args.device).eval()
    frames = frames_of(args.video.read_bytes())
    # the first seconds have fewer past frames than the lookback: one batch each, then batches at the full lookback
    groups = [[t] for t in range(min(args.lookback, len(frames)))]
    steady = list(range(min(args.lookback, len(frames)), len(frames)))
    groups += [steady[i:i + args.batch] for i in range(0, len(steady), args.batch)]
    rows = []
    for seconds in groups:
        loops = min(args.lookback, seconds[0])
        pasts = torch.stack([frames[t - loops:t].flip(0) for t in seconds])
        with torch.autocast(args.device.type, dtype=torch.bfloat16):
            texts = write(student, frames[seconds].to(args.device), pasts.to(args.device), loops=loops,
                          max_new=config.max_new)
        rows += [{"second": t, "caption": text} for t, text in zip(seconds, texts)]
    text = "".join(json.dumps(row) + "\n" for row in rows)
    if args.out is None:
        sys.stdout.write(text)
    else:
        args.out.write_text(text)


if __name__ == "__main__":
    main()
