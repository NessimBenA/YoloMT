"""The fluency scale of the reward: bits per character of the teacher's captions under LFM2.5-350M.

    python scripts/calibrate.py --labels <labels>/sft/llava_* --out data/fluency.json

20,000 greedy captions of training windows are drawn (the paper's: the LLaVA-Video-178K windows of the SFT labels)
and scored as written and with their words shuffled. The reward reads `p05` as b05, `shuffled_median` as b_shuf and
the 95th percentiles of repeated content words and bigrams as tau_c and tau_b (`yolomt.reward`).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from yolomt.data import windows
from yolomt.reward import SCORER, bits_per_character, load_scorer, repeats, stutters


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", type=Path, nargs="+", required=True, help="label folders of training windows")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--captions", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    items = windows(args.labels, {"train"}, derived=False)
    captions = [w.canonical for w in random.Random(args.seed).sample(items, min(args.captions, len(items)))]
    order = random.Random(args.seed + 1)
    shuffled = []
    for caption in captions:
        words = caption.split()
        order.shuffle(words)
        shuffled.append(" ".join(words))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scorer = load_scorer(device)
    bpc = np.asarray(bits_per_character(scorer, captions))
    salad = np.asarray(bits_per_character(scorer, shuffled))
    content, pairs = (np.asarray(x) for x in zip(*(repeats(c) for c in captions)))
    result = {"scorer": SCORER, "captions": len(captions), "device": device.type,
              "p05": float(np.percentile(bpc, 5)), "median": float(np.median(bpc)),
              "p95": float(np.percentile(bpc, 95)),
              "shuffled_p05": float(np.percentile(salad, 5)), "shuffled_median": float(np.median(salad)),
              "shuffled_p95": float(np.percentile(salad, 95)),
              "stutter_rate": sum(stutters(c) for c in captions) / len(captions),
              "content_repeats_p95": int(np.percentile(content, 95)),
              "bigram_repeats_p95": int(np.percentile(pairs, 95))}
    args.out.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
