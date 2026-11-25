"""The GRPO reward of a caption y, scored in text only.

    R = g (0.4 recall + 0.3 precision + 0.2 mcq) + 0.1 fluency
    R = 0 when y has more words than the budget (60) or does not parse

recall, precision and mcq come from `yolomt.facts.score` with Qwen3.5-9B. The two fluency signals are different:

    b(y)    = -(1 / C(y)) sum_i log2 p(y_i | y_<i)       bits per character under the frozen LFM2.5-350M
    fluency = clip((b_shuf - b) / (b_shuf - b05), 0, 1)
    g       = clip((s - 1) / 4, 0, 1) / (1 + e),         e = max(0, r_c - tau_c) + max(0, r_b - tau_b)

b05 is the 5th percentile of b over the teacher's training captions and b_shuf the median b of the same captions with
their words shuffled (`data/fluency.json`, from `scripts/calibrate.py`). C(y) counts the characters of y, s is the
judge's expected 1-5 fluency rating (`yolomt.facts.rated`), r_c and r_b count the repeats of content words and of
word bigrams (`repeats`), tau_c and tau_b are their 95th percentiles over the teacher's captions. The gate g scales
the factual terms down for list-like or repetitive captions, which a gate on b alone does not catch.

A caption parses when it is non-empty, has no replacement character and no special token, repeats no word three
times in a row and has b below b_shuf. A judge reply cut off by its token limit leaves the caption unscored
(`Parts.complete` is False) and the caller skips its group.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from yolomt.data import Target
from yolomt.facts import Run, rated, score

SCORER = "LiquidAI/LFM2.5-350M"
SCORER_REVISION = "9e6c6ccf47cd318696e137d381a7ded8fe4df09f"
STOPWORDS = frozenset(
    "a an the and or of in on at to from with by for over under near into onto out up down off is are was were be "
    "being been its it this that these those there here his her their them they he she we you as while who whom which "
    "what has have had does do did one two some another other each both very just also then than so".split()
)


@dataclass(frozen=True)
class Weights:
    """The weights of the four terms and the word budget."""

    recall: float = 0.4
    precision: float = 0.3
    mcq: float = 0.2
    fluency: float = 0.1
    budget: int = 60


@dataclass(frozen=True)
class Parts:
    """One caption's reward terms. fluency: the clipped term above; bpc: b; words: whitespace-separated words; parses:
    `parses(caption)` and, once b is measured, b < b_shuf; within: words <= the budget; complete: no judge reply was
    cut off; rating: s, 0 when not rated."""

    recall: float
    precision: float
    mcq: float
    fluency: float
    bpc: float
    words: int
    parses: bool
    within: bool
    complete: bool
    reward: float
    rating: float = 0.0


@dataclass(frozen=True)
class Scorer:
    model: torch.nn.Module
    tokenizer: Any


@dataclass(frozen=True)
class Fluency:
    """The frozen scorer and the calibration of `scripts/calibrate.py`: b05, b_shuf, and tau_c, tau_b as
    content_p95 and bigram_p95."""

    scorer: Scorer
    b05: float
    b_shuffled: float
    content_p95: int
    bigram_p95: int


def repeats(caption: str) -> tuple[int, int]:
    """Occurrences past the first of each content word (`STOPWORDS` excluded) and of each word
    bigram: in-context copies are cheap for the scorer, so a caption can buy bits per character by repeating."""
    words = re.findall(r"[a-z0-9']+", caption.lower())
    content = Counter(w for w in words if w not in STOPWORDS)
    pairs = Counter(zip(words, words[1:]))
    return sum(n - 1 for n in content.values()), sum(n - 1 for n in pairs.values())


def stutters(caption: str) -> bool:
    words = re.findall(r"[a-z0-9']+", caption.lower())
    return any(a == b == c for a, b, c in zip(words, words[1:], words[2:]))


def parses(caption: str) -> bool:
    return bool(caption.strip()) and "\ufffd" not in caption and "<|" not in caption and not stutters(caption)


def load_scorer(device: torch.device) -> Scorer:
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model: torch.nn.Module = AutoModelForCausalLM.from_pretrained(SCORER, revision=SCORER_REVISION, dtype=dtype)
    model.to(device).eval()
    model.requires_grad_(False)
    return Scorer(model, AutoTokenizer.from_pretrained(SCORER, revision=SCORER_REVISION))


def load_fluency(calibration: Path, device: torch.device) -> Fluency:
    """The scorer on `device` and the calibration json (`data/fluency.json`) written by `scripts/calibrate.py`."""
    bounds = json.loads(calibration.read_text())
    return Fluency(load_scorer(device), float(bounds["p05"]), float(bounds["shuffled_median"]),
                   int(bounds["content_repeats_p95"]), int(bounds["bigram_repeats_p95"]))


@torch.no_grad()
def bits_per_character(scorer: Scorer, captions: list[str], batch: int = 128) -> list[float]:
    """b of each caption: -sum log2 p(token | preceding) over its tokens after <|startoftext|>, divided by its number
    of characters. The scorer runs in `batch`es of right-padded captions."""
    out: list[float] = []
    tokenizer = scorer.tokenizer
    device = next(scorer.model.parameters()).device
    for start in range(0, len(captions), batch):
        texts = captions[start:start + batch]
        rows = [[tokenizer.bos_token_id, *tokenizer(t, add_special_tokens=False).input_ids] for t in texts]
        width = max(len(r) for r in rows)
        ids = torch.tensor([r + [0] * (width - len(r)) for r in rows], device=device)
        mask = torch.tensor([[1] * len(r) + [0] * (width - len(r)) for r in rows], device=device)
        logits = scorer.model(input_ids=ids, attention_mask=mask).logits.float()
        logp = F.log_softmax(logits[:, :-1], dim=-1).gather(-1, ids[:, 1:, None])[..., 0]
        nats = -(logp * mask[:, 1:]).sum(-1)
        out += [float(n) / math.log(2) / max(1, len(t)) for n, t in zip(nats.tolist(), texts)]
    return out


def normalized(fluency: Fluency, bpc: float) -> float:
    """The fluency term clip((b_shuf - b) / (b_shuf - b05), 0, 1)."""
    return min(1.0, max(0.0, (fluency.b_shuffled - bpc) / (fluency.b_shuffled - fluency.b05)))


def gate(fluency: Fluency, caption: str, rating: float) -> float:
    """g = clip((s - 1) / 4, 0, 1) / (1 + e) from the judge's expected 1-5 rating s (`yolomt.facts.rated`)."""
    content, pairs = repeats(caption)
    excess = max(0, content - fluency.content_p95) + max(0, pairs - fluency.bigram_p95)
    return min(1.0, max(0.0, (rating - 1) / 4)) / (1 + excess)


def rewards(judge: Run, fluency: Fluency, captions: list[str], targets: list[Target], weights: Weights) -> list[Parts]:
    """Reward parts of each caption against its target, in the order of `captions`. The judge is asked only about
    captions that pass the cheap checks (word budget, `parses`, b < b_shuf); the others get reward 0 and no call."""
    ok = [parses(c) and len(c.split()) <= weights.budget for c in captions]
    measured = [i for i, good in enumerate(ok) if good]
    bpcs = dict(zip(measured, bits_per_character(fluency.scorer, [captions[i] for i in measured])))
    kept = [i for i in measured if bpcs[i] < fluency.b_shuffled]
    scored = dict(zip(kept, score(judge, [captions[i] for i in kept], [targets[i] for i in kept])))
    ratings = dict(zip(kept, rated(judge, [captions[i] for i in kept])))
    out = []
    for i, caption in enumerate(captions):
        words = len(caption.split())
        if i not in scored:
            fluent = i not in bpcs or bpcs[i] < fluency.b_shuffled
            out.append(Parts(0.0, 0.0, 0.0, 0.0, bpcs.get(i, 0.0), words, parses(caption) and fluent,
                             words <= weights.budget, True, 0.0))
            continue
        s = scored[i]
        fl = normalized(fluency, bpcs[i])
        facts = weights.recall * s.recall + weights.precision * s.precision + weights.mcq * s.mcq
        reward = gate(fluency, caption, ratings[i]) * facts + weights.fluency * fl
        out.append(Parts(s.recall, s.precision, s.mcq, fl, bpcs[i], words, True, True, s.complete, reward, ratings[i]))
    return out
