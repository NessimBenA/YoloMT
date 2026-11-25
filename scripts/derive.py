"""Claims and questions of the RL windows, derived from their teacher captions by Qwen3.5-9B.

    python scripts/derive.py --labels <labels>/rl <labels>/val

For every `records_XXXXX.jsonl` under `--labels` without its `derived_XXXXX.jsonl.gz`, in batches of
`--batch-windows` windows, in four rounds of judge calls:

1. the atomic facts of each caption of a window (its kept samples, then the greedy caption);
2. one grouping of the window's facts into claims, each with the captions that make it; the union is every claim, the
   consensus the claims made by at least ceil(5/8 n) of the n kept samples (none when fewer than 6 are kept);
3. 8 multiple-choice questions on the consensus, options shuffled;
4. the leakage test: the judge answers each question with no caption under the 4 rotations of its options, and a
   question it gets right more than once is dropped; up to 6 are kept.

A derived file holds the records plus `derived` (false when a reply was cut off), `facts`, `claims`,
`consensus_facts`, `union_facts`, `mcq` and `rl_eligible` (a consensus and a question): RL trains on eligible windows.
Needs vLLM and a GPU.
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
from pathlib import Path
from typing import Any

from yolomt.data import Question, lines
from yolomt.facts import (MCQ_KEEP, MIN_SAMPLES, Claims, Run, answer_ask, cluster_ask, consensus, facts_ask, mcq_ask,
                          parse_answers, parse_claims, parse_facts, parse_mcq, rotate)
from yolomt.judge import QWEN, offline


def captions(record: dict[str, Any]) -> list[str]:
    """The kept samples, then the greedy caption."""
    return [*record["samples"], record["canonical"]]


def derive(run: Run, records: list[dict[str, Any]], *, seed: int) -> list[dict[str, Any]]:
    """The derived record of each record, in order; `seed` and the window shuffle each question's options."""
    replies = iter(run([facts_ask(c) for r in records for c in captions(r)]))
    facts: list[list[list[str]] | None] = []
    for r in records:
        found = [next(replies) for _ in captions(r)]
        facts.append([parse_facts(f) for f in found] if all(f.complete for f in found) else None)

    replies = iter(run([cluster_ask(f) for f in facts if f is not None and any(f)]))
    claims: list[Claims | None] = []
    for r, f in zip(records, facts):
        if f is None or not any(f):
            claims.append(None)
            continue
        reply = next(replies)
        claims.append(parse_claims(reply, f, prefer=len(r["samples"])) if reply.complete else None)
    agreed = [consensus(c, len(r["samples"])) if c is not None and len(r["samples"]) >= MIN_SAMPLES else []
              for r, c in zip(records, claims)]

    replies = iter(run([mcq_ask([c.claims[i] for i in picked]) for c, picked in zip(claims, agreed) if c and picked]))
    questions: list[list[Question]] = []
    for r, c, picked in zip(records, claims, agreed):
        if not (c and picked):
            questions.append([])
            continue
        reply = next(replies)
        key = f"{r['video_id']}@{r['t1']}"
        questions.append(parse_mcq(reply, random.Random(f"{seed}:{key}")) if reply.complete else [])

    blind = [answer_ask(None, [rotate(q, shift) for q in qs]) for qs in questions if qs for shift in range(4)]
    replies = iter(run(blind))
    kept: list[list[Question]] = []
    for qs in questions:
        if not qs:
            kept.append([])
            continue
        rotations = [next(replies) for _ in range(4)]
        if not all(reply.complete for reply in rotations):
            kept.append([])
            continue
        right = [0] * len(qs)
        for shift, reply in enumerate(rotations):
            for j, answer in enumerate(parse_answers(reply, len(qs))):
                right[j] += int(answer == rotate(qs[j], shift).answer)
        kept.append([q for q, n in zip(qs, right) if n <= 1][:MCQ_KEEP])

    out = []
    for r, f, c, picked, qs in zip(records, facts, claims, agreed, kept):
        if f is None or c is None:
            out.append({**r, "derived": False})
            continue
        out.append({**r, "derived": True, "facts": f,
                    "claims": [{"claim": claim, "makers": sorted(m)} for claim, m in zip(c.claims, c.makers)],
                    "consensus_facts": [c.claims[i] for i in picked], "union_facts": list(c.claims),
                    "mcq": [{"q": q.q, "options": list(q.options), "answer": q.answer} for q in qs],
                    "rl_eligible": bool(picked and qs)})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", type=Path, nargs="+", required=True, help="label roots with sampled captions")
    parser.add_argument("--batch-windows", type=int, default=1500, help="windows per batch of judge calls")
    parser.add_argument("--max-num-seqs", type=int, default=512, help="vLLM's concurrent sequences")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run = offline(QWEN, max_model_len=8192, memory=args.gpu_memory_utilization, max_num_seqs=args.max_num_seqs)

    def target(path: Path) -> Path:
        return path.with_name(path.name.split(".")[0].replace("records_", "derived_") + ".jsonl.gz")

    todo = [p for d in args.labels for p in sorted(d.rglob("records_*.jsonl")) if not target(p).exists()]
    while todo:
        batch: list[Path] = []
        records: list[dict[str, Any]] = []
        while todo and len(records) < args.batch_windows:
            batch.append(todo.pop(0))
            records += [json.loads(line) for line in lines(batch[-1])]
        derived = iter(derive(run, records, seed=args.seed))
        for path in batch:
            rows = [json.dumps(next(derived)) for _ in lines(path)]
            partial = target(path).with_suffix(".tmp")
            partial.write_bytes(gzip.compress("".join(row + "\n" for row in rows).encode()))
            partial.rename(target(path))


if __name__ == "__main__":
    main()
