"""The answerer and the caption sets it reads.

A caption set is `<captions>/<benchmark>/<name>.jsonl`, one JSON line per caption request with at least `video`, `t`,
`loops` and `caption`. `answer_sets` answers every question of every benchmark from each set with the answerer of
`yolomt.bench` (Mistral-Small-24B-Instruct-2501 at its pinned revision, bf16, vLLM, `llm.chat` on `bench.messages`,
temperature 0, seed 0, 8 new tokens, context 4096, one engine, one call per benchmark and set) and grades the replies
with `bench.grade`. The floor is the same question with no captions, named `floor`. It writes
`<out>/<benchmark>/<name>.jsonl`, one `bench.Answer` per question in question order, and never overwrites a file.

`cut` truncates one describer's captions, request by request, to the word count of another describer's caption for the
same request: the control that gives YoloMT the baseline's length.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download

from yolomt import bench

FLOOR = "floor"
SEED = 0
# the weights, the tokenizer and the configs of the pinned revision, without the repository's consolidated copy
ANSWERER_FILES = ["config.json", "generation_config.json", "model.safetensors.index.json", "model-*.safetensors",
                  "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]
Captions = dict[tuple[str, int], str]


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read_captions(path: Path) -> Captions:
    """A caption set: (video, t) -> caption."""
    rows = read_rows(path)
    captions = {(row["video"], row["t"]): row["caption"] for row in rows}
    assert len(captions) == len(rows), f"{path}: a (video, t) twice"
    return captions


def chats_of(questions: list[bench.Question], captions: Captions | None) -> list[list[dict[str, str]]]:
    """The answerer's chat of every question, shown the caption of each of its requests; None is the floor."""
    return [bench.messages(q, None if captions is None else tuple(captions[(c.video, c.t)] for c in q.captions))
            for q in questions]


def cut(source: list[dict[str, Any]], like: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The rows of `source` with each caption cut to its first n words, n the word count of the caption of `like` for
    the same request (video, t and loops); words are split on whitespace and joined by single spaces."""
    limits = {(r["video"], r["t"], r["loops"]): len(r["caption"].split()) for r in like}
    assert len(limits) == len(like), "a request twice in like"
    assert {(r["video"], r["t"], r["loops"]) for r in source} == limits.keys() and len(source) == len(like), \
        "source and like hold different requests"
    return [{**r, "caption": " ".join(r["caption"].split()[:limits[(r["video"], r["t"], r["loops"])]])} for r in source]


def write_cut(captions: Path, *, source: str, like: str, name: str) -> None:
    """Write the set `name` on every benchmark: the set `source` cut to the lengths of the set `like`."""
    check()
    targets = [captions / b / f"{name}.jsonl" for b in bench.BENCHMARKS]
    assert not any(t.exists() for t in targets), f"{name}.jsonl exists: caption sets are never overwritten"
    for b, target in zip(bench.BENCHMARKS, targets, strict=True):
        rows = cut(read_rows(captions / b / f"{source}.jsonl"), read_rows(captions / b / f"{like}.jsonl"))
        target.write_text("".join(json.dumps(row) + "\n" for row in rows))


def answer_sets(*, benchmarks: Path, captions: Path, names: Sequence[str], floor: bool, out: Path,
                tensor_parallel: int, memory: float) -> None:
    """Answer every benchmark from each caption set of `names` (and the floor); `benchmarks` holds
    `<benchmark>/questions.jsonl`. memory: the fraction of each GPU the engine may use."""
    import vllm

    check()
    sets = [*names, *([FLOOR] if floor else [])]
    assert sets and FLOOR not in names and len(set(sets)) == len(sets), f"sets to answer: {sets}"
    for b in bench.BENCHMARKS:
        for name in sets:
            assert not (out / b / f"{name}.jsonl").exists(), f"{out / b / name}.jsonl exists: never overwritten"
    questions = {b: bench.read_questions(benchmarks / b / "questions.jsonl") for b in bench.BENCHMARKS}
    chats = {(b, name): chats_of(questions[b], None if name == FLOOR else read_captions(captions / b / f"{name}.jsonl"))
             for b in bench.BENCHMARKS for name in sets}
    settings = {"dtype": "bfloat16", "max_model_len": bench.ANSWERER_CONTEXT, "tensor_parallel": tensor_parallel,
                "memory": memory, "seed": SEED, "temperature": 0.0, "max_tokens": bench.ANSWER_TOKENS}
    out.mkdir(parents=True, exist_ok=True)
    (out / f"run-answers-{time.strftime('%Y%m%d-%H%M%S')}.json").write_text(json.dumps(
        {"answerer": bench.ANSWERER, "answerer_revision": bench.ANSWERER_REVISION, "vllm": vllm.__version__,
         "settings": settings, "sets": sets, "prompts": sum(len(c) for c in chats.values()),
         "sample_prompt": chats[bench.BENCHMARKS[0], sets[0]][0]}, indent=1))
    folder = snapshot_download(bench.ANSWERER, revision=bench.ANSWERER_REVISION, allow_patterns=ANSWERER_FILES)
    engine = vllm.LLM(model=folder, tokenizer_mode="hf", dtype="bfloat16", max_model_len=bench.ANSWERER_CONTEXT,
                      gpu_memory_utilization=memory, tensor_parallel_size=tensor_parallel, seed=SEED)
    sampling = vllm.SamplingParams(temperature=0.0, max_tokens=bench.ANSWER_TOKENS)
    for (b, name), chat in chats.items():
        outputs = engine.chat(chat, sampling, use_tqdm=False)
        answers = [bench.grade(q, o.outputs[0].text) for q, o in zip(questions[b], outputs, strict=True)]
        (out / b).mkdir(exist_ok=True)
        (out / b / f"{name}.jsonl").write_text("".join(json.dumps(asdict(a)) + "\n" for a in answers))
        print(b, name, f"{sum(a.correct for a in answers) / len(answers):.4f}", flush=True)


def check() -> None:
    """Hand-computed values on tiny fixed inputs; runs at every launch."""
    bench.check()
    rows = [{"video": "v", "t": 3, "loops": 2, "caption": "one two  three\nfour five", "model": "m"},
            {"video": "v", "t": 4, "loops": 2, "caption": "one two", "model": "m"},
            {"video": "w", "t": 4, "loops": 0, "caption": "", "model": "m"}]
    likes = [{"video": "w", "t": 4, "loops": 0, "caption": "x y"}, {"video": "v", "t": 4, "loops": 2, "caption": ""},
             {"video": "v", "t": 3, "loops": 2, "caption": "a b c d"}]
    assert [r["caption"] for r in cut(rows, likes)] == ["one two three four", "", ""]
    assert cut(rows, likes)[0] == {**rows[0], "caption": "one two three four"}
    requests = (bench.Request("v", 7, 4), bench.Request("v", 9, 4))
    question = bench.Question("k", "x", "a", "v", 3, "Q?", ("one", "two"), 1, requests)
    assert chats_of([question], None) == [bench.messages(question, None)]
    shown = chats_of([question], {("v", 7): "first", ("v", 9): "second"})[0][1]["content"].splitlines()
    assert shown[1:3] == ["[at 4 s] first", "[at 6 s] second"]
