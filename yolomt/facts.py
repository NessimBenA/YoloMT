"""Claims and questions derived from the teacher's captions, and the scoring of a caption against them.

`scripts/derive.py` turns the captions of a window (up to 8 samples at temperature 0.7, then the greedy one) into
atomic facts, groups paraphrases into claims and keeps as consensus the claims made by at least ceil(5/8 n) of the n
kept samples (no consensus under `MIN_SAMPLES` = 6). The union is every claim. From the consensus it writes
`MCQ_CANDIDATES` = 8 multiple-choice questions and keeps up to `MCQ_KEEP` = 6 that the judge, given no caption,
answers right in at most 1 of the 4 `rotate`d orders of the options. A caption is scored against that target with
the same prompts (`score`):

    recall    = share of consensus claims the caption states or implies
    precision = share of the caption's own atomic facts that the union of claims states or implies
    mcq       = share of the window's questions a reader of the caption alone answers correctly
    f1        = 2 recall precision / (recall + precision)

A `Run` answers a batch of `Ask`s with `Reply`s (`yolomt.judge`). Every answer is constrained to a JSON schema, and a
reply cut off by its token limit is reported as incomplete, never parsed.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable

from yolomt.data import Question, Target

LETTERS = ("A", "B", "C", "D")
CONSENSUS_FRACTION = 5 / 8
MIN_SAMPLES = 6
MCQ_CANDIDATES = 8
MCQ_KEEP = 6

FACTS = """Break the description below into atomic facts.

Description: "{caption}"

An atomic fact is one short sentence that asserts exactly one thing: that something is there, one attribute of it (such as color, size, clothing or number), one action, or one location or relation. Split compound sentences. Name the subject in every fact instead of using pronouns. Keep the description's own words where possible and add nothing it does not say.

Example. Description: "Two men in white shirts arm-wrestle at a wooden table in a gym."
Facts: ["There are two men.", "The men wear white shirts.", "The men are arm-wrestling.", "The men are at a table.", "The table is wooden.", "The scene is a gym."]"""

CLUSTER = """Several independent descriptions of the same moment in a video were split into numbered facts, listed below by description. Group the facts that state the same thing in different words, for example "The man wears a red shirt." and "The man is dressed in a red top.".
- Only group facts that say exactly the same thing. Facts that differ in any detail, such as "There are two dogs." and "There are three dogs.", or "The dog runs." and "The dog runs on grass.", go in different groups.
- A group never holds two facts from the same description.
- A fact that matches nothing else is a group of its own.
- Every fact number appears in exactly one group.

{facts}"""

MCQ = """Several independent observers agreed on these facts about one moment in a video:
{facts}

Write {count} multiple-choice questions that check whether someone who read a short description of this moment learned these facts.
- Each question asks about one fact. Spread the questions over as many different facts as possible, and prefer details a guess cannot recover: colors, clothing, numbers, positions, what exactly someone holds or does.
- "correct" is the answer a fact states; "wrong" are three alternatives that are false for this moment.
- Someone who has not seen this moment must not be able to tell which option is correct: every wrong option must be just as likely as the correct one for a scene like this, and the question must not hint at the answer. Bad: "What tool drives the screw?" with a drill among a spoon, a comb and a book. Good, when a fact says the man holds a drill: "What is the man holding?" with a drill, a hammer, a saw and a wrench.
- Ask about the scene itself, for example "What color is the man's shirt?"; never mention a video, a description or the facts."""

ANSWER = """Description of a moment in a video: {caption}

Answer each question about that moment from the description alone. When the description does not settle a question, choose the option you find most likely.

{questions}"""

ENTAIL = """Description of a moment in a video: "{caption}"

For each numbered statement, answer "yes" if the description states it or clearly implies it, and "no" otherwise, including when the description does not mention it.

{statements}"""

SUPPORT = """Facts known about a moment in a video:
{facts}

For each numbered statement, answer "yes" if the known facts state it or clearly imply it, and "no" otherwise, including when they do not mention it.

{statements}"""

@dataclass(frozen=True)
class Ask:
    """A prompt answered as JSON under `schema`, or, when `choice` is set, as exactly one of those strings
    (a one-word JSON reply can run into whitespace the grammar allows)."""

    prompt: str
    schema: dict[str, Any]
    max_tokens: int
    choice: tuple[str, ...] = ()


@dataclass(frozen=True)
class Reply:
    """`probabilities`: for a choice ask, the judge's probability of each choice, from its token log-probabilities."""

    text: str
    complete: bool
    probabilities: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Claims:
    """claims[i] is made by the captions in makers[i] (indices into the window's caption list: the kept samples, then
    the canonical one); repaired: how many numbers and facts of the judge's grouping `parse_claims` had to fix."""

    claims: list[str]
    makers: list[frozenset[int]]
    repaired: int


@dataclass(frozen=True)
class Score:
    """facts: the number of atomic facts of the caption; complete: False when a judge reply was cut off, and the
    scores are then not to be used."""

    recall: float
    precision: float
    mcq: float
    facts: int
    complete: bool


Run = Callable[[list[Ask]], list[Reply]]


def strings(count: int | None = None) -> dict[str, Any]:
    items: dict[str, Any] = {"type": "array", "items": {"type": "string"}}
    if count is not None:
        items.update(minItems=count, maxItems=count)
    return items


def choices(values: tuple[str, ...], count: int) -> dict[str, Any]:
    return {"type": "array", "items": {"enum": list(values)}, "minItems": count, "maxItems": count}


def closed(properties: dict[str, Any]) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


def numbered(lines: list[str]) -> str:
    return "\n".join(f"{i}. {line}" for i, line in enumerate(lines, 1))


def bulleted(lines: list[str]) -> str:
    return "\n".join(f"- {line}" for line in lines)


def facts_ask(caption: str) -> Ask:
    schema = closed({"facts": {**strings(), "minItems": 1, "maxItems": 16}})
    return Ask(FACTS.format(caption=caption), schema, 400)


def parse_facts(reply: Reply) -> list[str]:
    assert reply.complete
    return [" ".join(f.split()) for f in json.loads(reply.text)["facts"] if f.strip()]


def cluster_ask(facts: list[list[str]]) -> Ask:
    """facts[c] are caption c's facts; they are numbered 1.. in caption order."""
    blocks, number = [], 1
    for c, caption in enumerate(facts):
        lines = []
        for fact in caption:
            lines.append(f"{number}. {fact}")
            number += 1
        blocks.append(f"Description {c + 1}:\n" + "\n".join(lines))
    schema = closed({"groups": {"type": "array", "minItems": 1,
                                "items": {"type": "array", "items": {"type": "integer"}, "minItems": 1}}})
    return Ask(CLUSTER.format(facts="\n\n".join(blocks)), schema, min(10 * number + 200, 3000))


def parse_claims(reply: Reply, facts: list[list[str]], prefer: int) -> Claims:
    """Claims from a partition of the numbered facts; `prefer` is the caption whose wording a claim takes when it can.

    The schema cannot say "every number exactly once, one per description", so
    the partition is repaired: numbers out of range or already used are dropped,
    a second fact from a caption already in the group starts a claim of its own,
    and a fact no group took is a claim of its own. A claim's text is a member
    fact, never a rewrite: the preferred caption's if it has one, else the
    shortest. `repaired` counts the numbers and facts touched.
    """
    assert reply.complete
    flat = [(c, fact) for c, caption in enumerate(facts) for fact in caption]
    seen: set[int] = set()
    groups: list[list[int]] = []
    repaired = 0
    for group in json.loads(reply.text)["groups"]:
        members: list[int] = []
        owners: set[int] = set()
        for number in group:
            if not 1 <= number <= len(flat) or number in seen:
                repaired += 1
                continue
            seen.add(number)
            owner = flat[number - 1][0]
            if owner in owners:
                groups.append([number])
                repaired += 1
                continue
            owners.add(owner)
            members.append(number)
        if members:
            groups.append(members)
    missing = [n for n in range(1, len(flat) + 1) if n not in seen]
    groups += [[n] for n in missing]
    claims, makers = [], []
    for members in groups:
        preferred = [n for n in members if flat[n - 1][0] == prefer]
        pick = preferred[0] if preferred else min(members, key=lambda n: len(flat[n - 1][1]))
        claims.append(flat[pick - 1][1])
        makers.append(frozenset(flat[n - 1][0] for n in members))
    return Claims(claims, makers, repaired + len(missing))


def consensus(claims: Claims, samples: int) -> list[int]:
    """Indices into `claims.claims` of those made by at least ceil(5/8 samples) of the samples, captions 0 ..
    samples-1; the canonical caption, last, never counts."""
    need = math.ceil(CONSENSUS_FRACTION * samples)
    return [i for i, makers in enumerate(claims.makers) if len({m for m in makers if m < samples}) >= need]


def mcq_ask(facts: list[str]) -> Ask:
    count = MCQ_CANDIDATES
    question = closed({"question": {"type": "string"}, "correct": {"type": "string"}, "wrong": strings(3)})
    schema = closed({"questions": {"type": "array", "items": question, "minItems": count, "maxItems": count}})
    return Ask(MCQ.format(facts=bulleted(facts), count=count), schema, 120 * count + 100)


def parse_mcq(reply: Reply, rng: random.Random) -> list[Question]:
    """Questions with the correct answer shuffled among the wrong ones; duplicates and empty options are dropped."""
    assert reply.complete
    out, asked = [], set()
    for item in json.loads(reply.text)["questions"]:
        question = " ".join(item["question"].split())
        options = [" ".join(o.split()) for o in [item["correct"], *item["wrong"]]]
        if not question or question.lower() in asked or len({o.lower() for o in options}) < 4 or not all(options):
            continue
        asked.add(question.lower())
        order = list(range(4))
        rng.shuffle(order)
        out.append(Question(question, tuple(options[i] for i in order), order.index(0)))
    return out


def rotate(question: Question, shift: int) -> Question:
    options = tuple(question.options[(i - shift) % 4] for i in range(4))
    return Question(question.q, options, (question.answer + shift) % 4)


def answer_ask(caption: str | None, questions: list[Question]) -> Ask:
    shown = f'"{caption}"' if caption is not None else "(none given)"
    blocks = [
        f"{i}. {q.q}\n" + "\n".join(f"{letter}. {option}" for letter, option in zip(LETTERS, q.options))
        for i, q in enumerate(questions, 1)
    ]
    schema = closed({"answers": choices(LETTERS, len(questions))})
    return Ask(ANSWER.format(caption=shown, questions="\n\n".join(blocks)), schema, 12 * len(questions) + 40)


def parse_answers(reply: Reply, count: int) -> list[int]:
    assert reply.complete
    answers = [LETTERS.index(a) for a in json.loads(reply.text)["answers"]]
    assert len(answers) == count
    return answers


def entail_ask(caption: str, statements: list[str]) -> Ask:
    schema = closed({"verdicts": choices(("yes", "no"), len(statements))})
    return Ask(ENTAIL.format(caption=caption, statements=numbered(statements)), schema, 8 * len(statements) + 40)


def support_ask(known: list[str], statements: list[str]) -> Ask:
    schema = closed({"verdicts": choices(("yes", "no"), len(statements))})
    return Ask(SUPPORT.format(facts=bulleted(known), statements=numbered(statements)), schema, 8 * len(statements) + 40)


def parse_verdicts(reply: Reply, count: int) -> list[bool]:
    assert reply.complete
    verdicts = [v == "yes" for v in json.loads(reply.text)["verdicts"]]
    assert len(verdicts) == count
    return verdicts


RATE = """A caption written for a moment in a video: "{caption}"

How fluent and grammatical is its English? 1: a list of words or broken fragments; 2: mostly broken; 3: understandable with clear errors; 4: minor errors; 5: fluent, natural sentences. Answer with the number."""


def rated(run: Run, captions: list[str]) -> list[float]:
    """Each caption's expected 1-5 fluency rating under the judge, sum_k k p(k), p renormalized over the five."""
    replies = run([Ask(RATE.format(caption=c), {}, 4, choice=("1", "2", "3", "4", "5")) for c in captions])
    out = []
    for r in replies:
        assert r.complete and r.text in ("1", "2", "3", "4", "5") and r.probabilities, r
        total = sum(r.probabilities.values())
        out.append(sum(int(k) * p for k, p in r.probabilities.items()) / total)
    return out


def memo(run: Run) -> Run:
    """`run` that answers a repeated ask from memory."""
    seen: dict[tuple[str, str, int], Reply] = {}

    def cached(asks: list[Ask]) -> list[Reply]:
        keys = [(a.prompt, json.dumps(a.schema, sort_keys=True), a.max_tokens) for a in asks]
        todo = {k: a for k, a in zip(keys, asks) if k not in seen}
        seen.update(zip(todo, run(list(todo.values()))))
        return [seen[k] for k in keys]

    return cached


def score(run: Run, captions: list[str], targets: list[Target]) -> list[Score]:
    """Recall, precision and MCQ accuracy of each caption against its window's target.

    Two rounds of calls: the caption's facts, its verdicts on the consensus and
    its answers go together; the union's verdicts on the facts need the facts.
    A target without consensus claims or questions scores 0 on that term.
    """
    assert len(captions) == len(targets)
    first: list[Ask] = []
    for caption, target in zip(captions, targets):
        first.append(facts_ask(caption))
        if target.consensus:
            first.append(entail_ask(caption, target.consensus))
        if target.mcq:
            first.append(answer_ask(caption, target.mcq))
    replies = iter(run(first))
    facts: list[list[str]] = []
    recall: list[float] = []
    mcq: list[float] = []
    complete: list[bool] = []
    for target in targets:
        extracted = next(replies)
        entailed = next(replies) if target.consensus else None
        answered = next(replies) if target.mcq else None
        ok = extracted.complete and (entailed is None or entailed.complete) and (answered is None or answered.complete)
        complete.append(ok)
        facts.append(parse_facts(extracted) if ok else [])
        recall.append(sum(parse_verdicts(entailed, len(target.consensus))) / len(target.consensus)
                      if ok and entailed is not None else 0.0)
        answers = parse_answers(answered, len(target.mcq)) if ok and answered is not None else []
        mcq.append(sum(a == q.answer for a, q in zip(answers, target.mcq)) / len(target.mcq) if answers else 0.0)
    second = [support_ask(t.union, f) for t, f in zip(targets, facts) if f]
    supported = iter(run(second))
    unions: list[list[bool]] = []
    for i, found in enumerate(facts):
        reply = next(supported) if found else None
        complete[i] = complete[i] and (reply is None or reply.complete)
        unions.append(parse_verdicts(reply, len(found)) if reply is not None and reply.complete else [])
    out = []
    for i, found in enumerate(facts):
        shown = sum(unions[i]) / len(found) if found and unions[i] and complete[i] else 0.0
        out.append(Score(recall[i], shown, mcq[i], len(found), complete[i]))
    return out


def f1(s: Score) -> float:
    total = s.recall + s.precision
    return 2 * s.recall * s.precision / total if total > 0 else 0.0
