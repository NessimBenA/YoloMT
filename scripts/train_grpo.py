"""Stage 4, GRPO with a KL penalty to the SFT model; its best checkpoint is YoloMT.

    CUDA_VISIBLE_DEVICES=0 python scripts/train_grpo.py --labels <labels>/rl --val <labels>/val \\
        --init <runs>/sft/best --out <runs>/grpo --judge-gpu 1

The student trains on `--device`; the reward judge (Qwen3.5-9B) and the held-out judge (phi-4) take turns on the
physical GPU `--judge-gpu`, served by vLLM. A step draws 64 eligible windows (`scripts/derive.py`) and one lookback
per window, L = m with probability 0.5 and uniform over 0 .. m otherwise, m = min(16, L_avail); the 8 captions of a
window, sampled at temperature 1, all read that L, so an advantage compares captions that read the same frames. With
R_i the reward of caption i (`yolomt.reward`) and t over its tokens:

    A_i   = (R_i - mean R) / std R            over the group; a group whose rewards do not vary is skipped
    rho_t = exp(log p_t - log p_old_t)        p_old: the policy that sampled the caption
    k3_t  = exp(d_t) - d_t - 1                d_t = log p_ref_t - log p_t, p_ref the frozen SFT model
    loss  = (1 / N) sum_i sum_t [ -min(rho_t A_i, clip(rho_t, 0.8, 1.2) A_i) + 0.04 k3_t ]

N the tokens of the kept captions. AdamW at 1e-6 (the vision encoder at 2e-7), gradient norm clipped to 1. Once more
than half of the groups were skipped on 3 steps in a row, windows are drawn in proportion to how much their last
rewards varied.

Every 50 steps the greedy captions at L = 16 of the validation windows with 16 past frames are scored by phi-4: the
fact-F1 times the gate g of `yolomt.reward`. That score selects `--out`/best (a model folder) and drives the exit
rule (3 evaluations without a 0.5% gain). Tripwire: after two evaluations in a row at which the training reward rose
and the held-out score did not, the reward's precision weight becomes 0.4 and its word budget 50, once.
`--out`/last.pt holds the optimizer, the reward state and the random streams; a run started again resumes from it.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import statistics
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from yolomt.data import Lookback, Window, assert_stored, collate, frame_name, frame_store, past_frames, windows
from yolomt.facts import Run, f1, memo, rated, score
from yolomt.judge import HELD_OUT, NO_THINKING, QWEN, Plan, taking_turns
from yolomt.model import (FolderConfig, Full, Student, Written, load_folder, load_state, parameter_groups, replay,
                          sample, save_folder, see, see_past, trained_state)
from yolomt.reward import Fluency, Parts, Weights, bits_per_character, gate, load_fluency, parses, rewards
from yolomt.schedule import ExitRule, evaluated, improved, stopped

K_MAX = 16
P_MAX = 0.5
WINDOWS = 64
GROUP = 8
TEMPERATURE = 1.0
MAX_NEW = 80
CLIP = 0.2
KL = 0.04
JUDGES = {
    "qwen": Plan(QWEN, port=8765, max_model_len=8192, memory=0.85, max_num_seqs=256, max_num_batched_tokens=None,
                 template=NO_THINKING, threads=192),
    "heldout": Plan(HELD_OUT, port=8775, max_model_len=4096, memory=0.98, max_num_seqs=16, max_num_batched_tokens=2048,
                    template={}, threads=128),
}


@dataclass(frozen=True)
class Slice:
    """Windows start .. stop-1 of a step, all read at lookback `loops`; rows start * group .. stop * group - 1."""

    loops: int
    start: int
    stop: int


@dataclass(frozen=True)
class Control:
    """The reward's weights and the state of the tripwire and of the resampling. interval: the mean reward of each
    step since the last evaluation, last_interval: of the interval before; heldout: the held-out score per validation
    window at the last evaluation."""

    weights: Weights
    fired: bool = False
    strikes: int = 0
    interval: tuple[float, ...] = ()
    last_interval: tuple[float, ...] = ()
    heldout: tuple[float, ...] = ()
    flat_steps: int = 0
    resampling: bool = False


def slices(loops: list[int], *, micro: int) -> list[Slice]:
    """loops: the lookback of each window of a step, ascending -> runs of one lookback, at most `micro` windows."""
    out: list[Slice] = []
    start = 0
    while start < len(loops):
        stop = start
        while stop < len(loops) and loops[stop] == loops[start] and stop - start < micro:
            stop += 1
        out.append(Slice(loops[start], start, stop))
        start = stop
    return out


def seen(student: Student, data: Lookback, part: Slice, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """(B, 64, d) tokens of the slice's frames t and (B, L, 64, d) of their past frames."""
    batch = collate([data[i] for i in range(part.start, part.stop)])
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        return (see(student, batch["frame_t"].to(device, non_blocking=True)),
                see_past(student, batch["pasts"].to(device, non_blocking=True)))


@torch.no_grad()
def rollouts(student: Student, data: Lookback, *, group: int, micro: int, temperature: float,
             generator: torch.Generator | None, device: torch.device) -> Written:
    """`group` captions per window of `data` (lookbacks ascending), row w * group + g, right-padded."""
    parts = []
    for part in slices(data.loops, micro=micro):
        tokens_t, tokens_past = seen(student, data, part, device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            parts.append(sample(student, tokens_t, tokens_past, loops=part.loops, group=group,
                                temperature=temperature, generator=generator, max_new=MAX_NEW))
    width = max(p.tokens.shape[1] for p in parts)

    def widened(x: torch.Tensor, fill: float | int | bool) -> torch.Tensor:
        return F.pad(x, (0, width - x.shape[1]), value=fill)

    return Written(torch.cat([widened(p.tokens, student.layout.pad) for p in parts]),
                   torch.cat([widened(p.logp, 0.0) for p in parts]), torch.cat([widened(p.mask, False) for p in parts]),
                   [c for p in parts for c in p.captions])


def swapped(student: Student, state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Copies `state` into the student's parameters and returns what it overwrote, to swap back."""
    live = dict(student.named_parameters())
    before = {n: live[n].detach().clone() for n in state}
    with torch.no_grad():
        for n, t in state.items():
            live[n].copy_(t)
    return before


@torch.no_grad()
def reference_logp(student: Student, reference: dict[str, torch.Tensor], data: Lookback, written: Written, *,
                   micro: int, device: torch.device) -> torch.Tensor:
    """(R, N) log-probabilities of the sampled tokens under the frozen SFT weights, swapped in for the pass."""
    current = swapped(student, reference)
    out = torch.zeros_like(written.logp)
    for part in slices(data.loops, micro=micro):
        rows = slice(part.start * GROUP, part.stop * GROUP)
        tokens_t, tokens_past = seen(student, data, part, device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out[rows] = replay(student, tokens_t, tokens_past, written.tokens[rows], written.mask[rows],
                               loops=part.loops, group=GROUP, temperature=TEMPERATURE)
    swapped(student, current)
    return out


def advantages(reward: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """reward: (R,) -> (R,) advantages normalized within each group by its population std, and (W,) the groups
    whose rewards vary."""
    grouped = reward.view(-1, GROUP)
    mean = grouped.mean(1, keepdim=True)
    std = grouped.std(1, keepdim=True, unbiased=False)
    normalized = torch.where(std > 1e-6, (grouped - mean) / std.clamp_min(1e-6), torch.zeros_like(grouped))
    return normalized.view(-1), grouped.std(1, unbiased=False) > 1e-6


def surrogate(new_logp: torch.Tensor, old_logp: torch.Tensor, mask: torch.Tensor, advantage: torch.Tensor,
              kept_rows: torch.Tensor, ref_logp: torch.Tensor) -> torch.Tensor:
    """The summed loss of the module docstring over the tokens of the kept rows; (R, N) log-probs, (R,) rest."""
    active = mask & kept_rows[:, None]
    ratio = (new_logp - old_logp).exp()
    per_token = advantage[:, None].expand_as(ratio)
    objective = torch.minimum(ratio * per_token, ratio.clamp(1 - CLIP, 1 + CLIP) * per_token)
    gap = ref_logp - new_logp
    return -objective[active].sum() + KL * (gap.exp() - gap - 1)[active].sum()


def update(student: Student, optimizer: torch.optim.Optimizer, data: Lookback, written: Written,
           advantage: torch.Tensor, kept: torch.Tensor, ref_logp: torch.Tensor, *, micro: int,
           device: torch.device) -> float:
    """Replays every kept caption with gradients, one slice at a time, and takes the step; the gradient norm."""
    params = [p for p in student.parameters() if p.requires_grad]
    kept_rows = kept.repeat_interleave(GROUP)
    tokens = float((written.mask & kept_rows[:, None]).sum())
    norm = 0.0
    if tokens > 0:
        for part in slices(data.loops, micro=micro):
            if not bool(kept[part.start:part.stop].any()):
                continue
            rows = slice(part.start * GROUP, part.stop * GROUP)
            tokens_t, tokens_past = seen(student, data, part, device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                new_logp = replay(student, tokens_t, tokens_past, written.tokens[rows], written.mask[rows],
                                  loops=part.loops, group=GROUP, temperature=TEMPERATURE)
            loss = surrogate(new_logp, written.logp[rows], written.mask[rows], advantage[rows], kept_rows[rows],
                             ref_logp[rows])
            torch.autograd.backward(loss / tokens)
        norm = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return norm


def scored(judge: Run, fluency: Fluency, weights: Weights, keys: list[str], captions: list[str],
           targets: dict[str, Any]) -> list[Parts]:
    """The reward parts of each caption; a (window, caption) pair repeated within the step is scored once."""
    todo = sorted(set(zip(keys, captions)))
    found = dict(zip(todo, rewards(judge, fluency, [c for _, c in todo], [targets[k] for k, _ in todo], weights)))
    return [found[(k, c)] for k, c in zip(keys, captions)]


@torch.no_grad()
def validate(student: Student, data: Lookback, judge: Run, fluency: Fluency, weights: Weights, *,
             device: torch.device) -> list[float]:
    """Per validation window: the held-out judge's fact-F1 of the greedy caption at L = 16, times the gate g from the
    same judge's fluency rating; 0 over the word budget, unparseable, or with a reply cut off."""
    captions = rollouts(student, data, group=1, micro=64, temperature=0.0, generator=None, device=device).captions
    targets = [w.target for w in data.items]
    assert all(t is not None for t in targets)
    within = [bool(c.strip()) and len(c.split()) <= weights.budget for c in captions]
    bpcs = bits_per_character(fluency.scorer, captions)
    parsed = [parses(c) and b < fluency.b_shuffled for c, b in zip(captions, bpcs)]
    scores = score(judge, captions, [t for t in targets if t is not None])
    gates = [gate(fluency, c, r) if ok else 0.0 for c, r, ok in zip(captions, rated(judge, captions), parsed)]
    return [(f1(s) if ok and s.complete else 0.0) * g for s, ok, g in zip(scores, within, gates)]


def rises(before: tuple[float, ...], after: tuple[float, ...], *, paired: bool) -> bool:
    """mean(after) - mean(before) > 2 standard errors of the difference; paired: one value per validation window."""
    if paired:
        gaps = [a - b for a, b in zip(after, before, strict=True)]
        return statistics.fmean(gaps) > 2 * statistics.stdev(gaps) / math.sqrt(len(gaps))
    error = math.sqrt(statistics.variance(before) / len(before) + statistics.variance(after) / len(after))
    return statistics.fmean(after) - statistics.fmean(before) > 2 * error


def eligible(dirs: list[Path], split: str) -> tuple[list[Window], list[tuple[str, ...]]]:
    """The eligible windows of a split and their past frames, found among all its windows."""
    every = windows(dirs, {split}, derived=True)
    pasts = past_frames(every, k_max=K_MAX)
    chosen = [i for i, w in enumerate(every) if w.eligible]
    return [every[i] for i in chosen], [pasts[i] for i in chosen]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", type=Path, nargs="+", required=True, help="derived label roots of the RL windows")
    parser.add_argument("--val", type=Path, nargs="+", required=True, help="derived label roots of the val windows")
    parser.add_argument("--init", type=Path, required=True, help="the model folder of stage 3")
    parser.add_argument("--out", type=Path, required=True, help="run folder")
    parser.add_argument("--judge-gpu", required=True, help="the physical GPU the judges take turns on")
    parser.add_argument("--fluency", type=Path, default=Path("data/fluency.json"))
    parser.add_argument("--micro", type=int, default=1, help="windows per replay pass, for memory")
    parser.add_argument("--max-steps", type=int, default=651)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--val-windows", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=torch.device, default=torch.device("cuda"))
    args = parser.parse_args()
    device = args.device
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "judges").mkdir(exist_ok=True)
    judges, stop_judges = taking_turns(JUDGES, device=args.judge_gpu, logs=str(args.out / "judges"))
    torch.manual_seed(args.seed)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    rng = random.Random(args.seed)
    loop_rng = random.Random(args.seed + 2)

    student = load_folder(args.init, dtype=torch.float32)
    assert isinstance(student.tuning, Full), "GRPO starts from the full SFT model"
    student.to(device)
    # nothing in the student drops out: eval mode only states that the sampling and the replayed policy are one
    student.eval()
    fluency = load_fluency(args.fluency, device)
    reference = {n: p.detach().clone() for n, p in student.named_parameters() if p.requires_grad}

    train_items, train_pasts = eligible(args.labels, "train")
    val_items, val_pasts = eligible(args.val, "val")
    deep = [i for i, p in enumerate(val_pasts) if len(p) >= K_MAX]
    deep = [deep[i] for i in random.Random(args.seed).sample(range(len(deep)), min(args.val_windows, len(deep)))]
    assert len(train_items) >= WINDOWS and deep, (len(train_items), len(deep))
    store = frame_store([*args.labels, *args.val])
    val = Lookback([val_items[i] for i in deep], [val_pasts[i] for i in deep], store, loops=[K_MAX] * len(deep))
    assert_stored(store, [n for w, p in [*zip(train_items, train_pasts), *zip(val.items, val.pasts)]
                          for n in (frame_name(w.video_id, w.t1), *p)])
    targets = {w.key: w.target for w in train_items}

    optimizer = torch.optim.AdamW(parameter_groups(student, lr=1e-6, vision_lr=2e-7), lr=1e-6)
    rule = ExitRule(higher_is_better=True, patience=3)
    control = Control(weights=Weights())
    uncertainty = [1.0] * len(train_items)
    step = 0
    last = args.out / "last.pt"
    if last.exists():
        state = torch.load(last, map_location="cpu", weights_only=True)
        load_state(student, state["model"])
        optimizer.load_state_dict(state["optimizer"])
        step, rule, uncertainty = state["step"], ExitRule(**state["rule"]), state["uncertainty"]
        control = Control(**{**state["control"], "weights": Weights(**state["control"]["weights"])})
        rng.setstate(state["rng"])
        loop_rng.setstate(state["loop_rng"])
        generator.set_state(state["generator"])

    def evaluate() -> tuple[float, ...]:
        nonlocal rule
        heldout = tuple(validate(student, val, judges["heldout"], fluency, control.weights, device=device))
        rule = evaluated(rule, sum(heldout) / len(heldout))
        print(f"step {step}: held-out gated fact-F1 {rule.last:.4f} best {rule.best:.4f}", file=sys.stderr, flush=True)
        if improved(rule):
            save_folder(student, args.out / "best",
                        FolderConfig(tuning=Full(), k_max=K_MAX, max_new=MAX_NEW, step=step))
        return heldout

    def save() -> None:
        state = {"model": trained_state(student), "optimizer": optimizer.state_dict(), "step": step,
                 "rule": asdict(rule), "control": asdict(control), "uncertainty": uncertainty,
                 "rng": rng.getstate(), "loop_rng": loop_rng.getstate(), "generator": generator.get_state()}
        torch.save(state, last.with_suffix(".part"))
        os.replace(last.with_suffix(".part"), last)

    if step == 0:
        control = replace(control, heldout=evaluate())
    while step < args.max_steps and not stopped(rule):
        picked = (rng.choices(range(len(train_items)), weights=uncertainty, k=WINDOWS) if control.resampling
                  else rng.sample(range(len(train_items)), WINDOWS))
        drawn = []
        for i in picked:
            top = min(K_MAX, len(train_pasts[i]))
            drawn.append(top if loop_rng.random() < P_MAX else loop_rng.randint(0, top))
        order = sorted(range(WINDOWS), key=drawn.__getitem__)
        chosen = [picked[i] for i in order]
        data = Lookback([train_items[i] for i in chosen], [train_pasts[i] for i in chosen], store,
                        loops=[drawn[i] for i in order])
        written = rollouts(student, data, group=GROUP, micro=WINDOWS, temperature=TEMPERATURE, generator=generator,
                           device=device)
        ref_logp = reference_logp(student, reference, data, written, micro=args.micro, device=device)
        keys = [train_items[i].key for i in chosen for _ in range(GROUP)]
        parts = scored(judges["qwen"], fluency, control.weights, keys, written.captions, targets)
        reward = torch.tensor([p.reward for p in parts], device=device)
        advantage, kept = advantages(reward)
        kept = kept & torch.tensor([p.complete for p in parts], device=device).view(-1, GROUP).all(1)
        for window, deviation in zip(chosen, reward.view(-1, GROUP).std(1, unbiased=False).tolist()):
            uncertainty[window] = 0.05 + deviation
        flat_steps = control.flat_steps + 1 if 1 - float(kept.float().mean()) > 0.5 else 0
        control = replace(control, flat_steps=flat_steps, resampling=control.resampling or flat_steps >= 3,
                          interval=(*control.interval, float(reward.mean())))
        norm = update(student, optimizer, data, written, advantage, kept, ref_logp, micro=args.micro, device=device)
        step += 1
        print(f"step {step}: reward {float(reward.mean()):.4f} kept groups {int(kept.sum())} grad_norm {norm:.3f}",
              file=sys.stderr, flush=True)
        if step % args.eval_every == 0:
            heldout = evaluate()
            if len(control.last_interval) > 1:
                rising = rises(control.last_interval, control.interval, paired=False)
                flat = not rises(control.heldout, heldout, paired=True)
                control = replace(control, strikes=control.strikes + 1 if rising and flat else 0)
                if control.strikes >= 2 and not control.fired:
                    control = replace(control, fired=True, weights=replace(control.weights, precision=0.4, budget=50))
            control = replace(control, last_interval=control.interval, interval=(), heldout=heldout)
            save()
    save()
    stop_judges()


if __name__ == "__main__":
    main()
