"""Stages 2 and 3, sandwich SFT: one set of weights learns to caption at every lookback L.

    python scripts/train_sft.py --labels <labels>/captioner --init <runs>/captioner/best --out <runs>/sandwich \\
        --k-max 5 --val-loops 0 1 2 3 4 5 --lr 2e-5 --max-new 48
    python scripts/train_sft.py --labels <labels>/sft --init <runs>/sandwich/best --out <runs>/sft --full \\
        --k-max 16 --val-loops 0 1 2 4 8 16 --lr 1e-5 --vision-lr 2e-6 --max-new 80 --checkpoint-layers

Windows are grouped by m_w = min(k_max, L_avail), L_avail the past frames a window has. Each micro-batch is scored at
L = 0, at one l drawn uniformly from 1 .. m-1 and at L = m, every frame encoded once, and the three losses take one
backward:

    loss = (1 / N) sum_{w in B} sum_{L in {0, l, m_w}} sum_i -log p(y^w_i | y^w_{<i}, c^w_L)

with B the `--batch` windows of a step, N their target tokens, y^w the teacher's caption and c^w_L the prompt of w
at lookback L. Without `--full` a LoRA `--init` keeps training its adapters and the projector; with it the adapters
are merged and every weight trains in float32, the vision encoder at `--vision-lr`. AdamW with the schedule of
`yolomt.schedule` over `--epochs`, gradient norm clipped to 1.

At step 0 and every `--eval-every` steps, the mean over `--val-loops` of the validation loss, on validation windows
with at least max(val-loops) past frames, selects `--out`/best (a model folder) and drives the exit rule;
`--out`/last.pt holds the optimizer state, and a run started again resumes from it.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from yolomt.data import Lookback, assert_stored, collate, frame_name, frame_store, past_frames, windows
from yolomt.model import (FolderConfig, Loss, Lora, Student, checkpoint_layers, load_folder, load_state, loss, merge,
                          parameter_groups, read_config, save_folder, see, see_past, trained_state)
from yolomt.schedule import ExitRule, evaluated, improved, learning_rate, stopped


def micro_batches(loops: list[int], micro: int, rng: random.Random) -> list[list[int]]:
    """loops: m of each window -> shuffled micro-batches of `micro` window indices sharing one m; the rest of each
    group sits the epoch out, so every step has the same number of windows."""
    groups: dict[int, list[int]] = defaultdict(list)
    for index, m in enumerate(loops):
        groups[m].append(index)
    out: list[list[int]] = []
    for m in sorted(groups):
        members = groups[m]
        rng.shuffle(members)
        out += [members[i:i + micro] for i in range(0, len(members) - micro + 1, micro)]
    rng.shuffle(out)
    return out


def sandwich(depth: int, rng: random.Random) -> tuple[int, ...]:
    """The lookbacks a micro-batch with m = depth is scored at: 0, one in 1 .. m-1, and m."""
    middle = (rng.randint(1, depth - 1),) if depth >= 2 else ()
    return tuple(sorted({0, *middle, depth}))


def losses(student: Student, part: dict[str, Any], loops: tuple[int, ...], device: torch.device) -> dict[int, Loss]:
    """The caption loss at every lookback of `loops`, each frame encoded once."""
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        tokens_t = see(student, part["frame_t"].to(device, non_blocking=True))
        tokens_past = see_past(student, part["pasts"][:, :max(loops)].to(device, non_blocking=True))
        return {k: loss(student, tokens_t, tokens_past, part["caption"], loops=k) for k in loops}


@torch.no_grad()
def validate(student: Student, data: Lookback, loops: tuple[int, ...], *, micro: int, workers: int,
             device: torch.device) -> float:
    """The token-weighted validation loss at each lookback of `loops`, averaged over them."""
    student.eval()
    total = dict.fromkeys(loops, 0.0)
    count = 0
    for part in DataLoader(data, batch_size=micro, num_workers=workers, collate_fn=collate):
        found = losses(student, part, loops, device)
        total = {k: total[k] + float(found[k].loss) * found[k].tokens for k in loops}
        count += found[loops[0]].tokens
    student.train()
    return sum(v / count for v in total.values()) / len(loops)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", type=Path, nargs="+", required=True, help="label roots: train and val windows")
    parser.add_argument("--init", type=Path, required=True, help="the model folder of the previous stage")
    parser.add_argument("--out", type=Path, required=True, help="run folder")
    parser.add_argument("--full", action="store_true", help="merge LoRA adapters and train every weight")
    parser.add_argument("--k-max", type=int, required=True, help="the largest lookback trained")
    parser.add_argument("--val-loops", type=int, nargs="+", required=True, help="the lookbacks validated")
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--vision-lr", type=float, default=None, help="with --full: the vision encoder's rate")
    parser.add_argument("--max-new", type=int, required=True, help="caption limit in tokens, kept in the model folder")
    parser.add_argument("--batch", type=int, default=256, help="windows per optimizer step")
    parser.add_argument("--micro", type=int, default=16, help="windows per forward pass; divides --batch")
    parser.add_argument("--epochs", type=float, default=2.0, help="the horizon of the cosine schedule")
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--val-windows", type=int, default=2000)
    parser.add_argument("--patience", type=int, default=100, help="exit-rule strikes")
    parser.add_argument("--checkpoint-layers", action="store_true", help="recompute activations in the backward pass")
    parser.add_argument("--workers", type=int, default=8, help="data-loading processes")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=torch.device, default=torch.device("cuda"))
    args = parser.parse_args()
    assert args.batch % args.micro == 0 and (args.vision_lr is None or args.full)
    val_loops = tuple(sorted(set(args.val_loops)))
    depth = max(val_loops)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = args.device
    args.out.mkdir(parents=True, exist_ok=True)

    train_items = windows(args.labels, {"train"}, derived=False)
    train_pasts = past_frames(train_items, k_max=args.k_max)
    val_items = windows(args.labels, {"val"}, derived=False)
    val_pasts = past_frames(val_items, k_max=depth)
    chosen = random.Random(args.seed).sample(range(len(val_items)), min(args.val_windows, len(val_items)))
    deep = [i for i in chosen if len(val_pasts[i]) >= depth]
    assert deep, f"no validation window has {depth} past frames"
    store = frame_store(args.labels)
    val = Lookback([val_items[i] for i in deep], [val_pasts[i] for i in deep], store, loops=[depth] * len(deep))
    assert_stored(store, [n for w, p in zip(train_items, train_pasts) for n in (frame_name(w.video_id, w.t1), *p)]
                  + [n for w, p in zip(val.items, val.pasts) for n in (frame_name(w.video_id, w.t1), *p)])
    loops = [min(args.k_max, len(p)) for p in train_pasts]
    per_step = args.batch // args.micro
    steps_per_epoch = len(micro_batches(loops, args.micro, random.Random(f"{args.seed}:0"))) // per_step
    assert steps_per_epoch > 0, f"{len(train_items)} train windows fill no optimizer step"
    total = max(1, round(args.epochs * steps_per_epoch))

    tuning = read_config(args.init).tuning
    full = args.full or not isinstance(tuning, Lora)
    student = load_folder(args.init, dtype=torch.float32 if full or device.type != "cuda" else torch.bfloat16)
    if full and isinstance(tuning, Lora):
        merge(student)
    if isinstance(student.tuning, Lora):
        student.pooler.requires_grad_(False)
    student.to(device)
    if args.checkpoint_layers:
        checkpoint_layers(student)
    params = [p for p in student.parameters() if p.requires_grad]
    vision_lr = args.lr if args.vision_lr is None else args.vision_lr
    optimizer = torch.optim.AdamW(parameter_groups(student, lr=args.lr, vision_lr=vision_lr), lr=args.lr)
    rule = ExitRule(higher_is_better=False, patience=args.patience)
    step, epoch, consumed = 0, 0, 0
    last = args.out / "last.pt"
    if last.exists():
        state = torch.load(last, map_location="cpu", weights_only=True)
        load_state(student, state["model"])
        optimizer.load_state_dict(state["optimizer"])
        step, epoch, consumed, rule = state["step"], state["epoch"], state["consumed"], ExitRule(**state["rule"])

    def evaluate() -> None:
        nonlocal rule
        rule = evaluated(rule, validate(student, val, val_loops, micro=args.micro, workers=args.workers, device=device))
        print(f"step {step}: val loss {rule.last:.4f} best {rule.best:.4f}", file=sys.stderr, flush=True)
        torch.save({"model": trained_state(student), "optimizer": optimizer.state_dict(), "step": step, "epoch": epoch,
                    "consumed": consumed, "rule": asdict(rule)}, last)
        if improved(rule):
            save_folder(student, args.out / "best", FolderConfig(tuning=student.tuning, k_max=args.k_max,
                                                                 max_new=args.max_new, step=step))

    if step == 0:
        evaluate()
    student.train()
    while not stopped(rule) and step < total:
        parts = micro_batches(loops, args.micro, random.Random(f"{args.seed}:{epoch}"))[consumed // args.micro:]
        loader = iter(DataLoader(Lookback(train_items, train_pasts, store, loops=loops), batch_sampler=parts,
                                 num_workers=args.workers, collate_fn=collate, pin_memory=device.type == "cuda"))
        for _ in range(len(parts) // per_step):
            scale = learning_rate(step, total, 1.0)
            for group in optimizer.param_groups:
                group["lr"] = group["peak"] * scale
            tokens, summed = 0, 0.0
            for j in range(per_step):
                part = next(loader)
                loops_j = sandwich(int(part["pasts"].shape[1]), random.Random(f"{args.seed}:{step}:{j}"))
                found = losses(student, part, loops_j, device)
                torch.autograd.backward(torch.stack([r.loss * r.tokens for r in found.values()]).sum())
                tokens += sum(r.tokens for r in found.values())
                summed += sum(float(r.loss) * r.tokens for r in found.values())
            for p in params:
                if p.grad is not None:
                    p.grad.div_(tokens)
            norm = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            consumed += args.batch
            if step % 10 == 0:
                print(f"step {step}: loss {summed / tokens:.4f} grad_norm {norm:.3f}", file=sys.stderr, flush=True)
            if step % args.eval_every == 0 or step == total:
                evaluate()
            if stopped(rule) or step >= total:
                break
        if not stopped(rule) and step < total:
            epoch += 1
            consumed = 0


if __name__ == "__main__":
    main()
