"""Stage 1, the 2 s captioner: LoRA SFT of LFM2.5-VL-450M on the teacher's captions of 2 s windows.

    python scripts/train_2s.py --labels <labels>/captioner --out <runs>/captioner

A window reads the pooled frame t-1 s and frame t, and after "Previous:" the caption of the window one second earlier
in its run: empty with probability 0.25 (a stream that just started) and another video's with probability 0.15 (a
scene cut), so that copying it cannot pass the loss. LoRA adapters of rank 64 and the projector train, the pooler
stays plain average pooling. The loss is the cross entropy of the caption and <|im_end|>, averaged over the tokens of
a step of `--batch` windows. AdamW, `--lr` with the schedule of `yolomt.schedule` over `--epochs`, gradient norm
clipped to 1.

Every `--eval-every` steps the loss on `--val-windows` validation windows selects `--out`/best (a model folder) and
the exit rule may stop the run; `--out`/last.pt holds the optimizer state, and a run started again resumes from it.
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from yolomt.data import FrameStore, Window, frame_name, frame_store, image, windows
from yolomt.model import FolderConfig, Lora, Student, load_state, loss, save_folder, see, trained_state
from yolomt.schedule import ExitRule, evaluated, improved, learning_rate, stopped

P_NONE = 0.25
P_STALE = 0.15


class Previous(Dataset[dict[str, Any]]):
    """Frames t and t-1 of each window, its caption, and the caption shown after "Previous:", drawn per epoch."""

    def __init__(self, items: list[Window], store: FrameStore, *, seed: int) -> None:
        self.items = items
        self.store = store
        self.seed = seed
        self.epoch = 0
        by_key = {(w.run, w.t1): w for w in items}
        self.before = [by_key.get((w.run, w.t1 - 1)) for w in items]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item, before = self.items[index], self.before[index]
        rng = random.Random(f"{self.seed}:{self.epoch}:{index}")
        draw = rng.random()
        if before is None or draw < P_NONE:
            previous = ""
        elif draw < P_NONE + P_STALE:
            other = self.items[rng.randrange(len(self.items))]
            previous = other.canonical if other.video_id != item.video_id else ""
        else:
            previous = before.canonical
        return {"frame_t": image(self.store, frame_name(item.video_id, item.t1)),
                "frame_prev": image(self.store, frame_name(item.video_id, item.t1 - 1)),
                "caption": item.canonical, "previous": previous}


def collate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {"frame_t": torch.stack([r["frame_t"] for r in rows]),
            "frame_prev": torch.stack([r["frame_prev"] for r in rows]),
            "caption": [r["caption"] for r in rows], "previous": [r["previous"] for r in rows]}


def caption_loss(student: Student, part: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, int]:
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        tokens_t = see(student, part["frame_t"].to(device, non_blocking=True))
        tokens_prev = see(student, part["frame_prev"].to(device, non_blocking=True))
        out = loss(student, tokens_t, tokens_prev[:, None], part["caption"], loops=1, previous=part["previous"])
    return out.loss, out.tokens


@torch.no_grad()
def validate(student: Student, data: Previous, *, micro: int, workers: int, device: torch.device) -> float:
    student.eval()
    total, count = 0.0, 0
    for part in DataLoader(data, batch_size=micro, num_workers=workers, collate_fn=collate):
        value, tokens = caption_loss(student, part, device)
        total += float(value) * tokens
        count += tokens
    student.train()
    return total / count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", type=Path, nargs="+", required=True, help="label roots of the 2 s windows")
    parser.add_argument("--out", type=Path, required=True, help="run folder")
    parser.add_argument("--batch", type=int, default=256, help="windows per optimizer step")
    parser.add_argument("--micro", type=int, default=64, help="windows per forward pass; divides --batch")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=float, default=3.0, help="the horizon of the cosine schedule")
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--val-windows", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=8, help="data-loading processes")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=torch.device, default=torch.device("cuda"))
    args = parser.parse_args()
    assert args.batch % args.micro == 0
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = args.device
    args.out.mkdir(parents=True, exist_ok=True)

    train_items = windows(args.labels, {"train"}, derived=False)
    val_items = windows(args.labels, {"val"}, derived=False)
    val_items = random.Random(args.seed).sample(val_items, min(args.val_windows, len(val_items)))
    assert len(train_items) >= args.batch and val_items, (len(train_items), len(val_items))
    store = frame_store(args.labels)
    train = Previous(train_items, store, seed=args.seed)
    val = Previous(val_items, store, seed=args.seed + 1)
    total = max(1, round(args.epochs * (len(train_items) // args.batch)))

    student = Student(tuning=Lora(rank=64), dtype=torch.bfloat16).to(device)
    student.pooler.requires_grad_(False)
    params = [p for p in student.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    rule = ExitRule(higher_is_better=False, patience=3)
    step, epoch, consumed = 0, 0, 0
    last = args.out / "last.pt"
    if last.exists():
        state = torch.load(last, map_location="cpu", weights_only=True)
        load_state(student, state["model"])
        optimizer.load_state_dict(state["optimizer"])
        step, epoch, consumed, rule = state["step"], state["epoch"], state["consumed"], ExitRule(**state["rule"])

    def save() -> None:
        torch.save({"model": trained_state(student), "optimizer": optimizer.state_dict(), "step": step, "epoch": epoch,
                    "consumed": consumed, "rule": asdict(rule)}, last)

    while not stopped(rule) and step < total:
        train.epoch = epoch
        order = torch.randperm(len(train_items), generator=torch.Generator().manual_seed(args.seed + epoch)).tolist()
        loader = iter(DataLoader(train, batch_size=args.micro, sampler=order[consumed:], num_workers=args.workers,
                                 collate_fn=collate, drop_last=True, pin_memory=device.type == "cuda"))
        for _ in range((len(train_items) - consumed) // args.batch):
            for group in optimizer.param_groups:
                group["lr"] = learning_rate(step, total, args.lr)
            tokens, summed = 0, 0.0
            for _ in range(args.batch // args.micro):
                value, count = caption_loss(student, next(loader), device)
                torch.autograd.backward(value * count)
                tokens += count
                summed += float(value) * count
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
                rule = evaluated(rule, validate(student, val, micro=args.micro, workers=args.workers, device=device))
                print(f"step {step}: val loss {rule.last:.4f} best {rule.best:.4f}", file=sys.stderr, flush=True)
                save()
                if improved(rule):
                    save_folder(student, args.out / "best", FolderConfig(tuning=student.tuning, k_max=1, max_new=48,
                                                                         step=step))
            if stopped(rule) or step >= total:
                break
        if not stopped(rule) and step < total:
            epoch += 1
            consumed = 0
    save()


if __name__ == "__main__":
    main()
