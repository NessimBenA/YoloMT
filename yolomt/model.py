"""The student: LFM2.5-VL-450M plus a pooler that caches each past frame as 16 tokens.

A 256 x 256 frame passes SigLIP2 and the projector once and becomes an 8 x 8 grid of 64 tokens x of width d = 1024.
A past frame is pooled: each 2 x 2 block is averaged and a learned residual, zero at initialisation, is added,

    z = x_bar + W x_bar + b,        x_bar the block average, W in R^{d x d}, b in R^d,

so it takes 16 tokens. At lookback L the prompt reads the pooled frames t-L s ... t-1 s, oldest first, then the 64
tokens of frame t, and the caption follows:

    <|startoftext|><|im_start|>user\\nPrevious: none\\n
    <|image_start|>{16 t-L}<|image_end|> ... <|image_start|>{16 t-1}<|image_end|>
    <|image_start|>{64 t}<|image_end|><|im_end|>\\n<|im_start|>assistant\\n{caption}<|im_end|>

The 2 s captioner, the first stage, reads one past frame and shows its last caption after "Previous:". The tied
embedding is the output head. A model folder holds the trained weights (model.safetensors) and config.json.
"""

from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, assert_never

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from huggingface_hub import snapshot_download
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file, save_file
from torch import nn
from transformers import AutoConfig, AutoModelForImageTextToText, AutoTokenizer

BASE = "LiquidAI/LFM2.5-VL-450M"
BASE_REVISION = "fc6221ca597f3315e4f82fc2df606783267b34ba"
SIDE = 256
PATCH = 16
GRID = SIDE // PATCH
# tokens of a frame after the projector merges each 2 x 2 block of the 16 x 16 patch grid
TOKENS = (GRID // 2) ** 2
POOL = 2
POOLED = TOKENS // POOL**2
# every linear layer of the LM's attention, short-convolution and MLP blocks
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "out_proj", "in_proj", "w1", "w2", "w3"]
WEIGHTS = "model.safetensors"
CONFIG = "config.json"


@dataclass(frozen=True)
class Lora:
    """LoRA adapters (alpha = rank, no dropout) on the LM train with the projector; the rest is frozen."""

    rank: int


@dataclass(frozen=True)
class Full:
    """Every weight trains."""


Tuning = Lora | Full


@dataclass(frozen=True)
class Layout:
    """Token ids of the prompt's special tokens; newline and assistant are the ids of "\\n" and "assistant\\n"."""

    bos: int
    im_start: int
    im_end: int
    image_start: int
    image_end: int
    image: int
    pad: int
    newline: list[int]
    assistant: list[int]


@dataclass(frozen=True)
class Batch:
    """ids, mask: (B, N) right-padded prompts and captions; targets: (B, N) the next token to predict, -100 for none."""

    ids: torch.Tensor
    mask: torch.Tensor
    targets: torch.Tensor


@dataclass(frozen=True)
class Loss:
    """loss: mean cross entropy over the caption tokens and the closing <|im_end|>; tokens: their count."""

    loss: torch.Tensor
    tokens: int


@dataclass(frozen=True)
class Written:
    """R sampled captions of at most N tokens. tokens: (R, N), <|pad|> after the end; logp: (R, N) the log-probability
    of each token under the sampling policy, 0 after the end; mask: (R, N) True on every token, the closing <|im_end|>
    included; captions: the R texts, special tokens kept."""

    tokens: torch.Tensor
    logp: torch.Tensor
    mask: torch.Tensor
    captions: list[str]


@dataclass(frozen=True)
class FolderConfig:
    """config.json of a model folder: the tuning of its weights, the largest lookback trained, the caption limit in
    tokens and the optimizer step the weights come from."""

    tuning: Tuning
    k_max: int
    max_new: int
    step: int


class Pooler(nn.Module):
    """(B, 64, d) tokens of an 8 x 8 grid, row-major -> (B, 16, d): z = x_bar + W x_bar + b over 2 x 2 blocks."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.residual = nn.Linear(width, width)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        side = GRID // 2
        grid = tokens.view(tokens.shape[0], side, side, tokens.shape[2]).permute(0, 3, 1, 2)
        pooled = F.avg_pool2d(grid, POOL).permute(0, 2, 3, 1).flatten(1, 2)
        out: torch.Tensor = pooled + self.residual(pooled)
        return out


def check_pooler() -> None:
    """On the grid holding its row-major index 8r + c, the block (R, C) averages 16R + 2C + {0, 1, 8, 9}."""
    grid = torch.arange(64, dtype=torch.float32).view(1, 64, 1)
    block = torch.arange(16)
    expected = (16 * (block // 4) + 2 * (block % 4) + 4.5).view(1, 16, 1)
    assert torch.equal(Pooler(1)(grid), expected), "the pooler is not 2 x 2 average pooling"


class Student(nn.Module):
    """The attribute names below are the weight names of model.safetensors; `position` is SigLIP's position
    embedding, a parameter of a full student and a frozen buffer of a LoRA one."""

    position: torch.Tensor

    def __init__(self, *, tuning: Tuning, dtype: torch.dtype, weights: bool = True) -> None:
        """weights: start from the base model's weights; without, only its configuration and tokenizer are read, for
        a caller that loads every weight."""
        super().__init__()
        check_pooler()
        from_config: Any = AutoModelForImageTextToText.from_config
        base = (AutoModelForImageTextToText.from_pretrained(BASE, revision=BASE_REVISION, dtype=dtype,
                                                            attn_implementation="sdpa")
                if weights else from_config(AutoConfig.from_pretrained(BASE, revision=BASE_REVISION), dtype=dtype,
                                            attn_implementation="sdpa"))
        self.tokenizer: Any = AutoTokenizer.from_pretrained(BASE, revision=BASE_REVISION)
        special: Any = self.tokenizer.convert_tokens_to_ids
        self.layout = Layout(
            bos=self.tokenizer.bos_token_id, im_start=special("<|im_start|>"), im_end=special("<|im_end|>"),
            image_start=special("<|image_start|>"), image_end=special("<|image_end|>"), image=special("<image>"),
            pad=self.tokenizer.pad_token_id, newline=self.tokenizer("\n", add_special_tokens=False).input_ids,
            assistant=self.tokenizer("assistant\n", add_special_tokens=False).input_ids)
        self.tuning = tuning
        self.checkpointing = False
        tower = base.model.vision_tower
        self.patch_embedding = tower.embeddings.patch_embedding
        position = tower.embeddings.position_embedding.weight.detach().clone()
        match tuning:
            case Lora():
                self.register_buffer("position", position, persistent=False)
            case Full():
                self.position = nn.Parameter(position)
            case _:
                assert_never(tuning)
        self.siglip = tower.encoder.layers
        self.post_layernorm = tower.post_layernorm
        self.projector = base.model.multi_modal_projector.float()
        lm = base.model.language_model
        self.embed = lm.embed_tokens
        self.lm: Any
        match tuning:
            case Lora(rank=rank):
                self.lm = get_peft_model(lm, LoraConfig(r=rank, lora_alpha=rank, lora_dropout=0.0,
                                                        target_modules=LORA_TARGETS))
                for module in (self.patch_embedding, self.siglip, self.post_layernorm, self.embed):
                    module.requires_grad_(False)
            case Full():
                self.lm = lm
            case _:
                assert_never(tuning)
        self.pooler = Pooler(lm.config.hidden_size)


def patches(frames: torch.Tensor) -> torch.Tensor:
    """(B, 3, 256, 256) uint8 -> (B, 256, 768): 16 x 16 patches row-major, each (row, col, channel), in [-1, 1]."""
    x = frames.float() / 127.5 - 1.0
    x = x.view(x.shape[0], 3, GRID, PATCH, GRID, PATCH).permute(0, 2, 4, 3, 5, 1)
    return x.reshape(x.shape[0], GRID * GRID, -1)


def see(student: Student, frames: torch.Tensor) -> torch.Tensor:
    """(B, 3, 256, 256) uint8 -> (B, 64, d) projector tokens. A LoRA student's encoder runs without grad."""
    match student.tuning:
        case Lora():
            grad: contextlib.AbstractContextManager[Any] = torch.no_grad()
        case Full():
            grad = contextlib.nullcontext()
        case _:
            assert_never(student.tuning)
    recompute = student.checkpointing and student.training and torch.is_grad_enabled()
    with grad:
        hidden = student.patch_embedding(patches(frames).to(student.position.dtype)) + student.position
        for layer in student.siglip:
            hidden = (torch.utils.checkpoint.checkpoint(layer, hidden, None, use_reentrant=False) if recompute
                      else layer(hidden, None))
        hidden = student.post_layernorm(hidden)
    grid = hidden.view(frames.shape[0], GRID, GRID, -1).to(student.projector.linear_1.weight.dtype)
    out: torch.Tensor = student.projector(grid).reshape(frames.shape[0], TOKENS, -1)
    return out


def see_past(student: Student, pasts: torch.Tensor) -> torch.Tensor:
    """(B, P, 3, 256, 256) uint8, frame t-1 s first -> (B, P, 64, d)."""
    count, depth = pasts.shape[:2]
    if depth == 0:
        return torch.zeros(count, 0, TOKENS, student.embed.weight.shape[1], device=pasts.device)
    return see(student, pasts.flatten(0, 1)).view(count, depth, TOKENS, -1)


def prompt(student: Student, *, loops: int, previous: str = "") -> list[int]:
    """The prompt ids at lookback `loops`, with `previous` after "Previous:" ("none" when empty)."""
    lay = student.layout
    text: list[int] = student.tokenizer(f"user\nPrevious: {previous or 'none'}\n", add_special_tokens=False).input_ids
    block = [lay.image_start, *[lay.image] * POOLED, lay.image_end]
    return [lay.bos, lay.im_start, *text, *block * loops, lay.image_start, *[lay.image] * TOKENS, lay.image_end,
            lay.im_end, *lay.newline, lay.im_start, *lay.assistant]


def batch(student: Student, heads: list[list[int]], captions: list[str] | None, device: torch.device) -> Batch:
    """Right-padded rows: each prompt of `heads`, then, when given, its caption's tokens and <|im_end|>."""
    rows, targets = [], []
    for i, head in enumerate(heads):
        tail: list[int] = []
        if captions is not None:
            tail = [*student.tokenizer(captions[i], add_special_tokens=False).input_ids, student.layout.im_end]
        rows.append(head + tail)
        targets.append([-100] * (len(head) - 1) + tail + [-100])
    width = max(len(r) for r in rows)

    def pad(values: list[list[int]], fill: int) -> torch.Tensor:
        return torch.tensor([v + [fill] * (width - len(v)) for v in values], device=device)

    return Batch(pad(rows, student.layout.pad), pad([[1] * len(r) for r in rows], 0).bool(), pad(targets, -100))


def visual(student: Student, tokens_t: torch.Tensor, tokens_past: torch.Tensor, *, loops: int) -> torch.Tensor:
    """tokens_t: (B, 64, d); tokens_past: (B, P, 64, d), t-1 s first, P >= loops -> (B, 16 loops + 64, d), the
    pooled frames t-loops s ... t-1 s then frame t, as the prompt's <image> slots hold them."""
    count, width = tokens_t.shape[0], tokens_t.shape[2]
    chosen = tokens_past[:, :loops].flip(1).flatten(0, 1)
    pooled = student.pooler(chosen).reshape(count, loops * POOLED, width) if loops else tokens_t[:, :0]
    return torch.cat([pooled, tokens_t], dim=1)


def embed(student: Student, ids: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
    """ids: (B, N); slots: (B, S, d) for the S <image> ids of each row -> (B, N, d) input embeddings."""
    embeds: torch.Tensor = student.embed(ids)
    where = ids == student.layout.image
    assert int(where.sum()) == slots.shape[0] * slots.shape[1]
    return embeds.masked_scatter(where[..., None], slots.to(embeds.dtype))


def loss(student: Student, tokens_t: torch.Tensor, tokens_past: torch.Tensor, captions: list[str], *, loops: int,
         previous: list[str] | None = None) -> Loss:
    """The teacher-forced caption loss at lookback `loops` (shapes as in `visual`); `previous`: per row, the caption
    shown after "Previous:", the 2 s captioner's prompt."""
    heads = ([prompt(student, loops=loops, previous=p) for p in previous] if previous is not None
             else [prompt(student, loops=loops)] * len(captions))
    rows = batch(student, heads, captions, tokens_t.device)
    inputs = embed(student, rows.ids, visual(student, tokens_t, tokens_past, loops=loops))
    states = student.lm(inputs_embeds=inputs, attention_mask=rows.mask.long()).last_hidden_state
    where = rows.targets != -100
    labels = rows.targets[where]
    return Loss(F.cross_entropy((states[where] @ student.embed.weight.T).float(), labels), int(labels.numel()))


def decoded(student: Student, tokens: list[int], *, special: bool) -> str:
    text = student.tokenizer.decode(tokens, skip_special_tokens=not special)
    assert isinstance(text, str)
    return text.strip()


@torch.no_grad()
def write(student: Student, frames_t: torch.Tensor, pasts: torch.Tensor, *, loops: int, max_new: int) -> list[str]:
    """frames_t: (B, 3, 256, 256) uint8; pasts: (B, P, 3, 256, 256), t-1 s first, P >= loops -> B greedy captions.
    Every prompt at one lookback has one length, so the rows share one cache; a finished row keeps decoding tokens
    that are never read."""
    device = student.embed.weight.device
    count = frames_t.shape[0]
    slots = visual(student, see(student, frames_t.to(device)), see_past(student, pasts[:, :loops].to(device)),
                   loops=loops)
    ids = torch.tensor([prompt(student, loops=loops)], device=device).expand(count, -1)
    result = student.lm(inputs_embeds=embed(student, ids, slots), use_cache=True)
    cache, h = result.past_key_values, result.last_hidden_state[:, -1]
    produced: list[list[int]] = [[] for _ in range(count)]
    done = [False] * count
    while True:
        tokens = (h @ student.embed.weight.T).argmax(-1)
        for r in range(count):
            if not done[r]:
                if int(tokens[r]) == student.layout.im_end or len(produced[r]) == max_new:
                    done[r] = True
                else:
                    produced[r].append(int(tokens[r]))
        if all(done):
            break
        step = student.lm(inputs_embeds=student.embed(tokens[:, None]), past_key_values=cache, use_cache=True)
        cache, h = step.past_key_values, step.last_hidden_state[:, -1]
    return [decoded(student, row, special=False) for row in produced]


def logits_of(states: torch.Tensor, head: torch.Tensor) -> torch.Tensor:
    """(..., d) states, (V, d) float32 head -> (..., V) float32 logits outside autocast: bf16 would quantize
    log-probabilities to about 0.1 nats."""
    with torch.autocast(device_type=states.device.type, enabled=False):
        return states.float() @ head.T


@torch.no_grad()
def sample(student: Student, tokens_t: torch.Tensor, tokens_past: torch.Tensor, *, loops: int, group: int,
           temperature: float, generator: torch.Generator | None, max_new: int) -> Written:
    """`group` captions per window at lookback `loops`, row w * group + g, drawn from softmax(logits / temperature)
    (greedy at temperature 0, logp then at temperature 1). <|im_end|> is an action: the policy chooses the length."""
    slots = visual(student, tokens_t, tokens_past, loops=loops).repeat_interleave(group, 0)
    rows = slots.shape[0]
    ids = torch.tensor([prompt(student, loops=loops)], device=slots.device).expand(rows, -1)
    lay = student.layout
    head = student.embed.weight.float()
    result = student.lm(inputs_embeds=embed(student, ids, slots), use_cache=True)
    cache, last = result.past_key_values, result.last_hidden_state[:, -1]
    done = torch.zeros(rows, dtype=torch.bool, device=slots.device)
    tokens, logps, masks = [], [], []
    for _ in range(max_new):
        logits = logits_of(last, head)
        if temperature > 0:
            logp_all = torch.log_softmax(logits / temperature, dim=-1)
            token = torch.multinomial(logp_all.exp(), 1, generator=generator)[:, 0]
        else:
            logp_all = torch.log_softmax(logits, dim=-1)
            token = logits.argmax(-1)
        token = torch.where(done, torch.full_like(token, lay.pad), token)
        tokens.append(token)
        logps.append(torch.where(done, 0.0, logp_all.gather(-1, token[:, None])[:, 0]))
        masks.append(~done)
        done = done | (token == lay.im_end)
        if bool(done.all()):
            break
        step = student.lm(inputs_embeds=student.embed(token[:, None]), past_key_values=cache, use_cache=True)
        cache, last = step.past_key_values, step.last_hidden_state[:, -1]
    drawn, mask = torch.stack(tokens, 1), torch.stack(masks, 1)
    captions = [decoded(student, [t for t, m in zip(row, keep) if m and t != lay.im_end], special=True)
                for row, keep in zip(drawn.tolist(), mask.tolist())]
    return Written(drawn, torch.stack(logps, 1), mask, captions)


def replay(student: Student, tokens_t: torch.Tensor, tokens_past: torch.Tensor, tokens: torch.Tensor,
           mask: torch.Tensor, *, loops: int, group: int, temperature: float) -> torch.Tensor:
    """tokens, mask: (R, N) of `sample`, R = B group -> (R, N) log-probability of each token under the current
    weights, 0 off the mask, in one parallel pass over prompt and tokens, right-padded (the LM is causal, so padding
    leaves the earlier states unchanged)."""
    slots = visual(student, tokens_t, tokens_past, loops=loops).repeat_interleave(group, 0)
    rows, width = tokens.shape
    ids = torch.tensor([prompt(student, loops=loops)], device=slots.device).expand(rows, -1)
    length = ids.shape[1]
    actions = torch.where(mask, tokens, student.layout.pad)
    attention = torch.cat([torch.ones_like(ids, dtype=torch.bool), mask], dim=1)
    # only the prompt's <image> ids are visual slots: a sampled <image> was fed back as its own embedding
    inputs = torch.cat([embed(student, ids, slots), student.embed(actions)], dim=1)
    states = student.lm(inputs_embeds=inputs, attention_mask=attention.long()).last_hidden_state
    # the state at position length - 1 + j predicts token j
    acting = states[:, length - 1:length - 1 + width][mask]
    logp = torch.log_softmax(logits_of(acting, student.embed.weight.float()) / max(temperature, 1e-6), dim=-1)
    taken = logp.gather(-1, tokens[mask][:, None])[:, 0]
    return torch.zeros(rows, width, device=slots.device, dtype=taken.dtype).masked_scatter(mask, taken)


def trained_state(student: Student) -> dict[str, torch.Tensor]:
    """The weights a stage changes, on the CPU: a LoRA student's adapters, projector and pooler; a full student's
    every weight, the tied embedding and head once, as embed.weight."""
    match student.tuning:
        case Lora():
            return {n: t.detach().cpu() for n, t in student.state_dict().items()
                    if "lora_" in n or n.startswith(("projector.", "pooler."))}
        case Full():
            return {n: p.detach().cpu() for n, p in student.named_parameters()}
        case _:
            assert_never(student.tuning)


def load_state(student: Student, state: dict[str, torch.Tensor]) -> None:
    """Every tensor `trained_state` names, cast to the student's dtype there; a missing or extra name fails."""
    assert set(state) == set(trained_state(student)), "the weights do not match the student's trained weights"
    student.load_state_dict(state, strict=False)


def merge(student: Student) -> None:
    """A LoRA student becomes a full one in float32: the adapters merged into the upcast base weights."""
    assert isinstance(student.tuning, Lora), student.tuning
    student.float()
    peft: Any = student.lm
    student.lm = peft.merge_and_unload()
    position = student.position.detach().clone()
    del student.position
    student.position = nn.Parameter(position)
    student.tuning = Full()
    student.requires_grad_(True)


def parameter_groups(student: Student, *, lr: float, vision_lr: float) -> list[dict[str, Any]]:
    """AdamW groups with their peak rate under "peak": a LoRA student's trainable weights at `lr`; a full student's
    vision encoder at `vision_lr` and every other weight at `lr`."""
    trainable = [p for p in student.parameters() if p.requires_grad]
    match student.tuning:
        case Lora():
            return [{"params": trainable, "peak": lr}]
        case Full():
            modules: list[nn.Module] = [student.patch_embedding, student.siglip, student.post_layernorm]
            vision = {id(student.position)} | {id(p) for m in modules for p in m.parameters()}
            assert sum(id(p) in vision for p in trainable) == len(vision), "the vision encoder trains"
            return [{"params": [p for p in trainable if id(p) not in vision], "peak": lr, "lr": lr},
                    {"params": [p for p in trainable if id(p) in vision], "peak": vision_lr, "lr": vision_lr}]
        case _:
            assert_never(student.tuning)


def checkpoint_layers(student: Student) -> None:
    """Recompute every SigLIP and LM layer's activations in the backward pass instead of keeping them."""
    student.checkpointing = True
    student.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})


def save_folder(student: Student, folder: Path, config: FolderConfig) -> None:
    """model.safetensors (a full student's weights in bf16, a LoRA student's trained weights as they are) and
    config.json."""
    state = trained_state(student)
    match student.tuning:
        case Lora(rank=rank):
            tuning: dict[str, Any] = {"tuning": "lora", "lora_rank": rank}
        case Full():
            state = {n: t.to(torch.bfloat16) for n, t in state.items()}
            tuning = {"tuning": "full"}
        case _:
            assert_never(student.tuning)
    folder.mkdir(parents=True, exist_ok=True)
    save_file(state, folder / WEIGHTS, metadata={"format": "pt"})
    raw = {"base": {"id": BASE, "revision": BASE_REVISION}, **tuning, "pool": POOL, "k_max": config.k_max,
           "max_new": config.max_new, "step": config.step}
    (folder / CONFIG).write_text(json.dumps(raw, indent=2) + "\n")


def folder_of(where: str) -> Path:
    """A model folder: a local directory, or a Hugging Face model repo id, downloaded unless cached."""
    if Path(where).is_dir():
        return Path(where)
    return Path(snapshot_download(where, allow_patterns=[WEIGHTS, CONFIG]))


def read_config(folder: Path) -> FolderConfig:
    raw = json.loads((folder / CONFIG).read_text())
    assert raw["base"] == {"id": BASE, "revision": BASE_REVISION} and raw["pool"] == POOL, raw
    assert raw["tuning"] in ("lora", "full"), raw["tuning"]
    tuning: Tuning = Lora(rank=int(raw["lora_rank"])) if raw["tuning"] == "lora" else Full()
    # the published folder records the step of the training checkpoint it was converted from under origin
    step = int(raw["step"]) if "step" in raw else int(raw["origin"]["step"])
    return FolderConfig(tuning=tuning, k_max=int(raw["k_max"]), max_new=int(raw["max_new"]), step=step)


def load_folder(folder: Path, *, dtype: torch.dtype) -> Student:
    """The student of a model folder, on the CPU, built in `dtype` (the projector and the pooler in float32). A full
    folder needs only the base's configuration and tokenizer; a LoRA one its weights too."""
    config = read_config(folder)
    student = Student(tuning=config.tuning, dtype=dtype, weights=isinstance(config.tuning, Lora))
    load_state(student, load_file(folder / WEIGHTS))
    return student
