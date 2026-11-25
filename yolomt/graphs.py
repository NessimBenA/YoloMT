"""The two graphs of the real-time engine, in PyTorch and as ONNX, and the plan of the prompts they serve.

vision: one 256 x 256 frame (uint8) -> its 64 projector tokens in FP16 and SigLIP's mean feature in float32. Every
LayerNorm computes in float32 inside the FP16 graph: SigLIP's residual stream reaches |x| of about 900, whose square
overflows FP16, and a plain FP16 engine drifts (cosine 0.67 against float32 on the vision tokens).

writer: T tokens of the LM from explicit states, so that one graph serves the prefill chunks (T up to 115) and the
decode steps (T = 1). The states are the 10 short-convolution windows (1, d, 2) and the 6 attention layers' K and V
caches (1, 8, T_MAX, 64); the position is a tensor, a token's key and value enter the cache by a one-hot matmul (no
in-place write), and an additive mask hides every cache column past the token. The graph returns the greedy next token
over the rows of the head kept by `data/vocab_trim.json`, its embedding (the head is tied), the next position, the last
hidden state and the new states.

At lookback L the prompt is 79 + 18 L tokens; the prefill cuts it into the fewest chunks of at most 115 tokens, as even
as possible. The pooler runs outside the engine, once per frame.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from yolomt.model import GRID, POOLED, SIDE, TOKENS, Student, patches, prompt

PREFILL_CHUNK = 115
# the writer engine's optimization profiles: two T = 1 profiles the decode steps alternate on, then one profile for
# every prefill chunk, T = 1 .. the longest chunk, tuned at it
DECODE = (0, 1)
EXTEND = 2
OPSET = 17


class Float32LayerNorm(nn.Module):
    """A LayerNorm that computes in float32 and returns the dtype of its input."""

    def __init__(self, norm: nn.LayerNorm) -> None:
        super().__init__()
        self.shape = tuple(norm.normalized_shape)
        self.eps = norm.eps
        self.weight = norm.weight
        self.bias = norm.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.layer_norm(x.float(), self.shape, self.weight.float(), self.bias.float(), self.eps)
        return out.to(x.dtype)


def float32_layer_norms(module: nn.Module) -> None:
    for name, child in module.named_children():
        if isinstance(child, nn.LayerNorm):
            setattr(module, name, Float32LayerNorm(child))
        else:
            float32_layer_norms(child)


def engine_precision(student: Student) -> None:
    """In place: every weight in FP16 but RoPE's inverse frequencies, and every LayerNorm in float32."""
    rotary: Any = student.lm.rotary_emb
    inv_freq = rotary.inv_freq.clone()
    student.half()
    rotary.inv_freq = inv_freq
    float32_layer_norms(student)


class Vision(nn.Module):
    def __init__(self, student: Student) -> None:
        super().__init__()
        self.student = student

    def forward(self, frame: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """frame: (1, 3, 256, 256) uint8 -> tokens (1, 64, d), siglip (1, 768)."""
        s = self.student
        hidden = s.patch_embedding(patches(frame).to(s.position.dtype)) + s.position
        for layer in s.siglip:
            hidden = layer(hidden, None)
        hidden = s.post_layernorm(hidden)
        tokens = s.projector(hidden.view(1, GRID, GRID, -1).to(s.projector.linear_1.weight.dtype))
        return tokens.reshape(1, TOKENS, -1), hidden.float().mean(1)


def qkv(attn: Any, h: torch.Tensor, rope: tuple[torch.Tensor, torch.Tensor],
        head_dim: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """q: (1, heads, T, head_dim), k and v: (1, kv_heads, T, head_dim), after per-head RMSNorm and RoPE."""
    shape = (1, h.shape[1], -1, head_dim)
    q = attn.q_layernorm(attn.q_proj(h).view(shape)).transpose(1, 2)
    k = attn.k_layernorm(attn.k_proj(h).view(shape)).transpose(1, 2)
    v = attn.v_proj(h).view(shape).transpose(1, 2)
    cos, sin = rope[0][:, None], rope[1][:, None]
    half = head_dim // 2

    def rotate(x: torch.Tensor) -> torch.Tensor:
        return x * cos + torch.cat((-x[..., half:], x[..., :half]), dim=-1) * sin

    return rotate(q), rotate(k), v


def attend(attn: Any, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor,
           groups: int) -> torch.Tensor:
    """Grouped-query attention by repeating each K/V head `groups` times; mask is additive, (1, 1, T, S)."""
    k = k.repeat_interleave(groups, dim=1)
    v = v.repeat_interleave(groups, dim=1)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=attn.scaling)
    result: torch.Tensor = attn.out_proj(out.transpose(1, 2).reshape(1, q.shape[2], -1))
    return result


class Chunk(nn.Module):
    """T tokens of the LM from explicit states:
    (embeds (1, T, d), position (1,), conv (1, d, 2) x 10, k and v (1, kv_heads, t_max, head_dim) x 6 each)
    -> (next_token (1,), next_embeds (1, 1, d), position + T, hidden (1, d), the new conv, k and v).

    Token i sits at position + i: its key and value replace the cache column position + i. A conv window is the last
    two columns of [window, B * x]. The next token is the argmax over the head's kept rows, `kept` their ids."""

    def __init__(self, lm: Any, *, kept: list[int], t_max: int) -> None:
        super().__init__()
        assert not lm.config.conv_bias, "Chunk drops the conv bias"
        self.lm = lm
        self.t_max = t_max
        weight = lm.embed_tokens.weight.detach()
        self.kept = torch.tensor(kept, device=weight.device)
        self.head = weight[self.kept].contiguous()
        self.attention = tuple(bool(layer.is_attention_layer) for layer in lm.layers)
        self.heads = lm.config.num_attention_heads
        self.kv_heads = lm.config.num_key_value_heads
        self.head_dim = lm.config.hidden_size // self.heads

    def forward(self, embeds: torch.Tensor, position: torch.Tensor, *states: torch.Tensor) -> tuple[torch.Tensor, ...]:
        n_conv = sum(not a for a in self.attention)
        n_attn = len(self.attention) - n_conv
        conv_in, k_in, v_in = states[:n_conv], states[n_conv:n_conv + n_attn], states[n_conv + n_attn:]
        conv_out: list[torch.Tensor] = []
        k_out: list[torch.Tensor] = []
        v_out: list[torch.Tensor] = []
        dtype = embeds.dtype
        positions = position + torch.arange(embeds.shape[1], device=embeds.device)
        rope = self.lm.rotary_emb(embeds, positions[None])
        index = torch.arange(self.t_max, device=embeds.device)
        mask = torch.where(index[None, :] > positions[:, None], torch.finfo(dtype).min, 0.0).to(dtype)[None, None]
        onehot = (index[:, None] == positions[None, :]).to(dtype)
        written = onehot.sum(-1)[None, None, :, None] > 0
        x = embeds
        for layer, attention in zip(self.lm.layers, self.attention):
            h = layer.operator_norm(x)
            if attention:
                q, k, v = qkv(layer.self_attn, h, rope, self.head_dim)
                k_new = torch.where(written, onehot @ k, k_in[len(k_out)])
                v_new = torch.where(written, onehot @ v, v_in[len(v_out)])
                k_out.append(k_new)
                v_out.append(v_new)
                y = attend(layer.self_attn, q, k_new, v_new, mask, self.heads // self.kv_heads)
            else:
                conv = layer.conv
                b, c, u = conv.in_proj(h).transpose(-1, -2).chunk(3, dim=-2)
                window = torch.cat((conv_in[len(conv_out)], b * u), dim=-1)
                z = F.conv1d(window, conv.conv.weight, None, groups=window.shape[1])
                conv_out.append(window[:, :, -(conv.conv_kernel_size - 1):])
                y = conv.out_proj((c * z).transpose(-1, -2))
            x = x + y
            x = x + layer.feed_forward(layer.ffn_norm(x))
        hidden = self.lm.embedding_norm(x[:, -1])
        best = (hidden @ self.head.T).argmax(-1)
        return (self.kept[best], self.head[best][:, None], position + embeds.shape[1], hidden,
                *conv_out, *k_out, *v_out)


def io_names(chunk: Chunk) -> tuple[list[str], list[str]]:
    """The writer graph's input names and output names, in order: conv windows, then K caches, then V caches."""
    n_conv = sum(not a for a in chunk.attention)
    n_attn = len(chunk.attention) - n_conv
    states = [f"conv{i}" for i in range(n_conv)] + [f"k{i}" for i in range(n_attn)] + [f"v{i}" for i in range(n_attn)]
    return (["embeds", "position", *states],
            ["next_token", "next_embeds", "next_position", "hidden", *[f"{s}_out" for s in states]])


def empty_states(chunk: Chunk, *, dtype: torch.dtype, device: torch.device) -> list[torch.Tensor]:
    """The zero conv windows (1, d, 2), then K and V caches (1, kv_heads, t_max, head_dim)."""
    width = chunk.lm.config.hidden_size
    conv = [torch.zeros(1, width, chunk.lm.config.conv_L_cache - 1, dtype=dtype, device=device)
            for attention in chunk.attention if not attention]
    kv = [torch.zeros(1, chunk.kv_heads, chunk.t_max, chunk.head_dim, dtype=dtype, device=device)
          for _ in range(2) for attention in chunk.attention if attention]
    return conv + kv


@dataclass(frozen=True)
class Call:
    """One call of the writer graph: its inputs (embeds, position, states) and its outputs."""

    inputs: tuple[torch.Tensor, ...]
    outputs: tuple[torch.Tensor, ...]


def prefill_calls(chunk: Chunk, embeds: torch.Tensor, *, chunks: list[int]) -> list[Call]:
    """embeds (1, N, d) through `chunk` from empty states at position 0 in consecutive calls of `chunks` tokens, each
    from the position and states the one before returned."""
    assert sum(chunks) == embeds.shape[1], (chunks, embeds.shape)
    feed = (torch.zeros(1, dtype=torch.long, device=embeds.device),
            *empty_states(chunk, dtype=embeds.dtype, device=embeds.device))
    calls = []
    start = 0
    for n in chunks:
        inputs = (embeds[:, start:start + n], *feed)
        outputs = chunk(*inputs)
        calls.append(Call(inputs, outputs))
        feed = (outputs[2], *outputs[4:])
        start += n
    return calls


def decode_call(chunk: Chunk, previous: Call) -> Call:
    """The decode step after `previous`: its next embedding and position and its states."""
    inputs = (previous.outputs[1], previous.outputs[2], *previous.outputs[4:])
    return Call(inputs, chunk(*inputs))


def chunks_of(length: int, *, chunk: int) -> list[int]:
    """A prompt of `length` tokens as the fewest consecutive chunks of at most `chunk` tokens, as even as possible,
    the longer first."""
    n = -(-length // chunk)
    sizes = [length // n + int(i < length % n) for i in range(n)]
    assert sum(sizes) == length and max(sizes) <= chunk, sizes
    return sizes


def t_max_of(longest: int, *, written: int) -> int:
    """Cache positions for prompts of at most `longest` tokens that write `written` tokens: the last token is fed at
    position longest + written - 1; rounded up to 16 for the FP16 kernels' tiles."""
    return -(-(longest + written) // 16) * 16


def check_plan() -> None:
    """The chunks of the paper's table at L = 0, 1, 2, 4, 8, 16 and the cache length of L = 16."""
    assert [chunks_of(n, chunk=PREFILL_CHUNK) for n in (79, 97, 115, 151, 223, 367)] == [
        [79], [97], [115], [76, 75], [112, 111], [92, 92, 92, 91]]
    assert t_max_of(367, written=80) == 448


@dataclass(frozen=True)
class Prompt:
    """The prompt at one lookback: embeds (N, d) FP16, the rows of its ids; chunks: the prefill's chunk lengths;
    pooled[j]: the first row of the j-th pooled block (16 rows), oldest frame first; current: the first of the 64 rows
    of frame t. The rows of the blocks and of frame t are placeholders until a frame fills them."""

    embeds: torch.Tensor
    chunks: list[int]
    pooled: list[int]
    current: int


@dataclass(frozen=True)
class Runtime:
    """What the runner needs besides the engines: prompts[L] for L = 0 .. k_max; the pooler's residual W (d, d) and
    b (d,) in float32; the id of <|im_end|>; the rows of the kept head; the cache length t_max; the most tokens a
    caption writes."""

    prompts: list[Prompt]
    pooler_weight: torch.Tensor
    pooler_bias: torch.Tensor
    im_end: int
    head_rows: int
    t_max: int
    written: int


def prompt_of(student: Student, *, loops: int) -> Prompt:
    ids = prompt(student, loops=loops)
    slots = [i for i, token in enumerate(ids) if token == student.layout.image]
    pooled = [slots[POOLED * j] for j in range(loops)]
    current = slots[POOLED * loops]
    assert slots == [*[at + i for at in pooled for i in range(POOLED)], *range(current, current + TOKENS)]
    embeds = student.embed(torch.tensor([ids], device=student.embed.weight.device))[0].detach().half().cpu()
    return Prompt(embeds=embeds, chunks=chunks_of(len(ids), chunk=PREFILL_CHUNK), pooled=pooled, current=current)


def runtime_of(student: Student, *, k_max: int, written: int, kept: list[int]) -> Runtime:
    """The runtime data of a float32 student for lookbacks 0 .. k_max."""
    check_plan()
    prompts = [prompt_of(student, loops=k) for k in range(k_max + 1)]
    assert [len(p.embeds) for p in prompts] == [79 + 18 * k for k in range(k_max + 1)]
    pooler = student.pooler.residual
    t_max = t_max_of(len(prompts[-1].embeds), written=written)
    return Runtime(prompts=prompts, pooler_weight=pooler.weight.detach().float().cpu(),
                   pooler_bias=pooler.bias.detach().float().cpu(), im_end=student.layout.im_end, head_rows=len(kept),
                   t_max=t_max, written=written)


def save_runtime(runtime: Runtime, path: Path) -> None:
    torch.save(asdict(runtime), path)


def load_runtime(path: Path) -> Runtime:
    raw = torch.load(path, weights_only=True)
    return Runtime(prompts=[Prompt(**p) for p in raw.pop("prompts")], **raw)


def export(module: nn.Module, args: tuple[torch.Tensor, ...], *, inputs: list[str], outputs: list[str],
           dynamic: dict[str, dict[int, str]], path: Path) -> None:
    torch.onnx.export(module, args, str(path), input_names=inputs, output_names=outputs, dynamic_axes=dynamic,
                      opset_version=OPSET, dynamo=False)


@torch.no_grad()
def export_vision(vision: Vision, path: Path) -> None:
    """vision.onnx of an FP16 `engine_precision` student on the CPU."""
    frame = torch.zeros(1, 3, SIDE, SIDE, dtype=torch.uint8)
    export(vision.eval(), (frame,), inputs=["frame"], outputs=["tokens", "siglip"], dynamic={}, path=path)


@torch.no_grad()
def export_writer(chunk: Chunk, path: Path, *, tokens: int) -> None:
    """writer.onnx of an FP16 chunk on the CPU, traced on a prompt of `tokens` rows, T dynamic."""
    inputs, outputs = io_names(chunk)
    embeds = torch.zeros(1, tokens, chunk.lm.config.hidden_size, dtype=torch.float16)
    args = (embeds, torch.zeros(1, dtype=torch.long), *empty_states(chunk, dtype=torch.float16, device=embeds.device))
    export(chunk.eval(), args, inputs=inputs, outputs=outputs, dynamic={"embeds": {1: "T"}}, path=path)
