"""The describers of the benchmark table: YoloMT, the open vision-language models it is compared with, the teacher.

Every describer reads the frames t - K .. t of a request (1 fps, 256 x 256, oldest first; K is the paper's lookback L)
and writes one caption of the last K + 1 seconds, greedily, in at most 80 new tokens (the teacher 120), whitespace
joined. The frames are the JPEGs of a frame store (`yolomt.data.frame_store`): a benchmark folder or a label root.

- YoloMT: the model folder of `yolomt.model`; frame t and the K frames before it pooled to 16 tokens each.
- A baseline: the model's Hugging Face repo at a pinned revision, one frozen wording of the request (`instruction`),
  one input form (K + 1 images in one user turn, or one video), one backend. Under vLLM the engine ignores the repo's
  own generation config (`generation_config="vllm"`); StreamingVLM answers a request with nothing to say with a
  silent token or an empty turn, so its silent tokens are never written and it writes at least one token.
- The teacher, Qwen3.8-27B-FP8 under vLLM, answers its labelling prompt (`yolomt.teacher`) on the same frames.

Models of the main environment (requirements.txt) run through `describe`; CUDA_VISIBLE_DEVICES picks the GPU and
`memory` is the share of its memory vLLM may use.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from importlib import metadata
from itertools import batched, product
from pathlib import Path
from typing import Any, assert_never

import torch
from PIL import Image

from yolomt.bench import HORIZON, frame_names, read_requests
from yolomt.data import FrameStore, assert_stored, frame_name, frame_store, image, listed, windows
from yolomt.teacher import TEACHER, TEACHER_REVISION, Horizon, horizon_request, limits

MAX_NEW = 80
FASTVLM_IMAGE_TOKEN = -200  # IMAGE_TOKEN_INDEX of the model's remote code: its input ids hold one per image
LIBRARIES = ("torch", "transformers", "vllm", "timm", "flash-attn", "qwen-vl-utils", "mamba-ssm", "causal-conv1d")


@dataclass(frozen=True)
class Plain:
    """Neutral: the frames, their timing and the request."""


@dataclass(frozen=True)
class Live:
    """A live narrator: what is happening now and what led to it; only the caption."""


@dataclass(frozen=True)
class Facts:
    """Concrete facts only: who and what is in view, the setting, the actions and the change; no speculation."""


Wording = Plain | Live | Facts
WORDINGS: tuple[Wording, ...] = (Plain(), Live(), Facts())


@dataclass(frozen=True)
class Images:
    """K + 1 images in one user turn, oldest first."""


@dataclass(frozen=True)
class Video:
    """The K + 1 frames as one 1 fps video; K = 0 is frame t as one image."""


Form = Images | Video


@dataclass(frozen=True)
class Vllm:
    """The model runs under vLLM, batched and greedy. remote_code: the repo's own modelling code is trusted. silent:
    strings of one token each that are never generated, and minimum: new tokens before the model may stop; penalty:
    the repetition penalty of the model's own generation config (vLLM is started with its defaults, not the repo's)."""

    remote_code: bool = False
    silent: tuple[str, ...] = ()
    minimum: int = 0
    penalty: float = 1.0


@dataclass(frozen=True)
class Transformers:
    """The model runs under transformers `generate`, left-padded batches and greedy, with its images at 384 x 384 and
    not split into tiles (SmolVLM2)."""


@dataclass(frozen=True)
class FastVlm:
    """FastVLM under its Hugging Face remote code and the library versions of `requirements/fastvlm.txt`, one clip at a
    time (a batch changes its float16 results). The code takes one <image> placeholder per image, so the K + 1 frames
    are K + 1 placeholders in the one user turn."""


Backend = Vllm | Transformers | FastVlm


@dataclass(frozen=True)
class Native:
    """The checkpoint's own dtype, bfloat16."""


@dataclass(frozen=True)
class Half:
    """float16."""


Precision = Native | Half


@dataclass(frozen=True)
class Student:
    """YoloMT. weights: a model folder or a Hugging Face model repo; chunk: windows per forward pass."""

    weights: str
    chunk: int = 32


@dataclass(frozen=True)
class Teacher:
    """The teacher as a describer. chunk: windows per `generate` call."""

    repo: str
    revision: str
    chunk: int = 256


@dataclass(frozen=True)
class Baseline:
    """An open model as a describer. repo, revision: its Hugging Face repo and commit; wording: the frozen wording of
    its request, chosen on held-out pilot windows by fact-F1; chunk: clips per `generate` call."""

    repo: str
    revision: str
    wording: Wording
    form: Form
    backend: Backend
    precision: Precision = Native()
    chunk: int = 64


Describer = Student | Teacher | Baseline

DESCRIBERS: dict[str, Teacher | Baseline] = {
    "lfm25-vl-450m": Baseline("LiquidAI/LFM2.5-VL-450M", "fc6221ca597f3315e4f82fc2df606783267b34ba", Live(), Images(),
                              Vllm()),
    "lfm25-vl-1.6b": Baseline("LiquidAI/LFM2.5-VL-1.6B", "919fde3d022e3f90a4716006f993938ee8c2eb97", Live(), Images(),
                              Vllm()),
    "smolvlm2-500m": Baseline("HuggingFaceTB/SmolVLM2-500M-Video-Instruct", "7b375e1b73b11138ff12fe22c8f2822d8fe03467",
                              Plain(), Images(), Transformers(), chunk=16),
    "smolvlm2-2.2b": Baseline("HuggingFaceTB/SmolVLM2-2.2B-Instruct", "482adb537c021c86670beed01cd58990d01e72e4",
                              Facts(), Images(), Transformers(), chunk=16),
    "internvl35-1b": Baseline("OpenGVLab/InternVL3_5-1B", "2f71cf52542334823e48a46ffba0e2bc9add3446", Plain(), Images(),
                              Vllm(remote_code=True)),
    "internvl35-2b": Baseline("OpenGVLab/InternVL3_5-2B", "7d7bd7bcc35b77b391883ebd7f9686cfa0e25cc0", Plain(), Images(),
                              Vllm(remote_code=True)),
    "qwen3-vl-2b": Baseline("Qwen/Qwen3-VL-2B-Instruct", "89644892e4d85e24eaac8bacfd4f463576704203", Plain(), Video(),
                            Vllm()),
    "qwen3-vl-4b": Baseline("Qwen/Qwen3-VL-4B-Instruct", "ebb281ec70b05090aa6165b016eac8ec08e71b17", Plain(), Video(),
                            Vllm()),
    "qwen3-vl-8b": Baseline("Qwen/Qwen3-VL-8B-Instruct", "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b", Facts(), Video(),
                            Vllm()),
    "qwen25-vl-7b": Baseline("Qwen/Qwen2.5-VL-7B-Instruct", "cc594898137f460bfe9f0759e9844b3ce807cfb5", Plain(),
                             Images(), Vllm()),
    "livecc-7b": Baseline("chenjoya/LiveCC-7B-Instruct", "dbe7415c5d2024c3e2d7f898ecbc58fd9fc8da2f", Live(), Images(),
                          Vllm()),
    "streamingvlm-7b": Baseline("mit-han-lab/StreamingVLM", "7ac1cef46d20432053a9814838760318cb26b348", Plain(),
                                Images(), Vllm(silent=("...", " ...", "…"), minimum=1, penalty=1.05)),
    "fastvlm-0.5b": Baseline("apple/FastVLM-0.5B", "16375720c2d673fa583e57e9876afde27549c7d0", Live(), Images(),
                             FastVlm(), Half(), chunk=1),
    "teacher27b": Teacher(TEACHER, TEACHER_REVISION),
}


@dataclass(frozen=True)
class Generated:
    """text: the caption; finish: "stop" or "length", prompt_tokens: the prompt's length in tokens, both absent from
    YoloMT's own decoder; visual: (kept, total) visual tokens of a model that drops some before its language model
    (prompt_tokens counts the kept)."""

    text: str
    finish: str | None = None
    prompt_tokens: int | None = None
    visual: tuple[int, int] | None = None


# a model's clips of one lookback: (B, K + 1, 3, 256, 256) uint8, oldest frame first
Engine = Callable[[torch.Tensor], list[Generated]]


def instruction(wording: Wording, *, loops: int) -> str:
    """The user text for the last K + 1 = loops + 1 frames."""
    n = loops + 1
    match wording:
        case Plain():
            if loops == 0:
                return "This image is the current frame of a video. Describe in 30 to 50 words what is happening."
            return (f"These {n} frames are from a video, sampled at 1 frame per second over the last {n} seconds, "
                    f"oldest first; the last frame is the current moment. Describe in 30 to 50 words what happens.")
        case Live():
            if loops == 0:
                return ("You are describing a live video once per second. This image is its current frame. Write one "
                        "caption of 30 to 50 words saying what is happening now. Output only the caption.")
            return (f"You are describing a live video once per second. These {n} frames are the video at 1 frame per "
                    f"second over the last {n} seconds, oldest first; the final frame is the present. Write one "
                    f"caption of 30 to 50 words saying what is happening now and what led to it. Output only the "
                    f"caption.")
        case Facts():
            if loops == 0:
                return ("This image is one frame of a video. In 30 to 50 words, state only concrete facts: who or "
                        "what is in view, the setting and what they are doing. No speculation, no preamble.")
            return (f"These {n} frames are consecutive frames at 1 frame per second covering the last {n} seconds of "
                    f"a video, oldest first. In 30 to 50 words, state only concrete facts: who or what is in view, "
                    f"the setting, what they do and how the scene changes from the first frame to the last. No "
                    f"speculation, no preamble.")
        case _:
            assert_never(wording)


def pil(clip: torch.Tensor) -> list[Image.Image]:
    """(F, 3, 256, 256) uint8 -> F RGB images."""
    return [Image.fromarray(frame.permute(1, 2, 0).contiguous().numpy()) for frame in clip]


def dtype_of(precision: Precision) -> torch.dtype:
    match precision:
        case Native():
            return torch.bfloat16
        case Half():
            return torch.float16
        case _:
            assert_never(precision)


def student_engine(model: Student) -> Engine:
    """YoloMT's own greedy decoder on frame t and the K frames before it, newest first, under bfloat16 autocast."""
    from yolomt.model import folder_of, load_folder, read_config, write

    folder = folder_of(model.weights)
    config = read_config(folder)
    student = load_folder(folder, dtype=torch.bfloat16).to("cuda").eval()

    def generate(clips: torch.Tensor) -> list[Generated]:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            texts = write(student, clips[:, -1].to("cuda"), clips[:, :-1].flip(1).to("cuda"), loops=clips.shape[1] - 1,
                          max_new=config.max_new)
        return [Generated(text) for text in texts]

    return generate


def video_request(clip: torch.Tensor, *, text: str, template: str) -> dict[str, Any]:
    """clip: (K + 1, 3, H, W) uint8, K >= 1, as a 1 fps video for Qwen3-VL's vLLM processor, built as the teacher's
    request is: frames_indices are the clip's seconds, an even count doubles the first frame (the processor pairs
    frames in time and pads the last), and `max_pixels` bounds the whole clip at K + 3 frames, so none is resized."""
    loops = clip.shape[0] - 1
    seconds = list(range(loops + 1))
    if len(seconds) % 2 == 0:
        seconds.insert(0, 0)
    metadata = {"fps": 1.0, "duration": float(loops + 1), "total_num_frames": len(seconds), "frames_indices": seconds,
                "video_backend": "pyav", "do_sample_frames": False}
    pixels = {"min_pixels": 64 * 32 * 32, "max_pixels": (loops + 3) * clip.shape[2] * clip.shape[3]}
    return {"prompt": template.replace("PROMPT_TEXT", text),
            "multi_modal_data": {"video": (clip[seconds].permute(0, 2, 3, 1).contiguous().numpy(), metadata)},
            "mm_processor_kwargs": pixels}


def vllm_engine(model: Baseline, backend: Vllm, *, memory: float) -> Engine:
    """vLLM, greedy: a video-form model reads the K + 1 frames as one video (K = 0: one image), the others as K + 1
    images in one user turn."""
    from vllm import LLM, SamplingParams

    video = isinstance(model.form, Video)
    llm = LLM(model=model.repo, revision=model.revision, tokenizer_revision=model.revision,
              code_revision=model.revision, trust_remote_code=backend.remote_code, dtype=dtype_of(model.precision),
              max_model_len=8192, gpu_memory_utilization=memory, seed=0, generation_config="vllm",
              limit_mm_per_prompt={"image": 1, "video": 1} if video else {"image": HORIZON + 1})
    tokenizer = llm.get_tokenizer()
    never: dict[int, float] = {}
    for word in backend.silent:
        (token,) = tokenizer.encode(word, add_special_tokens=False)
        never[token] = -100.0
    greedy = SamplingParams(n=1, temperature=0.0, max_tokens=MAX_NEW, min_tokens=backend.minimum,
                            logit_bias=never or None, repetition_penalty=backend.penalty)
    template = tokenizer.apply_chat_template(
        [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": "PROMPT_TEXT"}]}],
        tokenize=False, add_generation_prompt=True) if video else ""

    def generate(clips: torch.Tensor) -> list[Generated]:
        loops = clips.shape[1] - 1
        text = instruction(model.wording, loops=loops)
        match model.form:
            case Video() if loops > 0:
                outputs = llm.generate([video_request(clip, text=text, template=template) for clip in clips], greedy,
                                       use_tqdm=False)
            case Video() | Images():
                outputs = llm.chat([[{"role": "user", "content": [
                    *({"type": "image_pil", "image_pil": frame} for frame in pil(clip)),
                    {"type": "text", "text": text}]}] for clip in clips], greedy, use_tqdm=False)
            case _:
                assert_never(model.form)
        return [Generated(o.outputs[0].text, str(o.outputs[0].finish_reason), len(o.prompt_token_ids)) for o in outputs]

    return generate


def transformers_engine(model: Baseline) -> Engine:
    """transformers `generate`, greedy, the clips of a call left-padded to one length, K + 1 images in one user turn."""
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(model.repo, revision=model.revision)  # type: ignore[no-untyped-call]
    processor.tokenizer.padding_side = "left"
    net = AutoModelForImageTextToText.from_pretrained(model.repo, revision=model.revision,
                                                      dtype=dtype_of(model.precision))
    net = net.to("cuda").eval()  # type: ignore[arg-type]
    eos = net.generation_config.eos_token_id
    stops = torch.tensor(eos if isinstance(eos, list) else [eos], device="cuda")

    @torch.no_grad()
    def generate(clips: torch.Tensor) -> list[Generated]:
        loops = clips.shape[1] - 1
        turn = [{"role": "user", "content": [*([{"type": "image"}] * (loops + 1)),
                                             {"type": "text", "text": instruction(model.wording, loops=loops)}]}]
        prompt = processor.apply_chat_template(turn, add_generation_prompt=True)
        inputs = processor(text=[prompt] * len(clips), images=[pil(clip) for clip in clips], return_tensors="pt",
                           padding=True, do_image_splitting=False).to("cuda")
        made = net.generate(**inputs, do_sample=False, max_new_tokens=MAX_NEW)[:, inputs["input_ids"].shape[1]:]
        return [Generated(processor.decode(row, skip_special_tokens=True),
                          "stop" if bool(torch.isin(row, stops).any()) else "length", int(mask.sum()))
                for row, mask in zip(made, inputs["attention_mask"], strict=True)]

    return generate


def fastvlm_engine(model: Baseline) -> Engine:
    """FastVLM as its model card runs it: the chat template rendered to a string, the text around each <image> tokenized
    without special tokens with the image token id spliced in, pixel values from the model's own image processor (a
    256 x 256 frame is resized to 1024 x 1024, 256 visual tokens), `generate(inputs, attention_mask, images)`, greedy,
    stopping at the config's end-of-turn token. K = 0 is the card's own prompt."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = dtype_of(model.precision)
    tokenizer = AutoTokenizer.from_pretrained(model.repo, revision=model.revision, trust_remote_code=True)
    net = AutoModelForCausalLM.from_pretrained(model.repo, revision=model.revision, torch_dtype=dtype,
                                               trust_remote_code=True)
    net = net.to("cuda").eval()  # type: ignore[arg-type]
    tower = net.get_vision_tower()
    stops = torch.tensor([net.config.eos_token_id], device="cuda")

    @torch.inference_mode()
    def generate(clips: torch.Tensor) -> list[Generated]:
        assert len(clips) == 1, "FastVLM captions one clip at a time"
        loops = clips.shape[1] - 1
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": "<image>\n" * (loops + 1) + instruction(model.wording, loops=loops)}],
            add_generation_prompt=True, tokenize=False)
        assert isinstance(rendered, str)
        around = [tokenizer(part, return_tensors="pt", add_special_tokens=False).input_ids
                  for part in rendered.split("<image>")]
        marker = torch.tensor([[FASTVLM_IMAGE_TOKEN]], dtype=around[0].dtype)
        ids = torch.cat([around[0], *[torch.cat([marker, part], dim=1) for part in around[1:]]], dim=1).to("cuda")
        pixels = tower.image_processor(images=pil(clips[0]), return_tensors="pt")["pixel_values"].to("cuda", dtype)
        made = net.generate(inputs=ids, attention_mask=torch.ones_like(ids), images=pixels, do_sample=False,
                            max_new_tokens=MAX_NEW, eos_token_id=net.config.eos_token_id)
        (reply,) = tokenizer.batch_decode(made, skip_special_tokens=True)
        return [Generated(reply.strip(), "stop" if bool(torch.isin(made[0], stops).any()) else "length",
                          ids.shape[1] + (loops + 1) * (tower.num_patches - 1))]

    return generate


def teacher_engine(model: Teacher, *, memory: float) -> Engine:
    """The teacher's labelling request on the frames (`yolomt.teacher.horizon_request`), vLLM, greedy, thinking off."""
    from vllm import LLM, SamplingParams

    llm = LLM(model=model.repo, revision=model.revision, tokenizer_revision=model.revision, max_model_len=2048,
              max_num_seqs=128, gpu_memory_utilization=memory, limit_mm_per_prompt={"video": 1, "image": 0}, seed=0)
    template = llm.get_tokenizer().apply_chat_template(
        [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": "PROMPT_TEXT"}]}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    greedy = SamplingParams(n=1, temperature=0.0, max_tokens=2 * limits(Horizon()).words)

    def generate(clips: torch.Tensor) -> list[Generated]:
        loops = clips.shape[1] - 1
        asked = [horizon_request(dict(enumerate(clip.permute(0, 2, 3, 1).contiguous().numpy())), loops, lookback=loops,
                                 template=template) for clip in clips]
        outputs = llm.generate(asked, greedy, use_tqdm=False)
        return [Generated(o.outputs[0].text, str(o.outputs[0].finish_reason), len(o.prompt_token_ids)) for o in outputs]

    return generate


def baseline_engine(model: Baseline, *, memory: float) -> Engine:
    match model.backend:
        case Vllm() as backend:
            return vllm_engine(model, backend, memory=memory)
        case Transformers():
            return transformers_engine(model)
        case FastVlm():
            return fastvlm_engine(model)
        case _:
            assert_never(model.backend)


def engine(model: Describer, *, memory: float) -> Engine:
    """The model loaded on the GPU; memory: the share of its memory a vLLM model may use."""
    match model:
        case Student():
            return student_engine(model)
        case Teacher():
            return teacher_engine(model, memory=memory)
        case Baseline():
            return baseline_engine(model, memory=memory)
        case _:
            assert_never(model)


def describe(model: Describer, store: FrameStore, clips: Sequence[tuple[str, ...]], *,
             memory: float) -> list[Generated]:
    """The caption of each clip, in order; clips[i] holds the store names of its K + 1 frames, oldest first. The
    clips of one K go through the model together, `model.chunk` at a time."""
    assert_stored(store, [name for names in clips for name in names])
    generate = engine(model, memory=memory)
    done: dict[int, Generated] = {}
    for loops in sorted({len(names) - 1 for names in clips}):
        members = [i for i, names in enumerate(clips) if len(names) == loops + 1]
        for part in batched(members, model.chunk):
            frames = torch.stack([torch.stack([image(store, name) for name in clips[i]]) for i in part])
            for i, one in zip(part, generate(frames), strict=True):
                done[i] = replace(one, text=" ".join(one.text.split()))
        print(f"K = {loops}: {len(done)}/{len(clips)} clips", flush=True)
    return [done[i] for i in range(len(clips))]


def fields(name: str, model: Describer, one: Generated) -> dict[str, Any]:
    """The record of a caption besides its request: YoloMT's holds the caption only, a baseline's also how decoding
    ended, the prompt's length, the model and the wording (and the visual tokens kept, if it drops some)."""
    match model:
        case Student():
            return {"caption": one.text}
        case Teacher():
            wording: int | str = "horizon-w30-50"
        case Baseline():
            wording = WORDINGS.index(model.wording)
        case _:
            assert_never(model)
    assert one.finish is not None and one.prompt_tokens is not None
    kept = {} if one.visual is None else {"visual_kept": one.visual[0], "visual_total": one.visual[1]}
    return {"caption": one.text, "finish": one.finish, "prompt_tokens": one.prompt_tokens, "model": name,
            "wording": wording, **kept}


def bench_rows(name: str, model: Describer, folder: Path, *, loops: int | None,
               memory: float) -> list[dict[str, Any]]:
    """The record of each request of a benchmark folder, in the order of its requests.jsonl; loops: the lookback K
    capped at each request's own, or None for the request's own (the protocol)."""
    requests = read_requests(folder / "requests.jsonl")
    clips = [frame_names(request, loops=loops) for request in requests]
    made = describe(model, frame_store([folder]), clips, memory=memory)
    return [{"video": request.video, "t": request.t, "loops": len(names) - 1, **fields(name, model, one)}
            for request, names, one in zip(requests, clips, made, strict=True)]


def window_rows(name: str, model: Describer, listing: Path, labels: list[Path], *, loops: Sequence[int],
                memory: float) -> list[dict[str, Any]]:
    """The record of each window of a window list that the label roots hold, at each lookback in `loops`."""
    wanted = listed(listing)
    items = [w for w in windows(labels, set(wanted.values()), derived=False) if (w.video_id, w.t1) in wanted]
    assert items, f"the label roots hold no window of {listing}"
    pairs = list(product(items, loops))
    clips = [tuple(frame_name(w.video_id, w.t1 - j) for j in range(k, -1, -1)) for w, k in pairs]
    made = describe(model, frame_store(labels), clips, memory=memory)
    return [{"key": w.key, "loops": k, **fields(name, model, one)} for (w, k), one in zip(pairs, made, strict=True)]


def provenance() -> dict[str, Any]:
    """The code's git revision, whether its tree was dirty and the versions of the libraries installed."""
    git = ["git", "-C", str(Path(__file__).resolve().parents[1])]
    sha = subprocess.run([*git, "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    status = subprocess.run([*git, "status", "--porcelain", "--untracked-files=no"], check=True, capture_output=True,
                            text=True).stdout.strip()
    return {"git": sha, "dirty": bool(status),
            "libraries": {name: d.version for d in metadata.distributions()
                          if (name := d.metadata["Name"].lower().replace("_", "-")) in LIBRARIES}}
