"""Teacher labels of a window list, with the student frames the windows cover.

    python scripts/label.py --windows data/windows/captioner.tsv.gz --task 2s --out <labels>/captioner
    python scripts/label.py --windows data/windows/sft.tsv.gz --task 16s --out <labels>/sft
    python scripts/label.py --windows data/windows/rl.tsv.gz --task 16s --samples 8 --out <labels>/rl
    python scripts/label.py --windows data/windows/val.tsv.gz --task 16s --samples 8 --out <labels>/val

A window list (`data/windows/*.tsv.gz`) gives per video its archive, split and windows: the end seconds t of each run
of consecutive windows. Archive by archive (`data/archives.tsv`), the archive is downloaded into `--out`/downloads,
the listed videos are decoded, the teacher writes one greedy caption per window and, with `--samples S`, S captions
sampled at temperature 0.7 for `scripts/derive.py`, and the archive is deleted. Per chunk of videos the folder
`--out`/<source>/<archive>/ receives `frames_XXXXX.tar` and `records_XXXXX.jsonl` (`yolomt.data`); a finished chunk
or archive is not redone when the command runs again. Needs vLLM (`pip install -e .[vllm]`) and a GPU.
"""

from __future__ import annotations

import argparse
import gzip
import io
import itertools
import json
import shutil
import tarfile
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from yolomt.model import BASE, BASE_REVISION
from yolomt.teacher import TEACHER, Horizon, Short, Task, grid_span, kept, limits, request, start
from yolomt.video import Decoded, archives, decode, fetch, members, video_id_of


@dataclass(frozen=True)
class Video:
    """A listed video: runs[r] holds the end seconds of run r's windows."""

    video_id: str
    split: str
    runs: list[list[int]]


def listed(path: Path) -> dict[str, dict[str, Video]]:
    """The window list: per archive, its videos by id."""
    out: dict[str, dict[str, Video]] = defaultdict(dict)
    for line in gzip.decompress(path.read_bytes()).decode().splitlines()[1:]:
        archive, video_id, split, runs = line.split("\t")
        out[archive][video_id] = Video(video_id, split, [[int(t) for t in run.split(",")] for run in runs.split(";")])
    return out


def prepared(task: Task, video: Video, data: bytes) -> Decoded:
    ends = [t for run in video.runs for t in run]
    seconds = {s for t in ends for s in range(start(task, t), t + 1)}
    decoded = decode(data, seconds=seconds, grid={k for t in ends for k in grid_span(task, t)})
    assert set(decoded.student) == seconds, f"{video.video_id}: seconds {sorted(seconds - set(decoded.student))} missing"
    return decoded


def write_chunk(folder: Path, number: int, records: list[dict[str, Any]], frames: dict[str, bytes]) -> None:
    """The frames tar, then the records, renamed into place last: a chunk with records is finished."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for name in sorted(frames):
            info = tarfile.TarInfo(name)
            info.size = len(frames[name])
            tar.addfile(info, io.BytesIO(frames[name]))
    (folder / f"frames_{number:05d}.tar").write_bytes(buffer.getvalue())
    partial = folder / f"records_{number:05d}.jsonl.tmp"
    partial.write_text("".join(json.dumps(r) + "\n" for r in records))
    partial.rename(folder / f"records_{number:05d}.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--windows", type=Path, required=True, help="a window list of data/windows")
    parser.add_argument("--task", choices=["2s", "16s"], required=True, help="the kind of window")
    parser.add_argument("--out", type=Path, required=True, help="the label root to write")
    parser.add_argument("--samples", type=int, default=0, help="captions sampled per window besides the greedy one")
    parser.add_argument("--archives", type=Path, default=Path("data/archives.tsv"))
    parser.add_argument("--videos-per-chunk", type=int, default=48)
    parser.add_argument("--max-num-seqs", type=int, default=64, help="vLLM's concurrent sequences")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--workers", type=int, default=16, help="decoding threads")
    args = parser.parse_args()
    from vllm import LLM, SamplingParams

    task: Task = Short() if args.task == "2s" else Horizon()
    budget = 2 * limits(task).words
    greedy = SamplingParams(n=1, temperature=0.0, max_tokens=budget)
    sampled = SamplingParams(n=max(1, args.samples), temperature=0.7, top_p=1.0, max_tokens=budget)
    engine = LLM(model=TEACHER, max_model_len=2048, max_num_seqs=args.max_num_seqs, seed=0,
                 gpu_memory_utilization=args.gpu_memory_utilization, limit_mm_per_prompt={"video": 1, "image": 0})
    template = engine.get_tokenizer().apply_chat_template(
        [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": "PROMPT_TEXT"}]}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    tokenizer = AutoTokenizer.from_pretrained(BASE, revision=BASE_REVISION)
    known = archives(args.archives)
    pool = ThreadPoolExecutor(max_workers=args.workers)

    def label(folder: Path, number: int, videos: list[Video], futures: list[Future[Decoded]]) -> None:
        decoded = [f.result() for f in futures]
        ends = [(v, d, r, t) for v, d in zip(videos, decoded) for r, run in enumerate(v.runs) for t in run]
        prompts, params = [], []
        for v, d, _, t in ends:
            asked = request(task, d, t, template)
            if args.samples:
                prompts.append(asked)
                params.append(sampled)
            prompts.append(asked)
            params.append(greedy)
        outputs = iter(engine.generate(prompts, params, use_tqdm=False))
        records: list[dict[str, Any]] = []
        frames: dict[str, bytes] = {}
        for v, d, r, t in ends:
            many = next(outputs).outputs if args.samples else []
            one = next(outputs).outputs[0]
            samples = [text for text in (" ".join(o.text.split()) for o in many if o.finish_reason == "stop")
                       if kept(text, tokenizer, task)]
            canonical = " ".join(one.text.split())
            if one.finish_reason != "stop" or not kept(canonical, tokenizer, task):
                continue
            t0 = start(task, t)
            records.append({"video_id": v.video_id, "run": f"{v.video_id}#{r}", "t0": t0, "t1": t, "split": v.split,
                            "canonical": canonical, "samples": samples})
            frames |= {f"{v.video_id}/{s}.jpg": d.student[s] for s in range(t0, t + 1)}
        write_chunk(folder, number, records, frames)

    for name, wanted in listed(args.windows).items():
        archive = known[name]
        folder = args.out / archive.source / archive.name
        if (folder / "done").exists():
            continue
        folder.mkdir(parents=True, exist_ok=True)
        downloads = args.out / "downloads" / name
        path = fetch(archive, downloads)
        pending: tuple[int, list[Video], list[Future[Decoded]]] | None = None
        found = 0
        stream = ((wanted[i], data) for member, data in members(path) if (i := video_id_of(archive.source, member)) in wanted)
        for number, chunk in enumerate(itertools.batched(stream, args.videos_per_chunk)):
            found += len(chunk)
            if (folder / f"records_{number:05d}.jsonl").exists():
                continue
            futures = [pool.submit(prepared, task, v, data) for v, data in chunk]
            # the next chunk decodes while the teacher captions this one
            if pending is not None:
                label(folder, *pending)
            pending = (number, [v for v, _ in chunk], futures)
        if pending is not None:
            label(folder, *pending)
        assert found == len(wanted), f"{name}: {len(wanted) - found} listed videos are not in the archive"
        (folder / "done").touch()
        shutil.rmtree(downloads)


if __name__ == "__main__":
    main()
