"""The four benchmark subsets of `yolomt.bench`, rebuilt from their pinned Hugging Face revisions.

`build` reads a benchmark's annotation files and videos at the revisions of `data/bench.json`, decodes the seconds that
the captions of `yolomt.bench` read with the student's own decoder (`yolomt.video.decode`: 256 x 256 JPEG, quality 90)
and writes under `<out>/`:

    questions.jsonl      one `yolomt.bench.Question` per line, its captions fixed
    requests.jsonl       the captions to write, one per (video, t): video, t, loops, frames (oldest first), sorted
    frames_XXXXX.tar     one tar per group of source videos, members `<video id>/<second>.jpg`, sorted by name
    frames.sha256        SHA-256 and name of every frame, group by group
    manifest.json        the revision and annotation digests read, counts and software versions

Two questions of a video that ask the same second at different K share one request: the last question in file order
sets its K, and both questions read that caption (12 such seconds in MVBench).

OVO-Bench Real-Time: the annotations are a GitHub file; the videos are read in place from `src_videos.tar`, stored as
five parts of 10 GB that are downloaded, indexed by tar header and deleted once decoded. MVBench: 19 task annotation
files and eleven zips (17 GB), whose Episodic Reasoning videos are folders of 3 fps JPEG frames. StreamingBench: a CSV
and ten zips of 9 GB, one video per sample, each deleted once decoded; a video corrupt part way is decoded in a child
process and the questions after the corruption are dropped. Video-MME short: a parquet table (needs `pyarrow`) and the
300 short videos, read out of twenty 5 GB zips by HTTP range requests, so no zip is downloaded.

The build asserts the SHA-256 of every annotation file and, once written, of the ordered question keys, of
`questions.jsonl`, `requests.jsonl` and `frames.sha256`, against `data/bench.json`: a rebuild that differs fails. The
frame digests hold for the pinned `av` and `pillow` (requirements.txt).
"""

from __future__ import annotations

import ast
import csv
import hashlib
import io
import json
import math
import tarfile
import urllib.request
import zipfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, assert_never

import av
import PIL
from huggingface_hub import HfFileSystem, hf_hub_download
from PIL import Image

from yolomt.bench import (HORIZON, LETTERS, Question, Request, clip_requests, clock_seconds, frame_names, query_request,
                          tiled_requests)
from yolomt.video import SIDE, decode, decode_isolated, duration, jpeg

Name = Literal["ovo", "mvbench", "streaming", "videomme"]

OVO_TASKS = ("OCR", "ACR", "ATR", "STU", "FPD", "OJR")
OVO_PARTS = tuple(f"src_videos.tar.part{c}" for c in ("aa", "ab", "ac", "ad", "ae"))
TAR_BLOCK = 512
MVBENCH_ZIPS = ("star", "ssv2_video", "Moments_in_Time_Raw", "FunQA_test", "clevrer", "perception", "sta", "scene_qa",
                "vlnqa", "tvqa", "data0613")
STREAMING_CSV = "StreamingBench/Real_Time_Visual_Understanding.csv"
STREAMING_RANGES = tuple((low, low + 49) for low in range(1, 500, 50))
VIDEOMME_PARQUET = "videomme/test-00000-of-00001.parquet"
VIDEOMME_ZIPS = tuple(f"videos_chunked_{n:02d}.zip" for n in range(1, 21))
RANGE_BLOCK = 4 << 20


@dataclass(frozen=True)
class Pin:
    """repo, revision: the Hugging Face dataset repo the videos are read from and its commit; annotations: the SHA-256
    of each annotation file by name (a Hub path, or the pinned URL of a file that is not on the Hub); keys, questions,
    requests, frames: the SHA-256 of the ordered question keys and of the three files written."""

    repo: str
    revision: str
    annotations: dict[str, str]
    keys: str
    questions: str
    requests: str
    frames: str


@dataclass(frozen=True)
class Built:
    """frames: group of source videos -> {`<video id>/<second>.jpg`: JPEG}."""

    questions: list[Question]
    frames: dict[str, dict[str, bytes]]


@dataclass(frozen=True)
class Clip:
    """The seconds first .. last of a video that have a frame."""

    first: int
    last: int


@dataclass(frozen=True)
class Bound:
    """The question's clip is the seconds in [start, end) of its video; the whole video when both are None."""

    start: float | None
    end: float | None


@dataclass(frozen=True)
class Queried:
    """effective[t]: the second read for query second t (t, or the last decodable second when t is past it), absent when
    the video does not decode up to t; frames: second -> JPEG of the 16 seconds up to each effective t."""

    effective: dict[int, int]
    frames: dict[int, bytes]


@dataclass(frozen=True)
class Span:
    """A tar member's data in the concatenation of the parts."""

    offset: int
    size: int


@dataclass(frozen=True)
class OvoJob:
    """One video of the OVO-Bench tar: its path, the parts, its member's span in them and its query seconds."""

    video: str
    parts: tuple[Path, ...]
    span: Span
    times: tuple[int, ...]


@dataclass(frozen=True)
class Source:
    """The MVBench tasks whose `video` fields extend `prefix` in the archives; `name` is the source in video ids; a
    video is a folder of JPEG frames when `frames`."""

    prefix: str
    name: str
    tasks: tuple[str, ...]
    frames: bool = False


MVBENCH_SOURCES = (
    Source("star/Charades_v1_480/", "star", ("action_sequence", "action_prediction", "object_interaction")),
    Source("ssv2_video/", "ssv2", ("action_antonym",)),
    Source("Moments_in_Time_Raw/videos/", "mit", ("fine_grained_action",)),
    Source("FunQA_test/test/", "funqa", ("unexpected_action",)),
    Source("clevrer/video_validation/", "clevrer", ("object_existence", "moving_direction", "moving_count",
                                                    "moving_attribute", "counterfactual_inference")),
    Source("perception/videos/", "perception", ("object_shuffle", "action_count", "state_change", "character_order")),
    Source("sta/sta_video/", "sta", ("action_localization",)),
    Source("scene_qa/video/", "scene_qa", ("scene_transition",)),
    Source("vlnqa/", "vlnqa", ("egocentric_navigation",)),
    Source("tvqa/frames_fps3_hq/", "tvqa", ("episodic_reasoning",), frames=True),
)


@dataclass(frozen=True)
class MvRow:
    """An MVBench question: `index` is its row in the task's annotation file, `path` its video's in the archives."""

    source: Source
    task: str
    index: int
    video: str
    path: str
    bound: Bound
    text: str
    options: tuple[str, ...]
    answer: int
    fps: int


@dataclass(frozen=True)
class Entry:
    """A file of an MVBench zip: the zip, the member's name there and its CRC-32."""

    archive: Path
    member: str
    crc: int


@dataclass(frozen=True)
class Folder:
    """A video stored as `count` JPEG frames at `fps` frames per second."""

    fps: int
    count: int


@dataclass(frozen=True)
class MvJob:
    """The distinct bounds of the questions on one MVBench video; `folder` when its member is a folder of frames."""

    archive: Path
    member: str
    video: str
    bounds: tuple[Bound, ...]
    folder: Folder | None


@dataclass(frozen=True)
class MvResult:
    """clips: the clip of each bound; frames: second -> JPEG of the seconds the clips' requests read."""

    clips: dict[Bound, Clip]
    frames: dict[int, bytes]


@dataclass(frozen=True)
class StreamJob:
    """One StreamingBench video: its zip, its member and its query seconds."""

    archive: Path
    member: str
    times: tuple[int, ...]


@dataclass(frozen=True)
class VmmeJob:
    """The videos of `ids` that one Video-MME zip holds; `archive` is its path in the Hub filesystem."""

    archive: str
    ids: frozenset[str]


@dataclass(frozen=True)
class VmmeResult:
    """frames: second -> JPEG of the seconds the video's tiled requests read."""

    video: str
    clip: Clip
    frames: dict[int, bytes]


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 24):
            digest.update(block)
    return digest.hexdigest()


def pinned(path: Path, digest: str) -> Path:
    assert sha256_of(path) == digest, f"{path.name}: not the pinned file"
    return path


def hub_download(pin: Pin, name: str, *, folder: Path) -> Path:
    return Path(hf_hub_download(pin.repo, name, repo_type="dataset", revision=pin.revision, local_dir=folder))


def url_download(url: str, *, folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / url.rsplit("/", 1)[1]
    with urllib.request.urlopen(url, timeout=120) as response:
        path.write_bytes(response.read())
    return path


def named(video: str, seconds: dict[int, bytes]) -> dict[str, bytes]:
    return {f"{video}/{s}.jpg": data for s, data in seconds.items()}


def seconds_read(requests: tuple[Request, ...]) -> set[int]:
    return {s for r in requests for s in range(r.t - r.loops, r.t + 1)}


def seconds_of(times: set[int] | tuple[int, ...]) -> set[int]:
    """The seconds t - 16 .. t of each query second t."""
    return {s for t in times for s in range(max(0, t - HORIZON), t + 1)}


def contiguous(available: set[int], *, first: int, stop: float, video: str) -> Clip:
    """The clip of the integer seconds in [first, stop) that have a frame, which must be a gap-free run from `first`."""
    have = set(range(first, math.ceil(stop))) & available
    assert have, f"{video}: no frame in seconds [{first}, {stop})"
    last = max(have)
    assert have == set(range(first, last + 1)), f"{video}: gap in seconds {first} .. {last}"
    return Clip(first, last)


def options_of(texts: list[str], *, key: str) -> tuple[str, ...]:
    """The options of a question whose texts read "A. option", "B. option", ...: each prefix is its letter."""
    assert 2 <= len(texts) <= len(LETTERS), f"{key}: {len(texts)} options"
    for i, text in enumerate(texts):
        assert text.startswith(f"{LETTERS[i]}. "), f"{key}: option {i} is {text!r}"
    return tuple(text[3:] for text in texts)


def query_frames(data: bytes, *, times: tuple[int, ...]) -> Queried | None:
    """The frames of seconds t - 16 .. t for each query second t of one video; a t past the video's last decodable
    second is clamped to it (and then reads the 16 s before that). None when the video does not decode up to t."""
    wanted = seconds_of(times)
    decoded = decode_isolated(data, seconds=wanted, grid=set())
    if decoded is None:
        return None
    if wanted <= set(decoded.student):
        return Queried({t: t for t in times}, {s: decoded.student[s] for s in wanted})
    top = decoded.last // 2
    effective = {t: min(t, top) for t in times}
    again = seconds_of(set(effective.values()))
    redone = decode_isolated(data, seconds=again, grid=set())
    assert redone is not None and again <= set(redone.student), f"seconds {sorted(again)} do not decode"
    return Queried(effective, {s: redone.student[s] for s in again})


def read_at(parts: tuple[Path, ...], offset: int, size: int) -> bytes:
    """`size` bytes at `offset` of the parts laid end to end."""
    out: list[bytes] = []
    base = 0
    for path in parts:
        length = path.stat().st_size
        low, high = max(offset, base), min(offset + size, base + length)
        if low < high:
            with path.open("rb") as handle:
                handle.seek(low - base)
                out.append(handle.read(high - low))
        base += length
    data = b"".join(out)
    assert len(data) == size, f"{size} bytes at {offset}: {len(data)} read"
    return data


def tar_index(parts: tuple[Path, ...]) -> dict[str, Span]:
    """The regular members of the tar stored as `parts`, by name, found by hopping from header to header."""
    total = sum(path.stat().st_size for path in parts)
    index: dict[str, Span] = {}
    offset = 0
    long_name: str | None = None
    while offset < total:
        header = read_at(parts, offset, TAR_BLOCK)
        if header == bytes(TAR_BLOCK):
            break
        info = tarfile.TarInfo.frombuf(header, "utf-8", "surrogateescape")
        data = offset + TAR_BLOCK
        if info.type == tarfile.GNUTYPE_LONGNAME:
            long_name = read_at(parts, data, info.size).rstrip(b"\0").decode()
        elif info.type in (tarfile.REGTYPE, tarfile.AREGTYPE):
            index[long_name or info.name] = Span(data, info.size)
            long_name = None
        else:
            assert info.type == tarfile.DIRTYPE, f"{info.name}: tar member type {info.type!r}"
        offset = data + -(-info.size // TAR_BLOCK) * TAR_BLOCK
    return index


def ovo_rows(path: Path) -> list[dict[str, Any]]:
    rows = [r for r in json.loads(path.read_text()) if r["task"] in OVO_TASKS]
    assert len(rows) == 837 and len({r["video"] for r in rows}) == 237, "not OVO-Bench Real-Time"
    assert all(r["video"].endswith(".mp4") for r in rows)
    return rows


def ovo_questions(rows: list[dict[str, Any]], *, effective: dict[str, dict[int, int]]) -> list[Question]:
    """effective[video][t]: the second read for query second t, for the video's path in the tar."""
    questions = []
    for r in rows:
        video = f"ovo/{r['video'].removesuffix('.mp4')}"
        last = effective[r["video"]][math.floor(r["realtime"])]
        questions.append(Question(f"ovo/{r['id']}", "ovo", r["task"], video, 0, r["question"], tuple(r["options"]),
                                  r["gt"], (query_request(video, r["realtime"], last=last),)))
    return questions


def ovo_frames(job: OvoJob) -> Queried:
    queried = query_frames(read_at(job.parts, job.span.offset, job.span.size), times=job.times)
    assert queried is not None, f"{job.video}: does not decode"
    return queried


def build_ovo(pin: Pin, *, scratch: Path, workers: int) -> Built:
    [url] = pin.annotations
    rows = ovo_rows(pinned(url_download(url, folder=scratch / "ovo-annotation"), pin.annotations[url]))
    parts = tuple(hub_download(pin, name, folder=scratch / "ovo") for name in OVO_PARTS)
    index = {name.removeprefix("./src_videos/"): span for name, span in tar_index(parts).items()}
    times: dict[str, set[int]] = {}
    for r in rows:
        times.setdefault(r["video"], set()).add(math.floor(r["realtime"]))
    jobs = [OvoJob(video, parts, index[video], tuple(sorted(ts))) for video, ts in times.items()]
    with ProcessPoolExecutor(workers) as pool:
        queried = {job.video: q for job, q in zip(jobs, pool.map(ovo_frames, jobs, chunksize=1), strict=True)}
    for part in parts:
        part.unlink()
    frames: dict[str, dict[str, bytes]] = {}
    for video, q in queried.items():
        frames.setdefault(video.split("/")[0], {}).update(named(f"ovo/{video.removesuffix('.mp4')}", q.frames))
    return Built(ovo_questions(rows, effective={video: q.effective for video, q in queried.items()}), frames)


def student_jpeg(data: bytes) -> bytes:
    """A frame folder's JPEG as the student's frame: squashed to 256 x 256 (bicubic), JPEG quality 90."""
    image = Image.open(io.BytesIO(data)).convert("RGB")
    return jpeg(image.resize((SIDE, SIDE), Image.Resampling.BICUBIC))


def mvbench_rows(annotations: dict[str, list[dict[str, Any]]]) -> list[MvRow]:
    """annotations[task]: the rows of the task's annotation file; the rows in the order of MVBENCH_SOURCES."""
    rows = []
    for source in MVBENCH_SOURCES:
        for task in source.tasks:
            for index, r in enumerate(annotations[task]):
                path = source.prefix + r["video"]
                relative = r["video"] if source.frames else r["video"].rsplit(".", 1)[0]
                answers = [k for k, option in enumerate(r["candidates"]) if option.strip() == r["answer"].strip()]
                assert len(answers) == 1, f"{task} {index}: the answer is not exactly one candidate"
                rows.append(MvRow(source, task, index, f"mvbench/{source.name}/{relative}", path,
                                  Bound(r["start"], r["end"]) if "start" in r else Bound(None, None), r["question"],
                                  tuple(r["candidates"]), answers[0], r["fps"] if source.frames else 0))
    assert len(rows) == 3800, "not the 19 tasks of MVBench"
    return rows


def mvbench_questions(rows: list[MvRow], *, clips: dict[str, dict[Bound, Clip]]) -> list[Question]:
    """clips[video][bound]: the clip of the question's bound."""
    questions = []
    for row in rows:
        clip = clips[row.video][row.bound]
        questions.append(Question(f"mvbench/{row.task}/{row.index:03d}", "mvbench", row.task, row.video, clip.first,
                                  row.text, row.options, row.answer,
                                  clip_requests(row.video, first=clip.first, last=clip.last)))
    return questions


def mvbench_clips(job: MvJob) -> MvResult:
    """The clip of each bound of one video (its integer seconds in [start, end) that have a frame, from the video's
    first second when it has no start) and the frames of the seconds its requests read. A folder's frame for second s
    is index max(1, round(fps s)), as MVBench's own loader reads it."""
    firsts = {b: 0 if b.start is None else math.ceil(b.start) for b in job.bounds}
    with zipfile.ZipFile(job.archive) as archive:
        if job.folder is None:
            data = archive.read(job.member)
            whole = duration(data)
            stops = {b: whole if b.end is None else b.end for b in job.bounds}
            seconds = {s for b in job.bounds for s in range(firsts[b], math.ceil(stops[b]))}
            decoded = decode(data, seconds=seconds, grid=set()).student
        else:
            stops = {b: b.end for b in job.bounds if b.end is not None}
            assert len(stops) == len(job.bounds), f"{job.video}: a frame folder needs an end"
            seconds = {s for b in job.bounds for s in range(firsts[b], math.ceil(stops[b]))}
            at = {s: max(1, round(s * job.folder.fps)) for s in seconds}
            decoded = {s: student_jpeg(archive.read(f"{job.member}/{i:05d}.jpg")) for s, i in at.items()
                       if i <= job.folder.count}
    clips: dict[Bound, Clip] = {}
    frames: dict[int, bytes] = {}
    for b in job.bounds:
        clips[b] = contiguous(set(decoded), first=firsts[b], stop=stops[b], video=job.video)
        requests = clip_requests(job.video, first=clips[b].first, last=clips[b].last)
        frames |= {s: decoded[s] for s in seconds_read(requests)}
    return MvResult(clips, frames)


def mvbench_jobs(rows: list[MvRow], *, archives: dict[str, Path]) -> list[MvJob]:
    """One job per video of `rows`; its member is looked up in the zips `archives` by name ("tvqa" holds the frame
    folders, and a file that two zips both hold must be the same)."""
    where: dict[str, Entry] = {}
    counts: dict[str, int] = {}
    for path in archives.values():
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                canonical = info.filename.removeprefix("data0613/")
                if info.is_dir():
                    continue
                if canonical.endswith(".jpg"):
                    parent = canonical.rsplit("/", 1)[0]
                    counts[parent] = counts.get(parent, 0) + 1
                elif canonical in where:
                    assert where[canonical].crc == info.CRC, f"{canonical}: two archives hold different files"
                else:
                    where[canonical] = Entry(path, info.filename, info.CRC)
    first: dict[str, MvRow] = {}
    bounds: dict[str, dict[Bound, None]] = {}
    for row in rows:
        first.setdefault(row.path, row)
        bounds.setdefault(row.path, {})[row.bound] = None
    jobs = []
    for member, row in sorted(first.items()):
        if row.source.frames:
            jobs.append(MvJob(archives["tvqa"], member, row.video, tuple(bounds[member]),
                              Folder(row.fps, counts[member])))
        else:
            jobs.append(MvJob(where[member].archive, where[member].member, row.video, tuple(bounds[member]), None))
    return jobs


def build_mvbench(pin: Pin, *, scratch: Path, workers: int) -> Built:
    annotations: dict[str, list[dict[str, Any]]] = {}
    for source in MVBENCH_SOURCES:
        for task in source.tasks:
            name = f"json/{task}.json"
            annotations[task] = json.loads(pinned(hub_download(pin, name, folder=scratch / "mvbench"),
                                                  pin.annotations[name]).read_text())
    rows = mvbench_rows(annotations)
    archives = {name: hub_download(pin, f"video/{name}.zip", folder=scratch / "mvbench") for name in MVBENCH_ZIPS}
    jobs = mvbench_jobs(rows, archives=archives)
    with ProcessPoolExecutor(workers) as pool:
        results = {job.video: r for job, r in zip(jobs, pool.map(mvbench_clips, jobs, chunksize=4), strict=True)}
    for path in archives.values():
        path.unlink()
    frames: dict[str, dict[str, bytes]] = {}
    for video, r in results.items():
        frames.setdefault(video.split("/")[1], {}).update(named(video, r.frames))
    return Built(mvbench_questions(rows, clips={video: r.clips for video, r in results.items()}), frames)


def streaming_key(row: dict[str, str]) -> str:
    return f"streaming/{row['question_id'].removeprefix('Real-Time Visual Understanding_')}"


def streaming_video(row: dict[str, str]) -> str:
    return f"streaming/sample_{row['question_id'].split('_sample_')[1].split('_')[0]}"


def streaming_questions(rows: list[dict[str, str]], *, effective: dict[str, dict[int, int]]) -> list[Question]:
    """effective[video][t]: the second read for query second t; a question whose second is absent is lost."""
    questions = []
    for r in rows:
        video, t = streaming_video(r), clock_seconds(r["time_stamp"])
        if t in effective[video]:
            options = options_of(ast.literal_eval(r["options"]), key=r["question_id"])
            assert r["answer"] in LETTERS[:len(options)], f"{r['question_id']}: answer {r['answer']!r}"
            request = query_request(video, float(t), last=effective[video][t])
            questions.append(Question(streaming_key(r), "streaming", r["task_type"], video, 0, r["question"], options,
                                      LETTERS.index(r["answer"]), (request,)))
    return questions


def stream_frames(job: StreamJob) -> Queried:
    """`query_frames` of one video; when it does not decode, the latest query second is dropped and it is decoded
    again, so the questions before the corruption stay."""
    with zipfile.ZipFile(job.archive) as archive:
        data = archive.read(job.member)
    kept = list(job.times)
    while kept:
        queried = query_frames(data, times=tuple(kept))
        if queried is not None:
            return queried
        kept.pop()
    return Queried({}, {})


def build_streaming(pin: Pin, *, scratch: Path, workers: int) -> Built:
    table = pinned(hub_download(pin, STREAMING_CSV, folder=scratch / "streaming"), pin.annotations[STREAMING_CSV])
    with table.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2500, "not StreamingBench Real-Time"
    times: dict[str, set[int]] = {}
    for r in rows:
        times.setdefault(streaming_video(r), set()).add(clock_seconds(r["time_stamp"]))
    assert sorted(times) == sorted(f"streaming/sample_{n}" for n in range(1, 501)), "not 500 videos"
    effective: dict[str, dict[int, int]] = {}
    frames: dict[str, dict[str, bytes]] = {}
    names = [f"Real-Time Visual Understanding_{low}-{high}.zip" for low, high in STREAMING_RANGES]
    with ThreadPoolExecutor(3) as downloads:  # three zips download while one decodes
        coming = [downloads.submit(hub_download, pin, name, folder=scratch / "streaming") for name in names]
        for (low, high), download in zip(STREAMING_RANGES, coming, strict=True):
            archive = download.result()
            jobs = {f"streaming/sample_{n}": StreamJob(archive, f"sample_{n}/video.mp4",
                                                       tuple(sorted(times[f"streaming/sample_{n}"])))
                    for n in range(low, high + 1)}
            with ProcessPoolExecutor(workers) as pool:
                for video, q in zip(jobs, pool.map(stream_frames, jobs.values(), chunksize=1), strict=True):
                    effective[video] = q.effective
                    frames.setdefault(f"{low:03d}-{high:03d}", {}).update(named(video, q.frames))
            archive.unlink()
            print(f"{archive.name} decoded", flush=True)
    questions = streaming_questions(rows, effective=effective)
    lost = sorted({streaming_key(r) for r in rows} - {q.key for q in questions})
    assert len(lost) <= len(rows) // 20, f"{len(lost)} questions lost to undecodable video"
    print(f"questions lost to undecodable video: {lost}")
    return Built(questions, frames)


def videomme_rows(path: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    rows: list[dict[str, Any]] = [r for r in pq.read_table(path).to_pylist() if r["duration"] == "short"]
    assert len(rows) == 900 and len({r["videoID"] for r in rows}) == 300, "not Video-MME short"
    return rows


def videomme_questions(rows: list[dict[str, Any]], *, clips: dict[str, Clip]) -> list[Question]:
    """clips[video]: the clip of each video, by video id."""
    questions = []
    for r in rows:
        video = f"videomme/{r['videoID']}"
        options = options_of(list(r["options"]), key=r["question_id"])
        assert r["answer"] in LETTERS[:len(options)], f"{r['question_id']}: answer {r['answer']!r}"
        clip = clips[video]
        questions.append(Question(f"videomme/{r['question_id']}", "videomme", r["task_type"], video, clip.first,
                                  r["question"], options, LETTERS.index(r["answer"]),
                                  tiled_requests(video, first=clip.first, last=clip.last)))
    return questions


def videomme_clips(job: VmmeJob) -> list[VmmeResult]:
    """The clips of the wanted videos of one zip, read member by member through HTTP range requests; the zip checks
    each member's CRC-32 as it reads."""
    results = []
    with HfFileSystem().open(job.archive, "rb", block_size=RANGE_BLOCK) as handle, \
            zipfile.ZipFile(handle) as archive:
        for info in archive.infolist():
            found = info.filename.removeprefix("data/").removesuffix(".mp4")
            if info.filename.startswith("data/") and found in job.ids:
                data = archive.read(info)
                whole = duration(data)
                decoded = decode(data, seconds=set(range(math.ceil(whole))), grid=set()).student
                video = f"videomme/{found}"
                clip = contiguous(set(decoded), first=0, stop=whole, video=video)
                seconds = seconds_read(tiled_requests(video, first=clip.first, last=clip.last))
                results.append(VmmeResult(video, clip, {s: decoded[s] for s in seconds}))
    return results


def build_videomme(pin: Pin, *, scratch: Path, workers: int) -> Built:
    rows = videomme_rows(pinned(hub_download(pin, VIDEOMME_PARQUET, folder=scratch / "videomme"),
                                pin.annotations[VIDEOMME_PARQUET]))
    ids = frozenset(r["videoID"] for r in rows)
    jobs = [VmmeJob(f"datasets/{pin.repo}@{pin.revision}/{name}", ids) for name in VIDEOMME_ZIPS]
    with ProcessPoolExecutor(min(workers, len(jobs))) as pool:
        results = {r.video: r for batch in pool.map(videomme_clips, jobs, chunksize=1) for r in batch}
    assert sorted(results) == sorted(f"videomme/{v}" for v in ids), "the short videos are not in the zips exactly once"
    frames = {"short": {name: data for r in results.values() for name, data in named(r.video, r.frames).items()}}
    return Built(videomme_questions(rows, clips={video: r.clip for video, r in results.items()}), frames)


def write_tar(path: Path, frames: dict[str, bytes]) -> None:
    with tarfile.open(path, "w") as tar:
        for name in sorted(frames):
            info = tarfile.TarInfo(name)
            info.size = len(frames[name])
            tar.addfile(info, io.BytesIO(frames[name]))


def write_outputs(out: Path, pin: Pin, *, benchmark: str, built: Built) -> None:
    """The files of the module docstring, asserted equal to the pin."""
    stored = {name for chunk in built.frames.values() for name in chunk}
    requests: dict[tuple[str, int], Request] = {}
    for q in built.questions:
        for r in q.captions:
            assert all(name in stored for name in frame_names(r)), f"{q.key}: a frame of {r} was not decoded"
            requests[(r.video, r.t)] = r
    out.mkdir(parents=True, exist_ok=True)
    groups = sorted(built.frames)
    for number, group in enumerate(groups):
        write_tar(out / f"frames_{number:05d}.tar", built.frames[group])
    (out / "frames.sha256").write_text("".join(f"{hashlib.sha256(data).hexdigest()}  {name}\n" for group in groups
                                              for name, data in sorted(built.frames[group].items())))
    (out / "questions.jsonl").write_text("".join(json.dumps(asdict(q)) + "\n" for q in built.questions))
    (out / "requests.jsonl").write_text("".join(
        json.dumps({**asdict(requests[key]), "frames": list(frame_names(requests[key]))}) + "\n"
        for key in sorted(requests)))
    tasks: dict[str, int] = {}
    for q in built.questions:
        tasks[q.task] = tasks.get(q.task, 0) + 1
    manifest = {"benchmark": benchmark, "repo": pin.repo, "revision": pin.revision, "annotations": pin.annotations,
                "videos": len({q.video for q in built.questions}), "questions": len(built.questions), "tasks": tasks,
                "frames": len(stored), "captions": len(requests),
                "tars": [f"frames_{n:05d}.tar" for n in range(len(groups))],
                "software": {"av": av.__version__, "pillow": PIL.__version__}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    written = {"keys": hashlib.sha256("\n".join(q.key for q in built.questions).encode()).hexdigest(),
               "questions": sha256_of(out / "questions.jsonl"), "requests": sha256_of(out / "requests.jsonl"),
               "frames": sha256_of(out / "frames.sha256")}
    expected = {"keys": pin.keys, "questions": pin.questions, "requests": pin.requests, "frames": pin.frames}
    assert written == expected, f"{benchmark}: the rebuild differs from the pin: {written} != {expected}"


def read_pins(path: Path) -> dict[str, Pin]:
    return {name: Pin(**pin) for name, pin in json.loads(path.read_text()).items()}


def check() -> None:
    """Hand-computed values on tiny fixed inputs; runs at every build."""
    assert seconds_of((40, 3)) == {0, 1, 2, 3, *range(24, 41)} and seconds_read((Request("v", 5, 2),)) == {3, 4, 5}
    assert named("v", {3: b"x"}) == {"v/3.jpg": b"x"}
    assert contiguous({0, 1, 2, 3}, first=0, stop=3.2, video="v") == Clip(0, 3)
    assert contiguous({2, 3, 4, 6}, first=2, stop=5, video="v") == Clip(2, 4)
    assert options_of(["A. x", "B. y. z"], key="k") == ("x", "y. z")
    row = {"question_id": "Real-Time Visual Understanding_sample_332_4"}
    assert streaming_video(row) == "streaming/sample_332" and streaming_key(row) == "streaming/sample_332_4"
    ovo = {"id": 7, "task": "OCR", "video": "A/1.mp4", "realtime": 12.9, "question": "Q?", "options": ["x", "y"],
           "gt": 1}
    expected = Question("ovo/7", "ovo", "OCR", "ovo/A/1", 0, "Q?", ("x", "y"), 1, (Request("ovo/A/1", 12, 12),))
    assert ovo_questions([ovo], effective={"A/1.mp4": {12: 12}}) == [expected]
    assert ovo_questions([ovo], effective={"A/1.mp4": {12: 9}})[0].captions == (Request("ovo/A/1", 9, 9),)
    row["time_stamp"], row["task_type"], row["question"] = "00:01:05", "Counting", "Q?"
    row["options"], row["answer"] = "['A. x', 'B. y']", "B"
    kept = streaming_questions([row], effective={"streaming/sample_332": {65: 60}})
    assert kept == [Question("streaming/sample_332_4", "streaming", "Counting", "streaming/sample_332", 0, "Q?",
                             ("x", "y"), 1, (Request("streaming/sample_332", 60, 16),))]
    assert streaming_questions([row], effective={"streaming/sample_332": {}}) == []


def build(benchmark: Name, *, pins: Path, out: Path, scratch: Path, workers: int) -> None:
    """Downloads into `scratch` (archives are deleted once decoded), decodes with `workers` processes, writes `out`."""
    check()
    pin = read_pins(pins)[benchmark]
    match benchmark:
        case "ovo":
            built = build_ovo(pin, scratch=scratch, workers=workers)
        case "mvbench":
            built = build_mvbench(pin, scratch=scratch, workers=workers)
        case "streaming":
            built = build_streaming(pin, scratch=scratch, workers=workers)
        case "videomme":
            built = build_videomme(pin, scratch=scratch, workers=workers)
        case _:
            assert_never(benchmark)
    write_outputs(out, pin, benchmark=benchmark, built=built)
