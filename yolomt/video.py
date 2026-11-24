"""Video archives, video ids and frame decoding.

A student frame is the first decoded frame at or after a whole second, squashed to 256 x 256 (bicubic, no crop) and
stored as a JPEG at quality 90. Labelling, training and captioning all read these bytes. The teacher of the 2 s
windows also sees grid frames: grid frame k is the first decoded frame at or after k / 2 s, long side 448 px.
"""

from __future__ import annotations

import io
import re
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterator

import av
import numpy as np
from huggingface_hub import hf_hub_download
from numpy.typing import NDArray
from PIL import Image

SIDE = 256
FPS = 2.0
GRID_SIDE = 448
VIDEO_SUFFIXES = (".mp4", ".mkv", ".webm", ".mov", ".avi")


@dataclass(frozen=True)
class Archive:
    """A video archive of a Hugging Face dataset repo; `pieces`, when the repo stores the file split, are joined by
    concatenation into `file`. name: `<source>__<file stem>`, as the window lists name it."""

    name: str
    repo: str
    file: str
    source: str
    pieces: tuple[str, ...]


@dataclass(frozen=True)
class Decoded:
    """student[s]: the JPEG of second s; grid[k]: grid frame k as (H, W, 3) uint8; last: the last grid index reached."""

    student: dict[int, bytes]
    grid: dict[int, NDArray[np.uint8]]
    last: int


def archives(path: Path) -> dict[str, Archive]:
    """`data/archives.tsv`: repo, file, source and comma-separated pieces (empty for a whole file) per line."""
    out = {}
    for line in path.read_text().splitlines()[1:]:
        repo, file, source, pieces = line.split("\t")
        name = f"{source}__{Path(file).name.split('.')[0]}"
        out[name] = Archive(name, repo, file, source, tuple(pieces.split(",")) if pieces else ())
    return out


def fetch(archive: Archive, folder: Path) -> Path:
    """The archive's file downloaded into `folder`."""
    if not archive.pieces:
        return Path(hf_hub_download(archive.repo, archive.file, repo_type="dataset", local_dir=folder))
    whole = folder / archive.file
    with whole.open("wb") as out:
        for piece in archive.pieces:
            path = Path(hf_hub_download(archive.repo, piece, repo_type="dataset", local_dir=folder))
            with path.open("rb") as handle:
                while block := handle.read(1 << 24):
                    out.write(block)
            path.unlink()
    return whole


def members(path: Path) -> Iterator[tuple[str, bytes]]:
    """The video files of a .zip or .tar(.gz) archive, in archive order."""
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                if info.filename.lower().endswith(VIDEO_SUFFIXES):
                    yield info.filename, archive.read(info)
        return
    with tarfile.open(path, "r|*") as stream:
        for member in stream:
            if member.isfile() and member.name.lower().endswith(VIDEO_SUFFIXES):
                handle = stream.extractfile(member)
                assert handle is not None
                yield member.name, handle.read()


GENERIC = re.compile(r"^(split|clip|part|segment|video)_?\d+$")
YOUTUBE = r"[A-Za-z0-9_-]{11}"
# OpenVid-1M clip names by origin, mapped to the video they were cut from, so that its clips share one split
OPENVID_SOURCES = (
    (re.compile(rf"^({YOUTUBE})_\d+_\d+to\d+$"), "yt"),
    (re.compile(rf"^celebv_({YOUTUBE})_\d+(?:_\d+)?$"), "yt"),
    (re.compile(rf"^magictime_({YOUTUBE})(?:_\d{{4}})*$"), "yt"),
    (re.compile(r"^magictime_(.+?)(?:_\d{4})*$"), "magictime"),
    (re.compile(r"^pixabay_.*_(\d+)_\d+$"), "pixabay"),
    (re.compile(r"^pexels_.*_(\d+)_\d+$"), "pexels"),
    (re.compile(r"^mixkit_.*-(\d+)_\d+$"), "mixkit"),
)


def video_id_of(source: str, member: str) -> str:
    """`<source>/<stem>`; `<source>/<folder>__<stem>` when the stem is a generic clip name (YouCook2 keeps a video's
    clips in its folder as split_0.mp4 ...); `openvid/<origin>__<stem>` for OpenVid-1M."""
    path = PurePosixPath(member)
    if source == "openvid":
        origin = [f"{kind}-{m[1]}" for pattern, kind in OPENVID_SOURCES if (m := pattern.match(path.stem))]
        assert origin, f"{path.stem}: not an OpenVid-1M clip name"
        return f"openvid/{origin[0]}__{path.stem}"
    return f"{source}/{path.parent.name}__{path.stem}" if GENERIC.match(path.stem) else f"{source}/{path.stem}"


def jpeg(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


def rgb(data: bytes) -> NDArray[np.uint8]:
    """A JPEG -> (H, W, 3) uint8."""
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))


def resize_long(image: Image.Image, side: int) -> Image.Image:
    scale = side / max(image.size)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(size, Image.Resampling.BICUBIC)


def decode(data: bytes, *, seconds: set[int], grid: set[int]) -> Decoded:
    """The student JPEGs of `seconds` and the grid frames of `grid` of an encoded video; only those are converted."""
    container = av.open(io.BytesIO(data), mode="r")
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    assert stream.time_base is not None
    student: dict[int, bytes] = {}
    frames: dict[int, NDArray[np.uint8]] = {}
    stop = max(max(grid, default=0), 2 * max(seconds, default=0))
    k = 0
    for frame in container.decode(stream):
        if frame.pts is None:
            continue
        t = float(frame.pts * stream.time_base)
        if k / FPS > t + 1e-3:
            continue
        covered = []
        # a video below 2 fps repeats a frame on several grid indices instead of drifting
        while k / FPS <= t + 1e-3 and k <= stop:
            covered.append(k)
            k += 1
        taken = [j for j in covered if j in grid or (j % 2 == 0 and j // 2 in seconds)]
        if taken:
            image = frame.to_image()
            for j in taken:
                if j in grid:
                    frames[j] = np.asarray(resize_long(image, GRID_SIDE))
                if j % 2 == 0 and j // 2 in seconds:
                    student[j // 2] = jpeg(image.resize((SIDE, SIDE), Image.Resampling.BICUBIC))
        if k > stop:
            break
    container.close()
    return Decoded(student, frames, k - 1)


def duration(data: bytes) -> float:
    """Seconds of an encoded video."""
    container = av.open(io.BytesIO(data), mode="r")
    stream = container.streams.video[0]
    if stream.duration is not None and stream.time_base is not None:
        seconds = float(stream.duration * stream.time_base)
    else:
        assert container.duration is not None, "a video without a duration"
        seconds = container.duration / av.time_base
    container.close()
    return seconds
