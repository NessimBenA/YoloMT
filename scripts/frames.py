"""The student frames of a label folder of the labels dataset, decoded from the videos, without the teacher.

    hf download Nessim/YoloMT-labels --repo-type dataset --include "sft/*" --local-dir labels
    python scripts/frames.py --labels labels/sft

The dataset holds, per window list, the teacher's records (`records_XXXXX.jsonl.gz`, and `derived_XXXXX.jsonl.gz`
where `scripts/derive.py` ran) but no frames. For every archive (`data/archives.tsv`) that the records under
`--labels` name, the archive is downloaded into `--labels`/downloads, the seconds t0 .. t of every window are decoded
(`yolomt.video.decode`), each records file receives its `frames_XXXXX.tar` beside it, as `scripts/label.py` writes it,
and the archive is deleted. Records that have their frames are skipped, so the command resumes.
"""

from __future__ import annotations

import argparse
import itertools
import json
import shutil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from yolomt.data import frame_name, lines
from yolomt.video import archives, decode, fetch, members, video_id_of, write_frames


def tar_of(records: Path) -> Path:
    return records.with_name(records.name.split(".")[0].replace("records_", "frames_") + ".tar")


def seconds_of(records: Path) -> dict[str, set[int]]:
    """The seconds t0 .. t that the windows of a records file cover, per video."""
    out: dict[str, set[int]] = defaultdict(set)
    for r in map(json.loads, lines(records)):
        out[r["video_id"]] |= set(range(r["t0"], r["t1"] + 1))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", type=Path, required=True, help="a label folder of the labels dataset")
    parser.add_argument("--archives", type=Path, default=Path("data/archives.tsv"))
    parser.add_argument("--workers", type=int, default=16, help="decoding threads")
    args = parser.parse_args()
    known = archives(args.archives)
    todo: dict[str, list[Path]] = defaultdict(list)
    for records in sorted(args.labels.rglob("records_*.jsonl*")):
        if not tar_of(records).exists():
            todo[records.parent.name].append(records)
    pool = ThreadPoolExecutor(max_workers=args.workers)
    for name, files in todo.items():
        archive = known[name]
        wanted = {records: seconds_of(records) for records in files}
        seconds: dict[str, set[int]] = defaultdict(set)
        for covered in wanted.values():
            for video_id, s in covered.items():
                seconds[video_id] |= s
        downloads = args.labels / "downloads" / name
        stream = ((video_id, data) for member, data in members(fetch(archive, downloads))
                  if (video_id := video_id_of(archive.source, member)) in seconds)
        frames: dict[str, dict[int, bytes]] = {}
        for chunk in itertools.batched(stream, 64):
            decoded = pool.map(lambda item: decode(item[1], seconds=seconds[item[0]], grid=set()), chunk)
            for (video_id, _), d in zip(chunk, decoded):
                assert video_id not in frames, f"{name}: {video_id} twice in the archive"
                assert set(d.student) == seconds[video_id], f"{video_id}: {seconds[video_id] - set(d.student)} missing"
                frames[video_id] = d.student
        assert frames.keys() == seconds.keys(), f"{name}: {len(seconds) - len(frames)} videos not in the archive"
        for records, covered in wanted.items():
            write_frames(tar_of(records), {frame_name(v, s): frames[v][s] for v, ss in covered.items() for s in ss})
        shutil.rmtree(downloads)
        print(name, f"{len(frames)} videos, {len(files)} frame tars", flush=True)


if __name__ == "__main__":
    main()
