"""Cut one caption set to the word counts of another: the length control of the paper.

    python scripts/cut.py --captions <captions> --source yolomt-kprotocol --like lfm25-vl-450m-kprotocol \\
        --name yolomt-cut-lfm-kprotocol

On every benchmark, each caption of `--source` is truncated to its first n words, n the word count of the caption of
`--like` for the same request, and written to `<captions>/<benchmark>/<name>.jsonl`, never over an existing file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from yolomt.answer import write_cut


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--captions", type=Path, required=True)
    parser.add_argument("--source", required=True, help="the caption set to truncate")
    parser.add_argument("--like", required=True, help="the caption set whose word counts to match")
    parser.add_argument("--name", required=True, help="the new caption set")
    args = parser.parse_args()
    write_cut(args.captions, source=args.source, like=args.like, name=args.name)


if __name__ == "__main__":
    main()
