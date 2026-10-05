#!/usr/bin/env python3
"""Download the pinned official Gemma 4 12B QAT Q4_0 checkpoint for Metal."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import shutil
import subprocess
from pathlib import Path

REPOSITORY = "google/gemma-4-12B-it-qat-q4_0-gguf"
REVISION = "29d097773436b69ff9feafd636ab4cf873786537"
FILENAME = "gemma-4-12b-it-qat-q4_0.gguf"
SIZE = 6975879296
SHA256 = "93567e57a8fe10b23569b9d9ec38cd005deedf71e29477c421a4b83f418a538b"
ROOT = Path(__file__).resolve().parents[1]


def matches(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size != SIZE:
        return False
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() == SHA256


def download(destination: Path) -> Path:
    destination = destination.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.with_name(destination.name + ".lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if matches(destination):
            return destination
        partial = destination.with_name(destination.name + ".part")
        remaining = max(0, SIZE - (partial.stat().st_size if partial.exists() else 0))
        if shutil.disk_usage(destination.parent).free < remaining + 2**30:
            raise RuntimeError("Not enough free space for the 6.50 GiB checkpoint.")
        subprocess.run([
            "curl", "--fail", "--location", "--retry", "3", "--connect-timeout", "30",
            "--continue-at", "-", "--output", str(partial), "--",
            f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{FILENAME}",
        ], check=True)
        if not matches(partial):
            raise RuntimeError(f"Checkpoint verification failed; remove {partial} and retry.")
        partial.replace(destination)
        return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / ".models" / "12b" / FILENAME)
    args = parser.parse_args()
    print(download(args.output))


if __name__ == "__main__":
    main()
