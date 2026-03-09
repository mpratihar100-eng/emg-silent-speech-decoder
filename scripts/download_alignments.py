from __future__ import annotations

import argparse
import shutil
import tarfile
import zipfile
from pathlib import Path

import requests

from emg_ssd.config import ensure_dirs, load_config


def _extract_archive(path: Path, out_dir: Path) -> Path:
    extract_dir = out_dir / "extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".zip":
        with zipfile.ZipFile(path, "r") as zf:
            zf.extractall(extract_dir)
    elif path.suffix in {".gz", ".tgz"} or path.name.endswith(".tar.gz"):
        with tarfile.open(path, "r:gz") as tf:
            tf.extractall(extract_dir)
    else:
        shutil.copy2(path, extract_dir / path.name)
    return extract_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--url", default="")
    ap.add_argument("--archive", default="")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    raw_root = Path(cfg["paths"]["raw_alignments_root"])
    raw_root.mkdir(parents=True, exist_ok=True)

    if args.archive:
        archive = Path(args.archive)
    elif args.url:
        archive = raw_root / Path(args.url).name
        with requests.get(args.url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with archive.open("wb") as fp:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if chunk:
                        fp.write(chunk)
    else:
        raise RuntimeError("Provide --url or --archive")

    extracted = _extract_archive(archive, raw_root)
    print(f"Alignment archive ready at: {archive}")
    print(f"Extracted alignments to: {extracted}")


if __name__ == "__main__":
    main()
