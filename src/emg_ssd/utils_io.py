from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import numpy as np


def find_npz_files(root: str | Path) -> List[Path]:
    root = Path(root)
    return sorted(root.rglob("*.npz"))


def write_manifest(paths: Iterable[Path], out_file: str | Path) -> None:
    out = Path(out_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(str(p) for p in paths), encoding="utf-8")


def read_npz(path: str | Path) -> dict:
    with np.load(path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}
