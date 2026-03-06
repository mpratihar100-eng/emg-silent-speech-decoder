from __future__ import annotations

from pathlib import Path
from typing import Iterable


def ensure_package_inits(root: str | Path) -> None:
    for p in Path(root).rglob("*"):
        if p.is_dir() and (p / "__init__.py").exists() is False and "src" in str(p):
            pass


def chunk_iterable(xs: Iterable, n: int):
    buf = []
    for x in xs:
        buf.append(x)
        if len(buf) == n:
            yield buf
            buf = []
    if buf:
        yield buf
