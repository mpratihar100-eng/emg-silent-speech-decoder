from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class UtteranceItem:
    path: Path
    speaker_id: str
    session_id: str


def split_items(items: List[UtteranceItem], split_ratio: tuple[float, float, float]) -> tuple[List[UtteranceItem], List[UtteranceItem], List[UtteranceItem]]:
    n = len(items)
    n_train = int(n * split_ratio[0])
    n_val = int(n * split_ratio[1])
    train = items[:n_train]
    val = items[n_train:n_train + n_val]
    test = items[n_train + n_val:]
    return train, val, test
