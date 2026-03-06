from __future__ import annotations

from collections import deque
from typing import Deque, Iterator, Tuple

import numpy as np


class WindowBuffer:
    def __init__(self, sr: int, channels: int, window_ms: int = 300, hop_ms: int = 200) -> None:
        self.sr = sr
        self.channels = channels
        self.window = int(sr * window_ms / 1000)
        self.hop = int(sr * hop_ms / 1000)
        self.buf: Deque[np.ndarray] = deque()
        self.count = 0

    def push(self, sample: np.ndarray) -> Iterator[np.ndarray]:
        self.buf.append(sample.astype(np.float32))
        self.count += 1
        while len(self.buf) > self.window:
            self.buf.popleft()
        if self.count >= self.window and (self.count - self.window) % self.hop == 0:
            win = np.stack(list(self.buf), axis=0)
            yield win


def simulated_stream(npz_path: str, channels: int) -> Iterator[Tuple[float, np.ndarray]]:
    with np.load(npz_path, allow_pickle=True) as d:
        emg = d["emg"].astype(np.float32)
        sr = int(np.array(d["sr"]).item())
    if emg.shape[1] != channels:
        raise ValueError(f"Channel mismatch: expected {channels}, got {emg.shape[1]}")
    for i in range(emg.shape[0]):
        yield i / sr, emg[i]
