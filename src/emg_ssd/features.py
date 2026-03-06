from __future__ import annotations

from typing import Dict

import numpy as np
from scipy import signal


def raw_windows(x: np.ndarray, sr: int, frame_ms: int, hop_ms: int) -> np.ndarray:
    frame = max(1, int(sr * frame_ms / 1000))
    hop = max(1, int(sr * hop_ms / 1000))
    if x.shape[0] < frame:
        pad = np.zeros((frame - x.shape[0], x.shape[1]), dtype=x.dtype)
        x = np.vstack([x, pad])
    starts = np.arange(0, x.shape[0] - frame + 1, hop)
    return np.stack([x[s:s + frame] for s in starts], axis=0)


def stft_features(x: np.ndarray, sr: int, cfg: Dict) -> np.ndarray:
    n_fft = int(cfg["n_fft"])
    frame = int(sr * int(cfg["frame_ms"]) / 1000)
    hop = int(sr * int(cfg["hop_ms"]) / 1000)
    feats = []
    for c in range(x.shape[1]):
        _, _, z = signal.stft(
            x[:, c], fs=sr, nperseg=max(frame, 8), noverlap=max(frame - hop, 0), nfft=n_fft, boundary=None
        )
        p = np.log1p(np.abs(z) ** 2).T
        feats.append(p)
    m = min(f.shape[0] for f in feats)
    feats = [f[:m] for f in feats]
    return np.concatenate(feats, axis=1).astype(np.float32)


def extract_features(x: np.ndarray, sr: int, features_cfg: Dict) -> np.ndarray:
    if features_cfg["mode"] == "stft":
        return stft_features(x, sr, features_cfg)
    wins = raw_windows(x, sr, int(features_cfg["frame_ms"]), int(features_cfg["hop_ms"]))
    return wins.reshape(wins.shape[0], -1).astype(np.float32)
