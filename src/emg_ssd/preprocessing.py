from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
from scipy import signal


@dataclass
class PreprocessConfig:
    target_sr: int
    notch_enabled: bool
    notch_freq: float
    notch_q: float
    bandpass_enabled: bool
    bandpass_low: float
    bandpass_high: float
    bandpass_order: int
    rectify: bool
    envelope_enabled: bool
    envelope_lowpass: float
    norm_mode: str
    eps: float

    @staticmethod
    def from_cfg(cfg: Dict[str, Any]) -> "PreprocessConfig":
        p = cfg["preprocess"]
        d = cfg["data"]
        return PreprocessConfig(
            target_sr=int(d["target_sr"]),
            notch_enabled=bool(p["notch"]["enabled"]),
            notch_freq=float(p["notch"]["freq"]),
            notch_q=float(p["notch"]["q"]),
            bandpass_enabled=bool(p["bandpass"]["enabled"]),
            bandpass_low=float(p["bandpass"]["low_hz"]),
            bandpass_high=float(p["bandpass"]["high_hz"]),
            bandpass_order=int(p["bandpass"]["order"]),
            rectify=bool(p["rectify"]),
            envelope_enabled=bool(p["envelope"]["enabled"]),
            envelope_lowpass=float(p["envelope"]["lowpass_hz"]),
            norm_mode=str(p["normalization"]["mode"]),
            eps=float(p["normalization"]["eps"]),
        )


def _safe_filtfilt(b: np.ndarray, a: np.ndarray, x: np.ndarray) -> np.ndarray:
    if x.shape[0] < max(len(a), len(b)) * 3:
        return x
    return signal.filtfilt(b, a, x, axis=0)


def preprocess_emg(emg: np.ndarray, sr: int, cfg: PreprocessConfig) -> np.ndarray:
    x = np.asarray(emg, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"Expected [T, C], got {x.shape}")

    if sr != cfg.target_sr and sr > 0:
        num = int(round(x.shape[0] * cfg.target_sr / sr))
        x = signal.resample(x, num=num, axis=0).astype(np.float32)
        sr = cfg.target_sr

    if cfg.notch_enabled and 0 < cfg.notch_freq < sr / 2:
        b, a = signal.iirnotch(w0=cfg.notch_freq, Q=cfg.notch_q, fs=sr)
        x = _safe_filtfilt(b, a, x)

    if cfg.bandpass_enabled:
        low = cfg.bandpass_low / (sr / 2)
        high = min(cfg.bandpass_high / (sr / 2), 0.99)
        if 0 < low < high < 1:
            b, a = signal.butter(cfg.bandpass_order, [low, high], btype="band")
            x = _safe_filtfilt(b, a, x)

    if cfg.rectify:
        x = np.abs(x)

    if cfg.envelope_enabled and cfg.envelope_lowpass > 0:
        cut = min(cfg.envelope_lowpass / (sr / 2), 0.99)
        b, a = signal.butter(2, cut, btype="low")
        x = _safe_filtfilt(b, a, x)

    if cfg.norm_mode == "robust":
        med = np.median(x, axis=0, keepdims=True)
        mad = np.median(np.abs(x - med), axis=0, keepdims=True) + cfg.eps
        x = (x - med) / (1.4826 * mad)
    else:
        mu = np.mean(x, axis=0, keepdims=True)
        std = np.std(x, axis=0, keepdims=True) + cfg.eps
        x = (x - mu) / std

    return x.astype(np.float32)
