from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from emg_ssd.features import extract_features
from emg_ssd.preprocessing import PreprocessConfig, preprocess_emg


def load_internal_npz(path: str | Path) -> Dict:
    with np.load(path, allow_pickle=True) as d:
        return {k: d[k] for k in d.files}


def framewise_phoneme_segments(log_probs: torch.Tensor, tokens: List[str], blank_id: int, hop_ms: int) -> List[Dict]:
    probs = torch.softmax(log_probs, dim=-1)
    ids = torch.argmax(log_probs, dim=-1).cpu().tolist()
    segs: List[Dict] = []
    prev = ids[0] if ids else blank_id
    start = 0
    for i in range(1, len(ids) + 1):
        cur = ids[i] if i < len(ids) else None
        if cur != prev:
            if prev != blank_id and 0 <= prev < len(tokens):
                seg_probs = probs[start:i, prev]
                segs.append(
                    {
                        "token": tokens[prev],
                        "start_s": (start * hop_ms) / 1000.0,
                        "end_s": (i * hop_ms) / 1000.0,
                        "confidence": float(seg_probs.mean().item()) if seg_probs.numel() else 0.0,
                    }
                )
            start = i
            prev = cur if cur is not None else blank_id
    return segs


def plot_emg_channels(raw_emg: np.ndarray, proc_emg: np.ndarray, sr: int, out_path: str | Path, max_seconds: float = 6.0) -> None:
    t_raw = np.arange(raw_emg.shape[0]) / max(1, sr)
    t_proc = np.arange(proc_emg.shape[0]) / max(1, sr)
    n_chan = raw_emg.shape[1]
    limit_raw = min(raw_emg.shape[0], int(sr * max_seconds))
    limit_proc = min(proc_emg.shape[0], int(sr * max_seconds))

    fig, axes = plt.subplots(n_chan, 1, figsize=(14, max(4, n_chan * 1.6)), sharex=True)
    if n_chan == 1:
        axes = [axes]
    for c, ax in enumerate(axes):
        ax.plot(t_raw[:limit_raw], raw_emg[:limit_raw, c], color="#999999", linewidth=0.8, label="raw")
        ax.plot(t_proc[:limit_proc], proc_emg[:limit_proc, c], color="#0066cc", linewidth=0.8, label="preprocessed")
        ax.set_ylabel(f"Ch{c}")
        ax.grid(alpha=0.2)
        if c == 0:
            ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("EMG per Electrode (raw vs preprocessed)")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_phoneme_timeline(segments: List[Dict], out_path: str | Path, total_s: float, ref_phonemes: str = "") -> None:
    fig, ax = plt.subplots(figsize=(14, 2.5))
    if segments:
        y = 0.5
        for seg in segments:
            x0 = seg["start_s"]
            width = max(0.02, seg["end_s"] - seg["start_s"])
            ax.broken_barh([(x0, width)], (0.2, 0.6), facecolors="#2a9d8f", alpha=0.8)
            ax.text(x0 + width / 2, y, seg["token"], ha="center", va="center", fontsize=8, color="white")
    ax.set_xlim(0, max(0.1, total_s))
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Time (s)")
    title = "Predicted Phoneme Timeline"
    if ref_phonemes:
        title += f" | ref: {ref_phonemes}"
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def preprocess_and_features(raw_emg: np.ndarray, sr: int, cfg: Dict) -> tuple[np.ndarray, np.ndarray]:
    pp_cfg = PreprocessConfig.from_cfg(cfg)
    proc = preprocess_emg(raw_emg, sr, pp_cfg)
    feat = extract_features(proc, pp_cfg.target_sr, cfg["features"])
    return proc, feat
