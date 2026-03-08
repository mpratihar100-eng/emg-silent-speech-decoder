from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

from emg_ssd.data.npz_dataset import filter_split_files, load_split_files
from emg_ssd.preprocessing import PreprocessConfig, preprocess_emg


def temporal_envelope(x: np.ndarray) -> np.ndarray:
    # Compress multi-channel EMG into a stable DTW template signal.
    return np.sqrt(np.mean(np.square(x), axis=1)).astype(np.float32)


def downsample_1d(x: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1 or x.shape[0] < factor:
        return x
    return x[::factor].astype(np.float32)


def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return float("inf")
    dp = np.full((len(a) + 1, len(b) + 1), np.inf, dtype=np.float32)
    dp[0, 0] = 0.0
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = abs(float(a[i - 1]) - float(b[j - 1]))
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return float(dp[len(a), len(b)] / (len(a) + len(b)))


def build_text_prototypes(files: Sequence[Path], cfg: Dict) -> Dict[str, np.ndarray]:
    pp_cfg = PreprocessConfig.from_cfg(cfg)
    max_examples = int(cfg.get("alignment", {}).get("max_template_examples_per_text", 8))
    downsample = int(cfg.get("alignment", {}).get("dtw_downsample", 4))
    grouped: Dict[str, List[np.ndarray]] = defaultdict(list)

    for f in files:
        with np.load(f, allow_pickle=True) as d:
            text = str(d["text"].item()) if "text" in d.files else ""
            if not text.strip():
                continue
            emg = d["emg"].astype(np.float32)
            sr = int(np.array(d["sr"]).item())
        proc = preprocess_emg(emg, sr, pp_cfg)
        env = downsample_1d(temporal_envelope(proc), downsample)
        if len(grouped[text]) < max_examples:
            grouped[text].append(env)

    prototypes: Dict[str, np.ndarray] = {}
    for text, seqs in grouped.items():
        if not seqs:
            continue
        ref = max(seqs, key=len)
        scored = []
        for s in seqs:
            scored.append((dtw_distance(ref, s), s))
        scored.sort(key=lambda x: x[0])
        prototypes[text] = scored[0][1]
    return prototypes


def iter_filtered_split_files(cfg: Dict, split: str) -> List[Path]:
    files = load_split_files(cfg["paths"]["internal_root"], split)
    return filter_split_files(files, cfg)


def score_against_prototypes(proc_emg: np.ndarray, prototypes: Dict[str, np.ndarray], downsample: int) -> tuple[str, float]:
    env = downsample_1d(temporal_envelope(proc_emg), downsample)
    best_text = ""
    best_dist = float("inf")
    for text, proto in prototypes.items():
        d = dtw_distance(env, proto)
        if d < best_dist:
            best_dist = d
            best_text = text
    return best_text, best_dist


def save_preprocessed_npz(out_path: str | Path, raw_npz: Path, proc_emg: np.ndarray, sr: int, text: str, phonemes: str, speaker_id: str, session_id: str) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        emg=proc_emg.astype(np.float32),
        sr=np.array(sr),
        text=np.array(text),
        phonemes=np.array(phonemes),
        speaker_id=np.array(speaker_id),
        session_id=np.array(session_id),
        source_path=np.array(str(raw_npz)),
    )
