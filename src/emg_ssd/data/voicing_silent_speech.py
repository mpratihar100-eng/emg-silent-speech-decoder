from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np


FALLBACK_TEXTS = [
    "HELLO WORLD",
    "YES",
    "NO",
    "PLEASE STOP",
    "GO LEFT",
    "GO RIGHT",
]


def _simple_text_to_phonemes(text: str, lexicon: dict[str, list[str]]) -> str:
    toks = []
    for w in text.upper().split():
        toks.extend(lexicon.get(w, ["SP"]))
    return " ".join(toks)


def convert_generic_arrays_to_internal(source_dir: str | Path, out_dir: str | Path, expected_channels: int = 8, sr: int = 1000) -> int:
    source = Path(source_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = list(source.rglob("*.npy")) + list(source.rglob("*.npz"))
    n = 0
    lex = {
        "HELLO": ["HH", "AH", "L", "OW"],
        "WORLD": ["W", "ER", "L", "D"],
        "YES": ["Y", "EH", "S"],
        "NO": ["N", "OW"],
        "PLEASE": ["P", "L", "IY", "Z"],
        "STOP": ["S", "T", "AA", "P"],
        "GO": ["G", "OW"],
        "LEFT": ["L", "EH", "F", "T"],
        "RIGHT": ["R", "AY", "T"],
    }
    for i, f in enumerate(files):
        arr = None
        if f.suffix == ".npy":
            arr = np.load(f)
        else:
            with np.load(f, allow_pickle=True) as d:
                for k in ["emg", "data", "x"]:
                    if k in d.files:
                        arr = d[k]
                        break
                if arr is None:
                    continue
        arr = np.asarray(arr)
        if arr.ndim == 1:
            arr = arr[:, None]
        if arr.ndim != 2:
            continue
        if arr.shape[1] != expected_channels and arr.shape[0] == expected_channels:
            arr = arr.T
        if arr.shape[1] != expected_channels:
            continue

        text = FALLBACK_TEXTS[i % len(FALLBACK_TEXTS)]
        phones = _simple_text_to_phonemes(text, lex)
        split = "train" if i % 10 < 8 else ("val" if i % 10 == 8 else "test")
        tgt = out / split / f"utt_{i:05d}.npz"
        tgt.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            tgt,
            emg=arr.astype(np.float32),
            sr=np.array(sr),
            text=np.array(text),
            phonemes=np.array(phones),
            speaker_id=np.array("spk0"),
            session_id=np.array("sess0"),
        )
        n += 1
    return n


def generate_synthetic_internal(out_dir: str | Path, n_train: int = 48, n_val: int = 8, n_test: int = 8, sr: int = 1000, channels: int = 8) -> int:
    rng = np.random.default_rng(7)
    out = Path(out_dir)
    lex = {
        "HELLO WORLD": "HH AH L OW W ER L D",
        "YES": "Y EH S",
        "NO": "N OW",
        "PLEASE STOP": "P L IY Z S T AA P",
        "GO LEFT": "G OW L EH F T",
        "GO RIGHT": "G OW R AY T",
    }

    def make_one(i: int, split: str) -> None:
        text = list(lex.keys())[i % len(lex)]
        phones = lex[text]
        t = rng.integers(900, 1800)
        x = rng.normal(0, 0.05, size=(t, channels)).astype(np.float32)
        f0 = 3 + (i % 5)
        tt = np.arange(t) / sr
        for c in range(channels):
            x[:, c] += (0.2 + 0.03 * c) * np.sin(2 * np.pi * (f0 + c * 0.5) * tt).astype(np.float32)
        p = out / split / f"synthetic_{i:05d}.npz"
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            p,
            emg=x,
            sr=np.array(sr),
            text=np.array(text),
            phonemes=np.array(phones),
            speaker_id=np.array("synthetic_spk"),
            session_id=np.array("synthetic_sess"),
        )

    i = 0
    for _ in range(n_train):
        make_one(i, "train")
        i += 1
    for _ in range(n_val):
        make_one(i, "val")
        i += 1
    for _ in range(n_test):
        make_one(i, "test")
        i += 1
    return i


def iter_internal_npz(root: str | Path) -> Iterable[Path]:
    return Path(root).rglob("*.npz")
