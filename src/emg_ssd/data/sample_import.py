from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

import numpy as np


def _load_emg_file(path: Path, expected_channels: int) -> np.ndarray:
    suffix = path.suffix.lower()
    arr = None
    if suffix == ".npy":
        arr = np.load(path)
    elif suffix == ".npz":
        with np.load(path, allow_pickle=True) as d:
            for k in ["emg", "data", "x"]:
                if k in d.files:
                    arr = d[k]
                    break
    elif suffix == ".csv":
        data = np.genfromtxt(path, delimiter=",", dtype=np.float32, invalid_raise=False)
        if data.ndim == 1:
            data = data[None, :]
        data = data[~np.isnan(data).all(axis=1)]
        # If first column looks like timestamp, drop it.
        if data.shape[1] == expected_channels + 1:
            arr = data[:, 1:]
        else:
            arr = data
    else:
        raise ValueError(f"Unsupported sample file type: {path}")

    if arr is None:
        raise ValueError(f"No EMG array found in: {path}")
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array [T,C], got {arr.shape} for {path}")
    if arr.shape[1] != expected_channels and arr.shape[0] == expected_channels:
        arr = arr.T
    if arr.shape[1] != expected_channels:
        raise ValueError(f"Channel mismatch for {path}: expected {expected_channels}, got {arr.shape[1]}")
    return arr


def _read_manifest(manifest_path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            rows.append({k.strip(): (v.strip() if isinstance(v, str) else "") for k, v in row.items()})
    return rows


def write_manifest_template(manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    header = "file,sr,text,phonemes,speaker_id,session_id,split\n"
    sample = "sample_000.csv,1000,HELLO WORLD,HH AH L OW W ER L D,user0,sess0,test\n"
    manifest_path.write_text(header + sample, encoding="utf-8")


def import_sample_data(
    raw_dir: str | Path,
    internal_root: str | Path,
    expected_channels: int,
    target_sr: int,
    manifest_path: str | Path | None = None,
    prefix: str = "sample",
    default_split: str = "test",
) -> int:
    raw_dir = Path(raw_dir)
    internal_root = Path(internal_root)
    internal_root.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, str]]
    if manifest_path is not None and Path(manifest_path).exists():
        rows = _read_manifest(Path(manifest_path))
    else:
        files = sorted(list(raw_dir.glob("*.csv")) + list(raw_dir.glob("*.npy")) + list(raw_dir.glob("*.npz")))
        rows = [
            {
                "file": f.name,
                "sr": str(target_sr),
                "text": "",
                "phonemes": "",
                "speaker_id": "sample_user",
                "session_id": "sample_session",
                "split": default_split,
            }
            for f in files
        ]

    n = 0
    for i, row in enumerate(rows):
        rel = row.get("file", "")
        if not rel:
            continue
        src = raw_dir / rel
        if not src.exists():
            continue
        split = (row.get("split", "") or default_split).lower()
        if split not in {"train", "val", "test"}:
            split = default_split
        sr = int(float(row.get("sr", str(target_sr)) or target_sr))
        text = row.get("text", "")
        phonemes = row.get("phonemes", "")
        speaker_id = row.get("speaker_id", "sample_user")
        session_id = row.get("session_id", "sample_session")

        emg = _load_emg_file(src, expected_channels)
        out = internal_root / split / f"{prefix}_{i:05d}.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out,
            emg=emg.astype(np.float32),
            sr=np.array(sr),
            text=np.array(text),
            phonemes=np.array(phonemes),
            speaker_id=np.array(speaker_id),
            session_id=np.array(session_id),
        )
        n += 1
    return n
