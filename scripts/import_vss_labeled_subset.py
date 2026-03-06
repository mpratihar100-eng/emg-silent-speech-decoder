from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from emg_ssd.config import ensure_dirs, load_config

try:
    import pronouncing
except Exception:  # pragma: no cover
    pronouncing = None


def text_to_arpabet(text: str) -> str:
    words = re.findall(r"[A-Za-z']+", text.upper())
    phones: list[str] = []
    for w in words:
        ph = None
        if pronouncing is not None:
            cand = pronouncing.phones_for_word(w.lower())
            if cand:
                ph = cand[0]
        if ph:
            toks = [p.replace("0", "").replace("1", "").replace("2", "") for p in ph.split()]
            phones.extend(toks)
        else:
            phones.append("SP")
    return " ".join(phones)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--max_samples", type=int, default=500)
    ap.add_argument("--prefix", default="labeledvss")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)

    raw_root = Path(cfg["paths"]["data_root"]) / "voicing_silent_speech_raw" / "extracted" / "emg_data"
    if not raw_root.exists():
        raise RuntimeError(f"Raw VSS data not found at {raw_root}. Run download_data.py first.")

    out_root = Path(cfg["paths"]["internal_root"])
    out_root.mkdir(parents=True, exist_ok=True)
    target_sr = int(cfg["data"]["target_sr"])
    max_samples = int(args.max_samples)

    info_files = sorted(raw_root.rglob("*_info.json"))
    rows = []
    for info in info_files:
        try:
            meta = json.loads(info.read_text(encoding="utf-8"))
        except Exception:
            continue
        text = str(meta.get("text", "")).strip()
        if not text:
            continue
        emg_file = info.with_name(info.name.replace("_info.json", "_emg.npy"))
        if not emg_file.exists():
            continue
        phonemes = text_to_arpabet(text)
        if not phonemes.strip():
            continue
        rows.append((emg_file, text, phonemes))
        if len(rows) >= max_samples:
            break

    if not rows:
        raise RuntimeError("No labeled rows found in VSS metadata.")

    n = len(rows)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    written = 0

    for i, (emg_file, text, phonemes) in enumerate(rows):
        split = "train" if i < n_train else ("val" if i < n_train + n_val else "test")
        emg = np.load(emg_file).astype(np.float32)
        if emg.ndim == 1:
            emg = emg[:, None]
        if emg.ndim != 2:
            continue
        out = out_root / split / f"{args.prefix}_{i:05d}.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out,
            emg=emg,
            sr=np.array(target_sr),
            text=np.array(text),
            phonemes=np.array(phonemes),
            speaker_id=np.array("vss"),
            session_id=np.array(emg_file.parent.name),
        )
        written += 1

    print(f"Imported labeled VSS subset: {written} samples into {out_root}")
    print(f"Prefix: {args.prefix} | train/val/test ~= 80/10/10")


if __name__ == "__main__":
    main()
