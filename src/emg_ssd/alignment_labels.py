from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np


PHONE_KEYS = ("phonemes", "phones", "arpabet", "aligned_phonemes")
TEXT_KEYS = ("text", "transcript", "sentence", "utterance")


def _normalize_text(text: str) -> str:
    return " ".join(str(text).strip().upper().split())


def _normalize_phones(phones: str) -> str:
    phones = phones.replace(",", " ")
    phones = re.sub(r"\s+", " ", phones.strip().upper())
    return phones


def _infer_key_from_name(path: Path) -> str:
    stem = path.stem
    stem = re.sub(r"_audio$", "", stem)
    m = re.search(r"(\d+(?:-\d+)?(?:_\d+)?)", stem)
    if m:
        return m.group(1)
    return stem


def _extract_from_json(path: Path) -> Tuple[str, str]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return "", ""
    text = ""
    phones = ""
    for key in TEXT_KEYS:
        if key in data and str(data[key]).strip():
            text = _normalize_text(str(data[key]))
            break
    for key in PHONE_KEYS:
        if key in data and str(data[key]).strip():
            phones = _normalize_phones(str(data[key]))
            break
    return text, phones


def _extract_from_text(path: Path) -> Tuple[str, str]:
    raw = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not raw:
        return "", ""
    lines = [x.strip() for x in raw.splitlines() if x.strip()]
    if len(lines) == 1:
        return "", _normalize_phones(lines[0])
    return _normalize_text(lines[0]), _normalize_phones(lines[-1])


def _extract_from_textgrid(path: Path) -> Tuple[str, str]:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    # Minimal parser for Praat TextGrid long format.
    intervals: list[tuple[str, str]] = []
    current_tier = ""
    current_text = ""
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("name = "):
            current_tier = line.split("=", 1)[1].strip().strip('"')
        elif line.startswith("text = "):
            current_text = line.split("=", 1)[1].strip().strip('"')
            intervals.append((current_tier, current_text))
    words: list[str] = []
    phones: list[str] = []
    for tier, text in intervals:
        text = text.strip()
        if not text:
            continue
        tier_l = tier.lower()
        if "word" in tier_l:
            words.append(text)
        elif "phone" in tier_l or "phoneme" in tier_l:
            if text.lower() in {"sil", "sp", "spn"}:
                phones.append(text.upper())
            else:
                phones.append(re.sub(r"[0-9]", "", text.upper()))
    return _normalize_text(" ".join(words)), _normalize_phones(" ".join(phones))


def scan_alignment_dir(root: str | Path) -> Dict[str, Dict[str, str]]:
    root = Path(root)
    out: Dict[str, Dict[str, str]] = {}
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".json", ".txt", ".phn", ".textgrid"}:
            continue
        if p.suffix.lower() == ".json":
            text, phones = _extract_from_json(p)
        elif p.suffix.lower() == ".textgrid":
            text, phones = _extract_from_textgrid(p)
        else:
            text, phones = _extract_from_text(p)
        if not phones:
            continue
        key = _infer_key_from_name(p)
        out[key] = {"text": text, "phonemes": phones, "alignment_path": str(p)}
    return out


def patch_internal_labels(internal_root: str | Path, labels: Dict[str, Dict[str, str]]) -> int:
    root = Path(internal_root)
    n = 0
    for p in root.rglob("*.npz"):
        with np.load(p, allow_pickle=True) as d:
            payload = {k: d[k] for k in d.files}
        utt_id = str(np.array(payload.get("utterance_id", "")).item()) if "utterance_id" in payload else ""
        source_file = str(np.array(payload.get("source_file", "")).item()) if "source_file" in payload else ""
        key_candidates = [utt_id]
        if source_file:
            sf = Path(source_file)
            key_candidates.append(_infer_key_from_name(sf))
            stem_key = _infer_key_from_name(sf)
            parent = sf.parent.name.replace("_silent", "").replace("_voiced", "")
            bare = re.search(r"(\d+)", sf.stem)
            if parent and bare:
                key_candidates.append(f"{parent}_{bare.group(1)}")
        label = None
        for key in key_candidates:
            if key and key in labels:
                label = labels[key]
                break
        if label is None:
            continue
        if label.get("text"):
            payload["text"] = np.array(label["text"])
        payload["phonemes"] = np.array(label["phonemes"])
        payload["alignment_path"] = np.array(label.get("alignment_path", ""))
        np.savez_compressed(p, **payload)
        n += 1
    return n


def iter_label_issues(files: Iterable[Path], vocab: set[str]) -> Iterable[str]:
    for p in files:
        try:
            with np.load(p, allow_pickle=True) as d:
                text = str(d["text"].item()) if "text" in d.files else ""
                phonemes = str(d["phonemes"].item()) if "phonemes" in d.files else ""
                emg = d["emg"]
        except Exception:
            yield f"{p}: unreadable"
            continue
        if emg.ndim != 2 or emg.shape[0] == 0 or emg.shape[1] == 0:
            yield f"{p}: invalid emg shape {getattr(emg, 'shape', None)}"
        if not text.strip():
            yield f"{p}: empty text"
        if not phonemes.strip():
            yield f"{p}: empty phonemes"
            continue
        bad = [tok for tok in phonemes.split() if tok.upper() not in vocab]
        if bad:
            yield f"{p}: unknown tokens {bad[:5]}"
