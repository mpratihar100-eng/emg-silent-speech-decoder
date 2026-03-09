from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from emg_ssd.features import extract_features
from emg_ssd.preprocessing import PreprocessConfig, preprocess_emg
from emg_ssd.tokenizer import Tokenizer


class NPZUtteranceDataset(Dataset):
    def __init__(
        self,
        files: Sequence[Path],
        cfg: Dict,
        tokenizer: Tokenizer,
        expect_phonemes: bool = True,
        target_mode: str = "phoneme",
    ) -> None:
        self.files = list(files)
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.expect_phonemes = expect_phonemes
        self.target_mode = target_mode
        self.pp_cfg = PreprocessConfig.from_cfg(cfg)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict:
        p = self.files[idx]
        with np.load(p, allow_pickle=True) as d:
            emg = d["emg"].astype(np.float32)
            sr = int(np.array(d["sr"]).item())
            text = str(d["text"].item()) if "text" in d.files else ""
            phonemes = str(d["phonemes"].item()) if "phonemes" in d.files else ""
        emg = preprocess_emg(emg, sr, self.pp_cfg)
        feat = extract_features(emg, self.pp_cfg.target_sr, self.cfg["features"])
        if self.target_mode == "char":
            tgt = self.tokenizer.encode(text)
        else:
            tgt = self.tokenizer.encode(phonemes) if self.expect_phonemes else []
        return {
            "x": torch.from_numpy(feat),
            "x_len": feat.shape[0],
            "y": torch.tensor(tgt, dtype=torch.long),
            "y_len": len(tgt),
            "text": text,
            "phonemes": phonemes,
            "path": str(p),
        }


def ctc_collate(batch: List[Dict]) -> Dict:
    batch = sorted(batch, key=lambda b: b["x_len"], reverse=True)
    max_t = max(b["x_len"] for b in batch)
    d = batch[0]["x"].shape[1]
    x = torch.zeros((len(batch), max_t, d), dtype=torch.float32)
    x_lens = torch.tensor([b["x_len"] for b in batch], dtype=torch.long)
    ys = []
    y_lens = torch.tensor([b["y_len"] for b in batch], dtype=torch.long)
    for i, b in enumerate(batch):
        x[i, : b["x_len"]] = b["x"]
        ys.append(b["y"])
    y = torch.cat(ys, dim=0) if ys and ys[0].numel() > 0 else torch.zeros((0,), dtype=torch.long)
    return {
        "x": x,
        "x_lens": x_lens,
        "y": y,
        "y_lens": y_lens,
        "raw": batch,
    }


def load_split_files(internal_root: str | Path, split: str) -> List[Path]:
    p = Path(internal_root) / split
    files = sorted(p.rglob("*.npz"))
    return files


def filter_split_files(files: Sequence[Path], cfg: Dict) -> List[Path]:
    dcfg = cfg.get("data", {})
    include_prefixes = [str(x).lower() for x in dcfg.get("include_prefixes", [])]
    exclude_prefixes = [str(x).lower() for x in dcfg.get("exclude_prefixes", [])]
    require_text = bool(dcfg.get("require_nonempty_text", False))
    require_ph = bool(dcfg.get("require_nonempty_phonemes", False))
    require_alignment = bool(dcfg.get("require_alignment_path", False))

    out: List[Path] = []
    for f in files:
        stem = f.stem.lower()
        prefix = stem.split("_")[0] if "_" in stem else stem
        if include_prefixes and prefix not in include_prefixes:
            continue
        if exclude_prefixes and prefix in exclude_prefixes:
            continue
        if require_text or require_ph:
            try:
                with np.load(f, allow_pickle=True) as d:
                    text = str(d["text"].item()) if "text" in d.files else ""
                    phonemes = str(d["phonemes"].item()) if "phonemes" in d.files else ""
                    alignment_path = str(d["alignment_path"].item()) if "alignment_path" in d.files else ""
                if require_text and not text.strip():
                    continue
                if require_ph and not phonemes.strip():
                    continue
                if require_alignment and not alignment_path.strip():
                    continue
            except Exception:
                continue
        out.append(f)
    return out
