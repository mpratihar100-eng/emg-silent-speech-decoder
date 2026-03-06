from __future__ import annotations

from pathlib import Path
from typing import Dict, List


class Tokenizer:
    def __init__(self, vocab_path: str, add_blank: bool = True) -> None:
        lines = [l.strip().upper() for l in Path(vocab_path).read_text(encoding="utf-8").splitlines() if l.strip()]
        if add_blank and "BLANK" not in lines:
            lines = ["BLANK"] + lines
        self.tokens: List[str] = lines
        self.token_to_id: Dict[str, int] = {t: i for i, t in enumerate(self.tokens)}
        self.blank_id = self.token_to_id.get("BLANK", 0)

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    def encode(self, phoneme_str: str) -> List[int]:
        ids: List[int] = []
        for p in phoneme_str.strip().upper().split():
            if p in self.token_to_id:
                ids.append(self.token_to_id[p])
        return ids

    def decode_ids(self, ids: List[int]) -> str:
        toks = [self.tokens[i] for i in ids if 0 <= i < len(self.tokens)]
        return " ".join(toks)


class CharTokenizer:
    def __init__(self, add_blank: bool = True) -> None:
        base = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ '")
        self.tokens = (["BLANK"] + base) if add_blank else base
        self.token_to_id: Dict[str, int] = {t: i for i, t in enumerate(self.tokens)}
        self.blank_id = self.token_to_id.get("BLANK", 0)

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    def encode(self, text: str) -> List[int]:
        ids: List[int] = []
        for ch in text.upper():
            if ch in self.token_to_id:
                ids.append(self.token_to_id[ch])
        return ids

    def decode_ids(self, ids: List[int]) -> str:
        chars = [self.tokens[i] for i in ids if 0 <= i < len(self.tokens) and self.tokens[i] != "BLANK"]
        return "".join(chars).strip()


def build_tokenizer(cfg: dict):
    mode = cfg.get("targets", {}).get("mode", "phoneme")
    if mode == "char":
        return CharTokenizer(add_blank=bool(cfg["tokens"].get("add_blank", True)))
    return Tokenizer(cfg["tokens"]["vocab_path"], add_blank=bool(cfg["tokens"].get("add_blank", True)))
