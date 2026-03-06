from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple


def load_lexicon(path: str) -> Dict[str, List[str]]:
    lex: Dict[str, List[str]] = {}
    p = Path(path)
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            word = parts[0].upper()
            phones = " ".join(parts[1:]).replace("0", "").replace("1", "").replace("2", "")
            lex[word] = phones.split()
    if not lex:
        lex = {"HELLO": ["HH", "AH", "L", "OW"], "WORLD": ["W", "ER", "L", "D"]}
    try:
        import cmudict  # type: ignore

        cmu = cmudict.dict()
        for w, phones_list in cmu.items():
            if not phones_list:
                continue
            phones = [p.replace("0", "").replace("1", "").replace("2", "") for p in phones_list[0]]
            if w.upper() not in lex:
                lex[w.upper()] = phones
    except Exception:
        pass
    return lex


def _edit_distance(a: List[str], b: List[str]) -> int:
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        dp[i][0] = i
    for j in range(len(b) + 1):
        dp[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            c = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + c)
    return dp[-1][-1]


def phonemes_to_words(phonemes: List[str], lexicon: Dict[str, List[str]], max_edit: int = 2) -> Tuple[str, float]:
    if not phonemes:
        return "", 0.0

    n = len(phonemes)
    best = [(-1e9, []) for _ in range(n + 1)]
    best[0] = (0.0, [])
    lex_items = list(lexicon.items())
    max_word_len = max(len(v) for _, v in lex_items)

    for i in range(n):
        if best[i][0] < -1e8:
            continue
        for l in range(1, min(max_word_len, n - i) + 1):
            chunk = phonemes[i:i + l]
            for w, p in lex_items:
                d = _edit_distance(chunk, p)
                if d <= max_edit:
                    score = best[i][0] - d - 0.1 * abs(len(chunk) - len(p))
                    if score > best[i + l][0]:
                        best[i + l] = (score, best[i][1] + [w])

    if best[n][0] < -1e8:
        return " ".join(phonemes), 0.05

    conf = 1.0 / (1.0 + abs(best[n][0]))
    return " ".join(best[n][1]), float(conf)
