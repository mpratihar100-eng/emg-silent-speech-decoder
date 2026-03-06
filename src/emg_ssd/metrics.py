from __future__ import annotations

from typing import List


def edit_distance(a: List[str], b: List[str]) -> int:
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


def phoneme_error_rate(ref: str, hyp: str) -> float:
    r = [x for x in ref.strip().split() if x]
    h = [x for x in hyp.strip().split() if x]
    if not r:
        return 0.0 if not h else 1.0
    return edit_distance(r, h) / len(r)


def word_error_rate(ref: str, hyp: str) -> float:
    r = [x for x in ref.strip().upper().split() if x]
    h = [x for x in hyp.strip().upper().split() if x]
    if not r:
        return 0.0 if not h else 1.0
    return edit_distance(r, h) / len(r)
