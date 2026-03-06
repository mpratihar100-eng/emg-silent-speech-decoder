from __future__ import annotations

import math
from typing import List, Tuple

import torch


def ctc_greedy_decode(log_probs: torch.Tensor, blank_id: int) -> Tuple[List[int], float]:
    top = torch.argmax(log_probs, dim=-1).tolist()
    out: List[int] = []
    prev = None
    score = 0.0
    for t, idx in enumerate(top):
        score += float(log_probs[t, idx].item())
        if idx != blank_id and idx != prev:
            out.append(idx)
        prev = idx
    conf = float(math.exp(score / max(1, len(top))))
    return out, conf


def _log_add(a: float, b: float) -> float:
    if a == float("-inf"):
        return b
    if b == float("-inf"):
        return a
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))


def ctc_beam_decode(log_probs: torch.Tensor, blank_id: int, beam_size: int = 8) -> Tuple[List[int], float]:
    beams = {(): (0.0, float("-inf"))}
    T, V = log_probs.shape
    for t in range(T):
        next_beams = {}
        for prefix, (pb, pnb) in beams.items():
            for v in range(V):
                p = float(log_probs[t, v].item())
                if v == blank_id:
                    nb_pb, nb_pnb = next_beams.get(prefix, (float("-inf"), float("-inf")))
                    nb_pb = _log_add(nb_pb, pb + p)
                    nb_pb = _log_add(nb_pb, pnb + p)
                    next_beams[prefix] = (nb_pb, nb_pnb)
                    continue
                end = prefix[-1] if prefix else None
                new_prefix = prefix + (v,)
                nb_pb, nb_pnb = next_beams.get(new_prefix, (float("-inf"), float("-inf")))
                if v == end:
                    nb_pnb = _log_add(nb_pnb, pb + p)
                    cur_pb, cur_pnb = next_beams.get(prefix, (float("-inf"), float("-inf")))
                    cur_pnb = _log_add(cur_pnb, pnb + p)
                    next_beams[prefix] = (cur_pb, cur_pnb)
                else:
                    nb_pnb = _log_add(nb_pnb, pb + p)
                    nb_pnb = _log_add(nb_pnb, pnb + p)
                next_beams[new_prefix] = (nb_pb, nb_pnb)
        scored = sorted(next_beams.items(), key=lambda kv: _log_add(kv[1][0], kv[1][1]), reverse=True)
        beams = dict(scored[:beam_size])

    best_prefix, (pb, pnb) = max(beams.items(), key=lambda kv: _log_add(kv[1][0], kv[1][1]))
    conf = float(math.exp(_log_add(pb, pnb) / max(1, T)))
    return list(best_prefix), conf
