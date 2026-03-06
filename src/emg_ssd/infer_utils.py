from __future__ import annotations

import numpy as np
import torch

from emg_ssd.ctc_decode import ctc_beam_decode, ctc_greedy_decode
from emg_ssd.features import extract_features
from emg_ssd.preprocessing import PreprocessConfig, preprocess_emg


def infer_window(model: torch.nn.Module, x: np.ndarray, cfg: dict, blank_id: int, beam: bool = False):
    pp = PreprocessConfig.from_cfg(cfg)
    x = preprocess_emg(x, pp.target_sr, pp)
    feat = extract_features(x, pp.target_sr, cfg["features"])
    xt = torch.from_numpy(feat).unsqueeze(0)
    with torch.no_grad():
        logits = model(xt)
        log_probs = torch.log_softmax(logits[0], dim=-1)
    if beam:
        return ctc_beam_decode(log_probs, blank_id, int(cfg["decode"]["beam_size"]))
    return ctc_greedy_decode(log_probs, blank_id)
