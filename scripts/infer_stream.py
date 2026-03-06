from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from emg_ssd.config import load_config
from emg_ssd.ctc_decode import ctc_beam_decode, ctc_greedy_decode
from emg_ssd.features import extract_features
from emg_ssd.models.encoder_ctc import build_model
from emg_ssd.preprocessing import PreprocessConfig, preprocess_emg
from emg_ssd.serial_reader import SerialEMGReader
from emg_ssd.streaming import WindowBuffer, simulated_stream
from emg_ssd.tokenizer import build_tokenizer
from emg_ssd.train_utils import resolve_device
from emg_ssd.wordify import load_lexicon, phonemes_to_words


def _predict_window(model, win: np.ndarray, cfg: dict, tok, device: torch.device, beam: bool = True):
    pp = PreprocessConfig.from_cfg(cfg)
    x = preprocess_emg(win, pp.target_sr, pp)
    feat = extract_features(x, pp.target_sr, cfg["features"])
    xt = torch.from_numpy(feat).unsqueeze(0).to(device)
    with torch.no_grad():
        lp = torch.log_softmax(model(xt)[0], dim=-1).cpu()
    if beam:
        return ctc_beam_decode(lp, tok.blank_id, int(cfg["decode"]["beam_size"]))
    return ctc_greedy_decode(lp, tok.blank_id)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--simulate_file", default="")
    ap.add_argument("--decode", choices=["greedy", "beam"], default="beam")
    ap.add_argument("--max_windows", type=int, default=20)
    args = ap.parse_args()

    cfg = load_config(args.config)
    target_mode = cfg.get("targets", {}).get("mode", "phoneme")
    tok = build_tokenizer(cfg)

    ckpt_path = cfg.get("infer", {}).get("checkpoint", str(Path(cfg["paths"]["checkpoints"]) / "phoneme_ctc.pt"))
    ckpt = torch.load(ckpt_path, map_location="cpu")

    feat_mode = cfg["features"]["mode"]
    input_dim = int(cfg["data"]["expected_channels"]) * int(cfg["features"]["frame_ms"] * cfg["data"]["target_sr"] / 1000) if feat_mode == "raw" else int(cfg["data"]["expected_channels"]) * (int(cfg["features"]["n_fft"]) // 2 + 1)

    model = build_model(cfg, input_dim, tok.vocab_size)
    model.load_state_dict(ckpt["model_state"], strict=False)
    device = resolve_device(str(cfg["train"]["device"]))
    model.to(device).eval()

    lex = load_lexicon(cfg["paths"]["lexicon_path"])
    sr = int(cfg["data"]["target_sr"])
    ch = int(cfg["serial"]["channels"] if "serial" in cfg else cfg["data"]["expected_channels"])
    wb = WindowBuffer(sr=sr, channels=ch, window_ms=int(cfg["stream"]["window_ms"]), hop_ms=int(cfg["stream"]["hop_ms"]))

    if args.simulate:
        sim_path = args.simulate_file
        if not sim_path:
            train_dir = Path(cfg["paths"]["internal_root"]) / "test"
            candidates = sorted(train_dir.glob("*.npz"))
            if not candidates:
                raise RuntimeError("No internal test files for simulated stream")
            sim_path = str(candidates[0])
        source = simulated_stream(sim_path, ch)
    else:
        serial_cfg = cfg["serial"]
        reader = SerialEMGReader(
            port=str(serial_cfg["port"]),
            baud_rate=int(serial_cfg["baud_rate"]),
            channels=int(serial_cfg["channels"]),
            scale=float(serial_cfg.get("scale", 1.0)),
        )
        reader.__enter__()
        source = iter(reader)

    try:
        printed = 0
        for _, sample in source:
            for win in wb.push(sample):
                ids, conf = _predict_window(model, win, cfg, tok, device=device, beam=args.decode == "beam")
                ph = tok.decode_ids(ids)
                words, w_conf = (ph, conf) if target_mode == "char" else phonemes_to_words(ph.split(), lex, max_edit=int(cfg["decode"]["fuzzy_max_edit"]))
                print(f"phonemes={ph} | words={words} | conf={conf:.3f}/{w_conf:.3f}")
                printed += 1
                if args.max_windows > 0 and printed >= args.max_windows:
                    return
    finally:
        pass


if __name__ == "__main__":
    main()
