from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from emg_ssd.config import load_config
from emg_ssd.data.npz_dataset import load_split_files
from emg_ssd.models.encoder_ctc import build_model
from emg_ssd.tokenizer import build_tokenizer
from emg_ssd.visualize import (
    framewise_phoneme_segments,
    load_internal_npz,
    plot_emg_channels,
    plot_phoneme_timeline,
    preprocess_and_features,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--npz_path", default="")
    ap.add_argument("--split", default="test")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--out_dir", default="outputs/visuals")
    args = ap.parse_args()

    cfg = load_config(args.config)
    tok = build_tokenizer(cfg)

    if args.npz_path:
        npz_path = Path(args.npz_path)
    else:
        files = load_split_files(cfg["paths"]["internal_root"], args.split)
        if not files:
            raise RuntimeError(f"No internal samples found for split={args.split}")
        idx = min(max(0, args.index), len(files) - 1)
        npz_path = files[idx]

    d = load_internal_npz(npz_path)
    raw_emg = np.asarray(d["emg"], dtype=np.float32)
    sr = int(np.array(d["sr"]).item())
    ref_text = str(d["text"].item()) if "text" in d else ""
    ref_phonemes = str(d["phonemes"].item()) if "phonemes" in d else ""

    proc, feat = preprocess_and_features(raw_emg, sr, cfg)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    emg_plot = out_dir / f"{npz_path.stem}_emg.png"
    plot_emg_channels(raw_emg, proc, int(cfg["data"]["target_sr"]), emg_plot)
    print(f"Saved electrode visualization: {emg_plot}")

    ckpt_default = cfg.get("infer", {}).get("checkpoint", str(Path(cfg["paths"]["checkpoints"]) / "phoneme_ctc.pt"))
    ckpt_path = Path(args.checkpoint) if args.checkpoint else Path(ckpt_default)
    if not ckpt_path.exists():
        print(f"Checkpoint not found at {ckpt_path}; skipping phoneme mapping plot.")
        return

    model = build_model(cfg, feat.shape[1], tok.vocab_size)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()

    x = torch.from_numpy(feat).unsqueeze(0)
    with torch.no_grad():
        log_probs = torch.log_softmax(model(x)[0], dim=-1)
    hop_ms = int(cfg["features"]["hop_ms"])
    segments = framewise_phoneme_segments(log_probs, tok.tokens, tok.blank_id, hop_ms=hop_ms)
    total_s = (feat.shape[0] * hop_ms) / 1000.0

    timeline_plot = out_dir / f"{npz_path.stem}_phoneme_timeline.png"
    plot_phoneme_timeline(segments, timeline_plot, total_s=total_s, ref_phonemes=ref_phonemes)
    hyp = " ".join(seg["token"] for seg in segments)
    print(f"Saved phoneme timeline: {timeline_plot}")
    print(f"sample={npz_path}")
    print(f"ref_text={ref_text}")
    print(f"ref_phonemes={ref_phonemes}")
    print(f"hyp_segments={hyp}")


if __name__ == "__main__":
    main()
