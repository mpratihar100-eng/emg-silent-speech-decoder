from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.nn import CTCLoss
from torch.utils.data import DataLoader

from emg_ssd.config import load_config
from emg_ssd.data.npz_dataset import NPZUtteranceDataset, ctc_collate, load_split_files
from emg_ssd.models.encoder_ctc import build_model
from emg_ssd.tokenizer import build_tokenizer
from emg_ssd.train_utils import resolve_device, save_checkpoint


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    target_mode = cfg.get("targets", {}).get("mode", "phoneme")
    tok = build_tokenizer(cfg)
    files = load_split_files(cfg["paths"]["internal_root"], cfg["calibration"]["split"])
    if not files:
        raise RuntimeError("No calibration files. Run download_data first.")

    ds = NPZUtteranceDataset(files, cfg, tok, expect_phonemes=True, target_mode=target_mode)
    dl = DataLoader(ds, batch_size=int(cfg["train"]["batch_size"]), shuffle=True, collate_fn=ctc_collate)

    input_dim = ds[0]["x"].shape[1]
    model = build_model(cfg, input_dim, tok.vocab_size)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model_state"], strict=False)

    if bool(cfg["calibration"].get("adapter_only", True)):
        for n, p in model.named_parameters():
            p.requires_grad = ("adapter" in n or "classifier" in n)

    device = resolve_device(str(cfg["train"]["device"]))
    model.to(device)

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(cfg["calibration"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    ctc = CTCLoss(blank=tok.blank_id, zero_infinity=True)

    for ep in range(int(cfg["calibration"]["epochs"])):
        model.train()
        losses = []
        for b in dl:
            x = b["x"].to(device)
            y = b["y"].to(device)
            x_lens = b["x_lens"].to(device)
            y_lens = b["y_lens"].to(device)
            logits = model(x)
            lp = torch.log_softmax(logits, dim=-1).transpose(0, 1)
            loss = ctc(lp, y, x_lens, y_lens)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        print(f"calibration_epoch={ep+1} ctc_loss={sum(losses)/max(1,len(losses)):.4f}")

    out = Path(cfg["paths"]["checkpoints"]) / ("char_ctc_calibrated.pt" if target_mode == "char" else "phoneme_ctc_calibrated.pt")
    save_checkpoint(out, {"model_state": model.state_dict(), "config": cfg, "vocab": tok.tokens})
    print(f"Saved calibrated checkpoint: {out}")


if __name__ == "__main__":
    main()
