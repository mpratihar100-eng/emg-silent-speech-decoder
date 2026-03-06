from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.nn import CTCLoss
from torch.utils.data import DataLoader

from emg_ssd.config import ensure_dirs, load_config
from emg_ssd.data.npz_dataset import NPZUtteranceDataset, ctc_collate, load_split_files
from emg_ssd.models.encoder_ctc import build_model
from emg_ssd.tokenizer import build_tokenizer
from emg_ssd.train_utils import resolve_device, save_checkpoint, set_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--save_every_steps", type=int, default=0)
    ap.add_argument("--log_every_steps", type=int, default=10)
    ap.add_argument("--metrics_file", default="outputs/train_metrics.jsonl")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    set_seed(int(cfg.get("seed", 7)))

    target_mode = cfg.get("targets", {}).get("mode", "phoneme")
    tok = build_tokenizer(cfg)
    train_files = load_split_files(cfg["paths"]["internal_root"], "train")
    val_files = load_split_files(cfg["paths"]["internal_root"], "val")
    if not train_files:
        raise RuntimeError("No training .npz files found. Run scripts/download_data.py first.")

    ds = NPZUtteranceDataset(train_files, cfg, tok, expect_phonemes=True, target_mode=target_mode)
    dl = DataLoader(
        ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["train"]["num_workers"]),
        collate_fn=ctc_collate,
    )

    sample_dim = ds[0]["x"].shape[1]
    model = build_model(cfg, sample_dim, tok.vocab_size)
    device = resolve_device(str(cfg["train"]["device"]))
    model.to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    ctc = CTCLoss(blank=tok.blank_id, zero_infinity=True)

    ckpt_name = "char_ctc.pt" if target_mode == "char" else "phoneme_ctc.pt"
    ckpt = Path(args.checkpoint) if args.checkpoint else (Path(cfg["paths"]["checkpoints"]) / ckpt_name)
    metrics_path = Path(args.metrics_file)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    global_step = 0
    start_epoch = 0
    if args.resume and ckpt.exists():
        state = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(state["model_state"], strict=False)
        if "optimizer_state" in state:
            opt.load_state_dict(state["optimizer_state"])
        global_step = int(state.get("global_step", 0))
        start_epoch = int(state.get("epoch", 0))
        print(f"Resumed from {ckpt} at global_step={global_step}, epoch={start_epoch}")

    epochs = int(cfg["train"]["epochs"])
    run_steps = 0
    wall_start = time.time()
    def _save(ep_idx: int) -> None:
        save_checkpoint(
            ckpt,
            {
                "model_state": model.state_dict(),
                "optimizer_state": opt.state_dict(),
                "config": cfg,
                "vocab": tok.tokens,
                "val_count": len(val_files),
                "global_step": global_step,
                "epoch": ep_idx + 1,
            },
        )

    for ep in range(start_epoch, start_epoch + epochs):
        model.train()
        total = 0.0
        n = 0
        for b in dl:
            x = b["x"].to(device)
            x_lens = b["x_lens"].to(device)
            y = b["y"].to(device)
            y_lens = b["y_lens"].to(device)

            logits = model(x)
            log_probs = torch.log_softmax(logits, dim=-1).transpose(0, 1)
            loss = ctc(log_probs, y, x_lens, y_lens)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"]["clip_grad_norm"]))
            opt.step()

            total += float(loss.item())
            n += 1
            global_step += 1
            run_steps += 1
            if args.log_every_steps > 0 and (run_steps % args.log_every_steps == 0):
                elapsed = max(1e-9, time.time() - wall_start)
                sps = run_steps / elapsed
                eta = ((max(args.max_steps, run_steps) - run_steps) / sps) if args.max_steps > 0 else 0.0
                msg = (
                    f"progress run_steps={run_steps} global_step={global_step} "
                    f"loss={float(loss.item()):.4f} steps_per_sec={sps:.3f} "
                    f"elapsed_min={elapsed/60.0:.2f}"
                )
                if args.max_steps > 0:
                    msg += f" eta_min={eta/60.0:.2f}"
                print(msg)
                rec = {
                    "run_steps": run_steps,
                    "global_step": global_step,
                    "loss": float(loss.item()),
                    "steps_per_sec": sps,
                    "elapsed_sec": elapsed,
                    "eta_sec": eta if args.max_steps > 0 else None,
                }
                with metrics_path.open("a", encoding="utf-8") as fp:
                    fp.write(json.dumps(rec) + "\n")
            if args.save_every_steps > 0 and (run_steps % args.save_every_steps == 0):
                _save(ep)
                print(f"autosave checkpoint at run_steps={run_steps}, global_step={global_step}")
            if args.max_steps > 0 and run_steps >= args.max_steps:
                break

        print(f"epoch={ep + 1} train_ctc_loss={total / max(1, n):.4f}")
        if args.max_steps > 0 and run_steps >= args.max_steps:
            break

    _save(ep)
    print(f"Saved checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
