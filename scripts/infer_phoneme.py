from __future__ import annotations

import argparse
from pathlib import Path

import torch

from emg_ssd.config import load_config
from emg_ssd.ctc_decode import ctc_beam_decode, ctc_greedy_decode
from emg_ssd.data.npz_dataset import NPZUtteranceDataset, filter_split_files, load_split_files
from emg_ssd.models.encoder_ctc import build_model
from emg_ssd.tokenizer import build_tokenizer
from emg_ssd.train_utils import resolve_device
from emg_ssd.tts.elevenlabs_tts import elevenlabs_tts
from emg_ssd.wordify import load_lexicon, phonemes_to_words


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--num_samples", type=int, default=5)
    ap.add_argument("--decode", choices=["greedy", "beam"], default="beam")
    ap.add_argument("--tts", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    target_mode = cfg.get("targets", {}).get("mode", "phoneme")
    tok = build_tokenizer(cfg)

    files = load_split_files(cfg["paths"]["internal_root"], args.split)
    files = filter_split_files(files, cfg)
    if not files:
        raise RuntimeError("No samples found. Run scripts/download_data.py first.")
    print(f"Inference files: {len(files)} (split={args.split})")

    ds = NPZUtteranceDataset(files, cfg, tok, expect_phonemes=True, target_mode=target_mode)
    sample_dim = ds[0]["x"].shape[1]

    model = build_model(cfg, sample_dim, tok.vocab_size)
    ckpt_path = cfg.get("infer", {}).get("checkpoint", str(Path(cfg["paths"]["checkpoints"]) / "phoneme_ctc.pt"))
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"], strict=False)

    device = resolve_device(str(cfg["train"]["device"]))
    model.to(device).eval()
    lex = load_lexicon(cfg["paths"]["lexicon_path"])

    n = min(args.num_samples, len(ds))
    for i in range(n):
        item = ds[i]
        x = item["x"].unsqueeze(0).to(device)
        with torch.no_grad():
            lp = torch.log_softmax(model(x)[0], dim=-1).cpu()
        ids, conf = ctc_beam_decode(lp, tok.blank_id, int(cfg["decode"]["beam_size"])) if args.decode == "beam" else ctc_greedy_decode(lp, tok.blank_id)
        hyp_ph = tok.decode_ids(ids)

        if target_mode == "char" or cfg["decode"]["word_mode"] == "char_ctc":
            hyp_text = hyp_ph
            word_conf = conf
        else:
            hyp_text, word_conf = phonemes_to_words(hyp_ph.split(), lex, max_edit=int(cfg["decode"]["fuzzy_max_edit"]))

        print(f"[{i}] path={item['path']}")
        print(f"  ref_phonemes: {item['phonemes']}")
        print(f"  hyp_phonemes: {hyp_ph}")
        print(f"  text_ref: {item['text']}")
        print(f"  text_hyp: {hyp_text}")
        print(f"  conf_ph={conf:.3f} conf_word={word_conf:.3f}")

        if args.tts and hyp_text.strip():
            audio = elevenlabs_tts(hyp_text)
            if audio:
                out = Path("outputs") / f"sample_{i:03d}.mp3"
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(audio)
                print(f"  wrote_tts: {out}")


if __name__ == "__main__":
    main()
