from __future__ import annotations

import argparse
from collections import Counter

import torch
from torch.utils.data import DataLoader

from emg_ssd.config import load_config
from emg_ssd.ctc_decode import ctc_beam_decode, ctc_greedy_decode
from emg_ssd.data.npz_dataset import NPZUtteranceDataset, ctc_collate, filter_split_files, load_split_files
from emg_ssd.metrics import phoneme_error_rate, word_error_rate
from emg_ssd.models.encoder_ctc import build_model
from emg_ssd.tokenizer import build_tokenizer
from emg_ssd.train_utils import resolve_device
from emg_ssd.wordify import load_lexicon, phonemes_to_words


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--decode", choices=["greedy", "beam"], default="beam")
    args = ap.parse_args()

    cfg = load_config(args.config)
    target_mode = cfg.get("targets", {}).get("mode", "phoneme")
    tok = build_tokenizer(cfg)
    split = cfg.get("eval", {}).get("split", "test")
    files = load_split_files(cfg["paths"]["internal_root"], split)
    files = filter_split_files(files, cfg)
    if not files:
        raise RuntimeError(f"No files in split={split}")
    print(f"Evaluating files: {len(files)} (split={split})")

    ds = NPZUtteranceDataset(files, cfg, tok, expect_phonemes=True, target_mode=target_mode)
    dl = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=ctc_collate)

    sample_dim = ds[0]["x"].shape[1]
    model = build_model(cfg, sample_dim, tok.vocab_size)
    ckpt = torch.load(cfg["eval"]["checkpoint"], map_location="cpu")
    model.load_state_dict(ckpt["model_state"], strict=False)
    device = resolve_device(str(cfg["train"]["device"]))
    model.to(device)
    model.eval()

    lex = load_lexicon(cfg["paths"]["lexicon_path"])

    pers = []
    wers = []
    conf_mat = Counter()

    with torch.no_grad():
        for b in dl:
            x = b["x"].to(device)
            logits = model(x)[0]
            lp = torch.log_softmax(logits, dim=-1).cpu()
            ids, _ = ctc_beam_decode(lp, tok.blank_id, int(cfg["decode"]["beam_size"])) if args.decode == "beam" else ctc_greedy_decode(lp, tok.blank_id)
            hyp = tok.decode_ids(ids)
            ref_text = b["raw"][0]["text"]
            if target_mode == "char":
                if ref_text:
                    wers.append(word_error_rate(ref_text, hyp))
            else:
                ref = b["raw"][0]["phonemes"]
                per = phoneme_error_rate(ref, hyp)
                pers.append(per)
                hyp_words, _ = phonemes_to_words(hyp.split(), lex, max_edit=int(cfg["decode"]["fuzzy_max_edit"]))
                if ref_text:
                    wers.append(word_error_rate(ref_text, hyp_words))
                ref_toks = ref.split()
                hyp_toks = hyp.split()
                for r, h in zip(ref_toks, hyp_toks):
                    conf_mat[(r, h)] += 1

    if pers:
        print(f"split={split} PER={sum(pers)/max(1,len(pers)):.4f}")
    if wers:
        print(f"split={split} WER={sum(wers)/len(wers):.4f}")
    print("Top confusion pairs:")
    for (r, h), c in conf_mat.most_common(20):
        if r != h:
            print(f"  {r:>4s} -> {h:<4s}: {c}")


if __name__ == "__main__":
    main()
