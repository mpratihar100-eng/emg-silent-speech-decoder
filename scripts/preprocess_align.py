from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from emg_ssd.alignment import build_text_prototypes, iter_filtered_split_files, save_preprocessed_npz, score_against_prototypes
from emg_ssd.config import ensure_dirs, load_config
from emg_ssd.preprocessing import PreprocessConfig, preprocess_emg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--prefix", default="prep")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)

    pp_cfg = PreprocessConfig.from_cfg(cfg)
    template_split = str(cfg.get("alignment", {}).get("template_split", "train"))
    downsample = int(cfg.get("alignment", {}).get("dtw_downsample", 4))

    template_files = iter_filtered_split_files(cfg, template_split)
    if not template_files:
        raise RuntimeError(f"No template files found in split={template_split}")
    prototypes = build_text_prototypes(template_files, cfg)
    if not prototypes:
        raise RuntimeError("No text prototypes could be built from the filtered dataset.")

    manifest_path = Path(cfg["paths"]["manifests_root"]) / str(cfg.get("alignment", {}).get("manifest_name", "preprocessed_manifest.csv"))
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    total = 0
    for split in args.splits:
        files = iter_filtered_split_files(cfg, split)
        out_split = Path(cfg["paths"]["preprocessed_root"]) / split
        out_split.mkdir(parents=True, exist_ok=True)
        for i, f in enumerate(files):
            with np.load(f, allow_pickle=True) as d:
                emg = d["emg"].astype(np.float32)
                sr = int(np.array(d["sr"]).item())
                text = str(d["text"].item()) if "text" in d.files else ""
                phonemes = str(d["phonemes"].item()) if "phonemes" in d.files else ""
                speaker_id = str(d["speaker_id"].item()) if "speaker_id" in d.files else ""
                session_id = str(d["session_id"].item()) if "session_id" in d.files else ""
            proc = preprocess_emg(emg, sr, pp_cfg)
            dtw_text, dtw_dist = score_against_prototypes(proc, prototypes, downsample)
            out_file = out_split / f"{args.prefix}_{split}_{i:05d}.npz"
            save_preprocessed_npz(out_file, f, proc, pp_cfg.target_sr, text, phonemes, speaker_id, session_id)
            rows.append(
                {
                    "split": split,
                    "source_path": str(f),
                    "preprocessed_path": str(out_file),
                    "text": text,
                    "phonemes": phonemes,
                    "dtw_best_text": dtw_text,
                    "dtw_distance": f"{dtw_dist:.6f}",
                    "speaker_id": speaker_id,
                    "session_id": session_id,
                }
            )
            total += 1

    with manifest_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=["split", "source_path", "preprocessed_path", "text", "phonemes", "dtw_best_text", "dtw_distance", "speaker_id", "session_id"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Built {len(prototypes)} DTW prototypes from split={template_split}")
    print(f"Preprocessed {total} utterances into {cfg['paths']['preprocessed_root']}")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
