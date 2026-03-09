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
    ap.add_argument("--max_files", type=int, default=0, help="Optional cap per split for fast local verification.")
    ap.add_argument("--skip_dtw_scores", action="store_true", help="Skip DTW scoring and only preserve preprocessing + labels.")
    ap.add_argument("--log_every", type=int, default=100, help="Print progress every N files.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)

    pp_cfg = PreprocessConfig.from_cfg(cfg)
    template_split = str(cfg.get("alignment", {}).get("template_split", "train"))
    downsample = int(cfg.get("alignment", {}).get("dtw_downsample", 4))

    prototypes: dict[str, np.ndarray] = {}
    if not args.skip_dtw_scores:
        template_files = iter_filtered_split_files(cfg, template_split)
        if args.max_files > 0:
            template_files = template_files[: args.max_files]
        if not template_files:
            raise RuntimeError(f"No template files found in split={template_split}")
        print(f"Building DTW prototypes from {len(template_files)} files in split={template_split}")
        prototypes = build_text_prototypes(template_files, cfg)
        if not prototypes:
            raise RuntimeError("No text prototypes could be built from the filtered dataset.")
        print(f"Built {len(prototypes)} DTW prototypes from split={template_split}")
    else:
        print("Skipping DTW scoring; preprocessing-only smoke mode enabled")

    manifest_path = Path(cfg["paths"]["manifests_root"]) / str(cfg.get("alignment", {}).get("manifest_name", "preprocessed_manifest.csv"))
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    total = 0
    for split in args.splits:
        files = iter_filtered_split_files(cfg, split)
        if args.max_files > 0:
            files = files[: args.max_files]
        if not files:
            print(f"split={split}: no files matched after filtering")
            continue
        out_split = Path(cfg["paths"]["preprocessed_root"]) / split
        out_split.mkdir(parents=True, exist_ok=True)
        print(f"split={split}: preprocessing {len(files)} files into {out_split}")
        for i, f in enumerate(files):
            with np.load(f, allow_pickle=True) as d:
                emg = d["emg"].astype(np.float32)
                sr = int(np.array(d["sr"]).item())
                text = str(d["text"].item()) if "text" in d.files else ""
                phonemes = str(d["phonemes"].item()) if "phonemes" in d.files else ""
                speaker_id = str(d["speaker_id"].item()) if "speaker_id" in d.files else ""
                session_id = str(d["session_id"].item()) if "session_id" in d.files else ""
                utterance_id = str(d["utterance_id"].item()) if "utterance_id" in d.files else ""
                source_file = str(d["source_file"].item()) if "source_file" in d.files else ""
                alignment_path = str(d["alignment_path"].item()) if "alignment_path" in d.files else ""
            proc = preprocess_emg(emg, sr, pp_cfg)
            if args.skip_dtw_scores:
                dtw_text, dtw_dist = "", 0.0
            else:
                dtw_text, dtw_dist = score_against_prototypes(proc, prototypes, downsample)
            out_file = out_split / f"{args.prefix}_{split}_{i:05d}.npz"
            save_preprocessed_npz(
                out_file,
                f,
                proc,
                pp_cfg.target_sr,
                text,
                phonemes,
                speaker_id,
                session_id,
                utterance_id=utterance_id,
                source_file=source_file,
                alignment_path=alignment_path,
            )
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
                    "utterance_id": utterance_id,
                    "alignment_path": alignment_path,
                }
            )
            total += 1
            if args.log_every > 0 and ((i + 1) % args.log_every == 0 or (i + 1) == len(files)):
                print(f"split={split}: wrote {i + 1}/{len(files)} files")

    with manifest_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "split",
                "source_path",
                "preprocessed_path",
                "text",
                "phonemes",
                "dtw_best_text",
                "dtw_distance",
                "speaker_id",
                "session_id",
                "utterance_id",
                "alignment_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Preprocessed {total} utterances into {cfg['paths']['preprocessed_root']}")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
