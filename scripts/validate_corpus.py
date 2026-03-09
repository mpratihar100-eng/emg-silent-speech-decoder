from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from emg_ssd.alignment_labels import iter_label_issues
from emg_ssd.config import load_config
from emg_ssd.data.npz_dataset import filter_split_files, load_split_files


def _load_vocab(path: str | Path) -> set[str]:
    return {
        line.strip().upper()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--root", default="")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--require_alignment", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    root = Path(args.root) if args.root else Path(cfg["paths"]["internal_root"])
    vocab = _load_vocab(cfg["tokens"]["vocab_path"])

    bad = []
    total = 0
    missing_alignment = 0
    for split in args.splits:
        files = filter_split_files(load_split_files(root, split), cfg)
        total += len(files)
        bad.extend(iter_label_issues(files, vocab))
        if args.require_alignment or bool(cfg.get("alignment", {}).get("require_alignment", False)):
            for p in files:
                try:
                    with np.load(p, allow_pickle=True) as d:
                        alignment_path = str(d["alignment_path"].item()) if "alignment_path" in d.files else ""
                    if not alignment_path.strip():
                        missing_alignment += 1
                except Exception:
                    missing_alignment += 1

    print(f"Validated {total} filtered files in {root}")
    if total == 0:
        print("No filtered files found")
        raise SystemExit(1)
    if bad:
        for issue in bad[:50]:
            print(issue)
        raise SystemExit(1)
    if missing_alignment > 0:
        print(f"Missing alignment_path on {missing_alignment} files")
        raise SystemExit(1)
    print("Corpus validation passed")


if __name__ == "__main__":
    main()
