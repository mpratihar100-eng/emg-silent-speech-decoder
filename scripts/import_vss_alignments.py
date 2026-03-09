from __future__ import annotations

import argparse
from pathlib import Path

from emg_ssd.alignment_labels import patch_internal_labels, scan_alignment_dir
from emg_ssd.config import ensure_dirs, load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--alignment_dir", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    internal_root = Path(cfg["paths"]["internal_root"])

    labels = scan_alignment_dir(args.alignment_dir)
    if not labels:
        raise RuntimeError(f"No alignment labels found in {args.alignment_dir}")
    patched = patch_internal_labels(internal_root, labels)
    print(f"Patched {patched} utterances in {internal_root} using labels from {args.alignment_dir}")


if __name__ == "__main__":
    main()
