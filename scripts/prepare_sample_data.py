from __future__ import annotations

import argparse
from pathlib import Path

from emg_ssd.config import ensure_dirs, load_config
from emg_ssd.data.sample_import import import_sample_data, write_manifest_template


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--raw_dir", default="data/sample_raw")
    ap.add_argument("--manifest", default="data/sample_raw/manifest.csv")
    ap.add_argument("--prefix", default="sample")
    ap.add_argument("--default_split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--create_template", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)

    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest = Path(args.manifest)
    if args.create_template and not manifest.exists():
        write_manifest_template(manifest)
        print(f"Template manifest written: {manifest}")
        print("Place your sample files in the same raw_dir and rerun without --create_template")
        return

    n = import_sample_data(
        raw_dir=raw_dir,
        internal_root=cfg["paths"]["internal_root"],
        expected_channels=int(cfg["data"]["expected_channels"]),
        target_sr=int(cfg["data"]["target_sr"]),
        manifest_path=manifest if manifest.exists() else None,
        prefix=args.prefix,
        default_split=args.default_split,
    )
    print(f"Imported {n} sample utterances into {cfg['paths']['internal_root']}")


if __name__ == "__main__":
    main()
