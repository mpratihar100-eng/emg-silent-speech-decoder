from __future__ import annotations

import argparse
import shutil
import tarfile
import zipfile
from pathlib import Path

import requests

from emg_ssd.config import ensure_dirs, load_config
from emg_ssd.data.emg_uka import convert_emg_uka_trial_to_internal
from emg_ssd.data.voicing_silent_speech import convert_generic_arrays_to_internal, generate_synthetic_internal


def _download_zenodo_record(record_id: str, out_dir: Path) -> Path | None:
    out_dir.mkdir(parents=True, exist_ok=True)
    api = f"https://zenodo.org/api/records/{record_id}"
    try:
        meta = requests.get(api, timeout=60)
        meta.raise_for_status()
    except Exception:
        return None
    files = meta.json().get("files", [])
    if not files:
        return None
    preferred = None
    for f in files:
        key = str(f.get("key", "")).lower()
        if key.endswith(".zip") or key.endswith(".tar.gz") or key.endswith(".tgz"):
            preferred = f
            break
    if preferred is None:
        preferred = files[0]
    url = preferred.get("links", {}).get("self")
    key = preferred.get("key", "download.bin")
    if not url:
        return None
    out_path = out_dir / key
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with out_path.open("wb") as fp:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    fp.write(chunk)
    return out_path


def _extract_archive(path: Path, out_dir: Path) -> Path:
    extract_dir = out_dir / "extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".zip":
        with zipfile.ZipFile(path, "r") as zf:
            zf.extractall(extract_dir)
    elif path.suffix in {".gz", ".tgz"} or path.name.endswith(".tar.gz"):
        with tarfile.open(path, "r:gz") as tf:
            tf.extractall(extract_dir)
    else:
        dst = extract_dir / path.name
        shutil.copy(path, dst)
    return extract_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--source", default="zenodo_vss", choices=["zenodo_vss", "emg_uka_kaggle", "synthetic"])
    ap.add_argument("--record_id", default="4064408")
    ap.add_argument("--source_dir", default="")
    ap.add_argument("--archive", default="")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)

    data_root = Path(cfg["paths"]["data_root"])
    internal = Path(cfg["paths"]["internal_root"])
    internal.mkdir(parents=True, exist_ok=True)

    n = 0
    if args.source == "synthetic":
        n = generate_synthetic_internal(
            internal,
            sr=int(cfg["data"]["target_sr"]),
            channels=int(cfg["data"]["expected_channels"]),
        )
    elif args.source == "emg_uka_kaggle":
        source_dir = Path(args.source_dir) if args.source_dir else data_root / "emg_uka_raw"
        if source_dir.exists():
            n = convert_emg_uka_trial_to_internal(
                source_dir,
                internal,
                expected_channels=int(cfg["data"]["expected_channels"]),
                sr=int(cfg["data"]["target_sr"]),
            )
        if n == 0:
            n = generate_synthetic_internal(internal)
    else:
        raw_dir = data_root / "voicing_silent_speech_raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        direct_source = Path(args.source_dir) if args.source_dir else Path()
        extracted = None
        if direct_source and direct_source.exists():
            extracted = direct_source
        else:
            pre_extracted = raw_dir / "extracted"
            if pre_extracted.exists():
                extracted = pre_extracted
            else:
                archive = Path(args.archive) if args.archive else _download_zenodo_record(args.record_id, raw_dir)
                if archive is not None:
                    extracted = _extract_archive(archive, raw_dir)
        if extracted is not None:
            n = convert_generic_arrays_to_internal(
                extracted,
                internal,
                expected_channels=int(cfg["data"]["expected_channels"]),
                sr=int(cfg["data"]["target_sr"]),
                allowed_raw_roots=list(cfg.get("data", {}).get("allowed_raw_roots", [])),
            )
        if n == 0:
            n = generate_synthetic_internal(
                internal,
                sr=int(cfg["data"]["target_sr"]),
                channels=int(cfg["data"]["expected_channels"]),
            )

    print(f"Prepared {n} utterances in internal format at: {internal}")


if __name__ == "__main__":
    main()
