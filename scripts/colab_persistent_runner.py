from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], cwd: Path) -> None:
    print("RUN", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(cwd))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive_root", required=True)
    ap.add_argument("--repo_dir", default="/content/emg-silent-speech-decoder")
    ap.add_argument("--max_steps", type=int, default=200)
    ap.add_argument("--num_blocks", type=int, default=10)
    ap.add_argument("--record_id", default="4064408")
    ap.add_argument("--preprocess_config", default="configs/base.yaml")
    args = ap.parse_args()

    drive_root = Path(args.drive_root)
    repo_dir = Path(args.repo_dir)
    artifact_dir = drive_root / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    _run([sys.executable, "scripts/download_data.py", "--config", args.preprocess_config, "--source", "zenodo_vss", "--record_id", args.record_id], repo_dir)
    _run([sys.executable, "scripts/validate_corpus.py", "--config", args.preprocess_config], repo_dir)
    _run([sys.executable, "scripts/preprocess_align.py", "--config", args.preprocess_config], repo_dir)

    for block in range(args.num_blocks):
        _run(
            [
                sys.executable,
                "-u",
                "scripts/train_phoneme_ctc.py",
                "--config",
                "configs/train_preprocessed.yaml",
                "--max_steps",
                str(args.max_steps),
                "--resume",
                "--save_every_steps",
                "25",
                "--log_every_steps",
                "10",
                "--metrics_file",
                "outputs/train_metrics.jsonl",
            ],
            repo_dir,
        )
        ckpt = repo_dir / "checkpoints" / "phoneme_ctc.pt"
        metrics = repo_dir / "outputs" / "train_metrics.jsonl"
        if ckpt.exists():
            shutil.copy2(ckpt, artifact_dir / ckpt.name)
        if metrics.exists():
            shutil.copy2(metrics, artifact_dir / metrics.name)
        print(f"Finished block {block + 1}/{args.num_blocks}")

    _run([sys.executable, "scripts/eval_phoneme_ctc.py", "--config", "configs/eval_preprocessed.yaml", "--decode", "beam"], repo_dir)


if __name__ == "__main__":
    main()
