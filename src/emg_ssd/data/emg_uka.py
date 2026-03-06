from __future__ import annotations

from pathlib import Path

from .voicing_silent_speech import convert_generic_arrays_to_internal


def convert_emg_uka_trial_to_internal(source_dir: str | Path, out_dir: str | Path, expected_channels: int = 8, sr: int = 1000) -> int:
    return convert_generic_arrays_to_internal(source_dir, out_dir, expected_channels=expected_channels, sr=sr)
