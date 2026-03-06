import importlib


def test_imports() -> None:
    mods = [
        "emg_ssd.config",
        "emg_ssd.preprocessing",
        "emg_ssd.features",
        "emg_ssd.tokenizer",
        "emg_ssd.ctc_decode",
        "emg_ssd.wordify",
        "emg_ssd.metrics",
        "emg_ssd.models.encoder_ctc",
        "emg_ssd.data.sample_import",
        "emg_ssd.visualize",
    ]
    for m in mods:
        importlib.import_module(m)
