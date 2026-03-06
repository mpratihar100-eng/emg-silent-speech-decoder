from pathlib import Path

import torch
from torch.utils.data import DataLoader

from emg_ssd.config import load_config
from emg_ssd.data.synthetic import generate_synthetic_internal
from emg_ssd.data.npz_dataset import NPZUtteranceDataset, ctc_collate, load_split_files
from emg_ssd.models.encoder_ctc import build_model
from emg_ssd.tokenizer import Tokenizer


def test_tiny_forward_pass(tmp_path: Path) -> None:
    cfg = load_config("configs/base.yaml")
    cfg["paths"]["internal_root"] = str(tmp_path / "internal")
    cfg["paths"]["checkpoints"] = str(tmp_path / "ckpt")
    generate_synthetic_internal(cfg["paths"]["internal_root"], n_train=4, n_val=1, n_test=1)

    tok = Tokenizer(cfg["tokens"]["vocab_path"], add_blank=True)
    files = load_split_files(cfg["paths"]["internal_root"], "train")
    ds = NPZUtteranceDataset(files, cfg, tok)
    dl = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=ctc_collate)
    batch = next(iter(dl))

    input_dim = ds[0]["x"].shape[1]
    model = build_model(cfg, input_dim, tok.vocab_size)
    logits = model(batch["x"])
    assert logits.shape[0] == batch["x"].shape[0]
    assert logits.shape[1] == batch["x"].shape[1]
    assert logits.shape[2] == tok.vocab_size
