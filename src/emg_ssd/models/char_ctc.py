from __future__ import annotations

from torch import nn

from .encoder_ctc import CNNBiLSTMCTC


class CharCTCModel(CNNBiLSTMCTC):
    """Fallback direct EMG->character CTC model sharing the same backbone."""

    pass


def build_char_model(input_dim: int, vocab_size: int) -> nn.Module:
    return CharCTCModel(
        input_dim=input_dim,
        vocab_size=vocab_size,
        conv_channels=[32, 64],
        conv_kernel=5,
        lstm_hidden=128,
        lstm_layers=2,
        dropout=0.1,
        adapter_dim=64,
    )
