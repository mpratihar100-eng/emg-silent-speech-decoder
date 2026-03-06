from __future__ import annotations

import torch
from torch import nn


class ConvFrontend(nn.Module):
    def __init__(self, in_dim: int, conv_channels: list[int], kernel_size: int, dropout: float) -> None:
        super().__init__()
        layers = []
        d = in_dim
        for c in conv_channels:
            layers += [
                nn.Conv1d(d, c, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(c),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            d = c
        self.net = nn.Sequential(*layers)
        self.out_dim = d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.net(x)
        return x.transpose(1, 2)


class CNNBiLSTMCTC(nn.Module):
    def __init__(
        self,
        input_dim: int,
        vocab_size: int,
        conv_channels: list[int],
        conv_kernel: int,
        lstm_hidden: int,
        lstm_layers: int,
        dropout: float,
        adapter_dim: int = 64,
    ) -> None:
        super().__init__()
        self.front = ConvFrontend(input_dim, conv_channels, conv_kernel, dropout)
        self.rnn = nn.LSTM(
            input_size=self.front.out_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        self.adapter = nn.Sequential(nn.Linear(lstm_hidden * 2, adapter_dim), nn.ReLU(), nn.Linear(adapter_dim, lstm_hidden * 2))
        self.classifier = nn.Linear(lstm_hidden * 2, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.front(x)
        z, _ = self.rnn(z)
        z = z + self.adapter(z)
        return self.classifier(z)


class TransformerCTC(nn.Module):
    def __init__(
        self,
        input_dim: int,
        vocab_size: int,
        d_model: int = 128,
        nhead: int = 4,
        layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.classifier = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.proj(x)
        z = self.encoder(z)
        return self.classifier(z)


def build_model(cfg: dict, input_dim: int, vocab_size: int) -> nn.Module:
    m = cfg["model"]
    if m["arch"] == "transformer_ctc":
        return TransformerCTC(
            input_dim=input_dim,
            vocab_size=vocab_size,
            d_model=int(m["transformer_d_model"]),
            nhead=int(m["transformer_heads"]),
            layers=int(m["transformer_layers"]),
            dropout=float(m["dropout"]),
        )
    return CNNBiLSTMCTC(
        input_dim=input_dim,
        vocab_size=vocab_size,
        conv_channels=list(m["conv_channels"]),
        conv_kernel=int(m["conv_kernel"]),
        lstm_hidden=int(m["lstm_hidden"]),
        lstm_layers=int(m["lstm_layers"]),
        dropout=float(m["dropout"]),
        adapter_dim=int(m["adapter_dim"]),
    )
