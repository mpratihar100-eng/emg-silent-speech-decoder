from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np
import serial


class SerialEMGReader:
    """Reads lines formatted as: t,ch0,ch1,...,chN"""

    def __init__(self, port: str, baud_rate: int, channels: int, scale: float = 1.0, timeout: float = 0.1) -> None:
        self.port = port
        self.baud_rate = baud_rate
        self.channels = channels
        self.scale = scale
        self.timeout = timeout
        self._ser: serial.Serial | None = None

    def __enter__(self) -> "SerialEMGReader":
        self._ser = serial.Serial(self.port, self.baud_rate, timeout=self.timeout)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._ser is not None:
            self._ser.close()

    def __iter__(self) -> Iterator[Tuple[float, np.ndarray]]:
        if self._ser is None:
            raise RuntimeError("SerialEMGReader must be used as context manager")
        while True:
            line = self._ser.readline().decode("utf-8", errors="ignore").strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) != self.channels + 1:
                continue
            try:
                t = float(parts[0])
                vals = np.array([float(v) * self.scale for v in parts[1:]], dtype=np.float32)
            except ValueError:
                continue
            yield t, vals
