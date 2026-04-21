"""BPSK31 modem — no FEC, for comparison with QPSK31.

Identical symbol rate and carrier to QPSK31 but uses only two phases
(0° and 180°) and skips the convolutional encoder/Viterbi decoder.
Each varicode bit maps directly to a differential phase shift:
  0 → no change   1 → +180°
"""

import math
from typing import List

import numpy as np

from .modulator import DEFAULT_CARRIER, SAMPLE_RATE, SAMPLES_PER_SYMBOL
from .varicode import encode as vc_encode, decode as vc_decode

_PI2 = 2 * math.pi
_PRE = 32
_TAIL = 4


def encode(text: str, carrier_freq: float = DEFAULT_CARRIER) -> np.ndarray:
    bits = [0] * _PRE + vc_encode(text) + [0] * _TAIL
    return _modulate(bits, carrier_freq)


def decode(samples: np.ndarray, carrier_freq: float = DEFAULT_CARRIER) -> str:
    bits = _demodulate(samples, carrier_freq)
    return vc_decode(bits)


def _modulate(bits: List[int], carrier_freq: float) -> np.ndarray:
    N = SAMPLES_PER_SYMBOL
    out = np.empty(len(bits) * N, dtype=np.float32)
    phase = 0.0
    for i, b in enumerate(bits):
        if b == 1:
            phase += math.pi
        start = i * N
        t = (np.arange(N, dtype=np.float64) + start) / SAMPLE_RATE
        out[start : start + N] = np.cos(_PI2 * carrier_freq * t + phase).astype(
            np.float32
        )
    return out


def _demodulate(samples: np.ndarray, carrier_freq: float) -> List[int]:
    N = SAMPLES_PER_SYMBOL
    n_syms = len(samples) // N
    bits: List[int] = []
    prev_phase = 0.0
    scale = 2.0 / N
    for i in range(n_syms):
        start = i * N
        chunk = samples[start : start + N].astype(np.float64)
        t = (np.arange(N, dtype=np.float64) + start) / SAMPLE_RATE
        I = scale * np.dot(chunk, np.cos(_PI2 * carrier_freq * t))
        Q = scale * np.dot(chunk, -np.sin(_PI2 * carrier_freq * t))
        curr_phase = math.atan2(Q, I)
        delta = (curr_phase - prev_phase) % _PI2
        # 180° phase shift = bit 1, no shift = bit 0
        bits.append(1 if abs(delta - math.pi) < math.pi / 2 else 0)
        prev_phase = curr_phase
    return bits
