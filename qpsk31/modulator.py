"""QPSK31 modulator and demodulator.

Signal parameters:
  Sample rate:        8000 Hz
  Symbol rate:        31.25 baud  (= 8000 / 256)
  Samples/symbol:     256
  Default carrier:    1000 Hz  (= 32 * symbol_rate, so exactly 32 carrier cycles
                                 per symbol — ensures orthogonality of I/Q correlators)

Differential Grey-coded QPSK phase map (dibit -> phase delta in radians):
  (0,0) ->   0
  (0,1) ->  π/2
  (1,1) ->  π
  (1,0) -> 3π/2
"""

import math
from typing import List, Tuple

import numpy as np

SAMPLE_RATE: int = 8000
SAMPLES_PER_SYMBOL: int = 256
DEFAULT_CARRIER: float = 1000.0

_PI2 = 2 * math.pi

# Dibit -> phase delta (radians)
PHASE_DELTA: dict[Tuple[int, int], float] = {
    (0, 0): 0.0,
    (0, 1): math.pi / 2,
    (1, 1): math.pi,
    (1, 0): 3 * math.pi / 2,
}

# Phase delta index -> dibit (index = round(delta / (pi/2)) mod 4)
_IDX_TO_DIBIT: List[Tuple[int, int]] = [(0, 0), (0, 1), (1, 1), (1, 0)]


def modulate(
    dibits: List[Tuple[int, int]],
    carrier_freq: float = DEFAULT_CARRIER,
    sample_rate: int = SAMPLE_RATE,
    samples_per_symbol: int = SAMPLES_PER_SYMBOL,
) -> np.ndarray:
    """Convert a list of (g0,g1) dibits to a float32 audio sample array.

    Uses differential encoding: phase accumulates over the symbol stream.
    No pulse shaping is applied (rectangular window); the signal is strictly
    coherent across symbol boundaries.
    """
    N = samples_per_symbol
    n_syms = len(dibits)
    out = np.empty(n_syms * N, dtype=np.float32)

    phase = 0.0
    for i, dibit in enumerate(dibits):
        phase += PHASE_DELTA[dibit]
        start = i * N
        # Absolute sample indices for phase-coherent synthesis
        t = (np.arange(N, dtype=np.float64) + start) / sample_rate
        out[start : start + N] = np.cos(_PI2 * carrier_freq * t + phase).astype(
            np.float32
        )

    return out


def demodulate(
    samples: np.ndarray,
    carrier_freq: float = DEFAULT_CARRIER,
    sample_rate: int = SAMPLE_RATE,
    samples_per_symbol: int = SAMPLES_PER_SYMBOL,
) -> List[Tuple[int, int]]:
    """Recover the list of (g0,g1) dibits from audio samples.

    Uses I/Q correlation per symbol to estimate instantaneous phase, then
    differential decoding to map phase changes to dibits.
    """
    N = samples_per_symbol
    n_syms = len(samples) // N
    dibits: List[Tuple[int, int]] = []
    prev_phase = 0.0
    scale = 2.0 / N

    for i in range(n_syms):
        chunk = samples[i * N : (i + 1) * N].astype(np.float64)
        t = (np.arange(N, dtype=np.float64) + i * N) / sample_rate

        ref_cos = np.cos(_PI2 * carrier_freq * t)
        ref_sin = np.sin(_PI2 * carrier_freq * t)

        I = scale * np.dot(chunk, ref_cos)
        Q = scale * np.dot(chunk, -ref_sin)  # -sin gives Q = sin(phase)

        current_phase = math.atan2(Q, I)

        # Differential decode: unwrap phase delta to [0, 2pi)
        delta = (current_phase - prev_phase) % _PI2

        # Map to nearest of {0, pi/2, pi, 3pi/2}
        idx = round(delta / (math.pi / 2)) % 4
        dibits.append(_IDX_TO_DIBIT[idx])
        prev_phase = current_phase

    return dibits
