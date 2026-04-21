"""High-level QPSK31 encode/decode interface."""

import numpy as np

from . import varicode as vc
from . import convcode as cc
from .modulator import (
    DEFAULT_CARRIER,
    SAMPLE_RATE,
    SAMPLES_PER_SYMBOL,
    demodulate,
    modulate,
)

# Tail bits appended to flush the convolutional shift register (K-1 = 4 zeros)
# plus extra idle zeros for preamble / postamble synchronisation.
_PREAMBLE_ZEROS = 32
_TAIL_ZEROS = cc._MEM  # 4


def encode(
    text: str,
    carrier_freq: float = DEFAULT_CARRIER,
    sample_rate: int = SAMPLE_RATE,
    samples_per_symbol: int = SAMPLES_PER_SYMBOL,
) -> np.ndarray:
    """Encode text to a QPSK31 audio sample array.

    Pipeline:
      text -> varicode bits (with 00 separators)
           -> prepend preamble zeros + append tail zeros
           -> rate-1/2 convolutional encoding -> dibits
           -> differential QPSK modulation -> audio samples
    """
    bits = [0] * _PREAMBLE_ZEROS + vc.encode(text) + [0] * _TAIL_ZEROS
    dibits = cc.encode(bits)
    return modulate(dibits, carrier_freq, sample_rate, samples_per_symbol)


def decode(
    samples: np.ndarray,
    carrier_freq: float = DEFAULT_CARRIER,
    sample_rate: int = SAMPLE_RATE,
    samples_per_symbol: int = SAMPLES_PER_SYMBOL,
) -> str:
    """Decode a QPSK31 audio sample array back to text.

    Pipeline:
      audio samples -> I/Q demodulation -> dibits
                    -> Viterbi decode -> bits
                    -> varicode decode -> text
    """
    dibits = demodulate(samples, carrier_freq, sample_rate, samples_per_symbol)
    bits = cc.decode(dibits)
    return vc.decode(bits)
