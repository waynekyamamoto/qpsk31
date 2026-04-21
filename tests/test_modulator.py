"""Tests for the QPSK31 modulator and demodulator."""

import math

import numpy as np
import pytest

from qpsk31.modulator import (
    DEFAULT_CARRIER,
    SAMPLE_RATE,
    SAMPLES_PER_SYMBOL,
    PHASE_DELTA,
    demodulate,
    modulate,
)


# ---------------------------------------------------------------------------
# Modulator output properties
# ---------------------------------------------------------------------------

def test_modulate_output_length():
    dibits = [(0, 0)] * 10
    samples = modulate(dibits)
    assert len(samples) == 10 * SAMPLES_PER_SYMBOL


def test_modulate_empty():
    assert len(modulate([])) == 0


def test_modulate_output_dtype():
    samples = modulate([(0, 0)] * 4)
    assert samples.dtype == np.float32


def test_modulate_amplitude_bounded():
    # With no pulse shaping, amplitude is cos() so |s| <= 1.0
    dibits = [(0, 0), (0, 1), (1, 1), (1, 0)] * 4
    samples = modulate(dibits)
    assert np.all(np.abs(samples) <= 1.01)  # small float tolerance


def test_modulate_not_all_zeros():
    samples = modulate([(0, 0)] * 4)
    assert not np.all(samples == 0)


# ---------------------------------------------------------------------------
# Carrier frequency
# ---------------------------------------------------------------------------

def test_carrier_present_at_correct_frequency():
    """FFT of a constant-phase signal should peak at carrier frequency."""
    n_syms = 32
    dibits = [(0, 0)] * n_syms  # no phase change -> pure tone at fc
    samples = modulate(dibits, carrier_freq=1000.0)

    freqs = np.fft.rfftfreq(len(samples), d=1 / SAMPLE_RATE)
    mag = np.abs(np.fft.rfft(samples))
    peak_freq = freqs[np.argmax(mag)]
    assert abs(peak_freq - 1000.0) < 10.0  # within 10 Hz


def test_carrier_at_alternate_frequency():
    n_syms = 32
    dibits = [(0, 0)] * n_syms
    fc = 1500.0
    samples = modulate(dibits, carrier_freq=fc)
    freqs = np.fft.rfftfreq(len(samples), d=1 / SAMPLE_RATE)
    mag = np.abs(np.fft.rfft(samples))
    peak_freq = freqs[np.argmax(mag)]
    assert abs(peak_freq - fc) < 10.0


# ---------------------------------------------------------------------------
# Phase mapping
# ---------------------------------------------------------------------------

def _measure_phase(samples: np.ndarray, carrier_freq: float, symbol_index: int) -> float:
    """Return the measured phase of a single symbol via I/Q correlation."""
    N = SAMPLES_PER_SYMBOL
    chunk = samples[symbol_index * N : (symbol_index + 1) * N].astype(np.float64)
    t = (np.arange(N, dtype=np.float64) + symbol_index * N) / SAMPLE_RATE
    I = (2.0 / N) * np.dot(chunk, np.cos(2 * math.pi * carrier_freq * t))
    Q = (2.0 / N) * np.dot(chunk, -np.sin(2 * math.pi * carrier_freq * t))
    return math.atan2(Q, I)


@pytest.mark.parametrize("dibit,expected_delta", [
    ((0, 0), 0.0),
    ((0, 1), math.pi / 2),
    ((1, 1), math.pi),
    ((1, 0), 3 * math.pi / 2),
])
def test_phase_delta_applied(dibit, expected_delta):
    """Modulate two symbols: idle then the test dibit; check phase change."""
    dibits = [(0, 0), dibit]
    samples = modulate(dibits)
    phi0 = _measure_phase(samples, DEFAULT_CARRIER, 0)
    phi1 = _measure_phase(samples, DEFAULT_CARRIER, 1)
    measured_delta = (phi1 - phi0) % (2 * math.pi)
    assert abs(measured_delta - expected_delta) < 0.01


# ---------------------------------------------------------------------------
# Demodulator roundtrip
# ---------------------------------------------------------------------------

def test_demodulate_output_count():
    n = 8
    dibits = [(0, 1)] * n
    samples = modulate(dibits)
    recovered = demodulate(samples)
    assert len(recovered) == n


def test_demodulate_roundtrip_constant_dibit():
    for dibit in [(0, 0), (0, 1), (1, 1), (1, 0)]:
        dibits = [dibit] * 16
        samples = modulate(dibits)
        recovered = demodulate(samples)
        assert recovered == dibits, f"Failed for dibit {dibit}"


def test_demodulate_roundtrip_sequence():
    dibits = [(0, 0), (0, 1), (1, 1), (1, 0), (0, 0), (1, 0), (0, 1), (1, 1)]
    samples = modulate(dibits)
    recovered = demodulate(samples)
    assert recovered == dibits


def test_demodulate_roundtrip_all_phase_combinations():
    pattern = [(0, 0), (0, 1), (1, 1), (1, 0)] * 8
    samples = modulate(pattern)
    recovered = demodulate(samples)
    assert recovered == pattern


def test_demodulate_roundtrip_at_alternate_carrier():
    fc = 800.0
    dibits = [(0, 1), (1, 0), (0, 0), (1, 1)] * 4
    samples = modulate(dibits, carrier_freq=fc)
    recovered = demodulate(samples, carrier_freq=fc)
    assert recovered == dibits


def test_demodulate_ignores_extra_samples():
    """Leftover samples (not a full symbol) should be silently discarded."""
    dibits = [(0, 0)] * 4
    samples = modulate(dibits)
    samples_extra = np.concatenate([samples, np.zeros(10, dtype=np.float32)])
    recovered = demodulate(samples_extra)
    assert recovered == dibits


def test_demodulate_empty():
    assert demodulate(np.array([], dtype=np.float32)) == []
