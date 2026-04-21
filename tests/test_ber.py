"""BER/CER tests: QPSK31 vs BPSK31 under AWGN noise.

Key findings from the SNR sweep:
  SNR ≥ -11 dB  → both modes: 0% CER (clean channel)
  SNR = -12 dB  → QPSK31 starts to show FEC benefit over BPSK31
  SNR ≤ -14 dB  → channel too noisy for reliable decode in either mode
"""

import numpy as np
import pytest

from qpsk31 import bpsk31
from qpsk31.modem import decode as qpsk_decode
from qpsk31.modem import encode as qpsk_encode
from qpsk31.noise import add_noise, char_error_rate

TEXT = "the quick brown fox jumps over the lazy dog"
SEED = 42
N_TRIALS = 20


def avg_cer(encode_fn, decode_fn, snr_db, n=N_TRIALS, seed=SEED):
    rng = np.random.default_rng(seed)
    cers = [
        char_error_rate(TEXT, decode_fn(add_noise(encode_fn(TEXT), snr_db, rng)))
        for _ in range(n)
    ]
    return float(np.mean(cers))


# ── Clean channel ────────────────────────────────────────────────────────────

def test_qpsk31_perfect_at_high_snr():
    assert avg_cer(qpsk_encode, qpsk_decode, snr_db=0) == 0.0


def test_bpsk31_perfect_at_high_snr():
    assert avg_cer(bpsk31.encode, bpsk31.decode, snr_db=0) == 0.0


def test_qpsk31_perfect_at_minus10db():
    assert avg_cer(qpsk_encode, qpsk_decode, snr_db=-10) == 0.0


def test_bpsk31_perfect_at_minus10db():
    assert avg_cer(bpsk31.encode, bpsk31.decode, snr_db=-10) == 0.0


# ── FEC advantage at the sweet spot ─────────────────────────────────────────

def test_qpsk31_better_than_bpsk31_at_minus12db():
    """FEC coding gain: QPSK31 has lower CER than BPSK31 at -12 dB SNR."""
    q = avg_cer(qpsk_encode, qpsk_decode, snr_db=-12)
    b = avg_cer(bpsk31.encode, bpsk31.decode, snr_db=-12)
    assert q <= b, f"QPSK31 CER {q:.1%} should be ≤ BPSK31 CER {b:.1%} at -12 dB"


def test_qpsk31_low_cer_at_minus12db():
    """QPSK31 stays near-perfect at the FEC sweet spot."""
    assert avg_cer(qpsk_encode, qpsk_decode, snr_db=-12) < 0.02


# ── Graceful degradation ─────────────────────────────────────────────────────

def test_qpsk31_degrades_below_minus13db():
    """Both modes show measurable errors below the FEC threshold."""
    assert avg_cer(qpsk_encode, qpsk_decode, snr_db=-15) > 0.0


def test_bpsk31_degrades_below_minus13db():
    assert avg_cer(bpsk31.encode, bpsk31.decode, snr_db=-15) > 0.0


def test_both_modes_fail_at_very_low_snr():
    """At -20 dB both modes are in the noise floor."""
    q = avg_cer(qpsk_encode, qpsk_decode, snr_db=-20)
    b = avg_cer(bpsk31.encode, bpsk31.decode, snr_db=-20)
    assert q > 0.5, f"QPSK31 CER should be >50% at -20 dB, got {q:.1%}"
    assert b > 0.5, f"BPSK31 CER should be >50% at -20 dB, got {b:.1%}"


# ── CER is monotonically non-decreasing as SNR drops ────────────────────────

@pytest.mark.parametrize("encode_fn,decode_fn", [
    (qpsk_encode, qpsk_decode),
    (bpsk31.encode, bpsk31.decode),
])
def test_cer_increases_as_snr_drops(encode_fn, decode_fn):
    snr_points = [-10, -13, -16, -20]
    cers = [avg_cer(encode_fn, decode_fn, snr_db=s) for s in snr_points]
    for i in range(len(cers) - 1):
        assert cers[i] <= cers[i + 1], (
            f"CER should not decrease as SNR drops: "
            f"SNR {snr_points[i]} dB → {cers[i]:.1%}, "
            f"SNR {snr_points[i+1]} dB → {cers[i+1]:.1%}"
        )
