"""Integration tests for the full QPSK31 encode/decode pipeline."""

import numpy as np
import pytest

from qpsk31.modem import encode, decode


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------

def test_encode_returns_ndarray():
    samples = encode("hi")
    assert isinstance(samples, np.ndarray)


def test_encode_returns_float32():
    assert encode("e").dtype == np.float32


def test_encode_output_nonempty():
    assert len(encode("e")) > 0


def test_decode_returns_str():
    assert isinstance(decode(encode("hi")), str)


# ---------------------------------------------------------------------------
# Full roundtrip
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "e",
    "hi",
    "hello",
    "Hello",
    "hello world",
    "Hello, World!",
    "the quick brown fox jumps over the lazy dog",
    "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG",
    "0123456789",
    "CQ CQ CQ DE W1AW",
])
def test_roundtrip(text):
    assert decode(encode(text)) == text


def test_roundtrip_single_space():
    assert decode(encode(" ")) == " "


def test_roundtrip_single_digit():
    for ch in "0123456789":
        assert decode(encode(ch)) == ch, f"Failed for digit {ch!r}"


def test_roundtrip_all_lowercase():
    text = "abcdefghijklmnopqrstuvwxyz"
    assert decode(encode(text)) == text


def test_roundtrip_all_uppercase():
    text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    assert decode(encode(text)) == text


def test_roundtrip_common_punctuation():
    text = ".,!?-+:;=/()"
    assert decode(encode(text)) == text


def test_roundtrip_lf():
    assert decode(encode("\n")) == "\n"


def test_roundtrip_cr():
    assert decode(encode("\r")) == "\r"


def test_roundtrip_longer_message():
    text = "PSK31 is a digital amateur radio mode. QPSK31 adds FEC."
    assert decode(encode(text)) == text


# ---------------------------------------------------------------------------
# Output length sanity
# ---------------------------------------------------------------------------

def test_longer_text_produces_more_samples():
    short = encode("hi")
    long_ = encode("hello world")
    assert len(long_) > len(short)


def test_sample_count_is_multiple_of_256():
    from qpsk31.modulator import SAMPLES_PER_SYMBOL
    samples = encode("test")
    assert len(samples) % SAMPLES_PER_SYMBOL == 0


# ---------------------------------------------------------------------------
# Carrier frequency parameter
# ---------------------------------------------------------------------------

def test_roundtrip_at_800hz():
    text = "hello"
    assert decode(encode(text, carrier_freq=800.0), carrier_freq=800.0) == text


def test_roundtrip_at_1500hz():
    text = "hello"
    assert decode(encode(text, carrier_freq=1500.0), carrier_freq=1500.0) == text


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_encode_empty_string_is_decodeable():
    # Empty string: only preamble + tail bits are transmitted
    samples = encode("")
    result = decode(samples)
    assert result == ""


def test_decode_all_zeros_is_empty():
    # A buffer of silence should decode as empty
    silence = np.zeros(256 * 64, dtype=np.float32)
    result = decode(silence)
    assert result == ""


def test_mismatched_carrier_gives_wrong_output():
    """Decode with wrong carrier should not recover text (sanity check)."""
    text = "hello"
    samples = encode(text, carrier_freq=1000.0)
    result = decode(samples, carrier_freq=800.0)
    assert result != text
