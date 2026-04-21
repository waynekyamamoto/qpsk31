"""Tests for the convolutional encoder and Viterbi decoder."""

import pytest
from qpsk31.convcode import encode, decode, _enc_output, _next_state, _NUM_STATES


# ---------------------------------------------------------------------------
# Encoder unit tests
# ---------------------------------------------------------------------------

def test_all_zeros_input_gives_all_zero_dibits():
    bits = [0] * 20
    dibits = encode(bits)
    assert dibits == [(0, 0)] * 20


def test_encoder_output_count_equals_input_count():
    bits = [1, 0, 1, 1, 0]
    assert len(encode(bits)) == len(bits)


def test_encoder_empty_input():
    assert encode([]) == []


# Known test vector (computed from g0/g1 polynomials, verified by hand):
# Input:  [0,1,0,1,1,1,0,0,1,0,1,0,0,0,1,0,0,0,0,0]
# Output: first 5 dibits = [(0,0),(1,1),(1,0),(0,1),(0,0)]
_TEST_VECTOR_INPUT = [0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0]
_TEST_VECTOR_OUTPUT = [
    (0, 0), (1, 1), (1, 0), (0, 1), (0, 0),
    (0, 0), (0, 1), (0, 0), (0, 1), (0, 1),
    (0, 1), (1, 1), (0, 1), (0, 1), (0, 0),
    (1, 0), (1, 0), (0, 1), (1, 1), (0, 0),
]


def test_known_test_vector_first_five():
    dibits = encode(_TEST_VECTOR_INPUT)
    assert dibits[:5] == _TEST_VECTOR_OUTPUT[:5]


def test_known_test_vector_full():
    dibits = encode(_TEST_VECTOR_INPUT)
    assert dibits == _TEST_VECTOR_OUTPUT


# ---------------------------------------------------------------------------
# Encoder polynomial properties
# ---------------------------------------------------------------------------

def test_enc_output_all_zero_state_zero_input():
    assert _enc_output(0, 0) == (0, 0)


def test_enc_output_all_zero_state_one_input():
    assert _enc_output(0, 1) == (1, 1)


def test_enc_output_state1_input1():
    # state=1 (0b0001): x1=1, x2=0, x3=0, x4=0
    # g0 = 1^1^0^0 = 0, g1 = 1^0^0 = 1
    assert _enc_output(1, 1) == (0, 1)


def test_enc_output_state15_input0():
    # state=15 (0b1111): x1=1, x2=1, x3=1, x4=1
    # g0 = 0^1^1^1 = 1, g1 = 0^1^1 = 0
    assert _enc_output(15, 0) == (1, 0)


def test_next_state_from_zero_with_one():
    # state=0, input=1 -> new state has bit0=1, rest=0 -> 1
    assert _next_state(0, 1) == 1


def test_next_state_from_zero_with_zero():
    assert _next_state(0, 0) == 0


def test_next_state_shifts_correctly():
    # state=0b0101=5, input=1 -> 0b1011=11
    assert _next_state(5, 1) == 11


def test_next_state_drops_msb():
    # state=15 (0b1111), input=0: new bit=0 at bit0, old bits shift up, bit3 drops
    # new state = 0b1110 = 14
    assert _next_state(15, 0) == 14


def test_next_state_stays_in_range():
    for s in range(_NUM_STATES):
        for b in range(2):
            ns = _next_state(s, b)
            assert 0 <= ns < _NUM_STATES


def test_output_is_always_single_bits():
    for s in range(_NUM_STATES):
        for b in range(2):
            g0, g1 = _enc_output(s, b)
            assert g0 in (0, 1)
            assert g1 in (0, 1)


# ---------------------------------------------------------------------------
# Viterbi decoder
# ---------------------------------------------------------------------------

def test_decode_all_zeros():
    dibits = [(0, 0)] * 20
    bits = decode(dibits)
    assert len(bits) == 20
    # With only all-zero dibits, the most likely sequence is all zeros
    assert bits == [0] * 20


def test_decode_empty():
    assert decode([]) == []


def test_roundtrip_simple():
    bits_in = [1, 0, 1, 1, 0, 0, 0, 0]  # 4 tail zeros at end
    dibits = encode(bits_in)
    bits_out = decode(dibits)
    assert bits_out == bits_in


def test_roundtrip_longer():
    import random
    random.seed(42)
    bits_in = [random.randint(0, 1) for _ in range(40)] + [0] * 4  # tail
    dibits = encode(bits_in)
    bits_out = decode(dibits)
    assert bits_out == bits_in


def test_roundtrip_all_ones():
    bits_in = [1] * 20 + [0] * 4
    dibits = encode(bits_in)
    bits_out = decode(dibits)
    assert bits_out == bits_in


def test_roundtrip_alternating():
    bits_in = [0, 1] * 10 + [0] * 4
    dibits = encode(bits_in)
    bits_out = decode(dibits)
    assert bits_out == bits_in


def test_single_bit_error_corrected():
    # Single received dibit error should be correctable by Viterbi
    bits_in = [1, 0, 1, 0, 1, 0, 0, 0, 0, 0]  # last 4 are tail
    dibits = encode(bits_in)
    # Flip one bit in one dibit (introduce single error)
    dibits_err = list(dibits)
    dibits_err[3] = (dibits_err[3][0] ^ 1, dibits_err[3][1])
    bits_out = decode(dibits_err)
    assert bits_out == bits_in


def test_known_vector_roundtrip():
    bits_in = _TEST_VECTOR_INPUT
    dibits = encode(bits_in)
    assert dibits == _TEST_VECTOR_OUTPUT
    bits_out = decode(dibits)
    assert bits_out == bits_in
