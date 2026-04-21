"""Tests for the varicode encoder/decoder."""

import pytest
from qpsk31.varicode import VARICODE_TABLE, VARICODE_REVERSE, encode, decode


# ---------------------------------------------------------------------------
# Table integrity
# ---------------------------------------------------------------------------

def test_table_covers_all_ascii():
    assert len(VARICODE_TABLE) == 128
    assert set(VARICODE_TABLE.keys()) == set(range(128))


def test_reverse_table_size():
    # Every code must be unique (no two characters share the same code)
    assert len(VARICODE_REVERSE) == 128


def test_no_code_contains_double_zero():
    for ordinal, code in VARICODE_TABLE.items():
        assert "00" not in code, f"Code for char {ordinal!r} contains '00': {code!r}"


def test_all_codes_end_in_one():
    for ordinal, code in VARICODE_TABLE.items():
        assert code.endswith("1"), (
            f"Code for char {ordinal!r} does not end in '1': {code!r}"
        )


def test_all_codes_nonempty():
    for ordinal, code in VARICODE_TABLE.items():
        assert len(code) >= 1, f"Empty code for char {ordinal!r}"


def test_all_codes_are_binary_strings():
    for ordinal, code in VARICODE_TABLE.items():
        assert all(c in "01" for c in code), (
            f"Non-binary character in code for {ordinal!r}: {code!r}"
        )


# ---------------------------------------------------------------------------
# Known encode values (from PSK31 specification)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("char,expected_code", [
    (' ',  "1"),
    ('e',  "11"),
    ('t',  "101"),
    ('o',  "111"),
    ('a',  "1011"),
    ('i',  "1101"),
    ('n',  "1111"),
    ('s',  "10111"),
    ('r',  "10101"),
    ('h',  "101011"),
    ('0',  "10110111"),
    ('1',  "10111101"),
    ('A',  "1111101"),
    ('Z',  "1010101101"),
])
def test_known_encode_values(char, expected_code):
    bits = encode(char)
    # encode appends a '00' separator; strip it for comparison
    code_bits = bits[:-2]
    assert "".join(str(b) for b in code_bits) == expected_code


def test_encode_space_is_single_one():
    bits = encode(" ")
    # First bit is the varicode for space ('1'), next two are the separator
    assert bits[:1] == [1]
    assert bits[1:3] == [0, 0]


def test_encode_e_is_two_ones():
    bits = encode("e")
    assert bits[:2] == [1, 1]
    assert bits[2:4] == [0, 0]


# ---------------------------------------------------------------------------
# Separator structure
# ---------------------------------------------------------------------------

def test_separator_between_chars():
    bits = encode("et")
    # 'e' = "11" + "00" + 't' = "101" + "00"
    assert bits == [1, 1, 0, 0, 1, 0, 1, 0, 0]


def test_encode_empty_string():
    assert encode("") == []


# ---------------------------------------------------------------------------
# Decode
# ---------------------------------------------------------------------------

def test_decode_space():
    assert decode([1, 0, 0]) == " "


def test_decode_e():
    assert decode([1, 1, 0, 0]) == "e"


def test_decode_et():
    bits = [1, 1, 0, 0, 1, 0, 1, 0, 0]
    assert decode(bits) == "et"


def test_decode_ignores_leading_zeros():
    # Preamble zeros should produce no output
    bits = [0] * 10 + [1, 1, 0, 0]
    assert decode(bits) == "e"


def test_decode_ignores_trailing_zeros():
    bits = [1, 1, 0, 0] + [0] * 8
    assert decode(bits) == "e"


# ---------------------------------------------------------------------------
# Roundtrip
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "e",
    "hello",
    "Hello, World!",
    "the quick brown fox",
    "0123456789",
    "abcdefghijklmnopqrstuvwxyz",
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~",
])
def test_roundtrip(text):
    assert decode(encode(text)) == text


def test_roundtrip_all_printable_ascii():
    text = "".join(chr(i) for i in range(32, 127))
    assert decode(encode(text)) == text


def test_roundtrip_lf():
    assert decode(encode("\n")) == "\n"


def test_roundtrip_cr():
    assert decode(encode("\r")) == "\r"


def test_roundtrip_tab():
    assert decode(encode("\t")) == "\t"
