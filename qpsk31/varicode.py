"""PSK31 Varicode encoder/decoder.

Each ASCII character maps to a variable-length bit string (all 1-prefixed, no '00'
substring). Characters are separated by the 2-bit sequence '00' in the stream.
Table sourced from the PSK31 specification (Peter Martinez G3PLX) via Wikipedia.
"""

from typing import List

# Maps ASCII ordinal -> varicode bit string (string of '0'/'1' chars)
VARICODE_TABLE: dict[int, str] = {
    0:   "1010101011",  # NUL
    1:   "1011011011",  # SOH
    2:   "1011101101",  # STX
    3:   "1101110111",  # ETX
    4:   "1011101011",  # EOT
    5:   "1101011111",  # ENQ
    6:   "1011101111",  # ACK
    7:   "1011111101",  # BEL
    8:   "1011111111",  # BS
    9:   "11101111",    # HT
    10:  "11101",       # LF
    11:  "1101101111",  # VT
    12:  "1011011101",  # FF
    13:  "11111",       # CR
    14:  "1101110101",  # SO
    15:  "1110101011",  # SI
    16:  "1011110111",  # DLE
    17:  "1011110101",  # DC1
    18:  "1110101101",  # DC2
    19:  "1110101111",  # DC3
    20:  "1101011011",  # DC4
    21:  "1101101011",  # NAK
    22:  "1101101101",  # SYN
    23:  "1101010111",  # ETB
    24:  "1101111011",  # CAN
    25:  "1101111101",  # EM
    26:  "1110110111",  # SUB
    27:  "1101010101",  # ESC
    28:  "1101011101",  # FS
    29:  "1110111011",  # GS
    30:  "1011111011",  # RS
    31:  "1101111111",  # US
    32:  "1",           # SP
    33:  "111111111",   # !
    34:  "101011111",   # "
    35:  "111110101",   # #
    36:  "111011011",   # $
    37:  "1011010101",  # %
    38:  "1010111011",  # &
    39:  "101111111",   # '
    40:  "11111011",    # (
    41:  "11110111",    # )
    42:  "101101111",   # *
    43:  "111011111",   # +
    44:  "1110101",     # ,
    45:  "110101",      # -
    46:  "1010111",     # .
    47:  "110101111",   # /
    48:  "10110111",    # 0
    49:  "10111101",    # 1
    50:  "11101101",    # 2
    51:  "11111111",    # 3
    52:  "101110111",   # 4
    53:  "101011011",   # 5
    54:  "101101011",   # 6
    55:  "110101101",   # 7
    56:  "110101011",   # 8
    57:  "110110111",   # 9
    58:  "11110101",    # :
    59:  "110111101",   # ;
    60:  "111101101",   # <
    61:  "1010101",     # =
    62:  "111010111",   # >
    63:  "1010101111",  # ?
    64:  "1010111101",  # @
    65:  "1111101",     # A
    66:  "11101011",    # B
    67:  "10101101",    # C
    68:  "10110101",    # D
    69:  "1110111",     # E
    70:  "11011011",    # F
    71:  "11111101",    # G
    72:  "101010101",   # H
    73:  "1111111",     # I
    74:  "111111101",   # J
    75:  "101111101",   # K
    76:  "11010111",    # L
    77:  "10111011",    # M
    78:  "11011101",    # N
    79:  "10101011",    # O
    80:  "11010101",    # P
    81:  "111011101",   # Q
    82:  "10101111",    # R
    83:  "1101111",     # S
    84:  "1101101",     # T
    85:  "101010111",   # U
    86:  "110110101",   # V
    87:  "101011101",   # W
    88:  "101110101",   # X
    89:  "101111011",   # Y
    90:  "1010101101",  # Z
    91:  "111110111",   # [
    92:  "111101111",   # \
    93:  "111111011",   # ]
    94:  "1010111111",  # ^
    95:  "101101101",   # _
    96:  "1011011111",  # `
    97:  "1011",        # a
    98:  "1011111",     # b
    99:  "101111",      # c
    100: "101101",      # d
    101: "11",          # e
    102: "111101",      # f
    103: "1011011",     # g
    104: "101011",      # h
    105: "1101",        # i
    106: "111101011",   # j
    107: "10111111",    # k
    108: "11011",       # l
    109: "111011",      # m
    110: "1111",        # n
    111: "111",         # o
    112: "111111",      # p
    113: "110111111",   # q
    114: "10101",       # r
    115: "10111",       # s
    116: "101",         # t
    117: "110111",      # u
    118: "1111011",     # v
    119: "1101011",     # w
    120: "11011111",    # x
    121: "1011101",     # y
    122: "111010101",   # z
    123: "1010110111",  # {
    124: "110111011",   # |
    125: "1010110101",  # }
    126: "1011010111",  # ~
    127: "1110110101",  # DEL
}

# Reverse lookup: bit string -> character
VARICODE_REVERSE: dict[str, str] = {v: chr(k) for k, v in VARICODE_TABLE.items()}


def encode(text: str) -> List[int]:
    """Encode text to a flat list of bits (ints 0/1), with 00 separators between chars."""
    bits: List[int] = []
    for ch in text:
        code = VARICODE_TABLE[ord(ch)]
        bits.extend(int(b) for b in code)
        bits.extend([0, 0])
    return bits


def decode(bits: List[int]) -> str:
    """Decode a flat list of bits back to text, tolerating leading/trailing zeros."""
    result: List[str] = []
    current: List[int] = []

    i = 0
    while i < len(bits):
        b = bits[i]
        if b == 1:
            current.append(1)
            i += 1
        else:
            # Zero bit: peek ahead for the second zero (character boundary)
            if i + 1 < len(bits) and bits[i + 1] == 0:
                if current:
                    code = "".join(str(x) for x in current)
                    ch = VARICODE_REVERSE.get(code)
                    if ch is not None:
                        result.append(ch)
                    current = []
                i += 2  # consume both zeros
            else:
                # Single zero inside a codeword — shouldn't happen in valid varicode,
                # but some codes (e.g. 't'='101') have internal zeros.
                current.append(0)
                i += 1

    # Flush final character if stream ends without a trailing 00
    if current:
        code = "".join(str(x) for x in current)
        ch = VARICODE_REVERSE.get(code)
        if ch is not None:
            result.append(ch)

    return "".join(result)
