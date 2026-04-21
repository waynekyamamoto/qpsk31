"""Rate-1/2, constraint-length-5 convolutional encoder and Viterbi decoder.

Generator polynomials (taps on shift register x[0..4], x[0] = current input):
  g0: x[0] ^ x[1] ^ x[2] ^ x[4]   (0b10111 = 0x17)
  g1: x[0] ^ x[3] ^ x[4]           (0b11001 = 0x19)

State = 4-bit integer where bit 0 = most recent past input, bit 3 = oldest.
Each input bit produces one (g0, g1) dibit output.
"""

from typing import List, Tuple

_K = 5          # constraint length
_MEM = _K - 1  # 4 memory bits
_NUM_STATES = 1 << _MEM  # 16


def _enc_output(state: int, input_bit: int) -> Tuple[int, int]:
    x0 = input_bit
    x1 = (state >> 0) & 1
    x2 = (state >> 1) & 1
    x3 = (state >> 2) & 1
    x4 = (state >> 3) & 1
    g0 = x0 ^ x1 ^ x2 ^ x4
    g1 = x0 ^ x3 ^ x4
    return (g0, g1)


def _next_state(state: int, input_bit: int) -> int:
    # Shift register: new bit enters at position 0, oldest (bit 3) drops off
    return (input_bit & 1) | ((state & 0x7) << 1)


# Precomputed trellis tables
_OUTPUT: List[List[Tuple[int, int]]] = [
    [_enc_output(s, b) for b in range(2)] for s in range(_NUM_STATES)
]
_NEXT: List[List[int]] = [
    [_next_state(s, b) for b in range(2)] for s in range(_NUM_STATES)
]


def encode(bits: List[int]) -> List[Tuple[int, int]]:
    """Encode a list of bits to a list of (g0, g1) dibits.

    Caller should append K-1 = 4 tail zero bits before calling to flush the
    shift register back to state 0 for clean Viterbi termination.
    """
    state = 0
    dibits: List[Tuple[int, int]] = []
    for b in bits:
        dibits.append(_OUTPUT[state][b])
        state = _NEXT[state][b]
    return dibits


def decode(dibits: List[Tuple[int, int]]) -> List[int]:
    """Viterbi hard-decision decode: return the most likely input bit sequence."""
    T = len(dibits)
    if T == 0:
        return []

    INF = T * 3  # larger than any possible total Hamming distance

    pm = [INF] * _NUM_STATES
    pm[0] = 0

    # tb[t][s] stores the (previous_state, input_bit) that led to state s at time t
    tb_prev = [[-1] * _NUM_STATES for _ in range(T)]
    tb_bit  = [[-1] * _NUM_STATES for _ in range(T)]

    for t, (r0, r1) in enumerate(dibits):
        new_pm = [INF] * _NUM_STATES
        for s in range(_NUM_STATES):
            if pm[s] == INF:
                continue
            for b in range(2):
                g0, g1 = _OUTPUT[s][b]
                branch = (r0 != g0) + (r1 != g1)
                ns = _NEXT[s][b]
                metric = pm[s] + branch
                if metric < new_pm[ns]:
                    new_pm[ns] = metric
                    tb_prev[t][ns] = s
                    tb_bit[t][ns]  = b
        pm = new_pm

    # Traceback from state with minimum path metric
    best = min(range(_NUM_STATES), key=lambda s: pm[s])
    bits: List[int] = []
    s = best
    for t in range(T - 1, -1, -1):
        bits.append(tb_bit[t][s])
        s = tb_prev[t][s]

    bits.reverse()
    return bits
