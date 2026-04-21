"""BER/CER sweep — QPSK31 vs BPSK31 under AWGN.

Encodes a test string with both modes, adds Gaussian noise at a range of
SNR values, decodes, and measures Character Error Rate (Levenshtein distance).

Usage:
    python3 ber_sweep.py           # table only
    python3 ber_sweep.py --plot    # table + matplotlib chart
"""

import sys
import numpy as np

from qpsk31.modem import encode as qpsk_encode, decode as qpsk_decode
from qpsk31 import bpsk31
from qpsk31.noise import add_noise, char_error_rate

TEXT     = "the quick brown fox jumps over the lazy dog"
SNR_RANGE = list(range(0, -26, -1))   # 0 dB → -25 dB
N_TRIALS  = 20
SEED      = 42


def sweep(encode_fn, decode_fn, label):
    rng = np.random.default_rng(SEED)
    results = []
    for snr in SNR_RANGE:
        cers = [
            char_error_rate(TEXT, decode_fn(add_noise(encode_fn(TEXT), snr, rng)))
            for _ in range(N_TRIALS)
        ]
        results.append((snr, float(np.mean(cers)), float(np.std(cers))))
    return results


print(f"Text   : {TEXT!r}  ({len(TEXT)} chars)")
print(f"Trials : {N_TRIALS} per SNR point\n")

print(f"{'SNR (dB)':>9}  {'BPSK31 CER':>11}  {'QPSK31 CER':>11}  {'FEC gain':>10}")
print("─" * 48)

bpsk_results = sweep(bpsk31.encode, bpsk31.decode, "BPSK31")
qpsk_results = sweep(qpsk_encode, qpsk_decode, "QPSK31")

for (snr, b_mean, b_std), (_, q_mean, q_std) in zip(bpsk_results, qpsk_results):
    gain = b_mean - q_mean
    marker = " ◄" if abs(gain) >= 0.01 and gain > 0 else ""
    print(
        f"{snr:>8}  {b_mean:>10.1%}  {q_mean:>10.1%}  {gain:>+9.1%}{marker}"
    )

print("\n◄ = QPSK31 outperforms BPSK31 (FEC coding gain visible)")

# ── Optional matplotlib plot ─────────────────────────────────────────────────
if "--plot" in sys.argv:
    try:
        import matplotlib.pyplot as plt

        snrs  = [r[0] for r in bpsk_results]
        b_cer = [r[1] for r in bpsk_results]
        q_cer = [r[1] for r in qpsk_results]
        b_std = [r[2] for r in bpsk_results]
        q_std = [r[2] for r in qpsk_results]

        fig, ax = plt.subplots(figsize=(9, 5))

        ax.plot(snrs, b_cer, "o-", color="#f85149", label="BPSK31 (no FEC)", linewidth=2)
        ax.fill_between(snrs,
                        [max(0, m-s) for m,s in zip(b_cer, b_std)],
                        [min(1, m+s) for m,s in zip(b_cer, b_std)],
                        alpha=0.15, color="#f85149")

        ax.plot(snrs, q_cer, "s-", color="#58a6ff", label="QPSK31 (rate-½ K=5 FEC)", linewidth=2)
        ax.fill_between(snrs,
                        [max(0, m-s) for m,s in zip(q_cer, q_std)],
                        [min(1, m+s) for m,s in zip(q_cer, q_std)],
                        alpha=0.15, color="#58a6ff")

        ax.axvline(-12, color="#3fb950", linestyle="--", alpha=0.6, label="FEC sweet spot (−12 dB)")
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel("Character Error Rate")
        ax.set_title("QPSK31 vs BPSK31: CER under AWGN\n"
                     f"Text: \"{TEXT[:30]}…\"  ·  {N_TRIALS} trials/point")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        ax.invert_xaxis()
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("ber_curve.png", dpi=150)
        print("\nPlot saved to ber_curve.png")
        plt.show()
    except ImportError:
        print("\nmatplotlib not installed — run: pip install matplotlib")
