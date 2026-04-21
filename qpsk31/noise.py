"""AWGN noise injection and character error rate metric."""

import numpy as np


def add_noise(samples: np.ndarray, snr_db: float, rng=None) -> np.ndarray:
    """Add white Gaussian noise to samples at the given SNR (dB).

    SNR is defined as signal_power / noise_power where signal_power is
    measured from the samples themselves.
    """
    if rng is None:
        rng = np.random.default_rng()
    signal_power = float(np.mean(samples.astype(np.float64) ** 2))
    if signal_power == 0:
        return samples.copy()
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = rng.normal(0.0, np.sqrt(noise_power), len(samples)).astype(np.float32)
    return samples + noise


def char_error_rate(original: str, decoded: str) -> float:
    """Levenshtein edit distance normalised by the length of original.

    Returns 0.0 for a perfect match, 1.0 if every character is wrong,
    and >1.0 if the decoder inserted extra characters.
    """
    m, n = len(original), len(decoded)
    if m == 0:
        return 0.0
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if original[i - 1] == decoded[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[n] / m
