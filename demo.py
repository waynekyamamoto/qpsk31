"""Step-by-step demonstration of the QPSK31 pipeline."""

import struct
import wave

import numpy as np

from qpsk31 import varicode as vc
from qpsk31 import convcode as cc
from qpsk31.modulator import DEFAULT_CARRIER, SAMPLE_RATE, SAMPLES_PER_SYMBOL, modulate, demodulate
from qpsk31.modem import encode, decode, _PREAMBLE_ZEROS, _TAIL_ZEROS

TEXT = "Hello!"

print("=" * 60)
print(f"  QPSK31 pipeline demo  —  message: {TEXT!r}")
print("=" * 60)

# ── Step 1: Varicode ──────────────────────────────────────────────
print("\n[1] Varicode encoding")
for ch in TEXT:
    code = vc.VARICODE_TABLE[ord(ch)]
    print(f"    {ch!r:4s}  ->  {code}  ({len(code)} bits)")

vc_bits = vc.encode(TEXT)
print(f"\n    Full bit stream (with 00 separators):  {len(vc_bits)} bits")
print(f"    {vc_bits}")

# ── Step 2: Add preamble/tail ─────────────────────────────────────
print(f"\n[2] Framing: prepend {_PREAMBLE_ZEROS} preamble zeros, append {_TAIL_ZEROS} tail zeros")
bits = [0] * _PREAMBLE_ZEROS + vc_bits + [0] * _TAIL_ZEROS
print(f"    Total bits: {len(bits)}")

# ── Step 3: Convolutional encode ─────────────────────────────────
print("\n[3] Convolutional encoding  (rate 1/2, K=5, g0=0x17, g1=0x19)")
dibits = cc.encode(bits)
print(f"    {len(bits)} bits  ->  {len(dibits)} dibits")
print(f"    First 16 dibits: {dibits[:16]}")

# ── Step 4: Modulate ──────────────────────────────────────────────
print(f"\n[4] QPSK modulation  (fc={DEFAULT_CARRIER} Hz, Fs={SAMPLE_RATE} Hz, {SAMPLES_PER_SYMBOL} samples/symbol)")
samples = modulate(dibits)
print(f"    {len(dibits)} symbols  ->  {len(samples)} audio samples")
print(f"    Duration: {len(samples)/SAMPLE_RATE:.3f} s")
print(f"    Sample range: [{samples.min():.4f}, {samples.max():.4f}]")
print(f"    RMS level: {np.sqrt(np.mean(samples**2)):.4f}")

# ── Save WAV file ─────────────────────────────────────────────────
wav_path = "/tmp/qpsk31_demo.wav"
pcm16 = (samples * 32767).astype(np.int16)
with wave.open(wav_path, "w") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(SAMPLE_RATE)
    wf.writeframes(pcm16.tobytes())
print(f"\n[5] Saved WAV file: {wav_path}")
print(f"    {SAMPLE_RATE} Hz, 16-bit mono, {len(pcm16)} frames")

# ── Step 5: Reload WAV and decode ────────────────────────────────
print("\n[6] Reloading WAV and decoding end-to-end")
with wave.open(wav_path) as wf:
    raw = wf.readframes(wf.getnframes())
pcm_back = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0

dibits_rx = demodulate(pcm_back)
print(f"    Demodulated {len(dibits_rx)} dibits")

bits_rx = cc.decode(dibits_rx)
print(f"    Viterbi decoded {len(bits_rx)} bits")

text_rx = vc.decode(bits_rx)
print(f"\n    Decoded text: {text_rx!r}")

# ── Verdict ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
if text_rx == TEXT:
    print(f"  PASS  {TEXT!r}  ->  WAV file  ->  {text_rx!r}")
else:
    print(f"  FAIL  sent={TEXT!r}  got={text_rx!r}")
print("=" * 60)

# ── Batch roundtrip proof ─────────────────────────────────────────
print("\n[7] Batch roundtrip — 20 test strings")
test_strings = [
    "e", "hi", "hello", "Hello", "hello world",
    "Hello, World!", "CQ CQ DE W1AW",
    "the quick brown fox jumps over the lazy dog",
    "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG",
    "0123456789",
    "abcdefghijklmnopqrstuvwxyz",
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    ".,!?-+:;=()",
    "PSK31 uses FEC via QPSK31.",
    "73 de W1AW/KH6",
    "freq: 14.070 MHz",
    "RSID: <<QPSK-31>>",
    "Hello\nWorld",
    " ",
    "",
]

passed = failed = 0
for s in test_strings:
    got = decode(encode(s))
    ok = got == s
    status = "PASS" if ok else "FAIL"
    label = repr(s) if len(s) <= 30 else repr(s[:27] + "...")
    print(f"    {status}  {label}")
    if ok:
        passed += 1
    else:
        failed += 1
        print(f"         expected {s!r}")
        print(f"         got      {got!r}")

print(f"\n    {passed}/{passed+failed} passed")
