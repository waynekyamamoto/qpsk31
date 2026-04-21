# QPSK31

A complete Python implementation of the QPSK31 digital radio modem, plus an interactive single-page web demo that visualises every stage of the pipeline.

QPSK31 is a narrow-band (~62 Hz) digital mode used on amateur (ham) radio HF bands. It transmits text at ~50 WPM using Quadrature Phase Shift Keying at 31.25 baud, with a rate-½ convolutional code for forward error correction.

**Live demo:** http://qpsk31-demo.s3-website-us-west-2.amazonaws.com

---

## How it works

```
Text  →  Varicode bits  →  Convolutional encoding  →  QPSK symbols  →  Audio
                                                                           ↕
Text  ←  Varicode decode ←  Viterbi decoder        ←  I/Q demod    ←  Audio
```

| Stage | Description |
|---|---|
| **Varicode** | Variable-length binary codes (PSK31 spec). Common letters get shorter codes: `e`=2 bits, `t`=3 bits, space=1 bit. |
| **Convolutional code** | Rate ½, constraint length K=5. Each input bit produces a 2-bit output dibit. Generator polynomials g₀=0x17, g₁=0x19. |
| **QPSK modulation** | Each dibit maps to a differential phase rotation (0°, 90°, 180°, 270°). 256 samples/symbol at 8 kHz = 31.25 baud. Carrier at 1000 Hz. |
| **Viterbi decoder** | Hard-decision minimum-distance traceback through a 16-state trellis. Corrects single-symbol errors. |

---

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.9+ and NumPy.

---

## Usage

### Python library

```python
from qpsk31 import encode, decode

# Text → audio samples (numpy float32 array, 8 kHz)
samples = encode("Hello, World!")

# Audio samples → text
text = decode(samples)
print(text)  # "Hello, World!"
```

### Demo script

```bash
python3 demo.py
```

Runs the full pipeline for `"Hello!"`, prints each stage, and writes `/tmp/qpsk31_demo.wav`.

### Tests

```bash
python3 -m pytest tests/ -v
```

111 tests covering the varicode table, convolutional encoder test vectors, modulator/demodulator roundtrips, and full end-to-end pipeline.

---

## Web demo

Open `qpsk31_demo.html` directly in any browser (no server needed) or visit the hosted version above.

The demo shows:
1. **Varicode** — each character with its bit code, full bit stream colour-coded by character
2. **Convolutional encoding** — input bits, shift register state, output dibits
3. **QPSK constellation** — I/Q plane with the live phase trajectory of your message
4. **Waveform** — full signal overview + zoomed view showing carrier cycles and phase transitions at symbol boundaries
5. **Spectrum** — FFT showing the ~62 Hz bandwidth around the 1000 Hz carrier
6. **Decode pipeline** — received dibits, Viterbi decoded bits, and recovered text

Type any message in the input box and click **Encode**. Hit **Play** to hear the actual QPSK31 audio signal.

---

## Project structure

```
qpsk31/
  varicode.py    PSK31 Varicode table, encoder, decoder
  convcode.py    Rate-½ K=5 convolutional encoder + Viterbi decoder
  modulator.py   Differential QPSK modulator & demodulator
  modem.py       High-level encode(text) / decode(samples) interface
tests/
  test_varicode.py   37 tests
  test_convcode.py   24 tests
  test_modulator.py  18 tests
  test_modem.py      32 tests
qpsk31_demo.html     Self-contained interactive web demo
demo.py              Command-line pipeline walkthrough
```

---

## Limitations

The demodulator assumes ideal channel conditions: perfect symbol timing, no frequency offset, no noise, and no pulse shaping. It will not decode real over-the-air recordings without adding symbol timing recovery and carrier frequency tracking.

---

## References

- PSK31 specification — Peter Martinez G3PLX
- [Varicode — Wikipedia](https://en.wikipedia.org/wiki/Varicode)
- [PSK31 Convolutional Encoder — Lloyd Rochester](https://lloydrochester.com/psk31/cnvenc/)
