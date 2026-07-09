# mlx-audio-dsp

Shared **MLX-Swift audio-DSP primitives**: STFT framing, windows, spectra, and mel-filterbank
application. Extracted from `whisper-mlx-swift`'s parity-verified front-end and generalized for
kaldi-style feature extractors.

Module: `MLXAudioDSP`. Depends only on `MLX` + `MLXFFT`.

## What's here

- **Windows** — hann (periodic/symmetric), povey (kaldi `hann^0.85`)
- **Framing** — reflect padding, center-STFT framing, kaldi snip-edges framing
- **Kaldi frame prep** — per-frame DC-offset removal, pre-emphasis (`x[-1] ≡ x[0]`)
- **Spectra** — power/magnitude `|rfft|` with optional zero-pad-to-FFT-length (kaldi 400→512)
- **Mel** — filterbank application (`spectrum · filtersᵀ`)

Filterbank **generation is deliberately excluded** — consumers bake their filters as resources
(dumped from a reference implementation) and apply them here. This keeps every consumer's
transform bit-faithful to its own upstream.

## Consumers

- `whisper-mlx-swift` — Whisper's 128-mel log-mel head (refactor gated **bit-identical** to the
  pre-extraction implementation)
- `mlx-indextts2-swift` — w2v-BERT 2.0 SeamlessM4T feature extractor + CampPlus kaldi fbank +
  22 kHz ref-mel heads (each gated against HF/torchaudio/librosa goldens, cos ≈ 1.0)

## Usage

```swift
import MLXAudioDSP

let frames = AudioDSP.framedSnipEdges(waveform, frameLength: 400, hop: 160)!
let x = AudioDSP.preEmphasized(AudioDSP.removeDCOffset(frames), coefficient: 0.97)
let power = AudioDSP.powerSpectrum(x, window: AudioDSP.poveyWindow(400), fftLength: 512)
let mel = AudioDSP.applyMelFilterbank(power, filters: bakedFilters)  // (t, nMels)
```

MIT licensed.
