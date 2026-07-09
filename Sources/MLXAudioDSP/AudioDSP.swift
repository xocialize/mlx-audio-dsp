// AudioDSP.swift — shared MLX audio-DSP primitives.
//
// Extracted from whisper-mlx-swift `Audio.swift` (parity-verified there to ~1e-6) and
// generalized just enough for the kaldi-style heads (w2v-BERT / CampPlus): reflect padding,
// strided framing (both center-STFT and kaldi snip-edges shapes), hann/povey windows,
// DC-offset removal + pre-emphasis, power/magnitude spectra (with kaldi's
// pad-to-power-of-two), and mel-filterbank application.
//
// Conventions:
// - Signals are 1-D MLXArray; frames are (t, frameLength).
// - Windows are computed in Double loops then cast to fp32 (numpy-float64-then-astype
//   doctrine) — consumers may also keep using their baked window resources.
// - Filterbanks are (nMels, nBins) as in whisper's baked resource; application is
//   `spectrum.matmul(filters.T)`.

import Foundation
import MLX
import MLXFFT

public enum AudioDSP {

    // MARK: - Windows

    /// Hann window. `periodic: true` = numpy/scipy `hann(sym=False)` (STFT convention,
    /// denominator N — what Whisper/torch.hann_window use); `false` = symmetric (N−1).
    public static func hannWindow(_ length: Int, periodic: Bool = true) -> MLXArray {
        let denom = Double(periodic ? length : length - 1)
        let values = (0..<length).map { i in
            Float(0.5 - 0.5 * cos(2.0 * Double.pi * Double(i) / denom))
        }
        return MLXArray(values)
    }

    /// Povey window (kaldi): symmetric hann raised to 0.85 (`torchaudio.compliance.kaldi`).
    public static func poveyWindow(_ length: Int) -> MLXArray {
        let denom = Double(length - 1)
        let values = (0..<length).map { i in
            Float(pow(0.5 - 0.5 * cos(2.0 * Double.pi * Double(i) / denom), 0.85))
        }
        return MLXArray(values)
    }

    // MARK: - Padding / framing

    /// Reflect-pad a 1-D signal by `pad` samples on both sides (mx.pad has no reflect mode;
    /// this is the reversed-slice construction from whisper-mlx-swift).
    public static func reflectPadded(_ signal: MLXArray, pad: Int) -> MLXArray {
        let n = signal.dim(0)
        return concatenated([
            reversedSlice(signal, 1, pad + 1),
            signal,
            reversedSlice(signal, n - pad - 1, n - 1),
        ], axis: 0)
    }

    /// Strided framing: (t, frameLength) with t = (n − frameLength + hop) / hop.
    /// Center-STFT shape — reflect-pad first for whisper-style pipelines.
    public static func framed(_ signal: MLXArray, frameLength: Int, hop: Int) -> MLXArray {
        let t = (signal.dim(0) - frameLength + hop) / hop
        return asStrided(signal, [t, frameLength], strides: [hop, 1])
    }

    /// Kaldi snip-edges framing: t = 1 + (n − frameLength) / hop, no padding.
    /// Returns nil when the signal is shorter than one frame.
    public static func framedSnipEdges(_ signal: MLXArray, frameLength: Int, hop: Int) -> MLXArray? {
        let n = signal.dim(0)
        guard n >= frameLength else { return nil }
        let t = 1 + (n - frameLength) / hop
        return asStrided(signal, [t, frameLength], strides: [hop, 1])
    }

    // MARK: - Kaldi frame preprocessing

    /// Subtract the per-frame mean (kaldi `remove_dc_offset`).
    public static func removeDCOffset(_ frames: MLXArray) -> MLXArray {
        frames - frames.mean(axis: -1, keepDims: true)
    }

    /// Per-frame pre-emphasis: y[i] = x[i] − c·x[i−1], with x[−1] ≡ x[0]
    /// (kaldi convention: first sample uses itself).
    public static func preEmphasized(_ frames: MLXArray, coefficient: Float = 0.97) -> MLXArray {
        let first = frames[0..., 0..<1]
        let shifted = concatenated([first, frames[0..., 0..<(frames.dim(1) - 1)]], axis: 1)
        return frames - coefficient * shifted
    }

    // MARK: - Spectra

    /// |rfft(frames · window)|² → (t, fftLength/2 + 1).
    /// `fftLength` > frameLength zero-pads each frame first (kaldi rounds 400 → 512).
    public static func powerSpectrum(
        _ frames: MLXArray, window: MLXArray, fftLength: Int? = nil
    ) -> MLXArray {
        magnitudeSpectrum(frames, window: window, fftLength: fftLength).square()
    }

    /// |rfft(frames · window)| → (t, fftLength/2 + 1).
    public static func magnitudeSpectrum(
        _ frames: MLXArray, window: MLXArray, fftLength: Int? = nil
    ) -> MLXArray {
        var x = frames * window
        if let fftLength, fftLength > frames.dim(1) {
            x = padded(x, widths: [IntOrPair((0, 0)), IntOrPair((0, fftLength - frames.dim(1)))])
        }
        return abs(MLXFFT.rfft(x))
    }

    // MARK: - Mel

    /// Apply a (nMels, nBins) filterbank: spectrum (t, nBins) → (t, nMels).
    public static func applyMelFilterbank(_ spectrum: MLXArray, filters: MLXArray) -> MLXArray {
        spectrum.matmul(filters.transposed())
    }

    // MARK: - Internals

    /// x[lo..<hi] reversed (MLX-Swift has no `[::-1]`; build reversed indices and gather).
    static func reversedSlice(_ x: MLXArray, _ lo: Int, _ hi: Int) -> MLXArray {
        let idx = MLXArray(Array(lo ..< hi).reversed().map { Int32($0) })
        return take(x, idx, axis: 0)
    }
}
