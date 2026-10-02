// AudioDSPTests.swift — closed-form checks for the DSP primitives (CPU stream).
// The consumer-level parity gates live in the consumers (WhisperMLX old-vs-new bit-identity;
// IndexTTS2 P3 fbank heads vs HF goldens) — these tests pin the primitive semantics.

import XCTest
import MLX
@testable import MLXAudioDSP

final class AudioDSPTests: XCTestCase {

    override func invokeTest() { withMLXCPU { super.invokeTest() } }

    private func values(_ x: MLXArray) -> [Float] {
        eval(x)
        return x.asArray(Float.self)
    }

    func testHannPeriodic() {
        // hann(4, periodic): 0.5 - 0.5cos(2πi/4) = [0, 0.5, 1, 0.5]
        let w = values(AudioDSP.hannWindow(4, periodic: true))
        let expected: [Float] = [0, 0.5, 1, 0.5]
        for (a, b) in zip(w, expected) { XCTAssertEqual(a, b, accuracy: 1e-7) }
    }

    func testHannSymmetric() {
        // hann(5, symmetric): [0, 0.5, 1, 0.5, 0]
        let w = values(AudioDSP.hannWindow(5, periodic: false))
        let expected: [Float] = [0, 0.5, 1, 0.5, 0]
        for (a, b) in zip(w, expected) { XCTAssertEqual(a, b, accuracy: 1e-7) }
    }

    func testPovey() {
        // povey(5) = hann_sym(5)^0.85 → [0, 0.5^0.85, 1, 0.5^0.85, 0]
        let w = values(AudioDSP.poveyWindow(5))
        let half = Float(pow(0.5, 0.85))
        let expected: [Float] = [0, half, 1, half, 0]
        for (a, b) in zip(w, expected) { XCTAssertEqual(a, b, accuracy: 1e-6) }
    }

    func testReflectPadded() {
        // numpy reflect: [1,2,3,4,5] pad 2 → [3,2,1,2,3,4,5,4,3]
        let x = MLXArray([Float]([1, 2, 3, 4, 5]))
        let padded = values(AudioDSP.reflectPadded(x, pad: 2))
        XCTAssertEqual(padded, [3, 2, 1, 2, 3, 4, 5, 4, 3])
    }

    func testFramed() {
        // 0..9, len 4, hop 2 → 4 frames
        let x = MLXArray((0..<10).map(Float.init))
        let frames = AudioDSP.framed(x, frameLength: 4, hop: 2)
        XCTAssertEqual(frames.shape, [4, 4])
        XCTAssertEqual(values(frames[0]), [0, 1, 2, 3])
        XCTAssertEqual(values(frames[3]), [6, 7, 8, 9])
    }

    func testFramedSnipEdges() {
        // kaldi: n=10, len=4, hop=3 → t = 1 + (10-4)/3 = 3
        let x = MLXArray((0..<10).map(Float.init))
        let frames = AudioDSP.framedSnipEdges(x, frameLength: 4, hop: 3)
        XCTAssertEqual(frames?.shape, [3, 4])
        XCTAssertEqual(values(frames![2]), [6, 7, 8, 9])
        // shorter than one frame → nil
        XCTAssertNil(AudioDSP.framedSnipEdges(MLXArray([Float]([1, 2])), frameLength: 4, hop: 3))
    }

    func testRemoveDCOffset() {
        let frames = MLXArray([Float]([1, 2, 3, 4]), [1, 4])
        XCTAssertEqual(values(AudioDSP.removeDCOffset(frames)), [-1.5, -0.5, 0.5, 1.5])
    }

    func testPreEmphasis() {
        // x = [2,4,6], c=0.5, x[-1]≡x[0] → [1, 3, 4]
        let frames = MLXArray([Float]([2, 4, 6]), [1, 3])
        XCTAssertEqual(values(AudioDSP.preEmphasized(frames, coefficient: 0.5)), [1, 3, 4])
    }

    func testPowerSpectrumShapesAndDC() {
        // ones frame, ones window: DC bin = N², kaldi pad 400→512 → 257 bins.
        let frames = MLXArray.ones([2, 400])
        let window = MLXArray.ones([400])
        let plain = AudioDSP.powerSpectrum(frames, window: window)
        XCTAssertEqual(plain.shape, [2, 201])
        let padded = AudioDSP.powerSpectrum(frames, window: window, fftLength: 512)
        XCTAssertEqual(padded.shape, [2, 257])
        XCTAssertEqual(values(padded[0..., 0..<1])[0], Float(400 * 400), accuracy: 1.0)
    }

    func testApplyMelFilterbank() {
        let spectrum = MLXArray.ones([3, 201])
        let filters = MLXArray.ones([80, 201]) * 0.5
        let mel = AudioDSP.applyMelFilterbank(spectrum, filters: filters)
        XCTAssertEqual(mel.shape, [3, 80])
        XCTAssertEqual(values(mel[0..., 0..<1])[0], Float(201) * 0.5, accuracy: 1e-3)
    }
}
