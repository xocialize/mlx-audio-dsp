// swift-tools-version: 5.9
//
// mlx-audio-dsp — the shared MLX audio-DSP leaf: STFT framing, windows, spectra, and
// mel-filterbank application, extracted from whisper-mlx-swift's parity-verified Audio.swift.
//
// Consumers build per-model FRONT-END HEADS on these primitives (they do NOT share one
// extractor): Whisper's 128-mel log-mel (whisper-mlx-swift), w2v-BERT's Seamless-style
// normalized 80-mel fbank and CampPlus's kaldi 80-mel fbank (mlx-indextts2-swift).
// Mel filterbanks are BAKED resources in each consumer ("bake fixed transforms" rule) —
// this leaf applies filters, it does not generate them.
//
// Deliberately tiny and MLX-shaped: media-bridge (MLX-free, PCM acquisition) is the layer
// BELOW this; nothing here does file I/O or resampling.
import PackageDescription

let package = Package(
    name: "mlx-audio-dsp",
    platforms: [
        .macOS(.v14),
        .iOS(.v17),
        .visionOS(.v1),
    ],
    products: [
        .library(name: "MLXAudioDSP", targets: ["MLXAudioDSP"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.21.0"),
    ],
    targets: [
        .target(
            name: "MLXAudioDSP",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXFFT", package: "mlx-swift"),
            ]
        ),
        .testTarget(
            name: "MLXAudioDSPTests",
            dependencies: ["MLXAudioDSP"]
        ),
    ]
)
