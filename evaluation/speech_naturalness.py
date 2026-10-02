"""
speech_naturalness.py

Predicts how NATURAL/clear the generated speech sounds, using UTMOS --
an automatic Mean Opinion Score (MOS) predictor trained on large volumes of
REAL human quality ratings (VoiceMOS Challenge data). This is the standard,
widely-used proxy for "how would a human rate this audio's quality," without
needing an actual listening panel. It correlates ~0.84 with real human MOS
ratings (LCC, VoiceMOS Challenge 2022) -- a genuine ground-truth-correlated
metric, not just another embedding comparison.

This is DIFFERENT from, and complements, the other two metrics already built:
    - WER / content fidelity (compute_wer.py)          -- did it say the right words?
    - Voice similarity (voice_similarity_ecapa.py)      -- does it sound like THIS person?
    - Naturalness (this script)                         -- does it sound like natural,
                                                            good-quality speech AT ALL?

A pipeline can score well on one axis and poorly on another -- e.g. correct
words, wrong voice identity, but still perfectly natural-sounding audio.
Report all three separately; never average them into one number.

Scale: 1 (bad) to 5 (excellent), same scale as human MOS ratings.

Requires:
    pip install torch torchaudio --break-system-packages
    (UTMOS itself downloads automatically via torch.hub on first run)

Usage:
    python speech_naturalness.py --wav ../pipeline/final_output.wav

    # Compare several files at once (e.g. real original vs. generated):
    python speech_naturalness.py --wav real_original.wav generated.wav
"""

import argparse

import torch
import torchaudio


def load_predictor():
    print("Loading UTMOS (first run downloads it via torch.hub)...")
    return torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)


def predict_naturalness(predictor, wav_path: str) -> float:
    wave, sr = torchaudio.load(wav_path)
    if wave.shape[0] > 1:
        wave = wave.mean(dim=0, keepdim=True)
    score = predictor(wave, sr)
    return float(score)


def main():
    parser = argparse.ArgumentParser(description="Predict speech naturalness (UTMOS, 1-5 scale, no reference needed).")
    parser.add_argument("--wav", nargs="+", required=True, help="One or more .wav files to score")
    args = parser.parse_args()

    predictor = load_predictor()

    print(f"\n{'File':<60} {'UTMOS (1-5)':>12}")
    print("-" * 73)
    results = []
    for wav_path in args.wav:
        score = predict_naturalness(predictor, wav_path)
        results.append((wav_path, score))
        print(f"{wav_path:<60} {score:>12.2f}")

    if len(results) > 1:
        print("\nFor reference: typical ranges seen in TTS literature --")
        print("  1.0-2.0: poor/robotic/degraded    3.0-3.5: decent, synthetic-sounding")
        print("  3.5-4.0: good, natural-sounding    4.0-5.0: excellent, near-human")


if __name__ == "__main__":
    main()