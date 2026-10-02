"""
voice_similarity_ecapa.py

Same methodology as voice_similarity.py (calibrated against real same-speaker
and different-speaker baselines, with confidence intervals) but using
ECAPA-TDNN instead of Resemblyzer for the speaker embeddings.

Why swap: Resemblyzer's same-speaker vs. different-speaker baseline gap on
this dataset was only ~0.064 -- too thin to measure precisely even with 60
averaged samples. ECAPA-TDNN is the current standard for exactly this kind
of speaker-similarity evaluation in TTS/voice-cloning research (used as the
SIM-O metric in VALL-E, NaturalSpeech, and similar papers), and is
specifically documented to produce better-separated same/different-speaker
embeddings than the GE2E-style model Resemblyzer uses -- so it should need
far fewer samples for a tight, trustworthy result.

Requires:
    pip install speechbrain --break-system-packages

Usage:
    python voice_similarity_ecapa.py \
        --original ../M04/Session2/wav_headMic/0220.wav \
        --generated ../pipeline/final_output.wav \
        --speaker M04 \
        --n-other-clips 20
"""

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier

BASE_DIR = Path(__file__).parent.parent
MANIFEST_FEMALE = BASE_DIR / "manifests" / "manifest_female.csv"
MANIFEST_MALE = BASE_DIR / "manifests" / "manifest_male.csv"


def load_classifier():
    return EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="/tmp/spkrec-ecapa-voxceleb",
    )


def embed(classifier, wav_path: str) -> np.ndarray:
    signal, sr = torchaudio.load(wav_path)
    if signal.shape[0] > 1:  # downmix to mono if needed
        signal = signal.mean(dim=0, keepdim=True)
    if sr != 16000:
        signal = torchaudio.functional.resample(signal, sr, 16000)
    with torch.no_grad():
        emb = classifier.encode_batch(signal)
    return emb.squeeze().cpu().numpy()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def find_clips(speaker: str, min_duration: float = 3.0, n: int = 20) -> list[str]:
    clips = []
    for path in [MANIFEST_FEMALE, MANIFEST_MALE]:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                if row["speaker"] == speaker and row["status"] == "ok" and float(row["duration_sec"]) >= min_duration:
                    clips.append(row["wav_path"])
    random.Random(42).shuffle(clips)
    return clips[:n]


def find_other_speakers(exclude_speaker: str) -> list[str]:
    all_speakers = ["F01", "F03", "F04", "M01", "M02", "M03", "M04", "M05"]
    return [s for s in all_speakers if s != exclude_speaker]


def main():
    parser = argparse.ArgumentParser(description="Measure voice similarity using ECAPA-TDNN.")
    parser.add_argument("--original", required=True, help="The real reference clip used for cloning")
    parser.add_argument("--generated", required=True, help="The synthesized output")
    parser.add_argument("--speaker", required=True, help="Speaker name (e.g. M04)")
    parser.add_argument("--n-other-clips", type=int, default=20,
                         help="Number of DIFFERENT real clips to average against (not the cloning reference)")
    parser.add_argument("--n-baseline-clips", type=int, default=10,
                         help="Number of real clips to use for each baseline calculation")
    args = parser.parse_args()

    print("Loading ECAPA-TDNN (first run downloads the model)...")
    classifier = load_classifier()

    original_emb = embed(classifier, args.original)
    generated_emb = embed(classifier, args.generated)
    generated_vs_original = cosine_similarity(original_emb, generated_emb)

    # Same-speaker baseline
    same_speaker_clips = find_clips(args.speaker, n=args.n_baseline_clips)
    same_speaker_scores = []
    for i in range(len(same_speaker_clips)):
        for j in range(i + 1, len(same_speaker_clips)):
            emb_a = embed(classifier, same_speaker_clips[i])
            emb_b = embed(classifier, same_speaker_clips[j])
            same_speaker_scores.append(cosine_similarity(emb_a, emb_b))
    same_speaker_baseline = sum(same_speaker_scores) / len(same_speaker_scores) if same_speaker_scores else None
    same_speaker_std = np.std(same_speaker_scores) if same_speaker_scores else None

    # Different-speaker baseline, across all other speakers
    other_speakers = find_other_speakers(args.speaker)
    different_speaker_scores = []
    for other_speaker in other_speakers:
        other_clips = find_clips(other_speaker, n=3)
        for this_clip in same_speaker_clips[:3]:
            emb_this = embed(classifier, this_clip)
            for other_clip in other_clips:
                emb_other = embed(classifier, other_clip)
                different_speaker_scores.append(cosine_similarity(emb_this, emb_other))
    different_speaker_baseline = (sum(different_speaker_scores) / len(different_speaker_scores)
                                   if different_speaker_scores else None)
    different_speaker_std = np.std(different_speaker_scores) if different_speaker_scores else None

    # Channel-confound-safe comparison: generated vs. several DIFFERENT real clips
    candidate_other_clips = [c for c in find_clips(args.speaker, n=args.n_other_clips + 5) if c != args.original]
    other_clips_used = candidate_other_clips[: args.n_other_clips]
    other_clip_scores = []
    for clip in other_clips_used:
        emb = embed(classifier, clip)
        score = cosine_similarity(generated_emb, emb)
        other_clip_scores.append(score)
    other_clip_score = sum(other_clip_scores) / len(other_clip_scores) if other_clip_scores else None
    other_clip_std = np.std(other_clip_scores) if other_clip_scores else None

    print(f"\n=== ECAPA-TDNN results ===")
    print(f"Speaker: {args.speaker}")
    print(f"Generated vs. original (reference clip): {generated_vs_original:.3f}")
    if other_clip_score is not None:
        print(f"Generated vs. {len(other_clip_scores)} DIFFERENT real clips: "
              f"mean={other_clip_score:.3f}, std={other_clip_std:.3f}, "
              f"range=[{min(other_clip_scores):.3f}, {max(other_clip_scores):.3f}]")
    if same_speaker_baseline is not None:
        print(f"Same-speaker baseline: mean={same_speaker_baseline:.3f}, std={same_speaker_std:.3f} "
              f"(n={len(same_speaker_scores)} pairs)")
    if different_speaker_baseline is not None:
        print(f"Different-speaker baseline: mean={different_speaker_baseline:.3f}, std={different_speaker_std:.3f} "
              f"(n={len(different_speaker_scores)} pairs)")

    if same_speaker_baseline is not None and different_speaker_baseline is not None:
        span = same_speaker_baseline - different_speaker_baseline
        print(f"\nBaseline separation (same - different): {span:.3f}")
        print(f"(For comparison: Resemblyzer's separation on this same data was only 0.064)")

    print("\nReportable result (unconfounded):")
    if other_clip_score is not None and same_speaker_baseline is not None and different_speaker_baseline is not None:
        span = same_speaker_baseline - different_speaker_baseline
        if span > 0:
            raw_position = (other_clip_score - different_speaker_baseline) / span
            position = max(0.0, min(1.0, raw_position))
            sem = other_clip_std / (len(other_clip_scores) ** 0.5)
            ci_95 = 1.96 * sem
            position_ci = ci_95 / span * 100
            print(f"  Voice similarity: {position*100:.0f}%  (95% CI: +/- {position_ci:.0f} percentage points)")
            print(f"  Based on {len(other_clip_scores)} different real clips, NOT the exact cloning reference.")
    else:
        print("  Could not compute -- missing one or more required comparisons.")


if __name__ == "__main__":
    main()