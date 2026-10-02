"""
voice_similarity.py

Measures whether synthesized (voice-cloned) audio actually sounds like the
original speaker, using speaker embeddings (Resemblyzer) + cosine similarity
-- NOT word accuracy (that's what compute_wer.py is for). This answers a
genuinely different question: "does it sound like them," independent of
"did it say the right words."

Rather than asserting an absolute "good" threshold (which varies by model
and isn't something to take on faith), this calibrates the generated-vs-
original score against two real anchors from the speaker's own data:

    same_speaker_baseline:      similarity between two DIFFERENT real clips
                                 of the SAME speaker (the realistic ceiling --
                                 even two real recordings of one person aren't
                                 perfectly identical)
    different_speaker_baseline: similarity between this speaker and a
                                 DIFFERENT real speaker (the floor)
    generated_vs_original:      the actual number you care about

If generated_vs_original is close to same_speaker_baseline, the clone is
about as similar as two real recordings of that person. If it's close to
different_speaker_baseline, the clone doesn't really sound like them.

Requires:
    pip install resemblyzer --break-system-packages

Usage:
    python voice_similarity.py \
        --original ../M04/Session2/wav_headMic/0220.wav \
        --generated ../pipeline/final_output.wav \
        --speaker M04
"""

import argparse
import csv
import random
from pathlib import Path

import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav

BASE_DIR = Path(__file__).parent.parent
MANIFEST_FEMALE = BASE_DIR / "manifests" / "manifest_female.csv"
MANIFEST_MALE = BASE_DIR / "manifests" / "manifest_male.csv"


def embed(encoder: VoiceEncoder, wav_path: str) -> np.ndarray:
    wav = preprocess_wav(wav_path)
    return encoder.embed_utterance(wav)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def find_clips(speaker: str, min_duration: float = 3.0, n: int = 6) -> list[str]:
    """Returns up to n real, usable clip paths for a speaker (for baselines).
    n defaults to 6, not 2 -- a single pair makes the baseline noisy and
    unreliable (observed: one same-speaker pair and one different-speaker
    pair landed only 0.023 apart, making any percentage computed against
    them essentially meaningless)."""
    clips = []
    for path in [MANIFEST_FEMALE, MANIFEST_MALE]:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                if row["speaker"] == speaker and row["status"] == "ok" and float(row["duration_sec"]) >= min_duration:
                    clips.append(row["wav_path"])
    random.Random(42).shuffle(clips)
    return clips[:n]


def find_other_speakers(exclude_speaker: str) -> list[str]:
    """Returns ALL other speaker names, for a broader different-speaker baseline."""
    all_speakers = ["F01", "F03", "F04", "M01", "M02", "M03", "M04", "M05"]
    return [s for s in all_speakers if s != exclude_speaker]


def main():
    parser = argparse.ArgumentParser(description="Measure voice similarity between generated and original audio.")
    parser.add_argument("--original", required=True, help="The real reference clip used for cloning")
    parser.add_argument("--generated", required=True, help="The synthesized output")
    parser.add_argument("--speaker", required=True, help="Speaker name (e.g. M04), for baseline calibration")
    parser.add_argument("--n-other-clips", type=int, default=5,
                         help="Number of DIFFERENT real clips of the same speaker (not the reference used for "
                              "cloning) to average the generated audio against. A single clip is too noisy to "
                              "trust -- this should be averaged with the same rigor as the baselines.")
    args = parser.parse_args()

    print("Loading speaker embedding model...")
    encoder = VoiceEncoder()

    original_emb = embed(encoder, args.original)
    generated_emb = embed(encoder, args.generated)
    generated_vs_original = cosine_similarity(original_emb, generated_emb)

    # Same-speaker baseline: AVERAGE similarity across several pairs of real
    # clips of this speaker (a single pair is too noisy to trust -- see docstring).
    same_speaker_clips = find_clips(args.speaker, n=6)
    same_speaker_scores = []
    for i in range(len(same_speaker_clips)):
        for j in range(i + 1, len(same_speaker_clips)):
            emb_a = embed(encoder, same_speaker_clips[i])
            emb_b = embed(encoder, same_speaker_clips[j])
            same_speaker_scores.append(cosine_similarity(emb_a, emb_b))
    same_speaker_baseline = sum(same_speaker_scores) / len(same_speaker_scores) if same_speaker_scores else None

    # Different-speaker baseline: AVERAGE across ALL 7 other speakers, not
    # just one -- a single comparison speaker isn't a reliable floor.
    other_speakers = find_other_speakers(args.speaker)
    different_speaker_scores = []
    for other_speaker in other_speakers:
        other_clips = find_clips(other_speaker, n=2)
        for this_clip in same_speaker_clips[:2]:
            emb_this = embed(encoder, this_clip)
            for other_clip in other_clips:
                emb_other = embed(encoder, other_clip)
                different_speaker_scores.append(cosine_similarity(emb_this, emb_other))
    different_speaker_baseline = (sum(different_speaker_scores) / len(different_speaker_scores)
                                   if different_speaker_scores else None)

    # Channel-confound check: generated audio vs. SEVERAL DIFFERENT real clips
    # of the SAME speaker (not the exact reference used for cloning), AVERAGED
    # -- a single clip is too noisy to trust, same reasoning as the baselines.
    # If this stays high, that's real evidence of voice identity, not just
    # shared recording/channel characteristics from the reference clip itself.
    candidate_other_clips = [c for c in find_clips(args.speaker, n=args.n_other_clips + 3) if c != args.original]
    other_clips_used = candidate_other_clips[: args.n_other_clips]
    other_clip_scores = []
    for clip in other_clips_used:
        emb = embed(encoder, clip)
        score = cosine_similarity(generated_emb, emb)
        other_clip_scores.append(score)
        print(f"  [debug] {clip} -> {score:.3f}")
    other_clip_score = sum(other_clip_scores) / len(other_clip_scores) if other_clip_scores else None
    other_clip_std = np.std(other_clip_scores) if other_clip_scores else None

    print(f"\nSpeaker: {args.speaker}")
    print(f"Generated vs. original (reference clip) similarity: {generated_vs_original:.3f}")
    if other_clip_score is not None:
        print(f"Generated vs. {len(other_clip_scores)} DIFFERENT real {args.speaker} clips: "
              f"mean={other_clip_score:.3f}, std={other_clip_std:.3f}, "
              f"range=[{min(other_clip_scores):.3f}, {max(other_clip_scores):.3f}]")
        if other_clip_std is not None and same_speaker_baseline is not None and different_speaker_baseline is not None:
            baseline_span = same_speaker_baseline - different_speaker_baseline
            # Standard ERROR of the mean (std/sqrt(n)), not raw std -- this is
            # what actually determines whether the AVERAGE is trustworthy.
            # Raw std never shrinks with more samples; SEM does.
            sem = other_clip_std / (len(other_clip_scores) ** 0.5)
            if sem > baseline_span:
                print(f"  WARNING: standard error of the mean ({sem:.4f}) still exceeds the baseline signal "
                      f"span ({baseline_span:.3f}) even at n={len(other_clip_scores)}. The average is not yet "
                      f"trustworthy -- increase --n-other-clips further.")
    if same_speaker_baseline is not None:
        print(f"Same-speaker baseline (avg of {len(same_speaker_scores)} real pairs, same person): {same_speaker_baseline:.3f}")
    if different_speaker_baseline is not None:
        print(f"Different-speaker baseline (avg of {len(different_speaker_scores)} pairs, {args.speaker} vs all 7 others): {different_speaker_baseline:.3f}")

    print(f"\nChannel-confound check:")
    if other_clip_score is not None:
        drop = generated_vs_original - other_clip_score
        if drop > 0.05:
            print(f"  Similarity dropped by {drop:.3f} when averaged against {len(other_clip_scores)} DIFFERENT "
                  f"real clips instead of the exact reference used for cloning. This confirms the reference-clip "
                  f"comparison was inflated by shared recording characteristics, not pure voice identity -- "
                  f"use the score below instead.")
        else:
            print(f"  Similarity stayed about the same ({other_clip_score:.3f} vs {generated_vs_original:.3f}) "
                  f"even averaged against different real clips.")

    # The number to actually report: calculated from other_clip_score (the
    # UNCONFOUNDED, averaged comparison), never from generated_vs_original
    # (which is inflated by sharing recording characteristics with the exact
    # clip XTTS-v2 was conditioned on).
    print("\nReportable result (unconfounded):")
    if other_clip_score is not None and same_speaker_baseline is not None and different_speaker_baseline is not None:
        span = same_speaker_baseline - different_speaker_baseline
        if span > 0:
            raw_position = (other_clip_score - different_speaker_baseline) / span
            position = max(0.0, min(1.0, raw_position))

            sem = other_clip_std / (len(other_clip_scores) ** 0.5)
            ci_95 = 1.96 * sem
            position_ci = ci_95 / span * 100  # 95% CI half-width, in percentage-points

            print(f"  Voice similarity: {position*100:.0f}%  (95% CI: approx. {position*100:.0f}% "
                  f"+/- {position_ci:.0f} percentage points)")
            print(f"  (0% = as similar as a different person; 100% = as similar as {args.speaker}'s own "
                  f"real recordings are to each other. Based on {len(other_clip_scores)} different real clips, "
                  f"NOT the exact clip used for cloning.)")
            if position_ci > 20:
                print(f"\n  NOTE: this confidence interval is still wide. The point estimate ({position*100:.0f}%) "
                      f"is the best current guess, but report it WITH the interval, not as a bare number. "
                      f"Increasing --n-other-clips will narrow it further (CI shrinks with sqrt(n)).")
            if span < 0.05:
                print(f"\n  NOTE: the gap between the two baselines is only {span:.3f} -- this embedding model "
                      f"may not discriminate well between speakers on this dataset in general.")
    else:
        print("  Could not compute -- missing one or more required comparisons.")


if __name__ == "__main__":
    main()