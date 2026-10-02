"""
voice_similarity_general.py

Generalizes voice_similarity_ecapa.py to work for ANY speaker, not just the
8 TORGO speakers -- including a brand-new person recorded at inference time.

Two different things are needed, and they behave differently:

    1. "Different speaker" (stranger) baseline -- a general property of voices
       in general, not specific to any one new person. Computed ONCE from a
       fixed reference pool and CACHED to disk -- never recomputed per speaker.

    2. "Same speaker" baseline -- genuinely requires MULTIPLE real recordings
       of THIS SPECIFIC new person. This cannot be skipped or approximated:
       you cannot measure "does the clone sound like them" without real
       recordings of them to compare against. One clip is not enough (a
       single comparison is noisy -- see the project notes on n=5 vs n=40).

Practical requirement for a new speaker: provide at least ~6-10 short real
recordings upfront. Some are used as the voice-cloning reference; the REST
must be held out (never used for cloning) to serve as the comparison pool
here. More held-out clips = a tighter, more trustworthy percentage; fewer
clips still work, but the script reports a wider (honest) confidence
interval instead of a falsely precise number.

Usage:
    # One-time setup (run once, reused forever after):
    python voice_similarity_general.py --build-stranger-baseline

    # For any speaker (existing TORGO or a brand new person):
    python voice_similarity_general.py \
        --generated-audio /path/to/synthesized_output.wav \
        --held-out-clips clip1.wav clip2.wav clip3.wav clip4.wav clip5.wav
"""

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier

BASE_DIR = Path(__file__).parent.parent
MANIFEST_FEMALE = BASE_DIR / "manifests" / "manifest_female.csv"
MANIFEST_MALE = BASE_DIR / "manifests" / "manifest_male.csv"
STRANGER_CACHE_PATH = Path(__file__).parent / "stranger_baseline_cache.json"


def load_classifier():
    return EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb", savedir="/tmp/spkrec-ecapa-voxceleb"
    )


def embed(classifier, wav_path: str) -> np.ndarray:
    signal, sr = torchaudio.load(wav_path)
    if signal.shape[0] > 1:
        signal = signal.mean(dim=0, keepdim=True)
    if sr != 16000:
        signal = torchaudio.functional.resample(signal, sr, 16000)
    with torch.no_grad():
        emb = classifier.encode_batch(signal)
    return emb.squeeze().cpu().numpy()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def build_stranger_baseline(classifier, n_pairs: int = 100) -> dict:
    """
    Computes the GENERIC different-speaker baseline ONCE, using the 8 TORGO
    speakers as a diverse reference pool, and caches it to disk. This is
    reused for ALL future speakers (new or existing) -- never recomputed.

    Caveat, stated honestly: this pool is calibrated on TORGO's dysarthric
    speakers specifically. It's a reasonable general "stranger" floor, but
    may not perfectly match the true floor for a speaker with very different
    voice characteristics (e.g. a child, a very different accent). Treat it
    as a practical approximation, not an exact calibration for every case.
    """
    print(f"Building stranger baseline from {n_pairs} cross-speaker pairs (one-time setup)...")
    all_speakers = ["F01", "F03", "F04", "M01", "M02", "M03", "M04", "M05"]
    clips_by_speaker = {}
    for speaker in all_speakers:
        clips = []
        for path in [MANIFEST_FEMALE, MANIFEST_MALE]:
            with open(path, newline="") as f:
                for row in csv.DictReader(f):
                    if row["speaker"] == speaker and row["status"] == "ok" and float(row["duration_sec"]) >= 3.0:
                        clips.append(row["wav_path"])
        random.Random(42).shuffle(clips)
        clips_by_speaker[speaker] = clips[:5]

    scores = []
    rng = random.Random(42)
    attempts = 0
    while len(scores) < n_pairs and attempts < n_pairs * 3:
        attempts += 1
        sp_a, sp_b = rng.sample(all_speakers, 2)
        if not clips_by_speaker[sp_a] or not clips_by_speaker[sp_b]:
            continue
        clip_a = rng.choice(clips_by_speaker[sp_a])
        clip_b = rng.choice(clips_by_speaker[sp_b])
        scores.append(cosine_similarity(embed(classifier, clip_a), embed(classifier, clip_b)))

    result = {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "n_pairs": len(scores),
    }
    STRANGER_CACHE_PATH.write_text(json.dumps(result, indent=2))
    print(f"Stranger baseline: mean={result['mean']:.3f}, std={result['std']:.3f}, n={result['n_pairs']}")
    print(f"Cached to {STRANGER_CACHE_PATH} -- will be reused automatically from now on.")
    return result


def load_stranger_baseline() -> dict:
    if not STRANGER_CACHE_PATH.exists():
        raise FileNotFoundError(
            f"No cached stranger baseline found. Run once with --build-stranger-baseline first "
            f"(this only needs to happen one time, ever)."
        )
    return json.loads(STRANGER_CACHE_PATH.read_text())


def main():
    parser = argparse.ArgumentParser(description="Voice similarity evaluation for ANY speaker, new or existing.")
    parser.add_argument("--build-stranger-baseline", action="store_true",
                         help="One-time setup: computes and caches the reusable stranger baseline.")
    parser.add_argument("--generated-audio", default=None, help="The synthesized output to evaluate")
    parser.add_argument("--held-out-clips", nargs="*", default=None,
                         help="Real recordings of this speaker, NOT used for voice cloning. "
                              "Need at least 2; more gives a tighter confidence interval.")
    args = parser.parse_args()

    classifier = load_classifier()

    if args.build_stranger_baseline:
        build_stranger_baseline(classifier)
        return

    if not args.generated_audio or not args.held_out_clips:
        print("Provide --generated-audio and --held-out-clips (at least 2 real clips of this speaker).")
        return
    if len(args.held_out_clips) < 2:
        print("WARNING: fewer than 2 held-out clips -- the result will be very unreliable. "
              "Provide more real recordings of this speaker if at all possible.")

    stranger = load_stranger_baseline()

    generated_emb = embed(classifier, args.generated_audio)
    held_out_embs = [embed(classifier, c) for c in args.held_out_clips]

    # "Same speaker" baseline: real held-out clips compared to EACH OTHER
    same_speaker_scores = []
    for i in range(len(held_out_embs)):
        for j in range(i + 1, len(held_out_embs)):
            same_speaker_scores.append(cosine_similarity(held_out_embs[i], held_out_embs[j]))
    same_speaker_mean = sum(same_speaker_scores) / len(same_speaker_scores) if same_speaker_scores else None

    # Generated audio vs. each held-out real clip
    test_scores = [cosine_similarity(generated_emb, e) for e in held_out_embs]
    test_mean = sum(test_scores) / len(test_scores)
    test_std = np.std(test_scores)

    print(f"\nGenerated vs. {len(test_scores)} held-out real clips: mean={test_mean:.3f}, std={test_std:.3f}")
    if same_speaker_mean is not None:
        print(f"Same-speaker baseline (held-out clips vs. each other, n={len(same_speaker_scores)} pairs): "
              f"{same_speaker_mean:.3f}")
    print(f"Stranger baseline (cached, n={stranger['n_pairs']} pairs): {stranger['mean']:.3f}")

    if same_speaker_mean is not None:
        span = same_speaker_mean - stranger["mean"]
        if span > 0:
            position = max(0.0, min(1.0, (test_mean - stranger["mean"]) / span))
            sem = test_std / (len(test_scores) ** 0.5)
            ci = 1.96 * sem / span * 100
            print(f"\nVoice similarity: {position*100:.0f}%  (95% CI: +/- {ci:.0f} percentage points, "
                  f"based on {len(test_scores)} held-out clips)")
            if len(test_scores) < 10:
                print(f"NOTE: fewer than 10 held-out clips were used -- this confidence interval is likely "
                      f"wide. Add more held-out real recordings of this speaker for a tighter estimate.")
        else:
            print("\nCould not compute a meaningful baseline span (too few held-out clips, or they were too "
                  "similar/dissimilar to establish a reliable same-speaker reference).")
    else:
        print("\nNeed at least 2 held-out clips to establish a same-speaker baseline.")


if __name__ == "__main__":
    main()