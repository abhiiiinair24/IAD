"""
diagnose_bad_audio.py

Scans a manifest/split CSV for audio files likely to cause NaN/Inf loss during
training: silent (all-zero or near-zero amplitude), extremely short, or
unreadable. Run this BEFORE retrying a fold that diverged during training.

Usage:
    python diagnose_bad_audio.py --csv ../splits/fold_F04/train.csv
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import soundfile as sf


def main():
    parser = argparse.ArgumentParser(description="Find degenerate audio files that could cause training instability.")
    parser.add_argument("--csv", required=True, help="Manifest or split CSV with a wav_path column")
    parser.add_argument("--min-duration", type=float, default=0.1, help="Flag files shorter than this (seconds)")
    parser.add_argument("--max-abs-amplitude-threshold", type=float, default=1e-4,
                         help="Flag files whose max absolute sample value is below this (near-silent)")
    args = parser.parse_args()

    rows = list(csv.DictReader(open(args.csv, newline="")))
    problems = []

    for row in rows:
        wav_path = row.get("wav_path", "")
        if not wav_path:
            continue
        try:
            audio, sr = sf.read(wav_path)
        except Exception as e:
            problems.append((wav_path, f"unreadable: {e}"))
            continue

        if len(audio) == 0:
            problems.append((wav_path, "zero-length audio array"))
            continue

        duration = len(audio) / sr
        if duration < args.min_duration:
            problems.append((wav_path, f"very short: {duration:.4f}s"))

        max_amp = float(np.max(np.abs(audio)))
        if max_amp < args.max_abs_amplitude_threshold:
            problems.append((wav_path, f"near-silent: max amplitude {max_amp:.2e}"))

        if np.isnan(audio).any():
            problems.append((wav_path, "contains NaN samples"))
        if np.isinf(audio).any():
            problems.append((wav_path, "contains Inf samples"))

    print(f"Scanned {len(rows)} files from {args.csv}")
    if not problems:
        print("No problems found.")
        return

    print(f"\nFound {len(problems)} potentially problematic files:")
    for path, reason in problems:
        print(f"  {path}  ->  {reason}")


if __name__ == "__main__":
    main()
