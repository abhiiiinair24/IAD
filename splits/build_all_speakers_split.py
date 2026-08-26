"""
build_all_speakers_split.py

Builds a single train/test split using ALL 8 speakers, for training one
combined "base" model (as opposed to build_loso_splits.py's zero-shot,
one-speaker-held-out folds).

IMPORTANT - this answers a DIFFERENT question than the LOSO folds:
    LOSO WER  = "how well does this generalize to a brand-new, never-seen speaker"
    This split's WER = "how well does this model work for the 8 speakers it was
                        actually trained on" (same speakers in train and test,
                        just different utterances)
These two numbers are NOT directly comparable -- this one will be easier and
should come out lower, since the model has already heard each test speaker's
voice during training. Report them as separate metrics, not as a single
before/after story.

Split is done PER SPEAKER (not pooled-then-shuffled) so each speaker
contributes their own ~80/20 split -- this guarantees every speaker appears
in both train and test, in roughly their natural proportion.

Usage:
    python build_all_speakers_split.py \
        --manifests ../manifests/manifest_female.csv ../manifests/manifest_male.csv \
        --output-dir ../splits/all_speakers \
        --test-fraction 0.2
"""

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path


def load_usable_rows(manifest_paths: list[Path]) -> list[dict]:
    rows = []
    for path in manifest_paths:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                if row["status"] == "ok":
                    rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Build a per-speaker 80/20 all-speakers split.")
    parser.add_argument("--manifests", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    manifest_paths = [Path(p).expanduser() for p in args.manifests]
    rows = load_usable_rows(manifest_paths)
    if not rows:
        print("No usable rows found.")
        return

    fieldnames = list(rows[0].keys())
    by_speaker = defaultdict(list)
    for row in rows:
        by_speaker[row["speaker"]].append(row)

    train_rows, test_rows = [], []
    rng = random.Random(args.seed)

    print("Per-speaker split:")
    for speaker in sorted(by_speaker):
        speaker_rows = by_speaker[speaker].copy()
        rng.shuffle(speaker_rows)
        n_test = max(1, int(len(speaker_rows) * args.test_fraction))
        test_rows.extend(speaker_rows[:n_test])
        train_rows.extend(speaker_rows[n_test:])
        print(f"  {speaker}: train={len(speaker_rows) - n_test}  test={n_test}")

    output_dir = Path(args.output_dir).expanduser()
    write_csv(output_dir / "train.csv", train_rows, fieldnames)
    write_csv(output_dir / "test.csv", test_rows, fieldnames)

    train_hours = sum(float(r["duration_sec"]) for r in train_rows if float(r["duration_sec"]) > 0) / 3600
    test_hours = sum(float(r["duration_sec"]) for r in test_rows if float(r["duration_sec"]) > 0) / 3600

    print(f"\nTotal train: {len(train_rows)} utterances ({train_hours:.2f}h)")
    print(f"Total test:  {len(test_rows)} utterances ({test_hours:.2f}h)")
    print(f"Written to {output_dir}")


if __name__ == "__main__":
    main()
