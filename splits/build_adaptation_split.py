"""
build_adaptation_split.py

Takes ONE LOSO fold's test.csv (the held-out speaker's full data, currently
untouched by training) and splits it into two pieces:
    - adapt.csv (written as train.csv) -- a small slice used to further
      fine-tune the LOSO model on this specific speaker's own voice
    - test.csv -- the REMAINING data, still never touched by any training,
      used to measure how much adaptation actually helped

Output is written with filenames (train.csv / test.csv) that match what
finetune_whisper.py and run_finetuned_eval.py already expect, so both
existing scripts work unchanged against this new split directory.

Usage:
    python build_adaptation_split.py \
        --fold-dir ../splits/fold_M04 \
        --adapt-fraction 0.2 \
        --output-dir ../splits/adaptation/M04
"""

import argparse
import csv
import random
from pathlib import Path


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Split a LOSO fold's held-out speaker data into adapt/test slices.")
    parser.add_argument("--fold-dir", required=True, help="e.g. ../splits/fold_M04 (its test.csv is the source)")
    parser.add_argument("--adapt-fraction", type=float, default=0.2,
                         help="Fraction of the held-out speaker's data to use for adaptation (default 0.2)")
    parser.add_argument("--output-dir", required=True, help="e.g. ../splits/adaptation/M04")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    fold_dir = Path(args.fold_dir).expanduser()
    speaker = fold_dir.name.replace("fold_", "")
    test_csv = fold_dir / "test.csv"

    with open(test_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print(f"No rows found in {test_csv}")
        return

    fieldnames = list(rows[0].keys())
    rng = random.Random(args.seed)
    rng.shuffle(rows)

    n_adapt = max(1, int(len(rows) * args.adapt_fraction))
    adapt_rows = rows[:n_adapt]
    final_test_rows = rows[n_adapt:]

    output_dir = Path(args.output_dir).expanduser()
    # Named train.csv / test.csv deliberately -- matches what
    # finetune_whisper.py and run_finetuned_eval.py already expect.
    write_csv(output_dir / "train.csv", adapt_rows, fieldnames)
    write_csv(output_dir / "test.csv", final_test_rows, fieldnames)

    adapt_hours = sum(float(r["duration_sec"]) for r in adapt_rows if float(r["duration_sec"]) > 0) / 3600
    test_hours = sum(float(r["duration_sec"]) for r in final_test_rows if float(r["duration_sec"]) > 0) / 3600

    print(f"Speaker: {speaker}")
    print(f"  Adaptation slice: {len(adapt_rows)} utterances ({adapt_hours:.3f}h) -> {output_dir / 'train.csv'}")
    print(f"  Final test slice: {len(final_test_rows)} utterances ({test_hours:.3f}h) -> {output_dir / 'test.csv'}")
    print(f"  (Final test slice is still untouched by any training -- this is what you evaluate adaptation on.)")


if __name__ == "__main__":
    main()
