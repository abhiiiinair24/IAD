"""
build_correction_dataset.py

Aggregates (hypothesis, reference) pairs from your existing evaluation result
CSVs into a train/val set for training a text-correction model -- one that
learns to map noisy ASR output back toward the correct sentence.

Deliberately EXCLUDES results/baseline_whisper/ -- those are from the
unrelated pretrained whisper-small model and include severe hallucinations
(WER > 100%) that don't represent the error patterns your deployed
fine-tuned model actually produces. Training on that noise would teach the
correction model to "fix" errors it will never actually see in deployment.

Includes, by default:
    results/finetuned_whisper_medium/*.csv          (LOSO zero-shot results)
    results/finetuned_whisper_medium_all_speakers/*.csv  (deployed model's own results)
    results/adapted_whisper_medium/*.csv             (adaptation experiment results)

Usage:
    python build_correction_dataset.py \
        --results-dirs ../results/finetuned_whisper_medium ../results/finetuned_whisper_medium_all_speakers ../results/adapted_whisper_medium \
        --output-dir ../text_correction/data
"""

import argparse
import csv
import random
from pathlib import Path


def load_pairs(results_dirs: list[Path]) -> list[dict]:
    pairs = []
    for d in results_dirs:
        if not d.exists():
            print(f"[WARN] Directory not found, skipping: {d}")
            continue
        for csv_path in sorted(d.glob("*.csv")):
            if csv_path.name.endswith("_summary.csv"):
                continue  # summary files, not per-utterance hypothesis/reference pairs
            with open(csv_path, newline="") as f:
                reader = csv.DictReader(f)
                if "hypothesis" not in reader.fieldnames or "reference" not in reader.fieldnames:
                    continue
                for row in reader:
                    hyp = row["hypothesis"].strip()
                    ref = row["reference"].strip()
                    if hyp and ref:
                        pairs.append({"hypothesis": hyp, "reference": ref, "source": csv_path.name})
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Build a training set for the text-correction model.")
    parser.add_argument("--results-dirs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    results_dirs = [Path(d).expanduser() for d in args.results_dirs]
    pairs = load_pairs(results_dirs)

    if not pairs:
        print("No hypothesis/reference pairs found. Check --results-dirs.")
        return

    # De-duplicate identical (hypothesis, reference) pairs (e.g. the same utterance
    # appearing in both a LOSO fold's results and the all-speakers results).
    seen = set()
    unique_pairs = []
    for row in pairs:
        key = (row["hypothesis"], row["reference"])
        if key not in seen:
            seen.add(key)
            unique_pairs.append(row)

    exact_matches = sum(1 for r in unique_pairs if r["hypothesis"] == r["reference"])
    print(f"Total pairs: {len(pairs)} -> {len(unique_pairs)} after de-duplication")
    print(f"  Already-correct pairs (hypothesis == reference): {exact_matches} "
          f"(kept -- teaches the model not to 'fix' things that aren't broken)")

    rng = random.Random(args.seed)
    rng.shuffle(unique_pairs)
    n_val = max(1, int(len(unique_pairs) * args.val_fraction))
    val_rows = unique_pairs[:n_val]
    train_rows = unique_pairs[n_val:]

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in [("train.csv", train_rows), ("val.csv", val_rows)]:
        with open(output_dir / name, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["hypothesis", "reference", "source"])
            writer.writeheader()
            writer.writerows(rows)

    print(f"\nTrain: {len(train_rows)} pairs -> {output_dir / 'train.csv'}")
    print(f"Val:   {len(val_rows)} pairs -> {output_dir / 'val.csv'}")


if __name__ == "__main__":
    main()
