"""
build_loso_splits.py

Builds zero-shot Leave-One-Speaker-Out (LOSO) fold manifests from one or more
manifest CSVs (e.g. manifest_female.csv + manifest_male.csv).

For each speaker found in the combined data, creates a fold where:
    - train.csv = all usable (status=ok) utterances from every OTHER speaker
    - test.csv  = all usable (status=ok) utterances from THIS speaker

This is zero-shot by construction: the held-out speaker's data appears in
exactly one file (test.csv) and never in any train.csv. No held-out speaker
data is ever mixed into training for their own fold.

Usage:
    python build_loso_splits.py \
        --manifests ../manifests/manifest_female.csv ../manifests/manifest_male.csv \
        --output-dir ../splits

Output:
    splits/fold_<SPEAKER>/train.csv
    splits/fold_<SPEAKER>/test.csv
    splits/fold_summary.csv   <- row counts + hours per fold, for a sanity check
"""

import argparse
import csv
from pathlib import Path
from collections import defaultdict


def load_usable_rows(manifest_paths: list[Path]) -> list[dict]:
    rows = []
    for path in manifest_paths:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
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
    parser = argparse.ArgumentParser(description="Build zero-shot LOSO fold manifests.")
    parser.add_argument("--manifests", nargs="+", required=True, help="One or more manifest CSVs to combine")
    parser.add_argument("--output-dir", required=True, help="Directory to write fold_<speaker>/{train,test}.csv into")
    args = parser.parse_args()

    manifest_paths = [Path(p).expanduser() for p in args.manifests]
    for p in manifest_paths:
        if not p.exists():
            raise FileNotFoundError(f"Manifest not found: {p}")

    rows = load_usable_rows(manifest_paths)
    if not rows:
        print("No usable (status=ok) rows found across the given manifests. Nothing to split.")
        return

    fieldnames = list(rows[0].keys())
    speakers = sorted({r["speaker"] for r in rows})
    output_dir = Path(args.output_dir).expanduser()

    summary_rows = []
    for held_out in speakers:
        test_rows = [r for r in rows if r["speaker"] == held_out]
        train_rows = [r for r in rows if r["speaker"] != held_out]

        # Safety check: guarantee zero-shot property before writing anything.
        train_speakers = {r["speaker"] for r in train_rows}
        assert held_out not in train_speakers, (
            f"LOSO violation: held-out speaker {held_out} leaked into its own training set."
        )

        fold_dir = output_dir / f"fold_{held_out}"
        write_csv(fold_dir / "train.csv", train_rows, fieldnames)
        write_csv(fold_dir / "test.csv", test_rows, fieldnames)

        train_hours = sum(float(r["duration_sec"]) for r in train_rows if float(r["duration_sec"]) > 0) / 3600
        test_hours = sum(float(r["duration_sec"]) for r in test_rows if float(r["duration_sec"]) > 0) / 3600

        summary_rows.append({
            "held_out_speaker": held_out,
            "train_utterances": len(train_rows),
            "train_hours": round(train_hours, 3),
            "test_utterances": len(test_rows),
            "test_hours": round(test_hours, 3),
            "train_speakers": ",".join(sorted(train_speakers)),
        })

        print(f"fold_{held_out}: train={len(train_rows)} ({train_hours:.2f}h) "
              f"| test={len(test_rows)} ({test_hours:.2f}h)")

    summary_path = output_dir / "fold_summary.csv"
    write_csv(summary_path, summary_rows,
              ["held_out_speaker", "train_utterances", "train_hours",
               "test_utterances", "test_hours", "train_speakers"])

    print(f"\n{len(speakers)} LOSO folds written to {output_dir}")
    print(f"Fold summary: {summary_path}")


if __name__ == "__main__":
    main()
