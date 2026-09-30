"""
evaluate_phonetic_correction.py

Runs phonetic_correct() across every SHORT utterance (<= max_words) in a
hypothesis/reference CSV (e.g. data/val.csv) and reports what actually
happened -- not just one or two anecdotes. Every case falls into exactly one
bucket:

    already_correct_left_alone   hypothesis == reference, and we correctly left it unchanged
    already_correct_broken       hypothesis == reference, but correction WRONGLY changed it (bad -- a regression)
    wrong_fixed                  hypothesis != reference, and correction fixed it to match reference (good)
    wrong_still_wrong_changed    hypothesis != reference, correction changed it, but not to the right answer
    wrong_unchanged              hypothesis != reference, and correction left it as-is (no help, but no harm)

Usage:
    python evaluate_phonetic_correction.py --val-csv data/val.csv --vocab vocab.txt --max-words 2
"""

import argparse
import csv
from collections import Counter

from phonetic_correct import load_vocab, phonetic_correct


def main():
    parser = argparse.ArgumentParser(description="Bulk-evaluate the phonetic correction module.")
    parser.add_argument("--val-csv", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--max-words", type=int, default=2)
    parser.add_argument("--show-examples", action="store_true", help="Print every case, not just the summary")
    args = parser.parse_args()

    vocab = load_vocab(args.vocab)
    rows = list(csv.DictReader(open(args.val_csv, newline="")))

    short_rows = [r for r in rows if len(r["hypothesis"].strip().split()) <= args.max_words]
    print(f"{len(short_rows)} of {len(rows)} rows are short (<= {args.max_words} words) -- these are what phonetic correction targets")

    counts = Counter()
    for row in short_rows:
        hyp = row["hypothesis"].strip()
        ref = row["reference"].strip()
        corrected, was_corrected = phonetic_correct(hyp, vocab, args.max_words)

        hyp_was_correct = hyp.lower() == ref.lower()
        corrected_is_right = corrected.lower() == ref.lower()

        if hyp_was_correct and not was_corrected:
            bucket = "already_correct_left_alone"
        elif hyp_was_correct and was_corrected:
            bucket = "already_correct_broken"
        elif not hyp_was_correct and corrected_is_right:
            bucket = "wrong_fixed"
        elif not hyp_was_correct and was_corrected:
            bucket = "wrong_still_wrong_changed"
        else:
            bucket = "wrong_unchanged"
        counts[bucket] += 1

        if args.show_examples:
            print(f"[{bucket}]  hyp={hyp!r}  corrected={corrected!r}  ref={ref!r}")

    print("\nSummary:")
    for bucket in ["already_correct_left_alone", "already_correct_broken",
                   "wrong_fixed", "wrong_still_wrong_changed", "wrong_unchanged"]:
        print(f"  {bucket}: {counts[bucket]}")

    net_helped = counts["wrong_fixed"]
    net_harmed = counts["already_correct_broken"]
    print(f"\nNet effect: fixed {net_helped} genuine errors, broke {net_harmed} previously-correct answers")


if __name__ == "__main__":
    main()
