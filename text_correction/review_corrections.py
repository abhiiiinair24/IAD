"""
review_corrections.py

Runs the trained correction model on a sample of real val.csv rows and prints
hypothesis / corrected / reference side by side, so you can eyeball actual
correction quality instead of relying on the aggregate WER number alone.

Usage:
    python review_corrections.py --model-dir model/final --val-csv data/val.csv --n 15
"""

import argparse
import csv
import random

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

TASK_PREFIX = "fix asr transcription: "


def correct(text, tokenizer, model):
    inputs = tokenizer(TASK_PREFIX + text, return_tensors="pt", truncation=True, max_length=128)
    output_ids = model.generate(**inputs, max_length=128, num_beams=4)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--val-csv", required=True)
    parser.add_argument("--n", type=int, default=15)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir)

    rows = list(csv.DictReader(open(args.val_csv, newline="")))
    random.Random(args.seed).shuffle(rows)
    rows = rows[:args.n]

    unchanged = 0
    matched_ref = 0
    for row in rows:
        corrected = correct(row["hypothesis"], tokenizer, model)
        if corrected.strip().lower() == row["hypothesis"].strip().lower():
            unchanged += 1
        if corrected.strip().lower() == row["reference"].strip().lower():
            matched_ref += 1
        print(f"HYP: {row['hypothesis']}")
        print(f"COR: {corrected}")
        print(f"REF: {row['reference']}")
        print("---")

    print(f"\nOf {len(rows)} sampled: {unchanged} left unchanged, {matched_ref} exactly matched reference")


if __name__ == "__main__":
    main()
