"""
compute_wer.py

Model-agnostic WER/CER scoring. Import this module from any evaluation or
training script (Whisper, Wav2Vec2, WavLM, ...) rather than reimplementing
text normalization per model — keeping the metric consistent across every
model you compare is what makes the comparison fair.

Requires: pip install jiwer --break-system-packages

Usage as a library:
    from compute_wer import normalize_text, score_batch

    result = score_batch(references, hypotheses)
    print(result["wer"], result["cer"])

Usage as a CLI (scores a results CSV with columns: reference, hypothesis):
    python compute_wer.py --results-csv ../results/whisper_baseline/fold_F01.csv
"""

import argparse
import csv
import re
from pathlib import Path

import jiwer


def normalize_text(text: str) -> str:
    """
    Lowercase, strip punctuation, collapse whitespace. Applied identically to
    references and hypotheses before scoring so differences in casing/punctuation
    don't inflate WER for models that just format output differently.
    """
    text = text.lower()
    text = re.sub(r"[^\w\s']", " ", text)  # strip punctuation except apostrophes (contractions)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def score_batch(references: list[str], hypotheses: list[str]) -> dict:
    """
    Computes corpus-level WER and CER (not averaged per-utterance WER, which
    over-weights short utterances — corpus-level pools all words/chars first).
    Returns a dict with wer, cer, and counts for transparency.
    """
    if len(references) != len(hypotheses):
        raise ValueError(f"Mismatched lengths: {len(references)} references vs {len(hypotheses)} hypotheses")
    if not references:
        return {"wer": None, "cer": None, "num_utterances": 0}

    norm_refs = [normalize_text(r) for r in references]
    norm_hyps = [normalize_text(h) for h in hypotheses]

    # Drop pairs where the reference is empty after normalization (nothing to score against)
    pairs = [(r, h) for r, h in zip(norm_refs, norm_hyps) if r]
    if not pairs:
        return {"wer": None, "cer": None, "num_utterances": 0}
    filtered_refs, filtered_hyps = zip(*pairs)

    wer = jiwer.wer(list(filtered_refs), list(filtered_hyps))
    cer = jiwer.cer(list(filtered_refs), list(filtered_hyps))

    return {
        "wer": round(wer, 4),
        "cer": round(cer, 4),
        "num_utterances": len(filtered_refs),
        "num_skipped_empty_ref": len(references) - len(filtered_refs),
    }


def main():
    parser = argparse.ArgumentParser(description="Score a results CSV (columns: reference, hypothesis).")
    parser.add_argument("--results-csv", required=True)
    args = parser.parse_args()

    path = Path(args.results_csv).expanduser()
    references, hypotheses = [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            references.append(row["reference"])
            hypotheses.append(row["hypothesis"])

    result = score_batch(references, hypotheses)
    print(f"Scored {path}")
    print(f"  Utterances scored: {result['num_utterances']}")
    if result.get("num_skipped_empty_ref"):
        print(f"  Skipped (empty reference): {result['num_skipped_empty_ref']}")
    print(f"  WER: {result['wer']}")
    print(f"  CER: {result['cer']}")


if __name__ == "__main__":
    main()
