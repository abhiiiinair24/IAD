"""
evaluate_pipeline_results.py

Computes WER/accuracy metrics from a test_all_speakers.py summary.csv, at
each stage of the pipeline:

    Stage 1 (ASR):        raw_transcription   vs true_reference
    Stage 2 (correction):  corrected_text      vs true_reference
    Stage 3 (voice clone): roundtrip_transcription vs corrected_text

Reuses compute_wer.score_batch() -- the same scorer used everywhere else in
this project -- so these numbers are directly comparable to your other
reported results (baseline, LOSO, all-speakers).

Usage:
    python evaluate_pipeline_results.py --summary-csv all_speakers_test_outputs_v3/summary.csv
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "evaluation"))
from compute_wer import score_batch


def main():
    parser = argparse.ArgumentParser(description="Evaluate a test_all_speakers.py summary.csv.")
    parser.add_argument("--summary-csv", required=True)
    args = parser.parse_args()

    rows = list(csv.DictReader(open(args.summary_csv, newline="")))
    rows = [r for r in rows if r.get("status") == "ok"]

    if not rows:
        print("No successful rows found in this summary file.")
        return

    true_refs = [r["true_reference"] for r in rows]
    raw_hyps = [r["raw_transcription"] for r in rows]
    corrected_hyps = [r["corrected_text"] for r in rows]
    roundtrip_hyps = [r["roundtrip_transcription"] for r in rows]

    print(f"Evaluating {len(rows)} speakers from {args.summary_csv}\n")

    stage1 = score_batch(true_refs, raw_hyps)
    print("Stage 1 -- ASR (raw transcription vs. true reference)")
    print(f"  WER: {stage1['wer']}   |   Exact-match accuracy: {sum(1 for a,b in zip(true_refs, raw_hyps) if a.strip().lower()==b.strip().lower())}/{len(rows)}")

    stage2 = score_batch(true_refs, corrected_hyps)
    print("\nStage 2 -- Text correction (corrected text vs. true reference)")
    print(f"  WER: {stage2['wer']}   |   Exact-match accuracy: {sum(1 for a,b in zip(true_refs, corrected_hyps) if a.strip().lower()==b.strip().lower())}/{len(rows)}")

    stage3 = score_batch(corrected_hyps, roundtrip_hyps)
    print("\nStage 3 -- Voice cloning fidelity (round-trip transcription vs. corrected text)")
    print(f"  WER: {stage3['wer']}   |   Exact-match accuracy: {sum(1 for a,b in zip(corrected_hyps, roundtrip_hyps) if a.strip().lower()==b.strip().lower())}/{len(rows)}")

    if "verified" in rows[0]:
        verified_count = sum(1 for r in rows if r.get("verified") in ("True", "true", True))
        avg_attempts = sum(int(r.get("attempts_used", 1)) for r in rows) / len(rows)
        print(f"\nVoice-cloning verification: {verified_count}/{len(rows)} verified correct")
        print(f"Average attempts used: {avg_attempts:.2f}")

    print("\n" + "=" * 60)
    print("Interpretation:")
    print(f"  Stage 1->2 WER change: {stage1['wer']:.3f} -> {stage2['wer']:.3f} "
          f"({'improved' if stage2['wer'] < stage1['wer'] else 'no improvement' if stage2['wer']==stage1['wer'] else 'REGRESSED'})")
    print(f"  Stage 3 WER close to 0 means voice cloning faithfully reproduced the corrected text's words.")


if __name__ == "__main__":
    main()
