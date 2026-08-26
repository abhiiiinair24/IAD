"""
summarize_results.py

Merges baseline_summary.csv and finetuned_summary.csv (both keyed by speaker)
into one comparison table, computes macro-averaged WER/CER for each, and
prints + writes a clean summary suitable for a report, slide, or README.

Usage:
    python summarize_results.py \
        --baseline ../results/baseline_whisper/baseline_summary.csv \
        --finetuned ../results/finetuned_whisper_medium/finetuned_summary.csv \
        --output ../results/comparison_summary.md
"""

import argparse
import csv
from pathlib import Path


def load_summary(path: Path) -> dict:
    with open(path, newline="") as f:
        return {row["speaker"]: row for row in csv.DictReader(f)}


def macro_average(rows: dict, key: str) -> float:
    values = [float(r[key]) for r in rows.values() if r.get(key) not in (None, "", "None")]
    return sum(values) / len(values) if values else float("nan")


def main():
    parser = argparse.ArgumentParser(description="Summarize baseline vs fine-tuned WER/CER results.")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--finetuned", required=True)
    parser.add_argument("--all-speakers", default=None,
                         help="Optional: finetuned_summary.csv from the all-speakers (seen-speaker) run")
    parser.add_argument("--output", required=True, help="Path to write a markdown summary")
    args = parser.parse_args()

    baseline = load_summary(Path(args.baseline).expanduser())
    finetuned = load_summary(Path(args.finetuned).expanduser())

    speakers = sorted(set(baseline) | set(finetuned))

    lines = []
    lines.append("# Baseline vs Fine-tuned Whisper — Zero-shot LOSO Results\n")
    lines.append("| Speaker | Baseline WER | Fine-tuned WER | Improvement (pts) | Baseline CER | Fine-tuned CER |")
    lines.append("|---|---|---|---|---|---|")

    for sp in speakers:
        b = baseline.get(sp)
        f = finetuned.get(sp)
        b_wer = float(b["wer"]) * 100 if b else None
        f_wer = float(f["wer"]) * 100 if f else None
        b_cer = float(b["cer"]) * 100 if b else None
        f_cer = float(f["cer"]) * 100 if f else None
        improvement = (b_wer - f_wer) if (b_wer is not None and f_wer is not None) else None

        row = [
            sp,
            f"{b_wer:.1f}%" if b_wer is not None else "-",
            f"{f_wer:.1f}%" if f_wer is not None else "-",
            f"-{improvement:.1f}" if improvement is not None else "-",
            f"{b_cer:.1f}%" if b_cer is not None else "-",
            f"{f_cer:.1f}%" if f_cer is not None else "-",
        ]
        lines.append("| " + " | ".join(row) + " |")

    macro_baseline_wer = macro_average(baseline, "wer") * 100
    macro_finetuned_wer = macro_average(finetuned, "wer") * 100
    macro_baseline_cer = macro_average(baseline, "cer") * 100
    macro_finetuned_cer = macro_average(finetuned, "cer") * 100
    macro_improvement = macro_baseline_wer - macro_finetuned_wer

    lines.append("")
    lines.append(f"**Macro-average WER: {macro_baseline_wer:.2f}% → {macro_finetuned_wer:.2f}%  "
                 f"(-{macro_improvement:.2f} points)**")
    lines.append(f"**Macro-average CER: {macro_baseline_cer:.2f}% → {macro_finetuned_cer:.2f}%**")
    lines.append(f"\nBased on {len(speakers)} speakers, zero-shot Leave-One-Speaker-Out evaluation.")

    all_speakers_wer = None
    if args.all_speakers:
        all_sp = load_summary(Path(args.all_speakers).expanduser())
        if all_sp:
            row = next(iter(all_sp.values()))
            all_speakers_wer = float(row["wer"]) * 100
            all_speakers_cer = float(row["cer"]) * 100

            lines.append("\n---\n")
            lines.append("## Three-tier comparison\n")
            lines.append("| Model | WER | What it answers |")
            lines.append("|---|---|---|")
            lines.append(f"| Baseline (pretrained, zero-shot) | {macro_baseline_wer:.2f}% | "
                         f"How well does an out-of-the-box model handle dysarthric speech at all? |")
            lines.append(f"| Fine-tuned, LOSO zero-shot (macro-avg, {len(speakers)} folds) | {macro_finetuned_wer:.2f}% | "
                         f"How well does fine-tuning generalize to a brand-new, never-seen speaker? |")
            lines.append(f"| Fine-tuned, all-speakers (seen-speaker test) | {all_speakers_wer:.2f}% | "
                         f"How well does the system work for the specific people it was trained on? |")
            lines.append("")
            gap = macro_finetuned_wer - all_speakers_wer
            lines.append(f"**Note:** the all-speakers result ({all_speakers_wer:.2f}%) is NOT directly comparable "
                         f"to the LOSO zero-shot result above it — it answers a different question (performance on "
                         f"speakers seen during training, not generalization to a new speaker). The "
                         f"{gap:.2f}-point gap between the two is a rough upper bound on how much *knowing the "
                         f"specific person* could still help beyond generic dysarthric-speech fine-tuning alone — "
                         f"a natural target for a future speaker-adaptation experiment.")

    summary_text = "\n".join(lines)
    print(summary_text)

    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(summary_text)
    print(f"\nWritten to {output_path}")


if __name__ == "__main__":
    main()