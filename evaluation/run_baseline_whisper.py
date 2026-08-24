"""
run_baseline_whisper.py

Runs a PRETRAINED (not fine-tuned) Whisper model on every LOSO fold's test
set and scores it with compute_wer.py. This is the "how far is it from the
actual transcription" baseline number, before any fine-tuning happens.

Since no fine-tuning is involved, the same pretrained model is used for every
fold — but we still score it per-fold (using each fold's test.csv) so the
baseline numbers are directly comparable to the fine-tuned LOSO numbers later
on the exact same test sets.

Requires:
    pip install transformers torch soundfile jiwer accelerate --break-system-packages

Usage:
    python run_baseline_whisper.py \
        --splits-dir ../splits \
        --model-size small \
        --output-dir ../results/baseline_whisper

Whisper model sizes (tradeoff: accuracy vs speed/VRAM):
    tiny, base, small, medium, large-v3
    'small' is a reasonable default for a 4090 — good accuracy, fast enough
    to iterate. Use 'medium' or 'large-v3' for a stronger final baseline once
    the pipeline is validated end-to-end on 'small'.
"""

import argparse
import csv
import sys
from pathlib import Path

import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
from compute_wer import score_batch


def load_model(model_size: str, device: str):
    model_id = f"openai/whisper-{model_size}"
    print(f"Loading {model_id} ...")
    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id).to(device)
    model.eval()
    return processor, model


def transcribe(wav_path: str, processor, model, device: str) -> str:
    audio, sr = sf.read(wav_path)
    if sr != 16000:
        raise ValueError(
            f"{wav_path} is {sr}Hz, Whisper expects 16kHz. Resample TORGO audio before running this script."
        )
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt").input_features.to(device)
    with torch.no_grad():
        predicted_ids = model.generate(
            inputs,
            language="en",
            task="transcribe",
            no_repeat_ngram_size=3,   # blocks the same 3-gram from repeating, stops loops like "happy happy happy..."
            max_new_tokens=100,       # hard cap — TORGO utterances are short; runaway generation shouldn't need more
        )
    text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return text


def run_fold(fold_dir: Path, processor, model, device: str, output_dir: Path) -> dict:
    test_csv = fold_dir / "test.csv"
    speaker = fold_dir.name.replace("fold_", "")

    references, hypotheses, rows_out = [], [], []
    with open(test_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            hyp = transcribe(row["wav_path"], processor, model, device)
            references.append(row["transcript"])
            hypotheses.append(hyp)
            rows_out.append({
                "speaker": speaker, "utt_id": row["utt_id"],
                "reference": row["transcript"], "hypothesis": hyp,
            })

    fold_output = output_dir / f"fold_{speaker}.csv"
    fold_output.parent.mkdir(parents=True, exist_ok=True)
    with open(fold_output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["speaker", "utt_id", "reference", "hypothesis"])
        writer.writeheader()
        writer.writerows(rows_out)

    result = score_batch(references, hypotheses)
    result["speaker"] = speaker
    result["num_utterances_total"] = len(rows_out)
    print(f"  fold_{speaker}: WER={result['wer']}  CER={result['cer']}  (n={result['num_utterances']})")
    return result


def main():
    parser = argparse.ArgumentParser(description="Run pretrained Whisper zero-shot baseline on LOSO fold test sets.")
    parser.add_argument("--splits-dir", required=True, help="Directory containing fold_<speaker>/test.csv folders")
    parser.add_argument("--model-size", default="small",
                         choices=["tiny", "base", "small", "medium", "large-v3"])
    parser.add_argument("--output-dir", required=True, help="Where to write per-fold hypothesis CSVs + summary")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[WARN] No GPU detected — this will be slow. Check your CUDA/torch install if you expected the 4090.",
              file=sys.stderr)

    processor, model = load_model(args.model_size, device)

    splits_dir = Path(args.splits_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    fold_dirs = sorted(p for p in splits_dir.iterdir() if p.is_dir() and p.name.startswith("fold_"))

    if not fold_dirs:
        print(f"No fold_* directories found in {splits_dir}")
        return

    print(f"Running baseline over {len(fold_dirs)} folds ...")
    summary_rows = []
    for fold_dir in fold_dirs:
        result = run_fold(fold_dir, processor, model, device, output_dir)
        summary_rows.append(result)

    summary_path = output_dir / "baseline_summary.csv"
    fieldnames = ["speaker", "wer", "cer", "num_utterances", "num_utterances_total", "num_skipped_empty_ref"]
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in summary_rows:
            writer.writerow({k: r.get(k) for k in fieldnames})

    valid_wers = [r["wer"] for r in summary_rows if r["wer"] is not None]
    if valid_wers:
        macro_avg_wer = sum(valid_wers) / len(valid_wers)
        print(f"\nMacro-average WER across {len(valid_wers)} speakers: {macro_avg_wer:.4f}")
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()