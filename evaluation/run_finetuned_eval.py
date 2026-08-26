"""
run_finetuned_eval.py

Evaluates a FINE-TUNED Whisper checkpoint on its fold's held-out (zero-shot)
test speaker. This is the real "did fine-tuning help" number -- comparable
directly to run_baseline_whisper.py's per-speaker results, since it uses the
exact same test.csv and the same compute_wer.py scorer.

Usage:
    python run_finetuned_eval.py \
        --model-dir ../training_output/whisper-medium/fold_M04/final \
        --fold-dir ../splits/fold_M04 \
        --output-dir ../results/finetuned_whisper_medium
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
from compute_wer import score_batch


def transcribe(wav_path: str, processor, model, device: str) -> str:
    audio, sr = sf.read(wav_path)
    if sr != 16000:
        raise ValueError(f"{wav_path} is {sr}Hz, expected 16kHz.")
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt").input_features.to(device)
    with torch.no_grad():
        predicted_ids = model.generate(
            inputs, language="en", task="transcribe",
            no_repeat_ngram_size=3, max_new_tokens=100,
        )
    return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]


def main():
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned Whisper checkpoint on its held-out speaker.")
    parser.add_argument("--model-dir", required=True, help="Path to the fine-tuned model (the 'final/' directory)")
    parser.add_argument("--fold-dir", required=True, help="e.g. ../splits/fold_M04")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dir = Path(args.model_dir).expanduser()
    fold_dir = Path(args.fold_dir).expanduser()
    held_out_speaker = fold_dir.name.replace("fold_", "")

    print(f"Loading fine-tuned model from {model_dir} ...")
    processor = WhisperProcessor.from_pretrained(model_dir)
    model = WhisperForConditionalGeneration.from_pretrained(model_dir).to(device)
    model.generation_config.max_length = None
    model.eval()

    test_csv = fold_dir / "test.csv"
    references, hypotheses, rows_out = [], [], []
    eval_start = time.time()
    with open(test_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            hyp = transcribe(row["wav_path"], processor, model, device)
            references.append(row["transcript"])
            hypotheses.append(hyp)
            rows_out.append({
                "speaker": held_out_speaker, "utt_id": row["utt_id"],
                "reference": row["transcript"], "hypothesis": hyp,
            })
    eval_elapsed = time.time() - eval_start
    print(f"Transcription took {eval_elapsed/60:.1f} min ({eval_elapsed:.0f}s) for {len(rows_out)} utterances "
          f"({eval_elapsed/max(1,len(rows_out)):.2f}s/utterance)")

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = output_dir / f"fold_{held_out_speaker}.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["speaker", "utt_id", "reference", "hypothesis"])
        writer.writeheader()
        writer.writerows(rows_out)

    result = score_batch(references, hypotheses)
    print(f"\nZero-shot result for held-out speaker {held_out_speaker}:")
    print(f"  WER: {result['wer']}")
    print(f"  CER: {result['cer']}")
    print(f"  n:   {result['num_utterances']}")
    print(f"\nHypotheses written to {out_csv}")

    summary_path = output_dir / "finetuned_summary.csv"
    file_exists = summary_path.exists()
    with open(summary_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["speaker", "wer", "cer", "num_utterances"])
        if not file_exists:
            writer.writeheader()
        writer.writerow({"speaker": held_out_speaker, "wer": result["wer"],
                          "cer": result["cer"], "num_utterances": result["num_utterances"]})
    print(f"Appended to {summary_path}")


if __name__ == "__main__":
    main()