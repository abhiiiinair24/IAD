"""
finetune_whisper.py

Fine-tunes a pretrained Whisper checkpoint on one LOSO fold's training speakers.

IMPORTANT - avoiding validation leakage:
    The fold's test.csv belongs to the held-out (zero-shot) speaker and must
    NEVER be used during training for anything -- not for early stopping, not
    for picking the "best" checkpoint. Doing so would mean tuning decisions
    based on the exact data you're claiming the model never saw, undermining
    the zero-shot claim.

    Instead, this script carves a small validation slice OUT OF the training
    speakers' own data (default 10%) purely for checkpoint selection during
    training. The held-out speaker's test.csv is only ever touched by the
    separate evaluation script, after training is fully finished.

Requires:
    pip install transformers torch soundfile jiwer accelerate datasets --break-system-packages

Usage (single fold, to validate the pipeline before running all 8):
    python finetune_whisper.py \
        --fold-dir ../splits/fold_M04 \
        --model-size medium \
        --output-dir ../training_output/whisper-medium/fold_M04 \
        --num-epochs 10 \
        --batch-size 4
"""

import argparse
import csv
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import soundfile as sf
import torch
from torch.utils.data import Dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

sys.path.insert(0, str(Path(__file__).parent.parent / "evaluation"))
from compute_wer import score_batch  # shared scorer, same one baseline used


def load_rows(csv_path: Path) -> list[dict]:
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def split_train_val(rows: list[dict], val_fraction: float, seed: int = 42) -> tuple[list, list]:
    """Carve a validation slice out of the TRAINING speakers' own data (not the held-out speaker)."""
    rows = rows.copy()
    random.Random(seed).shuffle(rows)
    n_val = max(1, int(len(rows) * val_fraction))
    return rows[n_val:], rows[:n_val]


class TorgoDataset(Dataset):
    def __init__(self, rows: list[dict], processor: WhisperProcessor):
        self.rows = rows
        self.processor = processor

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        audio, sr = sf.read(row["wav_path"])
        if sr != 16000:
            raise ValueError(
                f"{row['wav_path']} is {sr}Hz, Whisper expects 16kHz. Resample TORGO audio first."
            )
        input_features = self.processor.feature_extractor(audio, sampling_rate=sr).input_features[0]
        labels = self.processor.tokenizer(row["transcript"]).input_ids
        return {"input_features": input_features, "labels": labels}


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: list[dict]) -> dict:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def make_compute_metrics(processor: WhisperProcessor):
    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        result = score_batch(label_str, pred_str)
        return {"wer": result["wer"] if result["wer"] is not None else 1.0,
                "cer": result["cer"] if result["cer"] is not None else 1.0}
    return compute_metrics


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Whisper on one LOSO fold.")
    parser.add_argument("--fold-dir", required=True, help="e.g. ../splits/fold_M04")
    parser.add_argument("--model-size", default="medium", choices=["tiny", "base", "small", "medium", "large-v3"])
    parser.add_argument("--init-from", default=None,
                         help="Optional: path to an existing local checkpoint (e.g. a LOSO fold's 'final/' dir) "
                              "to continue fine-tuning from, instead of starting from the pretrained "
                              "openai/whisper-{model_size} hub model. Used for speaker adaptation.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--val-fraction", type=float, default=0.1,
                         help="Fraction of TRAINING speakers' data held out for checkpoint selection (not the LOSO test speaker)")
    args = parser.parse_args()

    script_start = time.time()
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    fold_dir = Path(args.fold_dir).expanduser()
    train_rows_all = load_rows(fold_dir / "train.csv")
    train_rows, val_rows = split_train_val(train_rows_all, args.val_fraction)

    held_out_speaker = fold_dir.name.replace("fold_", "")
    print(f"Fold: held-out speaker = {held_out_speaker}")
    print(f"  Train: {len(train_rows)} utterances (from {sorted({r['speaker'] for r in train_rows})})")
    print(f"  Val (carved from train speakers, for checkpoint selection only): {len(val_rows)}")
    print(f"  NOTE: {held_out_speaker}'s test.csv is NOT loaded here — it stays untouched until separate evaluation.")

    model_id = args.init_from if args.init_from else f"openai/whisper-{args.model_size}"
    print(f"Initializing model from: {model_id}"
          f"{' (continuing from an existing checkpoint)' if args.init_from else ' (fresh pretrained hub model)'}")
    processor = WhisperProcessor.from_pretrained(model_id, language="en", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(model_id)
    model.generation_config.language = "en"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None

    train_ds = TorgoDataset(train_rows, processor)
    val_ds = TorgoDataset(val_rows, processor)
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=2,
        learning_rate=args.learning_rate,
        warmup_steps=50,
        num_train_epochs=args.num_epochs,
        fp16=torch.cuda.is_available(),
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=2,  # keep only the 2 most recent checkpoints -- avoids repeating the disk-fill incident
        predict_with_generate=True,
        generation_max_length=100,
        logging_steps=25,
        report_to=["none"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,   # validation slice from TRAIN speakers only -- never the held-out speaker
        data_collator=data_collator,
        compute_metrics=make_compute_metrics(processor),
        processing_class=processor.feature_extractor,
    )

    train_start = time.time()
    trainer.train()
    train_elapsed = time.time() - train_start
    print(f"\nTraining took {train_elapsed/60:.1f} min ({train_elapsed:.0f}s)")

    final_dir = Path(args.output_dir) / "final"
    trainer.save_model(str(final_dir))
    processor.save_pretrained(str(final_dir))

    total_elapsed = time.time() - script_start
    print(f"Saved fine-tuned model to {final_dir}")
    print(f"Total script time (incl. model load/setup): {total_elapsed/60:.1f} min")
    print(f"Run evaluation/run_finetuned_eval.py against {fold_dir / 'test.csv'} to get the zero-shot WER for {held_out_speaker}.")


if __name__ == "__main__":
    main()