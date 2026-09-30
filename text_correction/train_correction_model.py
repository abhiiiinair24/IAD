"""
train_correction_model.py

Fine-tunes a small text-to-text model (Flan-T5-small, ~80M params) to map
noisy ASR hypothesis text -> corrected reference text, using the pairs
built by build_correction_dataset.py.

This is a text-only model -- no audio involved. It sits AFTER Whisper in the
pipeline: Whisper produces a hypothesis, this model cleans it up, and the
cleaned text is what should be passed to the TTS/voice-cloning step.

Requires:
    pip install transformers torch sentencepiece accelerate jiwer --break-system-packages

Usage:
    python train_correction_model.py \
        --data-dir ../text_correction/data \
        --output-dir ../text_correction/model \
        --num-epochs 8
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    Seq2SeqTrainer, Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)

sys.path.insert(0, str(Path(__file__).parent.parent / "evaluation"))
from compute_wer import score_batch

TASK_PREFIX = "fix asr transcription: "


def load_rows(csv_path: Path) -> list[dict]:
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


class CorrectionDataset(Dataset):
    def __init__(self, rows: list[dict], tokenizer, max_length: int = 128):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        input_text = TASK_PREFIX + row["hypothesis"]
        target_text = row["reference"]

        model_inputs = self.tokenizer(input_text, max_length=self.max_length, truncation=True)
        labels = self.tokenizer(text_target=target_text, max_length=self.max_length, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


def make_compute_metrics(tokenizer):
    def compute_metrics(pred):
        pred_ids = pred.predictions
        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]
        label_ids = pred.label_ids

        # -100 is the "ignore" marker used for padding/loss-masking. It can show up
        # in BOTH predictions and labels depending on transformers version, and
        # decoding a -100 directly crashes the fast tokenizer with an OverflowError.
        pred_ids = np.where(pred_ids != -100, pred_ids, tokenizer.pad_token_id)
        label_ids = np.where(label_ids != -100, label_ids, tokenizer.pad_token_id)

        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        result = score_batch(label_str, pred_str)
        return {"wer": result["wer"] if result["wer"] is not None else 1.0}
    return compute_metrics


def main():
    parser = argparse.ArgumentParser(description="Fine-tune a small model to correct ASR output.")
    parser.add_argument("--data-dir", required=True, help="Directory with train.csv / val.csv")
    parser.add_argument("--base-model", default="google/flan-t5-small")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser()
    train_rows = load_rows(data_dir / "train.csv")
    val_rows = load_rows(data_dir / "val.csv")

    print(f"Train pairs: {len(train_rows)}  |  Val pairs: {len(val_rows)}")

    # Baseline: WER if we did NOTHING (hypothesis used as-is) -- the number the
    # trained correction model actually needs to beat to be worth using.
    baseline = score_batch([r["reference"] for r in val_rows], [r["hypothesis"] for r in val_rows])
    print(f"Baseline WER on val set (no correction applied): {baseline['wer']}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model)

    train_ds = CorrectionDataset(train_rows, tokenizer)
    val_ds = CorrectionDataset(val_rows, tokenizer)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        predict_with_generate=True,
        generation_max_length=128,
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
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=make_compute_metrics(tokenizer),
        processing_class=tokenizer,
    )

    start = time.time()
    trainer.train()
    print(f"\nTraining took {(time.time() - start)/60:.1f} min")

    final_dir = Path(args.output_dir) / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    final_metrics = trainer.evaluate()
    print(f"\nFinal val WER (hypothesis -> corrected, vs. reference): {final_metrics.get('eval_wer')}")
    print(f"Baseline val WER (hypothesis, uncorrected, vs. reference): {baseline['wer']}")
    print(f"Saved corrected model to {final_dir}")


if __name__ == "__main__":
    main()