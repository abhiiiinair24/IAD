"""
correct_text.py

Loads the trained correction model and cleans a piece of ASR output text.
This is the module that sits between Whisper (speech-to-text) and the future
TTS/voice-cloning step: Whisper output -> this -> cleaned text -> TTS.

Usage:
    python correct_text.py --model-dir ../text_correction/model/final --text "the passenger passenger passenger"
"""

import argparse

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

TASK_PREFIX = "fix asr transcription: "


def correct(text: str, tokenizer, model) -> str:
    inputs = tokenizer(TASK_PREFIX + text, return_tensors="pt", truncation=True, max_length=128)
    n_in = inputs["input_ids"].shape[1]
    # cap the output near the input length and forbid repeating any 3-word run: stops the
    # runaway loops seen on long spontaneous sentences
    output_ids = model.generate(**inputs, max_length=min(128, int(n_in * 1.3) + 8), num_beams=4,
                                no_repeat_ngram_size=3)
    out = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # length guard: for sentences of 4+ words the corrected text should be about as long as the input
    n_text, n_out = len(text.split()), len(out.split())
    if n_text >= 4 and not (0.6 <= n_out / n_text <= 1.4):
        return text
    return out


def main():
    parser = argparse.ArgumentParser(description="Correct a piece of ASR output text.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--text", required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir)

    corrected = correct(args.text, tokenizer, model)
    print(f"Input:     {args.text}")
    print(f"Corrected: {corrected}")


if __name__ == "__main__":
    main()
