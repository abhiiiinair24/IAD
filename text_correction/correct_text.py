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
    output_ids = model.generate(**inputs, max_length=128, num_beams=4)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


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
