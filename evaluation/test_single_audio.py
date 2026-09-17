"""
test_single_audio.py

Transcribes ONE audio file with a given Whisper checkpoint (pretrained or
fine-tuned) and prints the result.

Usage:
    python test_single_audio.py \
        --model ../training_output/whisper-medium/all_speakers/final \
        --wav /path/to/file.wav \
        --reference "store"
"""

import argparse

import torch
import soundfile as sf
from transformers import WhisperProcessor, WhisperForConditionalGeneration


def main():
    parser = argparse.ArgumentParser(description="Transcribe a single audio file and print the result.")
    parser.add_argument("--model", required=True,
                         help="HuggingFace model id or local path to a fine-tuned checkpoint's 'final/' folder")
    parser.add_argument("--wav", required=True, help="Path to the .wav file to transcribe")
    parser.add_argument("--reference", default=None, help="Optional: the true transcript, to print alongside for comparison")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model} on {device} ...")
    processor = WhisperProcessor.from_pretrained(args.model)
    model = WhisperForConditionalGeneration.from_pretrained(args.model).to(device)
    model.generation_config.max_length = None
    model.eval()

    audio, sr = sf.read(args.wav)
    if sr != 16000:
        raise ValueError(f"{args.wav} is {sr}Hz, expected 16kHz.")

    inputs = processor(audio, sampling_rate=sr, return_tensors="pt").input_features.to(device)
    with torch.no_grad():
        predicted_ids = model.generate(
            inputs, language="en", task="transcribe",
            no_repeat_ngram_size=3, max_new_tokens=100,
        )
    hypothesis = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    print(f"\nFile:       {args.wav}")
    print(f"Duration:   {len(audio) / sr:.2f}s")
    if args.reference:
        print(f"Reference:  {args.reference}")
    print(f"Hypothesis: {hypothesis.strip()}")


if __name__ == "__main__":
    main()
