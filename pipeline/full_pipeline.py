"""
full_pipeline.py

The complete Phase 1 + Phase 2a + Phase 2b pipeline, chained together:

    input .wav (dysarthric speech)
        -> Whisper (fine-tuned)         : audio -> raw hypothesis text
        -> unified_correct()             : raw hypothesis -> cleaned text
        -> XTTS-v2 (voice cloning)       : cleaned text -> speech, in the SAME speaker's own voice

This does NOT yet remove dysarthric acoustic character from the voice itself
(that is voice conversion, a separate unsolved research problem discussed
earlier) -- the output speaks the CORRECTED WORDS, but in a voice cloned
from the speaker's own (dysarthric) reference audio.

Requires everything already installed for the individual stages:
    transformers, torch, soundfile (Whisper)
    sentencepiece (text correction)
    coqui-tts, jellyfish (voice cloning + phonetic correction)

Usage:
    python full_pipeline.py \
        --wav ../M04/Session1/wav_headMic/0012.wav \
        --speaker M04 \
        --output final_output.wav
"""

import argparse
import sys
from pathlib import Path

import torch
import soundfile as sf
from transformers import (
    WhisperProcessor, WhisperForConditionalGeneration,
    AutoTokenizer, AutoModelForSeq2SeqLM,
)

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / "text_correction"))
sys.path.insert(0, str(BASE_DIR / "tts"))

from unified_correct import unified_correct
from phonetic_correct import load_vocab
from clone_voice import pick_reference_clip, pick_clearest_reference_clip
from TTS.api import TTS


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
    return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()


def main():
    parser = argparse.ArgumentParser(description="Full pipeline: dysarthric audio -> corrected text -> voice-cloned speech.")
    parser.add_argument("--wav", required=True, help="Input audio to transcribe")
    parser.add_argument("--speaker", required=True, help="Speaker for voice cloning reference, e.g. M04")
    parser.add_argument("--output", required=True, help="Output .wav path")
    parser.add_argument("--whisper-model-dir", default=str(BASE_DIR / "training_output" / "whisper-medium" / "all_speakers" / "final"))
    parser.add_argument("--seq2seq-model-dir", default=str(BASE_DIR / "text_correction" / "model" / "final"))
    parser.add_argument("--vocab", default=str(BASE_DIR / "text_correction" / "vocab.txt"))
    parser.add_argument("--reference-wav", default=None, help="Optional: exact clip to clone voice from")
    parser.add_argument("--max-words", type=int, default=2)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("STAGE 1: Transcription (Whisper)")
    print("=" * 60)
    processor = WhisperProcessor.from_pretrained(args.whisper_model_dir)
    whisper_model = WhisperForConditionalGeneration.from_pretrained(args.whisper_model_dir).to(device)
    whisper_model.generation_config.max_length = None
    whisper_model.eval()

    hypothesis = transcribe(args.wav, processor, whisper_model, device)
    print(f"Raw transcription: {hypothesis}")

    print("\n" + "=" * 60)
    print("STAGE 2: Text correction")
    print("=" * 60)
    vocab = load_vocab(args.vocab)
    tokenizer = AutoTokenizer.from_pretrained(args.seq2seq_model_dir)
    seq2seq_model = AutoModelForSeq2SeqLM.from_pretrained(args.seq2seq_model_dir)

    corrected_text, method = unified_correct(hypothesis, vocab, tokenizer, seq2seq_model, args.max_words)
    print(f"Corrected text: {corrected_text}  (method: {method})")

    print("\n" + "=" * 60)
    print("STAGE 3: Voice cloning (XTTS-v2)")
    print("=" * 60)
    reference_wav = args.reference_wav or pick_clearest_reference_clip(args.speaker, args.whisper_model_dir)

    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=torch.cuda.is_available())
    tts.tts_to_file(
        text=corrected_text,
        file_path=args.output,
        speaker_wav=reference_wav,
        language="en",
    )

    print(f"\nSaved final output to {args.output}")
    print(f"\nSummary:")
    print(f"  Input audio:      {args.wav}")
    print(f"  Raw transcription: {hypothesis}")
    print(f"  Corrected text:    {corrected_text}")
    print(f"  Voice reference:   {reference_wav}")
    print(f"  Output:            {args.output}")


if __name__ == "__main__":
    main()