"""
confidence_score.py

Transcribes audio AND returns a confidence score for each word, using
Whisper's own per-token generation probabilities (not a separate model).

At every generated token, Whisper computes a full probability distribution
over its vocabulary before picking the most likely one. This captures that
probability per token, then groups tokens back into words (Whisper's BPE
tokenizer marks the start of a new word with a leading space/'Ġ' character)
to get one confidence value per word: how sure the model was about that
specific word, from 0.0 (total guess) to 1.0 (completely certain).

This is exactly what would let you flag low-confidence words in a downstream
UI (e.g. highlight them, or route them to cross-checking against video, as
discussed for a possible future extension).

Usage:
    python confidence_score.py --model ../training_output/whisper-medium/all_speakers/final --wav /path/to/file.wav
"""

import argparse

import torch
import soundfile as sf
from transformers import WhisperProcessor, WhisperForConditionalGeneration


def transcribe_with_confidence(wav_path: str, processor, model, device: str):
    """
    Returns (full_text, [(word, confidence), ...], overall_confidence).
    Confidence is the model's own token-generation probability, averaged
    per word (and overall) -- NOT a separate confidence-estimation model.
    """
    audio, sr = sf.read(wav_path)
    if sr != 16000:
        raise ValueError(f"{wav_path} is {sr}Hz, expected 16kHz.")

    inputs = processor(audio, sampling_rate=sr, return_tensors="pt").input_features.to(device)
    with torch.no_grad():
        outputs = model.generate(
            inputs, language="en", task="transcribe",
            no_repeat_ngram_size=3, max_new_tokens=100,
            output_scores=True, return_dict_in_generate=True,
        )

    sequence = outputs.sequences[0]
    step_logits = outputs.scores  # one entry per generated token, each (1, vocab_size)

    # outputs.scores[i] are the logits used to produce sequence[i + <offset>];
    # the offset accounts for forced/special tokens (language, task, timestamp)
    # at the start that aren't part of outputs.scores. Align from the end
    # instead of the start to sidestep counting that offset exactly.
    generated_ids = sequence[-len(step_logits):] if len(step_logits) > 0 else sequence
    tokenizer = processor.tokenizer

    token_probs = []
    for logits, token_id in zip(step_logits, generated_ids):
        probs = torch.softmax(logits[0], dim=-1)
        token_probs.append(probs[token_id].item())

    tokens = tokenizer.convert_ids_to_tokens(generated_ids.tolist())

    words, word_confidences = [], []
    current_word, current_probs = "", []
    for tok, prob in zip(tokens, token_probs):
        if tok in tokenizer.all_special_tokens:
            continue
        starts_new_word = tok.startswith("Ġ") or tok.startswith(" ")
        piece = tok.lstrip("Ġ ")
        if starts_new_word and current_word:
            words.append(current_word)
            word_confidences.append(sum(current_probs) / len(current_probs))
            current_word, current_probs = piece, [prob]
        else:
            current_word += piece
            current_probs.append(prob)
    if current_word:
        words.append(current_word)
        word_confidences.append(sum(current_probs) / len(current_probs))

    full_text = processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0].strip()
    overall_confidence = sum(token_probs) / len(token_probs) if token_probs else 0.0

    return full_text, list(zip(words, word_confidences)), overall_confidence


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio with per-word confidence scores.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--wav", required=True)
    parser.add_argument("--low-confidence-threshold", type=float, default=0.7,
                         help="Words scoring below this are flagged (default 0.7)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = WhisperProcessor.from_pretrained(args.model)
    model = WhisperForConditionalGeneration.from_pretrained(args.model).to(device)
    model.generation_config.max_length = None
    model.eval()

    text, word_confidences, overall = transcribe_with_confidence(args.wav, processor, model, device)

    print(f"Full transcription: {text}")
    print(f"Overall confidence: {overall:.3f}\n")
    print("Per-word confidence:")
    for word, conf in word_confidences:
        flag = "  <-- LOW CONFIDENCE" if conf < args.low_confidence_threshold else ""
        print(f"  {word:20s} {conf:.3f}{flag}")


if __name__ == "__main__":
    main()
