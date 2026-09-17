"""
transcribe_nbest.py

Instead of only keeping Whisper's single best-guess transcription, this
captures its TOP-N candidate transcriptions (via beam search) along with
each one's confidence score. For short/word-level utterances, it then
prefers whichever candidate matches a known TORGO vocabulary word -- even if
it wasn't Whisper's #1 ranked guess.

This fixes the "corn" vs "car" problem: a pure text-correction model can't
fix this (both are valid English words, nothing LOOKS wrong), but if "car"
was Whisper's 2nd or 3rd choice internally, we can recover it directly from
the model's own candidates instead of guessing from text alone.

Usage (as a library):
    from transcribe_nbest import transcribe_nbest, vocab_aware_best

    candidates = transcribe_nbest(wav_path, processor, model, device, num_candidates=5)
    best_text = vocab_aware_best(candidates, vocab, max_words=2)

Usage (standalone test):
    python transcribe_nbest.py \
        --model-dir ../training_output/whisper-medium/all_speakers/final \
        --wav /path/to/clip.wav \
        --vocab ../text_correction/vocab.txt
"""

import argparse
import sys
from pathlib import Path

import torch
import soundfile as sf
from transformers import WhisperProcessor, WhisperForConditionalGeneration


def transcribe_nbest(wav_path: str, processor, model, device: str, num_candidates: int = 5) -> list[tuple[str, float]]:
    """
    Returns [(text, score), ...] sorted best-first. Score is the beam's
    sequence log-probability (higher / less negative = more confident).
    """
    audio, sr = sf.read(wav_path)
    if sr != 16000:
        raise ValueError(f"{wav_path} is {sr}Hz, expected 16kHz.")

    inputs = processor(audio, sampling_rate=sr, return_tensors="pt").input_features.to(device)
    with torch.no_grad():
        outputs = model.generate(
            inputs, language="en", task="transcribe",
            no_repeat_ngram_size=3, max_new_tokens=100,
            do_sample=True, temperature=0.8, top_k=50,
            num_return_sequences=num_candidates,
            output_scores=True, return_dict_in_generate=True,
        )

    texts = processor.batch_decode(outputs.sequences, skip_special_tokens=True)
    scores = getattr(outputs, "sequences_scores", None)
    if scores is not None:
        scores = scores.tolist()
    else:
        # Sampling doesn't populate sequences_scores the way beam search does --
        # compute each sequence's average per-token log-probability manually instead.
        transition_scores = model.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True
        )
        scores = transition_scores.mean(dim=-1).tolist()

    candidates = list(zip([t.strip() for t in texts], scores))
    candidates.sort(key=lambda x: -x[1])  # best (highest/least-negative score) first
    return candidates


def vocab_aware_best(candidates: list[tuple[str, float]], vocab: set[str], max_words: int = 2) -> tuple[str, bool]:
    """
    Returns (chosen_text, used_reranking). For short utterances, prefers the
    highest-ranked candidate that matches a known vocabulary word over
    Whisper's raw #1 pick, if they differ.
    """
    top_text = candidates[0][0]
    if len(top_text.split()) > max_words:
        return top_text, False  # sentence-level -- not what this vocab reranking is for

    if top_text.lower() in vocab:
        return top_text, False  # top pick is already a known valid word -- no need to rerank

    for text, _ in candidates:
        if text.lower() in vocab:
            return text, True  # found a vocab match further down the candidate list

    return top_text, False  # no candidate matched the vocabulary at all


def load_vocab(path: str) -> set[str]:
    with open(path) as f:
        return {line.strip().lower() for line in f if line.strip()}


def main():
    parser = argparse.ArgumentParser(description="Test n-best transcription with vocabulary-aware reranking.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--wav", required=True)
    parser.add_argument("--vocab", default=None, help="Optional vocab.txt for reranking")
    parser.add_argument("--num-candidates", type=int, default=5)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = WhisperProcessor.from_pretrained(args.model_dir)
    model = WhisperForConditionalGeneration.from_pretrained(args.model_dir).to(device)
    model.generation_config.max_length = None
    model.eval()

    candidates = transcribe_nbest(args.wav, processor, model, device, args.num_candidates)
    print("Top candidates (best first):")
    for text, score in candidates:
        print(f"  {score:.3f}  {text}")

    if args.vocab:
        vocab = load_vocab(args.vocab)
        best, reranked = vocab_aware_best(candidates, vocab)
        print(f"\nFinal choice: {best}")
        print(f"Reranked away from Whisper's top pick: {reranked}")


if __name__ == "__main__":
    main()