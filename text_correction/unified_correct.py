"""
unified_correct.py

The single entry point for Phase 2a (text correction), combining both
approaches built so far:

    - Short utterances (<= max_words):  phonetic vocabulary matching
      (phonetic_correct.py) -- fixes confidently-wrong-but-valid-word errors
      like "corn" -> "car" that a language model can't detect from text alone.

    - Longer utterances (> max_words):  the fine-tuned seq2seq model
      (correct_text.py / Flan-T5) -- fixes fluency/repetition issues in
      open-ended sentences.

This is what should sit between Whisper's output and the future TTS/voice-
cloning step: Whisper -> unified_correct() -> cleaned text -> TTS.

Usage:
    python unified_correct.py \
        --text "corn" \
        --vocab vocab.txt \
        --seq2seq-model model/final

    python unified_correct.py \
        --text "he slowly takes a turn to walk in the old air each day" \
        --vocab vocab.txt \
        --seq2seq-model model/final
"""

import argparse

from phonetic_correct import load_vocab, phonetic_correct
from correct_text import correct as seq2seq_correct
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def unified_correct(text: str, vocab: set[str], tokenizer, seq2seq_model, max_words: int = 1) -> tuple[str, str]:
    """
    Returns (corrected_text, method_used) where method_used is
    'phonetic', 'seq2seq', or 'unchanged'.

    max_words default is 1, not 2: TORGO's word-level target vocabulary is
    overwhelmingly single words. A 2-word hypothesis (e.g. "tear up") is not
    itself a real target word, so forcing it through phonetic matching --
    which can only output ONE vocabulary word -- can produce nonsense (e.g.
    "tear up" -> "drop"). Multi-word text is safer routed to the seq2seq
    model, which is built to handle phrases/sentences sensibly.
    """
    word_count = len(text.strip().split())
    if word_count <= max_words:
        corrected, was_corrected = phonetic_correct(text, vocab, max_words)
        return corrected, ("phonetic" if was_corrected else "unchanged")
    else:
        corrected = seq2seq_correct(text, tokenizer, seq2seq_model)
        return corrected, ("seq2seq" if corrected.strip().lower() != text.strip().lower() else "unchanged")


def main():
    parser = argparse.ArgumentParser(description="Unified text correction (phonetic + seq2seq).")
    parser.add_argument("--text", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--seq2seq-model", required=True)
    parser.add_argument("--max-words", type=int, default=1,
                         help="Utterances at or below this word count use phonetic correction; longer use seq2seq")
    args = parser.parse_args()

    vocab = load_vocab(args.vocab)
    tokenizer = AutoTokenizer.from_pretrained(args.seq2seq_model)
    seq2seq_model = AutoModelForSeq2SeqLM.from_pretrained(args.seq2seq_model)

    corrected, method = unified_correct(args.text, vocab, tokenizer, seq2seq_model, args.max_words)
    print(f"Input:     {args.text}")
    print(f"Corrected: {corrected}")
    print(f"Method:    {method}")


if __name__ == "__main__":
    main()