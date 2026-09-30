"""
phonetic_correct.py

Corrects SHORT ASR hypotheses (word-level utterances) by snapping them to the
nearest phonetically-similar word in TORGO's known closed vocabulary --
e.g. "corn" -> "car", "fog drought" -> "foxtrot".

This exists because those errors are NOT fixable by a text-only language
model: "corn" is a perfectly valid English word with no textual signal that
it should become "car". The two only relate phonetically (how they sound),
which a pure text-correction model has no way to know. This module uses
phonetic encoding (metaphone) + edit distance instead of language modeling.

For longer, sentence-level input this is NOT the right tool -- hand off to
the seq2seq correction model (train_correction_model.py / correct_text.py)
instead, which handles fluency/repetition issues in open-ended sentences.

Requires:
    pip install jellyfish --break-system-packages

Usage:
    python phonetic_correct.py --vocab vocab.txt --text "corn"
    python phonetic_correct.py --vocab vocab.txt --text "fog drought"
"""

import argparse

import jellyfish


def load_vocab(path: str) -> list[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def phonetic_correct(text: str, vocab: list[str], max_words: int = 3) -> tuple[str, bool]:
    """
    Returns (corrected_text, was_corrected). Only attempts correction for
    short text (<= max_words) -- longer sentences should go through the
    seq2seq correction model instead, not this word-level lookup.
    """
    text_clean = text.strip().lower()
    if len(text_clean.split()) > max_words:
        return text, False  # not a word-level utterance; leave for the seq2seq model

    if text_clean in vocab:
        return text, False  # already a known valid word -- nothing to correct

    # Compress spaces out for comparison, since a multi-token mis-hearing
    # (e.g. "fog drought") may correspond to a single target word ("foxtrot").
    compact = text_clean.replace(" ", "")
    hyp_code = jellyfish.metaphone(compact)

    best_word, best_score = None, None
    for candidate in vocab:
        candidate_compact = candidate.replace(" ", "")
        cand_code = jellyfish.metaphone(candidate_compact)

        raw_distance = jellyfish.levenshtein_distance(hyp_code, cand_code)
        # Normalize by code length -- otherwise short codes (e.g. "air") win
        # unfairly often just for being short, regardless of true similarity.
        longest_code = max(len(hyp_code), len(cand_code), 1)
        normalized_distance = raw_distance / longest_code

        # Tiebreaker: raw edit distance between the actual words (not their
        # phonetic codes), so ties are broken by genuine textual closeness
        # instead of arbitrary vocab-list ordering.
        word_distance = jellyfish.levenshtein_distance(compact, candidate_compact)
        score = (normalized_distance, word_distance)

        if best_score is None or score < best_score:
            best_score, best_word = score, candidate

    if best_word is not None and best_word != text_clean:
        return best_word, True
    return text, False


def main():
    parser = argparse.ArgumentParser(description="Phonetically correct a short ASR hypothesis against a known vocabulary.")
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--max-words", type=int, default=3)
    args = parser.parse_args()

    vocab = load_vocab(args.vocab)
    corrected, was_corrected = phonetic_correct(args.text, vocab, args.max_words)

    print(f"Input:     {args.text}")
    print(f"Corrected: {corrected}")
    print(f"Changed:   {was_corrected}")


if __name__ == "__main__":
    main()