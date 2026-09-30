"""
build_word_vocab.py

Extracts the closed vocabulary of TORGO's short, word-level target prompts
(e.g. "car", "foxtrot", "trouble") from the manifests. Sentence-level prompts
are open-ended English and not a fixed vocabulary, so they're excluded --
this vocab is specifically for phonetic correction of short utterances,
where the true answer is known to be one of a fixed set of words.

Also extracts the target word from bracket-style homograph disambiguation
prompts, e.g. `lead [as in "I will lead you"]` -> "lead", and
`tear [as in tear up that paper]` -> "tear". These are genuinely single
words the speaker says, but their FULL transcript is long (well over
--max-words), so without this they'd be missing from the vocabulary --
which was silently causing correct words like "lead" and "tear" to get
phonetically "corrected" into something else, since the matcher had no way
to recognize them as already valid.

Usage:
    python build_word_vocab.py \
        --manifests ../manifests/manifest_female.csv ../manifests/manifest_male.csv \
        --max-words 2 \
        --output vocab.txt
"""

import argparse
import csv
import re
from pathlib import Path

BRACKET_PATTERN = re.compile(r'^([A-Za-z\'-]+)\s*\[')


def main():
    parser = argparse.ArgumentParser(description="Build the closed word-level vocabulary from TORGO manifests.")
    parser.add_argument("--manifests", nargs="+", required=True)
    parser.add_argument("--max-words", type=int, default=2,
                         help="Max words in a transcript to count as a 'word-level' prompt (default 2)")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    vocab = set()
    bracket_extractions = 0
    for path in args.manifests:
        with open(Path(path).expanduser(), newline="") as f:
            for row in csv.DictReader(f):
                if row["status"] != "ok":
                    continue
                transcript = row["transcript"].strip()

                if 0 < len(transcript.split()) <= args.max_words:
                    vocab.add(transcript.lower())

                match = BRACKET_PATTERN.match(transcript)
                if match:
                    word = match.group(1).lower()
                    if word not in vocab:
                        bracket_extractions += 1
                    vocab.add(word)

    vocab = sorted(vocab)
    with open(args.output, "w") as f:
        f.write("\n".join(vocab))

    print(f"Extracted {len(vocab)} unique word-level prompts -> {args.output}")
    print(f"  (including {bracket_extractions} target words pulled from bracket-style disambiguation prompts)")
    print(f"Sample: {vocab[:15]}")


if __name__ == "__main__":
    main()
