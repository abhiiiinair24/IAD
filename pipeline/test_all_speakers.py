"""
test_all_speakers.py

Runs the full pipeline (transcribe -> correct -> voice-clone) on one real
SENTENCE-LEVEL utterance per speaker, across all 8 TORGO speakers. Prefers a
known ERROR case so text correction has something real to do, falling back
to any sentence-level utterance, then any usable utterance, if needed.

Uses synthesize_with_verification() so cloning reliability (not just
correction quality) is checked and retried automatically.

Usage:
    python test_all_speakers.py --output-dir all_speakers_test_outputs_v4
"""

import argparse
import csv
import sys
from pathlib import Path

import torch
import soundfile as sf
import librosa
from transformers import (
    WhisperProcessor, WhisperForConditionalGeneration,
    AutoTokenizer, AutoModelForSeq2SeqLM,
)

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / "text_correction"))
sys.path.insert(0, str(BASE_DIR / "tts"))
sys.path.insert(0, str(BASE_DIR / "evaluation"))

from unified_correct import unified_correct
from phonetic_correct import load_vocab
from clone_voice import pick_clearest_reference_clip, synthesize_with_verification
from compute_wer import score_batch
from TTS.api import TTS

SPEAKERS = ["F01", "F03", "F04", "M01", "M02", "M03", "M04", "M05"]
MIN_SENTENCE_WORDS = 4  # utterances with at least this many words count as "sentence-level"


def transcribe(wav_path: str, processor, model, device: str) -> str:
    audio, sr = sf.read(wav_path)
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt").input_features.to(device)
    with torch.no_grad():
        predicted_ids = model.generate(inputs, language="en", task="transcribe",
                                        no_repeat_ngram_size=3, max_new_tokens=100)
    return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()


def find_test_utterance(speaker: str) -> tuple[str, str]:
    """
    Returns (wav_path, reference_text). Prefers a SENTENCE-LEVEL utterance
    (>= MIN_SENTENCE_WORDS), specifically one that's a known error case if
    available -- this is what actually exercises the seq2seq correction
    model, unlike single words which mostly route to phonetic matching or
    pass through unchanged.
    """
    fold_csv = BASE_DIR / "results" / "finetuned_whisper_medium" / f"fold_{speaker}.csv"
    manifest_female = BASE_DIR / "manifests" / "manifest_female.csv"
    manifest_male = BASE_DIR / "manifests" / "manifest_male.csv"
    manifest_path = manifest_female if speaker.startswith("F") else manifest_male

    with open(manifest_path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["speaker"] == speaker and r["status"] == "ok"]
    sentence_rows = {r["utt_id"]: r for r in rows if len(r["transcript"].split()) >= MIN_SENTENCE_WORDS}

    if fold_csv.exists():
        with open(fold_csv, newline="") as f:
            for row in csv.DictReader(f):
                if (row["reference"].strip().lower() != row["hypothesis"].strip().lower()
                        and len(row["reference"].split()) >= MIN_SENTENCE_WORDS
                        and row["utt_id"] in sentence_rows):
                    r = sentence_rows[row["utt_id"]]
                    return r["wav_path"], r["transcript"]

    if sentence_rows:
        r = next(iter(sentence_rows.values()))
        return r["wav_path"], r["transcript"]

    if rows:
        return rows[0]["wav_path"], rows[0]["transcript"]
    raise ValueError(f"No usable utterance found for {speaker}")


def main():
    parser = argparse.ArgumentParser(description="Test the full pipeline across all 8 TORGO speakers.")
    parser.add_argument("--whisper-model-dir", default=str(BASE_DIR / "training_output" / "whisper-medium" / "all_speakers" / "final"))
    parser.add_argument("--seq2seq-model-dir", default=str(BASE_DIR / "text_correction" / "model" / "final"))
    parser.add_argument("--vocab", default=str(BASE_DIR / "text_correction" / "vocab.txt"))
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading models (once, reused across all 8 speakers) ...")
    processor = WhisperProcessor.from_pretrained(args.whisper_model_dir)
    whisper_model = WhisperForConditionalGeneration.from_pretrained(args.whisper_model_dir).to(device)
    whisper_model.generation_config.max_length = None
    whisper_model.eval()

    vocab = load_vocab(args.vocab)
    tokenizer = AutoTokenizer.from_pretrained(args.seq2seq_model_dir)
    seq2seq_model = AutoModelForSeq2SeqLM.from_pretrained(args.seq2seq_model_dir)

    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=torch.cuda.is_available())

    results = []
    for speaker in SPEAKERS:
        print(f"\n{'='*60}\n{speaker}\n{'='*60}")
        try:
            wav_path, reference = find_test_utterance(speaker)
            print(f"Test utterance: {wav_path}")
            print(f"True reference: {reference}")

            hypothesis = transcribe(wav_path, processor, whisper_model, device)
            print(f"Raw transcription: {hypothesis}")

            corrected, method = unified_correct(hypothesis, vocab, tokenizer, seq2seq_model)
            print(f"Corrected text: {corrected}  (method: {method})")

            ref_clip = pick_clearest_reference_clip(speaker, args.whisper_model_dir)

            output_wav = str(output_dir / f"{speaker}_output.wav")
            verified, attempts = synthesize_with_verification(
                tts, corrected, ref_clip, output_wav, args.whisper_model_dir, max_attempts=3
            )

            audio_16k, sr_16k = librosa.load(output_wav, sr=16000, mono=True)
            temp_16k_path = str(output_dir / f"{speaker}_output_16k.wav")
            sf.write(temp_16k_path, audio_16k, sr_16k)
            roundtrip_hyp = transcribe(temp_16k_path, processor, whisper_model, device)
            roundtrip_result = score_batch([corrected], [roundtrip_hyp])
            roundtrip_wer = roundtrip_result["wer"]

            print(f"Voice reference: {ref_clip}")
            print(f"Output: {output_wav}")
            print(f"Verified: {verified} (attempts used: {attempts})")
            print(f"Round-trip re-transcription: {roundtrip_hyp}  (WER vs corrected text: {roundtrip_wer})")

            results.append({
                "speaker": speaker, "wav_path": wav_path, "true_reference": reference,
                "raw_transcription": hypothesis, "corrected_text": corrected, "correction_method": method,
                "voice_reference": ref_clip, "output_wav": output_wav,
                "roundtrip_transcription": roundtrip_hyp, "roundtrip_wer": roundtrip_wer,
                "verified": verified, "attempts_used": attempts,
                "status": "ok",
            })
        except Exception as e:
            print(f"[ERROR] {speaker} failed: {e}")
            results.append({"speaker": speaker, "status": "error", "error": str(e)})

    summary_path = output_dir / "summary.csv"
    fieldnames = ["speaker", "wav_path", "true_reference", "raw_transcription", "corrected_text",
                  "correction_method", "voice_reference", "output_wav", "roundtrip_transcription",
                  "roundtrip_wer", "verified", "attempts_used", "status", "error"]
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    print(f"\n\nSummary written to {summary_path}")
    ok_count = sum(1 for r in results if r["status"] == "ok")
    print(f"{ok_count}/{len(SPEAKERS)} speakers completed successfully")


if __name__ == "__main__":
    main()
