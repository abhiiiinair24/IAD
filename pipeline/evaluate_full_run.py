"""
evaluate_full_run.py

Given an INPUT audio file and the pipeline's GENERATED output for it, this
breaks the result down into four INDEPENDENT metrics -- these answer
different questions and are deliberately NOT averaged into one score:

    1. ASR accuracy        -- did Whisper transcribe the input correctly?
    2. Text correction      -- did normalization improve on the raw transcription?
    3. Generation fidelity  -- does the generated audio actually SAY the
                               corrected text (re-verified by re-transcribing it)?
    4. Voice similarity      -- does the generated audio SOUND LIKE the
                               original speaker (ECAPA-TDNN, calibrated
                               against real same/different-speaker baselines)?

Metrics 1-3 are about WORDS. Metric 4 is about VOICE IDENTITY. A pipeline
can score well on one and poorly on the other -- report them separately,
never combined into a single "quality" number.

Usage:
    python evaluate_full_run.py \
        --input-audio ../M04/Session1/wav_arrayMic/0012.wav \
        --generated-audio ../pipeline/final_output_sentence.wav \
        --speaker M04 \
        --true-reference "Twice each day he plays skillfully and with zest upon our small organ."
"""

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio
import soundfile as sf
import librosa
from transformers import (
    WhisperProcessor, WhisperForConditionalGeneration,
    AutoTokenizer, AutoModelForSeq2SeqLM,
)
from speechbrain.inference.speaker import EncoderClassifier

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / "text_correction"))
sys.path.insert(0, str(BASE_DIR / "evaluation"))
from unified_correct import unified_correct
from phonetic_correct import load_vocab
from compute_wer import score_batch

MODEL_DIR = BASE_DIR / "training_output" / "whisper-medium" / "all_speakers" / "final"
SEQ2SEQ_MODEL_DIR = BASE_DIR / "text_correction" / "model" / "final"
VOCAB_PATH = BASE_DIR / "text_correction" / "vocab.txt"
MANIFEST_FEMALE = BASE_DIR / "manifests" / "manifest_female.csv"
MANIFEST_MALE = BASE_DIR / "manifests" / "manifest_male.csv"


# ---------- ASR helpers ----------

def transcribe(wav_path: str, processor, model, device: str) -> str:
    audio, sr = sf.read(wav_path)
    if sr != 16000:
        audio, sr = librosa.load(wav_path, sr=16000, mono=True)
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt").input_features.to(device)
    with torch.no_grad():
        predicted_ids = model.generate(inputs, language="en", task="transcribe",
                                        no_repeat_ngram_size=3, max_new_tokens=100)
    return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()


# ---------- ECAPA-TDNN voice similarity helpers ----------

def embed(classifier, wav_path: str) -> np.ndarray:
    signal, sr = torchaudio.load(wav_path)
    if signal.shape[0] > 1:
        signal = signal.mean(dim=0, keepdim=True)
    if sr != 16000:
        signal = torchaudio.functional.resample(signal, sr, 16000)
    with torch.no_grad():
        emb = classifier.encode_batch(signal)
    return emb.squeeze().cpu().numpy()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def find_clips(speaker: str, min_duration: float = 3.0, n: int = 20) -> list[str]:
    clips = []
    for path in [MANIFEST_FEMALE, MANIFEST_MALE]:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                if row["speaker"] == speaker and row["status"] == "ok" and float(row["duration_sec"]) >= min_duration:
                    clips.append(row["wav_path"])
    random.Random(42).shuffle(clips)
    return clips[:n]


def find_other_speakers(exclude_speaker: str) -> list[str]:
    all_speakers = ["F01", "F03", "F04", "M01", "M02", "M03", "M04", "M05"]
    return [s for s in all_speakers if s != exclude_speaker]


def voice_similarity(classifier, input_audio: str, generated_audio: str, speaker: str,
                      n_other: int = 20, n_baseline: int = 10) -> dict:
    input_emb = embed(classifier, input_audio)
    generated_emb = embed(classifier, generated_audio)

    same_speaker_clips = find_clips(speaker, n=n_baseline)
    same_scores = []
    for i in range(len(same_speaker_clips)):
        for j in range(i + 1, len(same_speaker_clips)):
            same_scores.append(cosine_similarity(embed(classifier, same_speaker_clips[i]),
                                                   embed(classifier, same_speaker_clips[j])))
    same_baseline = sum(same_scores) / len(same_scores) if same_scores else None

    diff_scores = []
    for other_speaker in find_other_speakers(speaker):
        other_clips = find_clips(other_speaker, n=3)
        for this_clip in same_speaker_clips[:3]:
            emb_this = embed(classifier, this_clip)
            for other_clip in other_clips:
                diff_scores.append(cosine_similarity(emb_this, embed(classifier, other_clip)))
    diff_baseline = sum(diff_scores) / len(diff_scores) if diff_scores else None

    candidate_clips = [c for c in find_clips(speaker, n=n_other + 5) if c != input_audio]
    test_clips = candidate_clips[:n_other]
    test_scores = [cosine_similarity(generated_emb, embed(classifier, c)) for c in test_clips]
    test_mean = sum(test_scores) / len(test_scores) if test_scores else None
    test_std = np.std(test_scores) if test_scores else None

    result = {
        "generated_vs_input_direct": cosine_similarity(input_emb, generated_emb),
        "generated_vs_other_real_clips_mean": test_mean,
        "generated_vs_other_real_clips_std": test_std,
        "n_other_clips": len(test_scores),
        "same_speaker_baseline": same_baseline,
        "different_speaker_baseline": diff_baseline,
    }
    if test_mean is not None and same_baseline is not None and diff_baseline is not None:
        span = same_baseline - diff_baseline
        if span > 0:
            position = max(0.0, min(1.0, (test_mean - diff_baseline) / span))
            sem = test_std / (len(test_scores) ** 0.5)
            ci = 1.96 * sem / span * 100
            result["voice_similarity_pct"] = position * 100
            result["voice_similarity_ci_pct"] = ci
    return result


def main():
    parser = argparse.ArgumentParser(description="Unified multi-metric evaluation of one pipeline run.")
    parser.add_argument("--input-audio", required=True)
    parser.add_argument("--generated-audio", required=True)
    parser.add_argument("--speaker", required=True)
    parser.add_argument("--true-reference", default=None,
                         help="Optional: known ground-truth transcript, if available")
    parser.add_argument("--n-other-clips", type=int, default=20)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading models...")
    processor = WhisperProcessor.from_pretrained(str(MODEL_DIR))
    whisper_model = WhisperForConditionalGeneration.from_pretrained(str(MODEL_DIR)).to(device)
    whisper_model.generation_config.max_length = None
    whisper_model.eval()

    vocab = load_vocab(str(VOCAB_PATH))
    seq2seq_tokenizer = AutoTokenizer.from_pretrained(str(SEQ2SEQ_MODEL_DIR))
    seq2seq_model = AutoModelForSeq2SeqLM.from_pretrained(str(SEQ2SEQ_MODEL_DIR))

    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb", savedir="/tmp/spkrec-ecapa-voxceleb"
    )

    # ---- Stage 1: ASR accuracy ----
    raw_transcription = transcribe(args.input_audio, processor, whisper_model, device)
    print(f"\n{'='*60}\n1. ASR ACCURACY\n{'='*60}")
    print(f"Raw transcription: {raw_transcription}")
    if args.true_reference:
        r1 = score_batch([args.true_reference], [raw_transcription])
        print(f"WER vs. true reference: {r1['wer']}")

    # ---- Stage 2: Text correction ----
    corrected_text, method = unified_correct(raw_transcription, vocab, seq2seq_tokenizer, seq2seq_model)
    print(f"\n{'='*60}\n2. TEXT CORRECTION\n{'='*60}")
    print(f"Corrected text: {corrected_text}  (method: {method})")
    if args.true_reference:
        r2 = score_batch([args.true_reference], [corrected_text])
        print(f"WER vs. true reference: {r2['wer']}")
        print(f"(Compare to raw transcription's WER above -- did correction help?)")

    # ---- Stage 3: Generation content fidelity ----
    print(f"\n{'='*60}\n3. GENERATION CONTENT FIDELITY\n{'='*60}")
    audio_16k, sr_16k = librosa.load(args.generated_audio, sr=16000, mono=True)
    tmp_path = "/tmp/eval_full_run_16k.wav"
    sf.write(tmp_path, audio_16k, sr_16k)
    roundtrip_text = transcribe(tmp_path, processor, whisper_model, device)
    r3 = score_batch([corrected_text], [roundtrip_text])
    print(f"Re-transcription of generated audio: {roundtrip_text}")
    print(f"WER vs. corrected text (did the audio say the right words?): {r3['wer']}")

    # ---- Stage 4: Voice similarity ----
    print(f"\n{'='*60}\n4. VOICE SIMILARITY\n{'='*60}")
    vs = voice_similarity(classifier, args.input_audio, args.generated_audio, args.speaker, args.n_other_clips)
    print(f"Generated vs. input (direct, may be channel-confounded): {vs['generated_vs_input_direct']:.3f}")
    if vs.get("generated_vs_other_real_clips_mean") is not None:
        print(f"Generated vs. {vs['n_other_clips']} different real {args.speaker} clips: "
              f"{vs['generated_vs_other_real_clips_mean']:.3f} (std={vs['generated_vs_other_real_clips_std']:.3f})")
    if vs.get("same_speaker_baseline") is not None:
        print(f"Same-speaker baseline: {vs['same_speaker_baseline']:.3f}")
    if vs.get("different_speaker_baseline") is not None:
        print(f"Different-speaker baseline: {vs['different_speaker_baseline']:.3f}")
    if "voice_similarity_pct" in vs:
        print(f"\nVoice similarity: {vs['voice_similarity_pct']:.0f}% "
              f"(95% CI: +/- {vs['voice_similarity_ci_pct']:.0f} percentage points)")

    # ---- Summary ----
    print(f"\n{'='*60}\nSUMMARY (independent metrics -- do not average these together)\n{'='*60}")
    if args.true_reference:
        print(f"  ASR accuracy (WER):          {r1['wer']*100:.1f}%")
        print(f"  After correction (WER):      {r2['wer']*100:.1f}%")
    print(f"  Generation content fidelity: {r3['wer']*100:.1f}% WER "
          f"({'PASS' if r3['wer'] == 0 else 'FAIL'} -- did it say the right words)")
    if "voice_similarity_pct" in vs:
        print(f"  Voice similarity:            {vs['voice_similarity_pct']:.0f}% "
              f"+/- {vs['voice_similarity_ci_pct']:.0f} points")


if __name__ == "__main__":
    main()