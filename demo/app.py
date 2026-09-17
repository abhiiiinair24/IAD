"""
app.py - Dysarthric Speech Recognition: About, Technical Details, Held-out
demo, and the full 3-stage Demo (detect / normalize / synthesize).

Four pages:
  1. About            -- what dysarthria is, why this project exists
  2. Technical Details -- pipeline, models, results (baseline vs fine-tuned chart)
  3. Held-out demo     -- known TORGO test samples with real, reproducible WER
  4. Demo              -- upload audio -> noise reduction -> (1) detect text,
                          (2) normalize/correct it, (3) synthesize the
                          corrected text back in the SAME uploaded speaker's
                          own cloned voice

Run:
    streamlit run app.py
"""

import csv
import io
import sys
import tempfile
import time
from pathlib import Path

import streamlit as st
import torch
import numpy as np
import librosa
import noisereduce as nr
import soundfile as sf
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import (
    WhisperProcessor, WhisperForConditionalGeneration,
    AutoTokenizer, AutoModelForSeq2SeqLM,
)

# --- Compatibility shim for XTTS-v2 on newer transformers versions ---
import transformers.pytorch_utils as _pytorch_utils
if not hasattr(_pytorch_utils, "isin_mps_friendly"):
    def _isin_mps_friendly(elements, test_elements):
        return torch.isin(elements, test_elements)
    _pytorch_utils.isin_mps_friendly = _isin_mps_friendly

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / "evaluation"))
sys.path.insert(0, str(BASE_DIR / "text_correction"))
sys.path.insert(0, str(BASE_DIR / "tts"))

from compute_wer import score_batch
from unified_correct import unified_correct
from phonetic_correct import load_vocab
from clone_voice import synthesize_with_verification
from confidence_score import transcribe_with_confidence

MODEL_DIR = BASE_DIR / "training_output" / "whisper-medium" / "all_speakers" / "final"
TEST_CSV = BASE_DIR / "splits" / "all_speakers" / "test.csv"
BASELINE_SUMMARY = BASE_DIR / "results" / "baseline_whisper" / "baseline_summary.csv"
FINETUNED_SUMMARY = BASE_DIR / "results" / "finetuned_whisper_medium" / "finetuned_summary.csv"
COMPARISON_MD = BASE_DIR / "results" / "comparison_summary.md"
SEQ2SEQ_MODEL_DIR = BASE_DIR / "text_correction" / "model" / "final"
VOCAB_PATH = BASE_DIR / "text_correction" / "vocab.txt"

REFERENCE_MIN_S = 3.0
REFERENCE_MAX_S = 10.0


@st.cache_resource
def load_whisper():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = WhisperProcessor.from_pretrained(str(MODEL_DIR))
    model = WhisperForConditionalGeneration.from_pretrained(str(MODEL_DIR)).to(device)
    model.generation_config.max_length = None
    model.eval()
    return processor, model, device


@st.cache_resource
def load_correction_model():
    tokenizer = AutoTokenizer.from_pretrained(str(SEQ2SEQ_MODEL_DIR))
    model = AutoModelForSeq2SeqLM.from_pretrained(str(SEQ2SEQ_MODEL_DIR))
    return tokenizer, model


@st.cache_resource
def load_tts():
    from TTS.api import TTS
    return TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=torch.cuda.is_available())


@st.cache_data
def load_results_table():
    baseline = {r["speaker"]: r for r in csv.DictReader(open(BASELINE_SUMMARY, newline=""))}
    finetuned = {r["speaker"]: r for r in csv.DictReader(open(FINETUNED_SUMMARY, newline=""))}
    speakers = sorted(set(baseline) | set(finetuned))
    data = []
    for sp in speakers:
        b = baseline.get(sp)
        f = finetuned.get(sp)
        data.append({
            "Speaker": sp,
            "Baseline WER (%)": round(float(b["wer"]) * 100, 1) if b else None,
            "Fine-tuned WER (%)": round(float(f["wer"]) * 100, 1) if f else None,
        })
    return pd.DataFrame(data)


@st.cache_data
def load_test_samples():
    with open(TEST_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    by_speaker = {}
    for row in rows:
        by_speaker.setdefault(row["speaker"], []).append(row)
    return by_speaker


def denoise_audio(audio: np.ndarray, sr: int) -> np.ndarray:
    """Spectral-gating noise reduction (stationary-noise assumption)."""
    return nr.reduce_noise(y=audio, sr=sr)


def plot_waveform_comparison(original: np.ndarray, cleaned: np.ndarray, sr: int):
    """Stacked original-vs-cleaned waveforms, sharing a time axis, with the
    region of largest change between the two highlighted so the effect of
    denoising is visible rather than just claimed."""
    n = min(len(original), len(cleaned))
    original = original[:n]
    cleaned = cleaned[:n]
    times = np.linspace(0, n / sr, num=n)

    diff = np.abs(original - cleaned)
    window = max(1, sr // 20)
    if n > window:
        kernel = np.ones(window) / window
        diff_smooth = np.convolve(diff, kernel, mode="same")
    else:
        diff_smooth = diff
    peak_idx = int(np.argmax(diff_smooth)) if n else 0
    peak_time = times[peak_idx] if n else 0
    band_half_width = 0.15

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 4.4), sharex=True)
    ax1.plot(times, original, linewidth=0.6, color="#D9534F")
    ax1.fill_between(times, original, color="#D9534F", alpha=0.2)
    ax1.set_title("Original (raw upload)", fontsize=10, loc="left")
    ax1.set_ylabel("Amplitude")

    ax2.plot(times, cleaned, linewidth=0.6, color="#4C9BE8")
    ax2.fill_between(times, cleaned, color="#4C9BE8", alpha=0.25)
    ax2.set_title("Cleaned (after noise reduction)", fontsize=10, loc="left")
    ax2.set_ylabel("Amplitude")
    ax2.set_xlabel("Time (s)")

    for ax in (ax1, ax2):
        if n:
            ax.axvspan(max(0, peak_time - band_half_width), peak_time + band_half_width,
                       color="#F0C419", alpha=0.25)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.set_facecolor("none")
        if n:
            ax.set_xlim(0, times[-1])

    if n:
        y_top = max(abs(original.max()), abs(original.min()), 0.01)
        ax1.annotate("Biggest change here", xy=(peak_time, 0), xytext=(peak_time, y_top * 1.15),
                     fontsize=8, color="#B58900", ha="center")

    fig.patch.set_alpha(0)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def animate_words(text: str, placeholder, delay: float = 0.18):
    """Reveals text one word at a time in the given placeholder, typewriter-style."""
    words = text.split()
    shown = ""
    for w in words:
        shown = (shown + " " + w).strip()
        placeholder.markdown(f"### {shown}")
        time.sleep(delay)


# ---------------- Page 1: About ----------------

def render_about_page():
    st.header("Dysarthric Speech Recognition")
    st.markdown(
        """
### What is dysarthria?

Dysarthria is a motor speech disorder caused by conditions such as cerebral
palsy, ALS, Parkinson's disease, or stroke. It affects the muscles used to
produce speech -- articulation, breath control, and vocal timing -- making
speech slower, less clear, and often difficult for both people and standard
speech-recognition systems to understand correctly.

### Why this matters

People with dysarthria frequently have their intended words correct in their
own mind, but the physical act of speaking distorts how those words come
out. Off-the-shelf automatic speech recognition (ASR) -- trained almost
entirely on typical speech -- performs dramatically worse on dysarthric
speech, which limits access to voice assistants, dictation tools, and other
speech-driven technology that many people rely on daily.

### What this project does

This project builds and tests a pipeline that:

1. **Transcribes** dysarthric speech into text, using a version of OpenAI's
   Whisper model fine-tuned specifically on dysarthric speech samples.
2. **Corrects** likely recognition errors in that text, so the output
   reflects what the speaker most likely intended to say.
3. **Synthesizes** that corrected text back into natural speech, in the
   *same speaker's own voice* -- so their words come through clearly,
   while the voice still sounds like them.

The goal is to explore whether this kind of pipeline could give people with
dysarthria a way to be understood more reliably -- by other people, and by
voice-driven technology -- without changing who they sound like.
        """
    )
    st.caption("Repo: [github.com/abhiiiinair24/IAD](https://github.com/abhiiiinair24/IAD)")


# ---------------- Page 2: Technical Details ----------------

def render_technical_page():
    st.header("Technical Details")
    st.markdown(
        """
**Dataset:** TORGO (University of Toronto) -- 8 speakers with dysarthria
(F01, F03, F04; M01-M05), ~3.3 hours of usable speech after cleaning
(removing noise markers, non-verbal phonation-task prompts, corrupt/too-short
clips, and non-literal prompt text).

**Pipeline:**
1. **Speech-to-text** -- Whisper-medium, fine-tuned using zero-shot
   Leave-One-Speaker-Out (LOSO) evaluation to measure genuine generalization
   to a new speaker, and a second model fine-tuned on all 8 speakers for
   deployment.
2. **Text correction** -- a two-part system: a fine-tuned Flan-T5 model
   corrects sentence-level fluency/repetition errors, while a phonetic
   (Metaphone + edit-distance) matcher corrects short/word-level errors
   against TORGO's known vocabulary -- catching error types a language
   model alone cannot detect from text.
3. **Voice cloning** -- XTTS-v2 synthesizes the corrected text in the
   original speaker's own voice, verified by automatically re-transcribing
   the output and retrying if it doesn't match.

**Models used:** `openai/whisper-small` (baseline), `openai/whisper-medium`
(fine-tuned), `google/flan-t5-small` (text correction), Metaphone +
Levenshtein distance (phonetic correction, non-ML), XTTS-v2 (voice cloning).
        """
    )

    st.subheader("Results: Baseline vs. Fine-tuned WER")
    df = load_results_table()
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.bar_chart(df.set_index("Speaker")[["Baseline WER (%)", "Fine-tuned WER (%)"]], stack=False)

    if COMPARISON_MD.exists():
        with st.expander("Full results writeup"):
            st.markdown(COMPARISON_MD.read_text())


# ---------------- Page 3: Held-out demo ----------------

def render_held_out_page(processor, model, device):
    st.header("Held-out demo (known, reproducible results)")
    st.caption(
        "These utterances are from the 20%-per-speaker test split the deployed "
        "model was NOT trained on -- results here are genuine, reproducible, and "
        "match the reported 15.83% held-out WER."
    )
    by_speaker = load_test_samples()

    speaker = st.selectbox("Speaker", sorted(by_speaker.keys()), key="ho_speaker")
    samples = by_speaker[speaker]
    options = [f"{row['utt_id']} — \"{row['transcript'][:40]}\"" for row in samples]
    choice_idx = st.selectbox("Utterance", range(len(options)), format_func=lambda i: options[i], key="ho_utt")
    chosen = samples[choice_idx]

    st.audio(chosen["wav_path"])
    st.write(f"**Reference (ground truth):** {chosen['transcript']}")

    if st.button("Transcribe", type="primary", key="ho_button"):
        with st.spinner("Running Whisper..."):
            audio, sr = sf.read(chosen["wav_path"])
            inputs = processor(audio, sampling_rate=sr, return_tensors="pt").input_features.to(device)
            with torch.no_grad():
                predicted_ids = model.generate(inputs, language="en", task="transcribe",
                                                no_repeat_ngram_size=3, max_new_tokens=100)
            hypothesis = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()

        placeholder = st.empty()
        animate_words(hypothesis, placeholder)

        result = score_batch([chosen["transcript"]], [hypothesis])
        if result["wer"] is not None:
            st.metric("WER for this utterance", f"{result['wer']*100:.1f}%")
        st.caption(
            "Single-utterance WER is noisy by nature (a 3-word phrase can only "
            "score 0%, 33%, 67%, or 100%) -- the 15.83% figure is averaged over "
            "622 held-out utterances, not this one sample."
        )


# ---------------- Page 4: Demo ----------------

def prepare_audio(uploaded_bytes: bytes) -> tuple[np.ndarray, str, float]:
    """Loads uploaded audio at 16kHz mono, saves it to a temp file, and
    returns (audio_array, temp_wav_path, duration_seconds)."""
    audio, sr = librosa.load(io.BytesIO(uploaded_bytes), sr=16000, mono=True)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, audio, sr)
    return audio, tmp.name, len(audio) / sr


def prepare_reference_clip(full_wav_path: str, duration: float) -> str:
    """Trims the uploaded audio to XTTS-v2's reliable reference-length window
    (see clone_voice.py: clips over ~10s destabilize generation). If the
    upload is shorter than the window, uses it as-is."""
    if duration <= REFERENCE_MAX_S:
        return full_wav_path
    audio, sr = sf.read(full_wav_path)
    trimmed = audio[: int(sr * REFERENCE_MAX_S)]
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, trimmed, sr)
    return tmp.name


def render_demo_page(processor, model, device):
    st.header("Try it: audio in, corrected speech back")
    st.caption(
        "Upload a speech sample. The pipeline detects the words spoken, corrects likely "
        "recognition errors, then speaks the corrected text back using the SAME uploaded "
        "audio as the voice reference -- so the output sounds like the original speaker."
    )
    st.warning(
        "Voice cloning reproduces the reference voice as heard in the upload, including any "
        "dysarthric acoustic character present. This corrects the recognized WORDS -- it does "
        "not alter or 'clean' the voice itself."
    )

    uploaded = st.file_uploader("Upload a .wav file", type=["wav"], key="demo_uploader")
    if uploaded is None:
        return

    st.audio(uploaded)

    if st.button("Run pipeline", type="primary"):
        audio_raw, wav_path, duration = prepare_audio(uploaded.getvalue())
        _, sr = sf.read(wav_path)

        # ---- Stage 0: Clean audio ----
        st.subheader("0. Noise reduction")
        with st.spinner("Cleaning audio..."):
            audio_clean = denoise_audio(audio_raw, sr)
            clean_wav_path = str(Path(tempfile.gettempdir()) / "demo_pipeline_clean_input.wav")
            sf.write(clean_wav_path, audio_clean, sr)
        col_a, col_b = st.columns(2)
        with col_a:
            st.caption("Original")
            st.audio(audio_raw, sample_rate=sr)
        with col_b:
            st.caption("Cleaned (used for transcription below)")
            st.audio(audio_clean, sample_rate=sr)
        with st.expander("Waveform: before vs. after"):
            plot_waveform_comparison(audio_raw, audio_clean, sr)

        # ---- Stage 1: Detect text ----
        st.subheader("1. Detected text")
        with st.spinner("Transcribing..."):
            full_text, word_confidences, overall_conf = transcribe_with_confidence(
                clean_wav_path, processor, model, device
            )
        st.write(f"**Transcription:** {full_text}")
        st.write(f"Overall confidence: {overall_conf:.2f}")
        low_conf_words = [w for w, c in word_confidences if c < 0.7]
        conf_display = "  ".join(
            f":red[{w}]" if c < 0.7 else w for w, c in word_confidences
        )
        st.markdown(f"Per-word confidence (low-confidence words highlighted): {conf_display}")
        if low_conf_words:
            st.caption(f"Low-confidence words: {', '.join(low_conf_words)}")

        # ---- Stage 2: Normalize ----
        st.subheader("2. Normalized text")
        st.caption("What we believe the speaker intended to say, after correcting likely recognition errors.")
        with st.spinner("Correcting..."):
            vocab = load_vocab(str(VOCAB_PATH))
            seq2seq_tokenizer, seq2seq_model = load_correction_model()
            corrected, method = unified_correct(full_text, vocab, seq2seq_tokenizer, seq2seq_model)
        st.write(f"**Normalized text:** {corrected}")
        st.caption(f"Correction method used: {method}")

        # ---- Stage 3: Audio ----
        st.subheader("3. Speech in the original speaker's voice")
        if duration > REFERENCE_MAX_S:
            st.caption(f"Your upload is {duration:.1f}s; using the first {REFERENCE_MAX_S:.0f}s as the voice "
                       f"reference (longer references have been found to destabilize voice cloning).")
        with st.spinner("Synthesizing voice (first run downloads XTTS-v2, ~2GB)..."):
            tts = load_tts()
            reference_wav = prepare_reference_clip(wav_path, duration)
            output_path = str(Path(tempfile.gettempdir()) / "demo_pipeline_output.wav")
            verified, attempts = synthesize_with_verification(
                tts, corrected, reference_wav, output_path, str(MODEL_DIR), max_attempts=3
            )

        if verified:
            st.success(f"Synthesized and verified correct (attempt {attempts}/3).")
        else:
            st.warning("Synthesized, but could not verify the output matches the corrected text "
                       "after 3 attempts. Result shown may be imperfect.")

        st.audio(output_path)
        with open(output_path, "rb") as f:
            st.download_button("Download .wav", f, file_name="corrected_speech.wav", mime="audio/wav")


def main():
    st.set_page_config(page_title="Dysarthric Speech ASR", layout="centered")

    with st.spinner("Loading model..."):
        processor, model, device = load_whisper()

    page = st.sidebar.radio("Page", ["About", "Technical Details", "Held-out demo", "Demo"])
    if page == "About":
        render_about_page()
    elif page == "Technical Details":
        render_technical_page()
    elif page == "Held-out demo":
        render_held_out_page(processor, model, device)
    else:
        render_demo_page(processor, model, device)


if __name__ == "__main__":
    main()