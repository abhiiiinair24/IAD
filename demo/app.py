"""
app.py - Streamlit demo for the TORGO dysarthric-speech ASR project.

Three tabs:
  1. About        -- what this project is, the pipeline, the models, the results
  2. Held-out demo -- transcribe known TORGO test utterances (reproducible, matches reported WER)
  3. Upload audio  -- transcribe any .wav you upload from your own machine (no ground truth,
                      so no WER guarantee -- see the caveat shown in that tab)

Run:
    streamlit run app.py
"""

import csv
import io
import sys
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
from transformers import WhisperProcessor, WhisperForConditionalGeneration

sys.path.insert(0, str(Path(__file__).parent.parent / "evaluation"))
from compute_wer import score_batch

BASE_DIR = Path(__file__).parent.parent
MODEL_DIR = BASE_DIR / "training_output" / "whisper-medium" / "all_speakers" / "final"
TEST_CSV = BASE_DIR / "splits" / "all_speakers" / "test.csv"
BASELINE_SUMMARY = BASE_DIR / "results" / "baseline_whisper" / "baseline_summary.csv"
FINETUNED_SUMMARY = BASE_DIR / "results" / "finetuned_whisper_medium" / "finetuned_summary.csv"
COMPARISON_MD = BASE_DIR / "results" / "comparison_summary.md"
MANIFEST_FEMALE = BASE_DIR / "manifests" / "manifest_female.csv"
MANIFEST_MALE = BASE_DIR / "manifests" / "manifest_male.csv"


@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = WhisperProcessor.from_pretrained(str(MODEL_DIR))
    model = WhisperForConditionalGeneration.from_pretrained(str(MODEL_DIR)).to(device)
    model.generation_config.max_length = None
    model.eval()
    return processor, model, device


@st.cache_data
def load_test_samples():
    with open(TEST_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    by_speaker = {}
    for row in rows:
        by_speaker.setdefault(row["speaker"], []).append(row)
    return by_speaker


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
def load_manifest_lookup():
    """Maps (speaker, session, utt_id) -> ground-truth transcript.

    TORGO reuses the same numeric filename (e.g. 0030.wav) across different
    sessions for the same speaker, with a DIFFERENT prompt each time -- so
    (speaker, utt_id) alone is ambiguous and will silently return the wrong
    transcript. Session must be part of the key.
    """
    lookup = {}
    for path in [MANIFEST_FEMALE, MANIFEST_MALE]:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                lookup[(row["speaker"], row["session"], row["utt_id"])] = row["transcript"]
    return lookup


@st.cache_data
def get_sessions_for_speaker(speaker: str) -> list[str]:
    lookup = load_manifest_lookup()
    sessions = sorted({s for (sp, s, _) in lookup.keys() if sp == speaker})
    return sessions


def transcribe_array(audio: np.ndarray, sr: int, processor, model, device: str) -> str:
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt").input_features.to(device)
    with torch.no_grad():
        predicted_ids = model.generate(
            inputs, language="en", task="transcribe",
            no_repeat_ngram_size=3, max_new_tokens=100,
        )
    return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()


def transcribe_file(wav_path: str, processor, model, device: str) -> str:
    audio, sr = sf.read(wav_path)
    if sr != 16000:
        raise ValueError(f"{wav_path} is {sr}Hz, expected 16kHz.")
    return transcribe_array(audio, sr, processor, model, device)


def render_about_tab():
    st.header("About this project")
    st.markdown(
        """
Dysarthria is a motor speech disorder that affects articulation, making speech
harder for standard ASR systems to transcribe accurately. This project fine-tunes
OpenAI's **Whisper** on the **TORGO** dysarthric speech corpus and measures how much
that closes the gap.

**Dataset:** TORGO (University of Toronto) — 8 speakers with dysarthria (3 female:
F01, F03, F04; 5 male: M01–M05), ~3.3 hours of usable speech after cleaning
(filtering noise markers, non-verbal phonation-task prompts, corrupt/too-short
audio clips, and image-description prompts that have no literal transcript).

**Method — zero-shot Leave-One-Speaker-Out (LOSO):** for each of the 8 speakers,
Whisper-medium is fine-tuned on the *other 7* and tested on the held-out speaker,
who the model has never heard during training. This measures genuine generalization
to a brand-new person, not memorization.

**Models compared:**
- **Baseline** — pretrained `openai/whisper-small`, no fine-tuning
- **Fine-tuned, LOSO** — Whisper-medium, fine-tuned per fold, tested zero-shot on the held-out speaker
- **Fine-tuned, all-speakers** — Whisper-medium fine-tuned on 80% of *every* speaker's data (this is the model powering the demo tabs above)
        """
    )

    st.subheader("Results")
    df = load_results_table()
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.bar_chart(df.set_index("Speaker")[["Baseline WER (%)", "Fine-tuned WER (%)"]])

    if COMPARISON_MD.exists():
        with st.expander("Full results writeup (comparison_summary.md)"):
            st.markdown(COMPARISON_MD.read_text())

    st.caption(
        "Repo: [github.com/abhiiiinair24/IAD](https://github.com/abhiiiinair24/IAD)"
    )


def render_held_out_tab(processor, model, device):
    st.header("Try a held-out TORGO sample")
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
            hypothesis = transcribe_file(chosen["wav_path"], processor, model, device)
        st.write(f"**Model hypothesis:** {hypothesis}")

        result = score_batch([chosen["transcript"]], [hypothesis])
        if result["wer"] is not None:
            st.metric("WER for this utterance", f"{result['wer']*100:.1f}%")
        st.caption(
            "Single-utterance WER is noisy by nature (a 3-word phrase can only "
            "score 0%, 33%, 67%, or 100%) -- the 15.83% figure is averaged over "
            "622 held-out utterances, not this one sample."
        )


def animate_words(text: str, placeholder, delay: float = 0.18):
    """Reveals text one word at a time in the given placeholder, typewriter-style."""
    words = text.split()
    shown = ""
    for w in words:
        shown = (shown + " " + w).strip()
        placeholder.markdown(f"### {shown}")
        time.sleep(delay)


def plot_waveform(audio: np.ndarray, sr: int):
    times = np.linspace(0, len(audio) / sr, num=len(audio))
    fig, ax = plt.subplots(figsize=(8, 2.2))
    ax.plot(times, audio, linewidth=0.6, color="#4C9BE8")
    ax.fill_between(times, audio, color="#4C9BE8", alpha=0.25)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    if len(times):
        ax.set_xlim(0, times[-1])
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


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

    # Smoothed absolute difference to find where denoising changed the signal most.
    diff = np.abs(original - cleaned)
    window = max(1, sr // 20)  # ~50ms smoothing window
    if n > window:
        kernel = np.ones(window) / window
        diff_smooth = np.convolve(diff, kernel, mode="same")
    else:
        diff_smooth = diff
    peak_idx = int(np.argmax(diff_smooth)) if n else 0
    peak_time = times[peak_idx] if n else 0
    band_half_width = 0.15  # seconds

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


def render_upload_tab(processor, model, device):
    st.header("Try a TORGO audio clip")
    st.caption(
        "Upload any .wav file from the TORGO dataset (from your desktop copy). "
        "Since these are known TORGO recordings, the real ground-truth transcript "
        "is looked up automatically once you pick the correct speaker below -- "
        "giving you a genuine accuracy score, not just a raw transcription."
    )

    speaker_options = ["F01", "F03", "F04", "M01", "M02", "M03", "M04", "M05"]
    speaker = st.selectbox("Which speaker is this clip from?", speaker_options, key="up_speaker")
    session_options = get_sessions_for_speaker(speaker)
    session = st.selectbox(
        "Which session? (TORGO reuses filenames like 0030.wav across sessions "
        "with different content -- this is required to look up the right transcript)",
        session_options, key="up_session",
    )

    if "uploader_counter" not in st.session_state:
        st.session_state.uploader_counter = 0
    if "up_result" not in st.session_state:
        st.session_state.up_result = None
    if "up_last_file_id" not in st.session_state:
        st.session_state.up_last_file_id = None

    uploaded = st.file_uploader(
        "Upload a TORGO .wav file",
        type=["wav"],
        key=f"uploader_{st.session_state.uploader_counter}",
    )

    if uploaded is not None:
        st.audio(uploaded)
        file_id = getattr(uploaded, "file_id", None) or (uploaded.name, uploaded.size)

        if file_id != st.session_state.up_last_file_id:
            # New file detected -- transcribe automatically, no extra button needed.
            with st.spinner("Loading audio..."):
                audio_raw, sr = librosa.load(io.BytesIO(uploaded.getvalue()), sr=16000, mono=True)
            with st.spinner("Cleaning audio (noise reduction)..."):
                audio_clean = denoise_audio(audio_raw, sr)
            with st.spinner("Running Whisper..."):
                hypothesis = transcribe_array(audio_clean, sr, processor, model, device)

            utt_id = Path(uploaded.name).stem
            reference = load_manifest_lookup().get((speaker, session, utt_id))

            st.session_state.up_result = {
                "audio_raw": audio_raw, "audio_clean": audio_clean, "sr": sr, "hypothesis": hypothesis,
                "reference": reference, "utt_id": utt_id, "speaker": speaker, "session": session,
            }
            st.session_state.up_last_file_id = file_id
            st.session_state.up_revealed = False

        result = st.session_state.up_result

        st.subheader("Compare: original vs. cleaned audio")
        st.caption("Same clip, side by side -- listen for the background noise/hiss disappearing in the cleaned version.")
        col_a, col_b = st.columns(2)
        with col_a:
            st.caption("Original (raw, resampled to 16kHz)")
            st.audio(result["audio_raw"], sample_rate=result["sr"])
        with col_b:
            st.caption("Cleaned (after noise reduction)")
            st.audio(result["audio_clean"], sample_rate=result["sr"])

        st.divider()
        if not st.session_state.get("up_revealed"):
            st.write("Listen to the clip above, then reveal the model's transcription and the ground truth.")
            if st.button("Reveal transcription", type="primary"):
                st.session_state.up_revealed = True
                st.rerun()
        else:
            st.write("**Model hypothesis:**")
            placeholder = st.empty()
            animate_words(result["hypothesis"], placeholder)

            if result["reference"]:
                st.write(f"**Ground truth ({result['speaker']}/{result['session']}/{result['utt_id']}):** {result['reference']}")
                wer_result = score_batch([result["reference"]], [result["hypothesis"]])
                if wer_result["wer"] is not None:
                    if wer_result["wer"] > 1.0:
                        st.warning(
                            f"Computed WER is {wer_result['wer']*100:.0f}%, over 100%. This usually means "
                            f"the selected speaker/session doesn't actually match the audio you uploaded -- "
                            f"TORGO reuses filenames like `{result['utt_id']}.wav` across many speakers and "
                            f"sessions with completely different content each time, so an incorrect selection "
                            f"can still return a real (but wrong) transcript. Double-check that "
                            f"{result['speaker']}/{result['session']} is really where this file came from -- "
                            f"the score below isn't a reliable accuracy measurement until that's confirmed."
                        )
                    accuracy = max(0.0, 1 - wer_result["wer"]) * 100
                    col1, col2 = st.columns(2)
                    col1.metric("Accuracy", f"{accuracy:.1f}%")
                    col2.metric("WER", f"{wer_result['wer']*100:.1f}%")
            else:
                st.info(
                    f"No matching transcript found for speaker={result['speaker']}, "
                    f"session={result['session']}, utt_id={result['utt_id']}. Double-check "
                    f"the speaker and session selections match the file you uploaded."
                )

            st.subheader("Waveform: before vs. after noise reduction")
            st.caption(
                "Top: your raw upload. Bottom: after spectral-gating noise reduction "
                "(this is the version actually fed to Whisper). The highlighted band "
                "marks where the two differ the most."
            )
            plot_waveform_comparison(result["audio_raw"], result["audio_clean"], result["sr"])

        if st.button("Upload another sample"):
            st.session_state.uploader_counter += 1
            st.session_state.up_result = None
            st.session_state.up_last_file_id = None
            st.session_state.up_revealed = False
            st.rerun()


def main():
    st.set_page_config(page_title="Dysarthric Speech ASR", layout="centered")
    st.title("Dysarthric Speech Recognition")

    with st.spinner("Loading model..."):
        processor, model, device = load_model()

    tab_about, tab_held_out, tab_upload = st.tabs(["About", "Held-out demo", "Try TORGO sample"])
    with tab_about:
        render_about_tab()
    with tab_held_out:
        render_held_out_tab(processor, model, device)
    with tab_upload:
        render_upload_tab(processor, model, device)


if __name__ == "__main__":
    main()