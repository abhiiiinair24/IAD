"""
clone_voice.py

Synthesizes arbitrary text in one of the 8 TORGO speakers' own voice, using
XTTS-v2 voice cloning. This clones the voice AS-IS -- including whatever
dysarthric acoustic character is present in the reference audio. It does NOT
yet "clean" the voice to sound non-dysarthric -- that is a separate, harder
research problem (voice conversion), not solved by voice cloning alone.

Requires:
    pip install coqui-tts --break-system-packages
    (community-maintained fork of the original Coqui TTS library)

Usage:
    # Auto-pick the longest usable reference clip for this speaker:
    python clone_voice.py --speaker M04 --text "This is a test sentence." --output out_M04.wav

    # Or specify an exact reference clip yourself:
    python clone_voice.py --speaker M04 --reference-wav ../M04/Session1/wav_headMic/0012.wav \
        --text "This is a test sentence." --output out_M04.wav
"""

import argparse
import csv
import sys
from pathlib import Path

import torch
import soundfile as sf
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration

sys.path.insert(0, str(Path(__file__).parent.parent / "evaluation"))
from compute_wer import score_batch

# --- Compatibility shim ---
import transformers.pytorch_utils as _pytorch_utils
if not hasattr(_pytorch_utils, "isin_mps_friendly"):
    def _isin_mps_friendly(elements, test_elements):
        return torch.isin(elements, test_elements)
    _pytorch_utils.isin_mps_friendly = _isin_mps_friendly

from TTS.api import TTS

BASE_DIR = Path(__file__).parent.parent
MANIFEST_FEMALE = BASE_DIR / "manifests" / "manifest_female.csv"
MANIFEST_MALE = BASE_DIR / "manifests" / "manifest_male.csv"


def pick_reference_clip(speaker: str, min_duration: float = 4.0, max_duration: float = 10.0) -> str:
    in_range, too_short_or_long = [], []
    for path in [MANIFEST_FEMALE, MANIFEST_MALE]:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                if row["speaker"] == speaker and row["status"] == "ok":
                    duration = float(row["duration_sec"])
                    if min_duration <= duration <= max_duration:
                        in_range.append((duration, row["wav_path"]))
                    elif duration > max_duration:
                        too_short_or_long.append((duration, row["wav_path"]))

    if in_range:
        in_range.sort(reverse=True)
        best_duration, best_path = in_range[0]
        print(f"Auto-selected reference clip for {speaker}: {best_path} ({best_duration:.1f}s, within {min_duration}-{max_duration}s target range)")
        return best_path

    if too_short_or_long:
        too_short_or_long.sort()
        _, fallback_path = too_short_or_long[0]
        print(f"No clip in {min_duration}-{max_duration}s range for {speaker}; "
              f"will trim {fallback_path} to {max_duration}s instead.")
        return trim_reference_clip(fallback_path, max_duration)

    raise ValueError(
        f"No usable reference clip found for speaker {speaker} at all. "
        f"Try passing --reference-wav explicitly."
    )


def trim_reference_clip(wav_path: str, max_duration: float, output_dir: str = "/tmp") -> str:
    import soundfile as sf
    audio, sr = sf.read(wav_path)
    trimmed = audio[: int(sr * max_duration)]
    out_path = str(Path(output_dir) / f"trimmed_{Path(wav_path).name}")
    sf.write(out_path, trimmed, sr)
    return out_path


def pick_clearest_reference_clip(speaker: str, whisper_model_dir: str,
                                  min_duration: float = 4.0, max_duration: float = 10.0,
                                  top_k_by_duration: int = 15) -> str:
    candidates = []
    for path in [MANIFEST_FEMALE, MANIFEST_MALE]:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                if row["speaker"] == speaker and row["status"] == "ok":
                    duration = float(row["duration_sec"])
                    if min_duration <= duration <= max_duration:
                        candidates.append((duration, row["wav_path"], row["transcript"]))

    if not candidates:
        return pick_reference_clip(speaker, min_duration, max_duration)

    candidates.sort(reverse=True)
    candidates = candidates[:top_k_by_duration]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = WhisperProcessor.from_pretrained(whisper_model_dir)
    model = WhisperForConditionalGeneration.from_pretrained(whisper_model_dir).to(device)
    model.generation_config.max_length = None
    model.eval()

    scored = []
    for duration, wav_path, transcript in candidates:
        audio, sr = sf.read(wav_path)
        inputs = processor(audio, sampling_rate=sr, return_tensors="pt").input_features.to(device)
        with torch.no_grad():
            predicted_ids = model.generate(inputs, language="en", task="transcribe",
                                            no_repeat_ngram_size=3, max_new_tokens=100)
        hypothesis = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
        result = score_batch([transcript], [hypothesis])
        wer = result["wer"] if result["wer"] is not None else 1.0
        scored.append((wer, -duration, wav_path, duration))

    scored.sort()
    best_wer, _, best_path, best_duration = scored[0]
    print(f"Auto-selected CLEAREST reference clip for {speaker}: {best_path} "
          f"({best_duration:.1f}s, WER={best_wer:.2f} on its own transcript)")
    return best_path


def synthesize_with_verification(tts, text: str, reference_wav: str, output_path: str,
                                  whisper_model_dir: str, max_attempts: int = 3, language: str = "en") -> tuple[bool, int]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = WhisperProcessor.from_pretrained(whisper_model_dir)
    whisper_model = WhisperForConditionalGeneration.from_pretrained(whisper_model_dir).to(device)
    whisper_model.generation_config.max_length = None
    whisper_model.eval()

    for attempt in range(1, max_attempts + 1):
        tts.tts_to_file(text=text, file_path=output_path, speaker_wav=reference_wav, language=language)

        audio_16k, sr_16k = librosa.load(output_path, sr=16000, mono=True)
        inputs = processor(audio_16k, sampling_rate=sr_16k, return_tensors="pt").input_features.to(device)
        with torch.no_grad():
            predicted_ids = whisper_model.generate(inputs, language="en", task="transcribe",
                                                     no_repeat_ngram_size=3, max_new_tokens=100)
        roundtrip_text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()

        result = score_batch([text], [roundtrip_text])
        wer = result["wer"] if result["wer"] is not None else 1.0

        if wer == 0.0:
            print(f"  Attempt {attempt}/{max_attempts}: verified correct (\"{roundtrip_text}\")")
            return True, attempt
        else:
            print(f"  Attempt {attempt}/{max_attempts}: round-trip said \"{roundtrip_text}\" "
                  f"(expected \"{text}\") -- retrying" if attempt < max_attempts else
                  f"  Attempt {attempt}/{max_attempts}: round-trip said \"{roundtrip_text}\" "
                  f"(expected \"{text}\") -- giving up, keeping this output")

    return False, max_attempts


def main():
    parser = argparse.ArgumentParser(description="Clone a TORGO speaker's voice to speak new text (XTTS-v2).")
    parser.add_argument("--speaker", required=True, help="e.g. M04")
    parser.add_argument("--text", required=True, help="Text to synthesize in the speaker's voice")
    parser.add_argument("--output", required=True, help="Output .wav path")
    parser.add_argument("--reference-wav", default=None,
                         help="Optional: exact reference clip to clone from. If omitted, "
                              "auto-picks the speaker's longest usable clip.")
    parser.add_argument("--min-ref-duration", type=float, default=4.0,
                         help="Minimum reference clip duration to consider when auto-selecting (default 4.0s)")
    parser.add_argument("--reference-selection", choices=["longest", "clearest"], default="clearest",
                         help="'longest': just the longest clip. 'clearest': longest clip the model ALSO "
                              "transcribes correctly -- better when severity varies utterance-to-utterance.")
    parser.add_argument("--whisper-model-dir", default=None,
                         help="Required when --reference-selection=clearest, or when --verify is used")
    parser.add_argument("--verify", action="store_true",
                         help="Re-transcribe the output and retry (up to --max-attempts times) if it doesn't "
                              "match the requested text. XTTS-v2 occasionally produces a wrong/garbled result "
                              "by chance on short text; retrying usually fixes it.")
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--language", default="en")
    args = parser.parse_args()

    if args.verify and not args.whisper_model_dir:
        raise ValueError("--whisper-model-dir is required when --verify is used")

    if args.reference_wav:
        reference_wav = args.reference_wav
    elif args.reference_selection == "clearest":
        if not args.whisper_model_dir:
            raise ValueError("--whisper-model-dir is required when --reference-selection=clearest")
        reference_wav = pick_clearest_reference_clip(args.speaker, args.whisper_model_dir, args.min_ref_duration)
    else:
        reference_wav = pick_reference_clip(args.speaker, args.min_ref_duration)

    print("Loading XTTS-v2 (first run downloads the model, ~2GB) ...")
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)

    print(f"Synthesizing: \"{args.text}\"")
    if args.verify:
        verified, attempts = synthesize_with_verification(
            tts, args.text, reference_wav, args.output, args.whisper_model_dir, args.max_attempts, args.language
        )
        status = "VERIFIED correct" if verified else "NOT verified -- kept best-effort output"
        print(f"Saved to {args.output} ({status} after {attempts} attempt(s))")
    else:
        tts.tts_to_file(
            text=args.text,
            file_path=args.output,
            speaker_wav=reference_wav,
            language=args.language,
        )
        print(f"Saved to {args.output}")

    print("\nNOTE: this clones the reference voice AS-IS, including any dysarthric "
          "acoustic character present in the reference clip. It does not produce "
          "'cleaned'/non-dysarthric speech -- that is a separate research step.")


if __name__ == "__main__":
    main()
