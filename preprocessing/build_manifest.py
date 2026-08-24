"""
build_manifest.py

Walks raw TORGO speaker directories (e.g. F01/Session1/, F01/Session2/, ...)
and produces a single clean manifest CSV pairing each usable utterance's
audio file with its transcript.

TORGO layout assumed per session:
    <speaker>/<session>/prompts/<utt_id>.txt
    <speaker>/<session>/wav_headMic/<utt_id>.wav
    <speaker>/<session>/wav_arrayMic/<utt_id>.wav

Usage:
    python build_manifest.py \
        --data-root ~/MunisaiAbhishek_Project/IAD/Phase1-SpeechRecognition \
        --speakers F01 F03 F04 \
        --mic headMic \
        --output ../manifests/manifest_female.csv

Output CSV columns:
    speaker, session, utt_id, wav_path, transcript, duration_sec, mic, status

`status` is one of: ok, skipped_noise, skipped_image_prompt, skipped_missing_audio,
skipped_empty_transcript — kept in the manifest (not silently dropped) so you
can audit exactly how much of the raw data got filtered and why. Downstream
training code should filter to status == "ok".
"""

import argparse
import csv
import sys
from pathlib import Path

try:
    import soundfile as sf
except ImportError:
    sf = None


def get_duration(wav_path: Path) -> float:
    """Return duration in seconds, or -1.0 if it can't be read."""
    if sf is None:
        return -1.0
    try:
        info = sf.info(str(wav_path))
        return round(info.frames / info.samplerate, 4)
    except Exception:
        return -1.0


def classify_prompt(raw_text: str) -> tuple[str, str]:
    """
    Returns (status, cleaned_transcript).
    status is 'ok', 'skipped_noise', 'skipped_image_prompt', 'skipped_empty_transcript',
    or 'skipped_task_instruction'.
    """
    text = raw_text.strip()
    if not text:
        return "skipped_empty_transcript", ""
    if text.lower() == "xxx":
        return "skipped_noise", text
    if text.lower().endswith(".jpg"):
        return "skipped_image_prompt", text
    if text.startswith("[") and text.endswith("]"):
        # TORGO bracketed prompts are task INSTRUCTIONS (e.g. "[say 'Ah-P-Eee' repeatedly]",
        # "[relax your mouth in its normal position]"), not literal words the speaker said.
        # Scoring ASR output against this text as if it were the true transcript is invalid —
        # the audio contains non-verbal phonation/articulation exercises, not this sentence.
        return "skipped_task_instruction", text
    return "ok", text


def process_speaker(speaker_dir: Path, mic: str, rows: list) -> None:
    speaker = speaker_dir.name
    session_dirs = sorted(p for p in speaker_dir.iterdir() if p.is_dir() and p.name.lower().startswith("session"))

    if not session_dirs:
        print(f"  [WARN] No Session* directories found under {speaker_dir}", file=sys.stderr)
        return

    fallback_mic = "arrayMic" if mic == "headMic" else "headMic"

    for session_dir in session_dirs:
        prompts_dir = session_dir / "prompts"

        if not prompts_dir.exists():
            print(f"  [WARN] Missing prompts/ in {session_dir}", file=sys.stderr)
            continue

        wav_dir = session_dir / f"wav_{mic}"
        session_mic = mic
        if not wav_dir.exists():
            fallback_dir = session_dir / f"wav_{fallback_mic}"
            if fallback_dir.exists():
                print(f"  [INFO] wav_{mic}/ missing in {session_dir}, falling back to wav_{fallback_mic}/", file=sys.stderr)
                wav_dir = fallback_dir
                session_mic = fallback_mic
            else:
                print(f"  [WARN] Missing wav_{mic}/ and wav_{fallback_mic}/ in {session_dir}", file=sys.stderr)
                continue

        for prompt_file in sorted(prompts_dir.glob("*.txt")):
            utt_id = prompt_file.stem
            raw_text = prompt_file.read_text(errors="ignore")
            status, transcript = classify_prompt(raw_text)

            wav_path = wav_dir / f"{utt_id}.wav"
            if not wav_path.exists():
                rows.append({
                    "speaker": speaker, "session": session_dir.name, "utt_id": utt_id,
                    "wav_path": "", "transcript": transcript, "duration_sec": -1.0,
                    "mic": session_mic, "status": "skipped_missing_audio",
                })
                continue

            if wav_path.stat().st_size == 0:
                rows.append({
                    "speaker": speaker, "session": session_dir.name, "utt_id": utt_id,
                    "wav_path": str(wav_path), "transcript": transcript, "duration_sec": -1.0,
                    "mic": session_mic, "status": "skipped_empty_audio",
                })
                continue

            duration = get_duration(wav_path) if status == "ok" else -1.0

            if status == "ok" and duration <= 0:
                # File exists and is non-empty but soundfile couldn't read valid
                # audio from it (corrupt/truncated header). Don't let a bad
                # duration silently pass through as status=ok.
                rows.append({
                    "speaker": speaker, "session": session_dir.name, "utt_id": utt_id,
                    "wav_path": str(wav_path), "transcript": transcript, "duration_sec": -1.0,
                    "mic": session_mic, "status": "skipped_corrupt_audio",
                })
                continue

            rows.append({
                "speaker": speaker, "session": session_dir.name, "utt_id": utt_id,
                "wav_path": str(wav_path), "transcript": transcript, "duration_sec": duration,
                "mic": session_mic, "status": status,
            })


def main():
    parser = argparse.ArgumentParser(description="Build a TORGO manifest CSV from raw speaker directories.")
    parser.add_argument("--data-root", required=True, help="Directory containing speaker folders (e.g. F01, F03, F04)")
    parser.add_argument("--speakers", nargs="+", required=True, help="Speaker folder names, e.g. F01 F03 F04")
    parser.add_argument("--mic", default="headMic", choices=["headMic", "arrayMic"], help="Which microphone's audio to use")
    parser.add_argument("--output", required=True, help="Path to write the output manifest CSV")
    args = parser.parse_args()

    if sf is None:
        print("[WARN] soundfile not installed — durations will be -1.0. "
              "Install with: pip install soundfile --break-system-packages", file=sys.stderr)

    data_root = Path(args.data_root).expanduser()
    rows = []

    for speaker in args.speakers:
        speaker_dir = data_root / speaker
        if not speaker_dir.exists():
            print(f"[WARN] Speaker directory not found, skipping: {speaker_dir}", file=sys.stderr)
            continue
        print(f"Processing {speaker} ...")
        process_speaker(speaker_dir, args.mic, rows)

    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["speaker", "session", "utt_id", "wav_path", "transcript", "duration_sec", "mic", "status"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)
    ok = sum(1 for r in rows if r["status"] == "ok")
    ok_hours = sum(r["duration_sec"] for r in rows if r["status"] == "ok" and r["duration_sec"] > 0) / 3600

    print(f"\nWrote {total} rows to {output_path}")
    print(f"  usable (status=ok): {ok}  (~{ok_hours:.2f} hours)")
    for status in ["skipped_noise", "skipped_image_prompt", "skipped_empty_transcript",
                   "skipped_task_instruction", "skipped_missing_audio", "skipped_empty_audio",
                   "skipped_corrupt_audio"]:
        n = sum(1 for r in rows if r["status"] == status)
        if n:
            print(f"  {status}: {n}")


if __name__ == "__main__":
    main()