"""
build_stutter_manifest.py

Turns FluencyBank Voices-AWS interviews (CHAT .cha transcripts + one 16 kHz wav per
interview) into a clip-level test set: cuts each participant utterance using its
timestamp, cleans the CHAT markup, and writes a manifest in the same column layout as
the TORGO manifests.

Two references per clip:
  transcript          = intended message: retraced/repeated material and fillers removed
  transcript_verbatim = words as spoken, including repetitions and fillers
"""
import argparse
import csv
import re
from collections import Counter
from pathlib import Path

import soundfile as sf

BULLET_STRICT = re.compile(r"\x15(\d+)_(\d+)\x15")
BULLET_LOOSE = re.compile(r"\s(\d+)_(\d+)\s*$")
RETRACE_GROUP = r"<[^<>]*>\s*\[/{1,3}\]"
RETRACE_ANY = r"\[/{1,3}\]"


def read_utterances(cha_path):
    """Yield (speaker_code, text, start_ms, end_ms) for each main-tier line."""
    lines = Path(cha_path).read_text(encoding="utf8", errors="replace").splitlines()
    tiers, cur = [], None
    for line in lines:
        if line.startswith("*"):
            cur = [line]
            tiers.append(cur)
        elif line.startswith("\t") and cur is not None:
            cur.append(line.strip())
        else:
            cur = None
    for parts in tiers:
        m = re.match(r"\*(\w+):\s*(.*)", " ".join(parts))
        if not m:
            continue
        spk, text = m.group(1), m.group(2)
        bullets = BULLET_STRICT.findall(text)
        if bullets:
            text = BULLET_STRICT.sub(" ", text)
        else:
            lm = BULLET_LOOSE.search(text)
            bullets = [lm.groups()] if lm else []
            if lm:
                text = text[:lm.start()]
        if bullets:
            start = min(int(b[0]) for b in bullets)
            end = max(int(b[1]) for b in bullets)
        else:
            start = end = None
        yield spk, text.strip(), start, end


def _common_clean(s):
    s = re.sub(r"↫[^↫]{0,20}↫", " ", s)   # sound repetition  ↫c-c-c↫credits -> credits
    s = s.replace("↫", " ")                          # stray/unpaired marks
    s = s.replace("≠", "")                           # block marker  ≠boy -> boy
    s = s.replace("^", "")                                # mid-word break  b^unny -> bunny
    s = re.sub(r"&=\S+", " ", s)           # events (&=laughs)
    s = re.sub(r"&[+~]\S*", " ", s)        # fragments / non-words
    s = re.sub(r"\(\.+\)", " ", s)         # timed pauses
    s = re.sub(r"\+\S*", " ", s)           # trailing-off / interruption codes
    s = re.sub(r"\[[^\]]*\]", " ", s)      # remaining [..] codes
    s = re.sub(r"@\w+", "", s)             # word@b style tags
    s = s.replace("_", " ")                # you_know -> you know
    s = re.sub(r"\((\w+)\)", r"\1", s)     # (be)cause -> because
    s = re.sub(r"(?<=\w):(?=\w)", "", s)   # prolongation marker wo:rd
    s = re.sub(r"\b0\w+", " ", s)          # omitted-word codes
    s = re.sub(r'[.?!,;:"<>^]', " ", s)
    s = re.sub(r"\b(xxx|yyy|www)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip().lower()


def clean_chat(text):
    """Return (intended, verbatim, n_retraces, has_unintelligible)."""
    unintelligible = bool(re.search(r"\b(xxx|yyy|www)\b", text))
    n_retraces = len(re.findall(RETRACE_ANY, text))

    v = re.sub(r"<([^<>]*)>\s*\[/{1,3}\]", r"\1", text)   # keep repeated words
    v = re.sub(RETRACE_ANY, " ", v)
    v = re.sub(r"&-(\w+)", r"\1", v)                      # fillers kept: &-um -> um
    verbatim = _common_clean(v)

    i = re.sub(RETRACE_GROUP, " ", text)                  # drop the retraced material
    i = re.sub(r"(\S+)\s*" + RETRACE_ANY, " ", i)
    i = re.sub(r"&-\w+", " ", i)                          # fillers dropped
    intended = _common_clean(i)
    return intended, verbatim, n_retraces, unintelligible


def count_sld(text):
    """Stuttering-like disfluency marks (FluencyBank CHAT conventions): sound repetitions,
    blocks, prolongations, mid-word breaks."""
    return (text.count("↫") // 2 + text.count("≠") + len(re.findall(r"(?<=[A-Za-z]):", text))
            + text.count("^"))


def media_id(cha_path):
    for line in Path(cha_path).read_text(encoding="utf8", errors="replace").splitlines():
        if line.startswith("@Media:"):
            return line.split("\t", 1)[1].split(",")[0].strip()
    return Path(cha_path).stem


def participant_codes(cha_path):
    """Speaker codes of the people being studied (everyone except investigators/interviewers),
    read from the @Participants header. Falls back to PAR if the header is missing."""
    skip = ("investigator", "interviewer", "examiner", "researcher")
    for line in Path(cha_path).read_text(encoding="utf8", errors="replace").splitlines():
        if line.startswith("@Participants:"):
            codes = set()
            for entry in line.split("\t", 1)[1].split(","):
                parts = entry.strip().split()
                if len(parts) >= 2 and not any(k in " ".join(parts[1:]).lower() for k in skip):
                    codes.add(parts[0])
            return codes or {"PAR"}
    return {"PAR"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcripts", required=True)
    ap.add_argument("--audio-dir", required=True)
    ap.add_argument("--clips-dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--min-dur", type=float, default=1.0)
    ap.add_argument("--max-dur", type=float, default=28.0)
    ap.add_argument("--min-words", type=int, default=3)
    ap.add_argument("--pad", type=float, default=0.15)
    args = ap.parse_args()

    clips_dir = Path(args.clips_dir)
    clips_dir.mkdir(parents=True, exist_ok=True)
    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
    fields = ["speaker", "session", "utt_id", "wav_path", "transcript", "duration_sec", "mic", "status",
              "transcript_verbatim", "n_retraces", "n_sld"]
    status_counts, per_speaker, rows = Counter(), Counter(), []
    ok_dur, ok_retr, ok_sld = 0.0, 0, 0

    for cha in sorted(Path(args.transcripts).rglob("*.cha")):
        mid = media_id(cha)
        codes = participant_codes(cha)
        wav = Path(args.audio_dir) / f"{mid}.wav"
        audio = sr = None
        if wav.exists():
            audio, sr = sf.read(str(wav))
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
        for i, (spk, text, start, end) in enumerate(read_utterances(cha)):
            if spk not in codes:
                continue
            intended, verbatim, n_retr, unintel = clean_chat(text)
            uid = f"{mid}_{i:04d}"
            dur = (end - start) / 1000.0 if start is not None else 0.0
            if audio is None:
                status = "skipped_missing_audio"
            elif start is None:
                status = "skipped_no_timestamp"
            elif unintel:
                status = "skipped_unintelligible"
            elif len(intended.split()) < args.min_words:
                status = "skipped_too_few_words"
            elif dur < args.min_dur:
                status = "skipped_too_short_audio"
            elif dur > args.max_dur:
                status = "skipped_too_long_audio"
            else:
                a = max(0, int((start / 1000.0 - args.pad) * sr))
                b = min(len(audio), int((end / 1000.0 + args.pad) * sr))
                if b - a < int(args.min_dur * sr):
                    status = "skipped_out_of_range"
                else:
                    status = "ok"
            wav_path = ""
            if status == "ok":
                wav_path = str(clips_dir / f"{uid}.wav")
                sf.write(wav_path, audio[a:b], sr)
                ok_dur += dur
                ok_retr += int(n_retr > 0)
                ok_sld += int(count_sld(text) > 0)
                per_speaker[mid] += 1
            status_counts[status] += 1
            rows.append({"speaker": mid, "session": "interview", "utt_id": uid, "wav_path": wav_path,
                         "transcript": intended, "duration_sec": round(dur, 2), "mic": "video",
                         "status": status, "transcript_verbatim": verbatim, "n_retraces": n_retr,
                         "n_sld": count_sld(text)})

    with open(args.manifest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    n_ok = status_counts["ok"]
    print("Status counts:", dict(status_counts))
    print(f"Usable clips: {n_ok} from {len(per_speaker)} interviews, {ok_dur / 60:.1f} min of speech")
    if n_ok:
        print(f"Clips containing a word/phrase repetition [/]: {ok_retr} ({100 * ok_retr / n_ok:.0f}%)")
        print(f"Clips containing a sound-level stutter mark (↫ ≠ : ^): {ok_sld} ({100 * ok_sld / n_ok:.0f}%)")
    print("Manifest written to", args.manifest)


if __name__ == "__main__":
    main()
