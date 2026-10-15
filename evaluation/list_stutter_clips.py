"""
list_stutter_clips.py

Lists the stutter test clips with the most marked stuttering, so you know which audio files to
listen to. For each clip it prints the original transcript text WITH the transcriber's marks still
in it (so you can hear what each mark refers to) and what stock Whisper-medium transcribed.

Two lists:
  1. Sound-level stutter marks: sound repetitions (c-c-c), blocks, prolongations, mid-word breaks
  2. Word or phrase repetitions only (to [/] to avoid)

How to read the marks in the original text:
  ↫c-c-c↫credits   sound repetition       ≠word    block
  wo:rd            prolongation           b^unny   break inside a word
  <it's hard> [/]  repeated phrase        to [/]   repeated word

These marks are the transcribers' notes, not a clinical severity rating. A clip with no marks can
still contain stuttering, and most sound-level marks come from one interview (54f).

Reads existing files and runs no models. Can be run from any folder:
    python evaluation/list_stutter_clips.py
"""
import argparse
import csv
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / "preprocessing"))
from build_stutter_manifest import media_id, read_utterances  # noqa: E402

MIN_DURATION = 2.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(BASE_DIR / "manifests" / "manifest_stutter.csv"))
    ap.add_argument("--stock", default=str(BASE_DIR / "results" / "stutter_baseline_stock_v2" / "stutter_baseline_results.csv"))
    ap.add_argument("--top", type=int, default=8)
    args = ap.parse_args()

    rows = [r for r in csv.DictReader(open(args.manifest, newline="")) if r["status"] == "ok"
            and float(r["duration_sec"]) >= MIN_DURATION]
    stock = {}
    if Path(args.stock).exists():
        stock = {r["utt_id"]: r["hyp_raw"] for r in csv.DictReader(open(args.stock, newline=""))}

    cha_by_media = {media_id(p): p for p in (BASE_DIR / "stutter_data" / "Voices-AWS" / "transcripts").rglob("*.cha")}
    cache = {}

    def chat_text(utt_id):
        mid, idx = utt_id.rsplit("_", 1)
        if mid not in cache:
            cache[mid] = list(read_utterances(cha_by_media[mid])) if mid in cha_by_media else []
        try:
            return cache[mid][int(idx)][1]
        except IndexError:
            return "(original text not found)"

    def show(title, selected, key_name):
        print(f"\n=== {title} ===")
        for r in selected:
            print(f"\n{r['utt_id']}.wav   {float(r['duration_sec']):.0f} s   {key_name}: {r[key_name]}")
            print(f"  original text : {chat_text(r['utt_id'])[:220]}")
            if r["utt_id"] in stock:
                print(f"  stock Whisper : {stock[r['utt_id']][:220]}")

    sld = [r for r in rows if int(r["n_sld"]) > 0]
    sld.sort(key=lambda r: (-int(r["n_sld"]) / float(r["duration_sec"]), -int(r["n_sld"])))
    show("Clips with the most sound-level stutter marks (per second)", sld[: args.top], "n_sld")

    rep = [r for r in rows if int(r["n_sld"]) == 0 and int(r["n_retraces"]) > 0]
    rep.sort(key=lambda r: (-int(r["n_retraces"]) / float(r["duration_sec"]), -int(r["n_retraces"])))
    show("Clips with the most word or phrase repetitions (no sound-level marks)", rep[: args.top], "n_retraces")

    by_interview = {}
    for r in sld:
        by_interview[r["speaker"]] = by_interview.get(r["speaker"], 0) + 1
    print(f"\nClips with sound-level marks, by interview: {by_interview}")


if __name__ == "__main__":
    main()