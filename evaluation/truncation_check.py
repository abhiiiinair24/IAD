"""
truncation_check.py

Checks whether the recognizers stop early on longer clips, and which clips BOTH models fail on.

For every stutter test clip it compares the number of words in the transcript with the number of
words in the reference, grouped by clip length, for the fine-tuned model and for stock
Whisper-medium. WER is split into deleted, substituted and inserted words: an early stop shows up
as a high deletion rate.

Clips where both models output almost nothing are listed with the reference text, both
transcripts, and how many seconds of the clip actually contain speech. If two different models
fail the same way, the audio and the reference text may not match (for example a clip cut from
the wrong place), which is a data problem rather than a model problem.

Reads result files that already exist; runs no models. Can be run from any folder:
    python evaluation/truncation_check.py
"""
import argparse
import csv
import re
import statistics
from collections import Counter
from pathlib import Path

import jiwer
import numpy as np
import soundfile as sf

BASE_DIR = Path(__file__).parent.parent
BUCKETS = [("under 5 s", 0, 5), ("5 to 8 s", 5, 8), ("8 to 11 s", 8, 11), ("11 s and over", 11, 999)]
SHORT_RATIO = 0.75   # a transcript with under 75% of the reference's words counts as "clearly short"
FAIL_RATIO = 0.30    # "both models fail": each outputs under 30% of the reference's words


def norm(s):
    return " ".join(re.sub(r"[^\w\s']", " ", s.lower()).split())


def load_results(path):
    return {r["utt_id"]: r for r in csv.DictReader(open(path, newline=""))}


def measure(ref, hyp):
    ref, hyp = norm(ref), norm(hyp)
    n_ref = len(ref.split())
    o = jiwer.process_words(ref, hyp or "x")
    n_hyp = len(hyp.split())
    return {"n_ref": n_ref, "ratio": n_hyp / n_ref, "del": o.deletions, "sub": o.substitutions,
            "ins": o.insertions}


def corpus_wer(rows):
    n_ref = sum(r["n_ref"] for r in rows)
    return sum(r["del"] + r["sub"] + r["ins"] for r in rows) / n_ref if n_ref else float("nan")


def summarize(rows):
    n_ref = sum(r["n_ref"] for r in rows)
    pct = lambda k: 100 * sum(r[k] for r in rows) / n_ref
    short = 100 * sum(r["ratio"] < SHORT_RATIO for r in rows) / len(rows)
    return (f"{pct('del') + pct('sub') + pct('ins'):5.1f}%  del {pct('del'):4.1f}%  sub {pct('sub'):4.1f}%  "
            f"ins {pct('ins'):4.1f}%  short {short:4.0f}%")


def active_seconds(wav_path):
    """Rough amount of speech in a clip: seconds where the 25 ms energy is above 10% of the loud level."""
    try:
        audio, sr = sf.read(str(BASE_DIR / wav_path))
    except Exception:
        return None
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    win = int(0.025 * sr)
    n = len(audio) // win
    if n == 0:
        return None
    rms = np.sqrt((audio[: n * win].reshape(n, win) ** 2).mean(axis=1))
    return float((rms > 0.1 * np.percentile(rms, 95)).sum() * 0.025)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(BASE_DIR / "manifests" / "manifest_stutter.csv"))
    ap.add_argument("--finetuned", default=str(BASE_DIR / "results" / "stutter_baseline_v2" / "stutter_baseline_results.csv"))
    ap.add_argument("--stock", default=str(BASE_DIR / "results" / "stutter_baseline_stock_v2" / "stutter_baseline_results.csv"))
    args = ap.parse_args()

    man = {r["utt_id"]: r for r in csv.DictReader(open(args.manifest, newline="")) if r["status"] == "ok"}
    ft, st = load_results(args.finetuned), load_results(args.stock)

    clips = []
    for u, m in man.items():
        if u in ft and u in st and norm(m["transcript"]):
            clips.append({
                "id": u, "dur": float(m["duration_sec"]), "ref": m["transcript"], "wav": m["wav_path"],
                "ft": measure(m["transcript"], ft[u]["hyp_raw"]),
                "st": measure(m["transcript"], st[u]["hyp_raw"]),
                "ft_text": ft[u]["hyp_raw"], "st_text": st[u]["hyp_raw"],
            })
    print(f"{len(clips)} clips with both models' transcripts\n")
    print("WER split by clip length (del = words missing, sub = wrong words, ins = extra words).")
    print("'short' = share of clips whose transcript has under 75% of the reference's words.\n")

    for name, key in (("FINE-TUNED (your model)", "ft"), ("STOCK Whisper-medium", "st")):
        print(name)
        for label, lo, hi in BUCKETS:
            rows = [c[key] for c in clips if lo <= c["dur"] < hi]
            if rows:
                print(f"  {label:14s} n={len(rows):3d}  WER {summarize(rows)}")
            else:
                print(f"  {label:14s} n=  0")
        print()

    long_clips = [c for c in clips if c["dur"] >= 8]
    if long_clips:
        mf = statistics.median(c["ft"]["ratio"] for c in long_clips)
        ms = statistics.median(c["st"]["ratio"] for c in long_clips)
        print(f"Clips of 8 s and over (n={len(long_clips)}): median transcript length as a share of the reference: "
              f"fine-tuned {100 * mf:.0f}%, stock {100 * ms:.0f}%\n")

    both = [c for c in clips if c["ft"]["ratio"] < FAIL_RATIO and c["st"]["ratio"] < FAIL_RATIO]
    rest = [c for c in clips if c not in both]
    print(f"Clips where BOTH models output under {int(FAIL_RATIO * 100)}% of the reference's words: "
          f"{len(both)} of {len(clips)}")
    print("  by interview:", dict(Counter(c["id"].split("_")[0] for c in both)))
    if both and rest:
        print(f"  WER on all clips:        fine-tuned {100 * corpus_wer([c['ft'] for c in clips]):.1f}%, "
              f"stock {100 * corpus_wer([c['st'] for c in clips]):.1f}%")
        print(f"  WER without those clips: fine-tuned {100 * corpus_wer([c['ft'] for c in rest]):.1f}%, "
              f"stock {100 * corpus_wer([c['st'] for c in rest]):.1f}%   (diagnostic only)")

    print("\nThose clips with the most reference words (check them by ear):")
    for c in sorted(both, key=lambda c: -c["ft"]["n_ref"])[:8]:
        act = active_seconds(c["wav"])
        act_txt = f"{act:.1f} s of speech" if act is not None else "speech time n/a"
        print(f"\n  {c['id']}  clip {c['dur']:.0f} s, {act_txt}")
        print(f"    reference : {c['ref'][:160]}")
        print(f"    fine-tuned: {c['ft_text'][:160]}")
        print(f"    stock     : {c['st_text'][:160]}")


if __name__ == "__main__":
    main()