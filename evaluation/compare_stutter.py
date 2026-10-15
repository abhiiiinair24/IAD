"""
compare_stutter.py

Compares fine-tuned vs stock Whisper-medium on the stutter test set and tests whether
marked clips are harder, using speaker-level resampling (the speaker, not the clip,
is the unit that matters with so few interviews).
"""
import argparse
import csv
import random
import re
from collections import defaultdict

import jiwer


def norm(s):
    return " ".join(re.sub(r"[^\w\s']", " ", s.lower()).split())


def errs(ref, hyp):
    ref, hyp = norm(ref), norm(hyp)
    if not ref:
        return None
    o = jiwer.process_words(ref, hyp or "x")
    return o.substitutions + o.deletions + o.insertions, o.hits + o.substitutions + o.deletions


def wer(recs, key):
    n = sum(r["n"] for r in recs)
    return sum(r[key] for r in recs) / n if n else float("nan")


def load(path):
    return {r["utt_id"]: r for r in csv.DictReader(open(path, newline=""))}


def pct(x):
    return f"{100 * x:5.1f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--finetuned", required=True)
    ap.add_argument("--stock", required=True)
    ap.add_argument("--boots", type=int, default=2000)
    ap.add_argument("--manifest", required=True, help="rebuilt manifest_stutter.csv (reference text + stutter flags)")
    ap.add_argument("--flag", choices=["sld", "retrace", "any"], default="sld",
                    help="which marks define a 'marked' clip: sound-level (sld), word/phrase repeats (retrace), or any")
    args = ap.parse_args()

    ft, st = load(args.finetuned), load(args.stock)
    man = {r["utt_id"]: r for r in csv.DictReader(open(args.manifest, newline=""))}
    for d in (ft, st):
        for u in list(d):
            m = man.get(u)
            if not m or m["status"] != "ok":
                del d[u]
                continue
            n_r, n_s = int(m["n_retraces"]), int(m["n_sld"])
            d[u]["ref_intended"] = m["transcript"]
            d[u]["n_retraces"] = str({"sld": n_s, "retrace": n_r, "any": n_r + n_s}[args.flag])
    print(f"Marked clips = clips with {args.flag} marks\n")
    recs = []
    for u in ft:
        if u not in st:
            continue
        a = errs(ft[u]["ref_intended"], ft[u]["hyp_raw"])
        if a is None:
            continue
        recs.append({
            "spk": ft[u]["speaker"], "rep": int(ft[u]["n_retraces"]) > 0, "n": a[1],
            "ft": a[0],
            "ftc": errs(ft[u]["ref_intended"], ft[u]["hyp_corrected"])[0],
            "st": errs(st[u]["ref_intended"], st[u]["hyp_raw"])[0],
        })
    spks = sorted({r["spk"] for r in recs})
    by = defaultdict(list)
    for r in recs:
        by[r["spk"]].append(r)

    print(f"{len(recs)} clips, {len(spks)} interviews, {sum(r['rep'] for r in recs)} marked clips\n")
    print("Overall WER (intended reference):")
    print(f"  fine-tuned {pct(wer(recs, 'ft'))} | fine-tuned + correction {pct(wer(recs, 'ftc'))} | "
          f"stock {pct(wer(recs, 'st'))}")

    random.seed(0)
    d_ft_st, d_cor = [], []
    for _ in range(args.boots):
        s = [r for sp in random.choices(spks, k=len(spks)) for r in by[sp]]
        d_ft_st.append(wer(s, "ft") - wer(s, "st"))
        d_cor.append(wer(s, "ftc") - wer(s, "ft"))
    d_ft_st.sort(); d_cor.sort()
    lo, hi = int(0.025 * args.boots), int(0.975 * args.boots) - 1
    print(f"\nFine-tuned minus stock:      {100 * (wer(recs, 'ft') - wer(recs, 'st')):+.1f} pts, "
          f"95% CI [{100 * d_ft_st[lo]:+.1f}, {100 * d_ft_st[hi]:+.1f}] (resampling interviews)")
    print(f"Correction minus no-correction: {100 * (wer(recs, 'ftc') - wer(recs, 'ft')):+.1f} pts, "
          f"95% CI [{100 * d_cor[lo]:+.1f}, {100 * d_cor[hi]:+.1f}]")

    print("\nPer interview (n clips | n marked clips | fine-tuned | stock):")
    for sp in spks:
        g = by[sp]
        print(f"  {sp:6s} n={len(g):3d} rep={sum(r['rep'] for r in g):3d}   {pct(wer(g, 'ft'))}  {pct(wer(g, 'st'))}")
    wins = sum(wer(by[sp], "st") < wer(by[sp], "ft") for sp in spks)
    print(f"  stock better in {wins} of {len(spks)} interviews")

    print("\nMarked-clip effect WITHIN each interview (marked clips minus other clips, stock model):")
    diffs_st, diffs_ft = [], []
    for sp in spks:
        a = [r for r in by[sp] if r["rep"]]
        b = [r for r in by[sp] if not r["rep"]]
        if len(a) >= 3 and len(b) >= 3:
            ds, df = wer(a, "st") - wer(b, "st"), wer(a, "ft") - wer(b, "ft")
            diffs_st.append(ds); diffs_ft.append(df)
            print(f"  {sp:6s} rep n={len(a):2d} {pct(wer(a, 'st'))} vs other n={len(b):2d} {pct(wer(b, 'st'))}"
                  f"   diff {100 * ds:+.1f} pts (fine-tuned {100 * df:+.1f})")
        else:
            print(f"  {sp:6s} rep n={len(a):2d}, other n={len(b):2d}  (too few for a comparison)")
    if diffs_st:
        print(f"  Interviews with enough clips: {len(diffs_st)}; marked clips harder in "
              f"{sum(d > 0 for d in diffs_st)} of them (stock). Mean difference {100 * sum(diffs_st) / len(diffs_st):+.1f} pts.")


if __name__ == "__main__":
    main()
