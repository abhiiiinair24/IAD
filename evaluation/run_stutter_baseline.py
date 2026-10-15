"""
run_stutter_baseline.py

Baseline on the stutter test set: run the current recognizer (and, optionally, the text
correction stage) on every clip in manifest_stutter.csv and report WER. No retraining.
Results are split by whether the clip contains stutter repetitions.
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / "evaluation"))
sys.path.insert(0, str(BASE_DIR / "text_correction"))

MAX_WORDS_FOR_CORRECTION = 60   # Flan-T5-small truncates long inputs


def wer_of(score_batch, refs, hyps):
    pairs = [(r, h) for r, h in zip(refs, hyps) if r.strip()]
    if not pairs:
        return None
    return score_batch([p[0] for p in pairs], [p[1] for p in pairs])["wer"]


def fmt(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def summarize(rows, score_batch, has_correction):
    def block(name, subset):
        if not subset:
            print(f"  {name}: no clips")
            return
        ref_i = [r["ref_intended"] for r in subset]
        ref_v = [r["ref_verbatim"] for r in subset]
        raw = [r["hyp_raw"] for r in subset]
        line = (f"  {name} (n={len(subset)}): raw vs intended {fmt(wer_of(score_batch, ref_i, raw))}"
                f" | raw vs verbatim {fmt(wer_of(score_batch, ref_v, raw))}")
        if has_correction:
            cor = [r["hyp_corrected"] for r in subset]
            line += f" | corrected vs intended {fmt(wer_of(score_batch, ref_i, cor))}"
        print(line)

    print("\n=== Stutter baseline (WER, lower is better) ===")
    block("All clips", rows)
    block("Clips WITH repetitions", [r for r in rows if r["n_retraces"] > 0])
    block("Clips WITHOUT repetitions", [r for r in rows if r["n_retraces"] == 0])

    by_spk = defaultdict(list)
    for r in rows:
        by_spk[r["speaker"]].append(r)
    per = []
    print("\nPer interview (raw vs intended):")
    for spk in sorted(by_spk):
        sub = by_spk[spk]
        w = wer_of(score_batch, [r["ref_intended"] for r in sub], [r["hyp_raw"] for r in sub])
        print(f"  {spk:8s} n={len(sub):3d}  {fmt(w)}")
        if w is not None and len(sub) >= 3:
            per.append(w)
    if per:
        print(f"Macro-average over {len(per)} interviews with >=3 clips: {fmt(sum(per) / len(per))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output", required=True, help="output folder")
    ap.add_argument("--whisper-model-dir",
                    default=str(BASE_DIR / "training_output" / "whisper-medium" / "all_speakers" / "final"))
    ap.add_argument("--seq2seq-model-dir", default=str(BASE_DIR / "text_correction" / "model" / "final"))
    ap.add_argument("--vocab", default=str(BASE_DIR / "text_correction" / "vocab.txt"))
    ap.add_argument("--no-correction", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import soundfile as sf
    import torch
    from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, WhisperForConditionalGeneration,
                              WhisperProcessor)
    from compute_wer import score_batch

    manifest = [r for r in csv.DictReader(open(args.manifest, newline="")) if r["status"] == "ok"]
    if args.limit:
        manifest = manifest[: args.limit]
    print(f"{len(manifest)} clips to run")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = WhisperProcessor.from_pretrained(args.whisper_model_dir)
    model = WhisperForConditionalGeneration.from_pretrained(args.whisper_model_dir).to(device)
    model.generation_config.max_length = None
    model.eval()

    correct = None
    if not args.no_correction:
        from phonetic_correct import load_vocab
        from unified_correct import unified_correct
        vocab = load_vocab(args.vocab)
        tok = AutoTokenizer.from_pretrained(args.seq2seq_model_dir)
        s2s = AutoModelForSeq2SeqLM.from_pretrained(args.seq2seq_model_dir)
        correct = lambda text: unified_correct(text, vocab, tok, s2s)[0]

    rows = []
    for i, m in enumerate(manifest, 1):
        wav_path = Path(m["wav_path"])
        if not wav_path.is_absolute():
            wav_path = BASE_DIR / wav_path
        audio, sr = sf.read(str(wav_path))
        feats = processor(audio, sampling_rate=sr, return_tensors="pt").input_features.to(device)
        with torch.no_grad():
            ids = model.generate(feats, language="en", task="transcribe",
                                 no_repeat_ngram_size=3, max_new_tokens=200)
        hyp = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
        cor = hyp
        if correct and hyp and len(hyp.split()) <= MAX_WORDS_FOR_CORRECTION:
            cor = correct(hyp)
        rows.append({"utt_id": m["utt_id"], "speaker": m["speaker"], "n_retraces": int(m["n_retraces"]),
                     "ref_intended": m["transcript"], "ref_verbatim": m["transcript_verbatim"],
                     "hyp_raw": hyp, "hyp_corrected": cor})
        if i % 25 == 0:
            print(f"  {i}/{len(manifest)}")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "stutter_baseline_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    summarize(rows, score_batch, correct is not None)
    print("\nPer-clip results:", out / "stutter_baseline_results.csv")


if __name__ == "__main__":
    main()
