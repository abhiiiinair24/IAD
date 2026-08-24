"""
explore_dataset.py

Reads a manifest CSV produced by preprocessing/build_manifest.py and generates
exploratory plots + a summary report. Fully decoupled from preprocessing —
point it at any manifest CSV with the expected columns.

Usage:
    python explore_dataset.py --manifest ../manifests/manifest_female.csv --out-dir ./plots
"""

import argparse
import csv
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")  # headless-safe for a training server with no display
import matplotlib.pyplot as plt


def load_manifest(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def summary_report(rows: list[dict]) -> str:
    by_speaker = defaultdict(lambda: {"ok": 0, "hours": 0.0, "skipped": 0})
    for r in rows:
        sp = r["speaker"]
        if r["status"] == "ok":
            by_speaker[sp]["ok"] += 1
            dur = float(r["duration_sec"])
            if dur > 0:
                by_speaker[sp]["hours"] += dur / 3600
        else:
            by_speaker[sp]["skipped"] += 1

    lines = ["=== Dataset Summary ===", ""]
    total_ok = sum(v["ok"] for v in by_speaker.values())
    total_hours = sum(v["hours"] for v in by_speaker.values())
    lines.append(f"Total usable utterances: {total_ok}")
    lines.append(f"Total usable audio: {total_hours:.2f} hours")
    lines.append("")
    lines.append(f"{'Speaker':<10}{'Usable':<10}{'Hours':<10}{'Skipped':<10}")
    for sp in sorted(by_speaker):
        v = by_speaker[sp]
        lines.append(f"{sp:<10}{v['ok']:<10}{v['hours']:<10.2f}{v['skipped']:<10}")
    return "\n".join(lines)


def plot_duration_histogram(rows: list[dict], out_dir: Path):
    durations = [float(r["duration_sec"]) for r in rows if r["status"] == "ok" and float(r["duration_sec"]) > 0]
    plt.figure(figsize=(8, 5))
    plt.hist(durations, bins=40, color="#4C72B0", edgecolor="black")
    plt.xlabel("Utterance duration (seconds)")
    plt.ylabel("Count")
    plt.title("Distribution of utterance durations")
    plt.tight_layout()
    plt.savefig(out_dir / "duration_histogram.png", dpi=150)
    plt.close()


def plot_utterances_per_speaker(rows: list[dict], out_dir: Path):
    counts = defaultdict(int)
    for r in rows:
        if r["status"] == "ok":
            counts[r["speaker"]] += 1
    speakers = sorted(counts)
    plt.figure(figsize=(6, 5))
    plt.bar(speakers, [counts[s] for s in speakers], color="#55A868")
    plt.xlabel("Speaker")
    plt.ylabel("Usable utterances")
    plt.title("Usable utterances per speaker")
    plt.tight_layout()
    plt.savefig(out_dir / "utterances_per_speaker.png", dpi=150)
    plt.close()


def plot_status_breakdown(rows: list[dict], out_dir: Path):
    counts = defaultdict(int)
    for r in rows:
        counts[r["status"]] += 1
    labels = sorted(counts)
    plt.figure(figsize=(7, 5))
    plt.bar(labels, [counts[l] for l in labels], color="#C44E52")
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Count")
    plt.title("Utterance status breakdown (filtering audit)")
    plt.tight_layout()
    plt.savefig(out_dir / "status_breakdown.png", dpi=150)
    plt.close()


def plot_word_count_histogram(rows: list[dict], out_dir: Path):
    word_counts = [len(r["transcript"].split()) for r in rows if r["status"] == "ok"]
    plt.figure(figsize=(8, 5))
    plt.hist(word_counts, bins=range(0, max(word_counts, default=1) + 2), color="#8172B2", edgecolor="black")
    plt.xlabel("Words per transcript")
    plt.ylabel("Count")
    plt.title("Distribution of transcript lengths")
    plt.tight_layout()
    plt.savefig(out_dir / "word_count_histogram.png", dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Explore a TORGO manifest CSV.")
    parser.add_argument("--manifest", required=True, help="Path to manifest CSV")
    parser.add_argument("--out-dir", default="./plots", help="Directory to write plots + report to")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_manifest(manifest_path)
    if not rows:
        print("Manifest is empty — nothing to plot.")
        return

    report = summary_report(rows)
    print(report)
    (out_dir / "summary.txt").write_text(report)

    plot_duration_histogram(rows, out_dir)
    plot_utterances_per_speaker(rows, out_dir)
    plot_status_breakdown(rows, out_dir)
    plot_word_count_histogram(rows, out_dir)

    print(f"\nPlots + summary.txt written to {out_dir}")


if __name__ == "__main__":
    main()

