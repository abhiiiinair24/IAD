# Baseline vs Fine-tuned Whisper — Zero-shot LOSO Results

| Speaker | Baseline WER | Fine-tuned WER | Improvement (pts) | Baseline CER | Fine-tuned CER |
|---|---|---|---|---|---|
| F01 | 82.6% | 30.3% | -52.3 | 58.4% | 20.9% |
| F03 | 34.5% | 17.2% | -17.3 | 20.0% | 9.7% |
| F04 | 19.8% | 7.5% | -12.3 | 10.6% | 4.5% |
| M01 | 79.2% | 21.9% | -57.3 | 49.8% | 13.9% |
| M02 | 73.6% | 27.5% | -46.1 | 47.7% | 19.2% |
| M03 | 9.9% | 4.2% | -5.7 | 5.6% | 2.6% |
| M04 | 90.8% | 50.5% | -40.3 | 62.3% | 34.0% |
| M05 | 55.6% | 21.5% | -34.1 | 35.2% | 13.3% |

**Macro-average WER: 55.74% → 22.57%  (-33.18 points)**
**Macro-average CER: 36.19% → 14.74%**

Based on 8 speakers, zero-shot Leave-One-Speaker-Out evaluation.

---

## Three-tier comparison

| Model | WER | What it answers |
|---|---|---|
| Baseline (pretrained, zero-shot) | 55.74% | How well does an out-of-the-box model handle dysarthric speech at all? |
| Fine-tuned, LOSO zero-shot (macro-avg, 8 folds) | 22.57% | How well does fine-tuning generalize to a brand-new, never-seen speaker? |
| Fine-tuned, all-speakers (seen-speaker test) | 15.83% | How well does the system work for the specific people it was trained on? |

**Note:** the all-speakers result (15.83%) is NOT directly comparable to the LOSO zero-shot result above it — it answers a different question (performance on speakers seen during training, not generalization to a new speaker). The 6.74-point gap between the two is a rough upper bound on how much *knowing the specific person* could still help beyond generic dysarthric-speech fine-tuning alone — a natural target for a future speaker-adaptation experiment.