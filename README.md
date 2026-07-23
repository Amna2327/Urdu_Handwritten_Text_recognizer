# Urdu Handwritten Text Recognizer

An offline handwriting recognition system for Urdu, built on a CNN + BiLSTM +
CTC architecture. This README documents the debugging journey, the
experiments run, current results, and the concrete next steps identified
through error analysis.

## Summary

- **Best model:** 3-layer BiLSTM, fc+conv dropout, trained with data
  augmentation
- **Character Error Rate (CER):** 11.66% (validation split)
- **Root cause of an early, complete training collapse:** identified and
  fixed (see below)
- **Primary remaining error source:** word-spacing / boundary errors — not
  general character misrecognition

## Architecture

```
Input image (grayscale, resized to height=64, padded to width=1400)
        |
   CNN Encoder (6 conv layers, batchnorm, LeakyReLU, dropout, max-pooling)
        |
   BiLSTM (3 stacked layers, hidden_size=256, bidirectional, dropout)
        |
   Linear -> log_softmax
        |
   CTC Loss (training) / Beam search decode (inference)
```

## The debugging journey

Training initially collapsed to predicting blank (or a handful of generic
classes) almost everywhere, with a plateaued loss curve regardless of
hyperparameter tuning. Ruled out along the way: dead/vanishing gradients
(confirmed gradients were flowing and non-trivial), and CNN feature
collapse (confirmed CNN outputs genuinely differed between distinct input
images).

**Root cause, found via a first-principles overfit sanity check:** the CTC
`input_length` passed to `torch.nn.CTCLoss` was fixed at 175 for every
sample, regardless of that sample's real content width. Since images were
padded to a fixed canvas (max width 1400px) before being fed to the CNN,
but the *median* image's real content only occupied roughly 65% of that
canvas, CTC was being told "175 valid timesteps" when a large fraction of
those timesteps were actually blank padding. This forced CTC to find
alignments that stretched labels across mostly-empty regions, systematically
pushing the model toward predicting blank everywhere.

**Fix:** compute each image's true content width post-resize, divide by the
model's total downsampling factor (8x, from three (2,2) pooling steps), and
pass that per-sample value as `input_length` instead of a fixed constant.

**Verification:** an overfit sanity check on 5 training samples, which had
previously plateaued at a high loss, converged to exact predictions after
~400 epochs once the fix was applied — confirming the pipeline
(data loading, label encoding, CTC alignment, and the model itself) was
fundamentally sound.

## Experiments

All experiments trained on the full training split, evaluated on the
validation split, LR=3e-4, Adam, `ReduceLROnPlateau` scheduler, early
stopping (patience=8 unless noted).

| Config | Best val_loss | Epoch | Train loss (at best) | Train/val gap | CER |
|---|---|---|---|---|---|
| No dropout | 0.8053 | 9 | 0.2220 | 0.68 | — |
| fc-dropout only | 0.8182 | 9 | 0.2733 | 0.64 | — |
| fc+conv dropout, 1-layer BiLSTM | 0.6439 | 34 | 0.4520 | 0.21 | 13.50% |
| fc+conv dropout, 2-layer BiLSTM | 0.5972 | 31 | 0.3862 | 0.21 | 12.29% |
| fc+conv dropout, 3-layer BiLSTM | 0.6046 | 32 | 0.3359 | 0.27 | 11.67% |
| fc+conv dropout, 3-layer BiLSTM + augmentation | **0.5834** | 31 | 0.3887 | **0.19** | **11.66%** |

**Findings:**
- Dropout (conv + fc) was essential for generalization — removing it caused
  early collapse into severe overfitting (train/val gap of 0.64–0.68 vs.
  0.19–0.27 with dropout).
- Increasing BiLSTM depth from 1→2→3 layers gave consistent CER
  improvements (13.50% → 12.29% → 11.67%), even though val_loss and CER
  didn't always move in the same direction across configs — a reminder that
  CTC loss (probability calibration) and CER (final decoded accuracy) are
  related but distinct metrics, and CER is the one that matters for
  real-world usefulness.
- A 4th BiLSTM layer was not tested — 3-layer already showed a widening
  train/val gap (0.27), suggesting depth was approaching a capacity ceiling
  for this data volume (~8000 training samples).
- Data augmentation (mild rotation, elastic distortion, brightness/contrast
  jitter, applied to training data only) improved generalization
  calibration (gap shrank from 0.27 → 0.19) but did **not** meaningfully
  improve CER (11.67% → 11.66%). Combined with the error analysis below,
  this suggests the augmentation is not addressing the specific pattern
  causing most remaining errors.

## Error analysis

A confusion-matrix analysis was run on validation predictions (edit-distance
alignment between predicted and reference strings, classifying each error
as a substitution, insertion, or deletion).

**Headline finding: word-spacing dominates the error distribution.**

| Error type | Count |
|---|---|
| Space deletions | 539 |
| Space insertions | 327 |
| **Total space-related errors** | **866** |
| Next-largest single-character deletion (ی) | 172 |

Space-related errors are roughly **5x** larger than the next-largest
individual character error, and total deletions (~2,700) outnumber
insertions (~740) by a wide margin — the model tends to drop characters far
more than it hallucinates extra ones.

A diagnostic check (stripping spaces from both predictions and references
before recomputing CER) showed CER got *worse*, not better, after removing
spaces — counterintuitive at first, but explained by how CER is computed:
spaces act as alignment anchors in the edit-distance calculation, containing
the "blast radius" of any single error to one word. Removing them lets a
single spacing error cascade into misalignment across the rest of the line,
inflating the apparent error count. This is further evidence that spacing
errors are not just costing the space character itself but are propagating
damage into the surrounding text.

**Why this matters:** word-boundary placement in cursive Urdu (Nastaliq
script) is often genuinely ambiguous from the image alone — inter-word and
intra-word gaps can look visually similar. This is a strong signal that a
**language model prior**, not more visual augmentation, is the right next
lever: deciding where word boundaries plausibly fall is a linguistic
decision, not a visual one.

## Comparison to prior work

> Maqsood, A., Riaz, N., Ul-Hasan, A., & Shafait, F. (2023). *A Unified
> Architecture for Urdu Printed and Handwritten Text Recognition.*
> National Center of Artificial Intelligence (NCAI), National University of
> Sciences and Technology (NUST), Islamabad, Pakistan.

This prior work from the same lab used a CNN + Transformer encoder-decoder
architecture, with a Transformer decoder pretrained on a 1M-document Urdu
news corpus (Urdu News Dataset 1M), achieving **5.9–6.2% CER** on
NUST-UHWR.

Their own reported baselines for simpler CNN+RNN+CTC architectures
(comparable in category to this project's model) scored:

| Model | val CER | test CER |
|---|---|---|
| CNN-RNN | 13.25% | 14.12% |
| BiGRU | 13.50% | 13.28% |
| MDLSTM | 14.11% | 19.15% |
| Conv. Recursive (+ n-gram LM) | 7.25% | 7.35% |
| Conv. Transformer | 6.0% | 6.4% |

This project's 11.66% CER lands in-line with, and modestly better than,
their own CNN-RNN/BiGRU baselines — a reasonable result for a CTC-only
architecture built from scratch. Notably, their largest single jump in
performance (13–14% → 7.25%) came not from a more complex encoder, but from
**adding an n-gram language model** as a spelling-correction step on top of
a comparably simple CNN-RNN-CTC model — directly consistent with what this
project's own error analysis independently found.

## Next steps

1. **N-gram / language model rescoring on top of beam search decoding.**
   Directly targets the dominant error source (spacing/word-boundary
   ambiguity) identified through error analysis, and is supported by prior
   lab results showing this specific intervention produced the largest
   single accuracy jump in comparable architectures.
2. Re-run the confusion matrix analysis after the above change to confirm
   it's actually closing the spacing gap rather than shifting error mass
   elsewhere.
3. Longer-term, targeted synthetic data generation for word-boundary and
   weak-character examples is a promising direction, contingent on access
   to appropriate generation tooling — not yet scoped in detail.

## Dataset

This project uses the **NUST-UHWR (National University of Sciences and
Technology - Urdu Handwriting Recognition)** dataset, introduced in:

> Zia, N., Naeem, M. F., Raza, S. M. K., Khan, M. M., Ul-Hasan, A., &
> Shafait, F. (2022). *A Convolutional Recursive Deep Architecture for
> Unconstrained Urdu Handwriting Recognition.* Neural Computing and
> Applications, 34(2), 1635–1648.

**Dataset features:**
- 10,606 handwritten Urdu text-line image samples, collected from a variety
  of websites including social networking and news sites
- Each sample is a scanned/photographed image of a single handwritten line,
  paired with its ground-truth transcription
- Variable image widths (samples with width > 1400px are excluded during
  preprocessing in this project; see `dataset.py`)
- Standard train/val/test splits as defined by the dataset's original
  release, used consistently with prior work on this dataset for
  comparability (see [Comparison to prior work](#comparison-to-prior-work))

**Access:** the dataset is not included in this repository (not licensed
for redistribution here). It is available by request from the original
authors / NCAI, NUST. A community-uploaded mirror is also available on
Kaggle: [Nust-UHWR dataset](https://www.kaggle.com/datasets/ameerhamzaqamar28/nust-uhwr-dataset)
(third-party upload, not officially maintained by the original authors —
verify sample count and splits match before relying on it for exact
reproducibility). To reproduce this project's results, obtain the dataset
and place it at `data/DataSet/UHWR/` with `train.txt`, `val.txt`, and
`test.txt` label files (tab-separated: `<relative_image_path>\t<label>`),
matching the format expected by `dataset.py`.

## Checkpoints and large artifacts

Trained model checkpoints (`.pt` files) are **not** committed to this
repository, to keep repo size and clone times reasonable. The
`checkpoints/` directory is git-ignored.

- To reproduce the best checkpoint, run `python src/train.py` following the
  configuration documented in the [Experiments](#experiments) table above.
- [Add a link here once checkpoints are hosted externally, e.g. Google
  Drive or Hugging Face Hub, if you want to share pretrained weights
  directly.]

The character-level confusion matrix (`confusion_matrix.csv`, generated by
`src/confusion_analysis.py`) **is** included in the repo under `results/` —
it's small (aggregated counts only, no images or raw text) and is directly
referenced in the [Error analysis](#error-analysis) section above.

## Reproducing results

```bash
# Train
python src/train.py

# Evaluate (CER + prediction/reference pairs)
python src/eval/run_eval.py

# Character-level confusion matrix analysis
# (writes results/confusion_matrix.csv)
python src/confusion_analysis.py
```

Checkpoints are saved to `checkpoints/` (git-ignored) on validation-loss
improvement. Dataset expected at `data/DataSet/UHWR/` — see
[Dataset](#dataset) above for access details and format.
