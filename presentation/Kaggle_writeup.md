# 6th place solution: frozen speech foundation models, honest validation, and a twin lock

Thanks to the organisers for an interesting small-data problem, and congratulations to the top teams. This post describes our 6th-place solution (private macro-F1 **0.9434**, public 0.9423). All code and the full experiment log, including everything that failed, are at **https://github.com/blowmuffin/kaggle**.

## TL;DR

1. **About 30% of clips are noisy near-duplicates.** Random folds overstate CV by about 0.04. We validated only on folds grouped by near-duplicate recording.
2. **Frozen encoders with small heads beat fine-tuning** at 826 clips: Whisper-large-v3, w2v-BERT 2.0 and Qwen2.5-Omni-7B features with logistic-regression probes.
3. **A 6-member fixed-weight blend** (audio, audio-LLM, transcript text, and an attentive audio-text head), with members chosen for diversity rather than individual strength.
4. **Twin lock:** test clips with a near-identical training clip take that clip's label (99.1% agreement in train). This affects 78 of 209 test clips.
5. **Every idea was pre-registered** and gated on 3 grouped CV splits. About 50 ideas were tested and 4 were adopted.

## 1. Data: the hidden structure

- 826 train / 209 test clips, 16 kHz mono, 1.0–11.2 s (median 2.9 s), balanced classes (205/192/217/212).
- Many clips are the same recording re-saved with noise: same duration, zero lag, residual SNR of 18–30 dB. Only 12% are byte-identical, so hashing misses most of them.
- We grouped clips by **same duration and Whisper-L32 cosine > 0.9, or identical hash**, which turns 826 clips into **689 groups**. Twins share their label **99.1%** of the time. 60 test clips have a training twin, and 16 have a test twin.

## 2. Validation

| Scoring of the same first model | Macro-F1 |
|---|---|
| Random 5-fold CV | 0.930 |
| Random folds, clips without a cross-fold twin | 0.913 |
| **StratifiedGroupKFold over near-duplicate groups** | **0.891** |
| Public leaderboard | 0.917 |

The public score matched the honest number, so all decisions after that used grouped CV. Because blend CV moves about ±0.01 between re-splits, each idea was **written down before scoring**, evaluated on **3 grouped splits** (seeds 42, 7, 11) with all members retrained, and adopted only if the **mean gain was ≥ +0.005 and the gain was positive on every split**.

## 3. Members

All encoders are frozen. Features are mean + std pooled over valid frames and cached once as fp16 `[clips × layers × 2·hidden]`, so a new head or blend costs minutes. Each probe is StandardScaler + LogisticRegression, with C chosen by out-of-fold log-loss. Scores below are grouped-CV macro-F1, averaged over the 3 splits.

| Member | Details | Alone |
|---|---|---|
| Whisper-large-v3 encoder | Layers 24 + 32 concatenated. **Audio zero-padded to 30 s**: without padding the top layers fall from 0.913 to 0.833. Trained on the original plus 4 augmented views (white noise at 20/30 dB SNR, 0.25/0.5 s leading silence), predicting the mean over 5 views. | 0.902 |
| w2v-BERT 2.0 | Layers 10 + 16. Emotion peaks mid-network and the top layer collapses (L24: 0.59). Deliberately **not** augmented: augmentation raised its own score but made its errors Whisper-like and cost the blend 0.004. | 0.812 |
| Qwen2.5-Omni-7B hidden states | Mean LLM hidden state over audio tokens at layers 14 and 28 (Kaggle 2×T4, fp16, about 10 min for all clips), then a probe. | 0.820 |
| Qwen2.5-Omni-7B zero-shot | First-token log-probs of the 4 labels from a typed prompt (about 95% of probability mass), then a probe. | 0.738 |
| Transcript text | Whisper-large-v3 Hindi ASR, then multilingual-e5-large ("query: " prefix, mean pool), then LR. Repeated sentences keep their emotion (label purity 0.97). | 0.725 |
| Attentive audio-text fusion head | Whisper L24‖L32 frames (2,560-d, standardised) → Linear 256 + GELU + dropout 0.3 → 4-head attentive statistics pooling, concatenated with e5 (1,024 → 64) → Linear 4. AdamW lr 1e-3, wd 1e-2, 30 epochs, batch 16, label smoothing 0.1, cosine schedule, 3 seeds per fold. | **0.916** |

## 4. Blend, twin lock, final refit

- **Blend (fixed weights, set before scoring):** audio = 0.8·(⅓ Whisper + ⅓ w2v-BERT + ⅙ Omni hidden + ⅙ Omni log-probs) + 0.2·fusion head; final = 0.75·audio + 0.25·text, then argmax. Learned combiners all lost: an LR stacker −0.005 to −0.021, per-class bias −0.011, majority vote −0.002, and calibration ±0.
- **Twin lock:** a test clip takes its training twin's label when the audio matches (identical hash, or same duration and Whisper-L32 cos > 0.9), or otherwise when a transcript of 3+ words matches a training transcript exactly (35/35 agreement in train). Audio matches win over text. This locked 69 clips by audio and 9 by text. It uses only competition training labels.
- **Full-data refit:** the learning curve was still rising at 660 clips (0.843 / 0.882 / 0.896 at 330 / 495 / 660), so every member was refit on all 826 clips for the final predictions.

## 5. How the score was built (grouped CV, seed 42)

| Step | CV | Public | Private |
|---|---|---|---|
| Whisper L32 + w2v-BERT L16 probes | 0.891 | 0.917 | 0.888 |
| + Omni (hidden + log-probs) + e5 text, + twin lock | 0.911 | 0.931 | 0.936 |
| Whisper L24+L32 | 0.916 | 0.942 | 0.927 |
| + augmentation views, w2v-BERT L10+L16 (**final pick**) | 0.923 | 0.942 | **0.943** |
| + fusion head, full refit (final pick) | 0.930 | 0.942 | 0.935 |

The public board saturated at 0.942 (about 90 clips), which is why CV drove every decision.

## 6. What did not work

| Idea | Result |
|---|---|
| Fine-tuning Whisper (full, and LP-FT on a cached trunk) | 0.55; LP-FT fell below its own initialisation (660 clips per fold) |
| emotion2vec+, CLAP, openSMILE eGeMAPS/ComParE | 0.56–0.73 alone; no blend gain |
| WavLM, Whisper-v2, Dasheng-1.2B, Qwen3-ASR encoder, Gemma-4 audio encoder | 0.81–0.89 alone but redundant with Whisper-v3 (blend Δ below +0.005) |
| Audio-LLM zero-shot (Qwen2-Audio, Gemma-4, Step-Audio-2) | 0.31–0.75 |
| Pseudo-labels, label spreading, kNN interpolation | no robust gain |
| LR stacker, class-bias tuning, calibration, majority vote | −0.021 to 0.000 |
| Speaker normalisation (x-vector clusters) | −0.006 to −0.026 (clusters partly encode emotion) |
| External CC-BY Indian SER data; English translation + English emotion models | −0.009 / −0.005 |

Main lesson: **a stronger member is not a better blend member.** The Gemma-4 encoder scored 0.855 alone and still lowered the blend. The only late addition that helped was the one carrying different information (the transcript, fused early).

## 7. Error analysis

Out-of-fold predictions averaged over 3 grouped splits give per-class F1 of angry 0.935, happy 0.926, **neutral 0.886** and sad 0.920. Clips under 2 s err 14% of the time, against 4% for 3–4 s clips.

## 8. Final selection and the lesson

Our two picks (A″ and the A‴ full refit) differed on only one test clip. A dropped candidate (A‴ + Gemma-4 LLM members), rejected for a lower public score (0.930), scored **0.952 private**. With 90 public and about 125 private clips, the two leaderboards disagree by a few clips in both directions. In hindsight, the second pick should have been chosen for diversity, not public score.

## Compute

A local RTX 4050 (6 GB) for feature extraction and probes, plus about 3 GPU-hours on Kaggle 2×T4 for the 7B audio LLMs.

Questions and feedback are welcome!
