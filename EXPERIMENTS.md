# DataVerse Hindi SER — experiment log

Metric: Macro-F1 (4 classes). CV = 5-fold StratifiedGroupKFold (groups = exact-audio hash), `feats/folds.csv`, seed 42.
Final score = 0.4·public + 0.6·private. Deadline 2026-10-10 18:30 UTC. 5 subs/day.

## Data facts (src/eda.py, 2026-10-05)
- 826 train / 209 test, all 16 kHz mono; 1.0–11.2 s, median 2.9 s. Test IDs interleaved with train (random split, no order leak).
- 80 train files are PCM_32, rest PCM_16 — does NOT predict label.
- 14 duplicate rows in train (exact audio); 1 conflicting pair (Id 636 sad / Id 75 neutral).
- 8 test clips have an exact train twin, all labelled sad (Ids 272, 317, 410, 450, 509, 514, 579, 775).
- Loudness (RMS) differs by class: angry > happy > sad > neutral.
- **Near-duplicates (2026-10-05 night):** ~30% of clips are noisy copies of another clip (same length, lag 0, residual SNR 18–30 dB or 70–85 dB re-save). Whisper-L32 cosine >0.95 → 221 train clips have a twin, label agreement 99.1%; only 12% are hash-exact. **60 test clips have a train twin** (current sub agrees 60/60), 16 test clips have a test twin.
- **Honest CV = OOF on clips without a cross-fold twin.** Probe blend: all 0.930 / twin-clips 0.995 / **no-twin 0.913**. Judge every idea on no-twin OOF (and re-split folds by near-dup groups).

## Public reference
- kunalinfinite WavLM Base+/Large fine-tune, 5-fold: OOF 0.862, public LB 0.879. LB top 0.965.

## Frozen-feature linear probes (src/extract.py + src/probe.py)
| features | best layer | OOF macro-F1 | notes |
|---|---|---|---|
| wavlm-large (mean+std) | 13 (of 0–24) | **0.860** (C=0.1) | layers 9–19 all ≥0.84; ≈ public fine-tuned ensemble (0.862) |
| w2v-bert-2.0 | 16 (of 0–24) | 0.869 (C=0.01) | mid layers best, top layers collapse (L24 0.687) |
| whisper-large-v3 encoder (30-s pad, pool valid frames) | 32 = last | **0.916** (C=0.1) | rises steadily with depth; clear winner |
| blend whisper_L32 + w2vbert_L16 (C=0.03, mean probs) | — | **0.930** | best of 2-way blends (some selection bias); 3-way no better |
| blend whisper_L32 + w2vbert_L16 + wavlm_L15 | — | 0.925 | wavlm adds nothing |

| whisper-large-v3, batch-longest input (NO 30-s pad), src/check_trim.py | L16 0.846 / L24 0.842 / L32 **0.833** | — | top layers degrade without padding (padded L32 0.913); lower layers unaffected |
| whisper-large-v3, audio zero-padded to 12 / 20 / 30 s, pool valid frames | L32: 0.904 / **0.916** / 0.913 | — | padding (trailing silence) is what matters; 20 s = full quality at 0.67× compute, 12 s ≈ at 0.4×. 30 s control reproduces HF path → custom forward correct. **Use ≥12 s padding in fine-tunes.** |


## Tier 3 (src/t3_eval.py, 2026-10-05 night) — metric: all / **no-twin** OOF macro-F1 (C picked by log-loss)
| model / addition | alone (all / no-twin) | base blend + it (no-twin) | notes |
|---|---|---|---|
| base = whisper_L32 + w2vbert_L16 | 0.927 / **0.909** | — | reference |
| whisper L24–32 mean | 0.911 / 0.891 | 0.912 (+0.003) | marginal |
| openSMILE eGeMAPS (88) | 0.578 / 0.584 | 0.895 ✗ | too weak |
| openSMILE ComParE (6373) | 0.726 / 0.673 | 0.889 ✗ | too weak |
| CLAP larger_clap_general emb | 0.638 / 0.600 | 0.886 ✗ | too weak |
| CLAP zero-shot prompts | 0.301 / 0.307 | — | ≈ chance |
| label spreading (whisper_L32 kNN10, α 0.2, train+test graph, fold-honest) | 0.745 / 0.721 | 0.922 (+0.014) at fixed weight 0.25 | **✗ NOT robust**: k=5 −0.000, k=20 +0.006; L24-32 graph −0.004…+0.004; whisper+w2vbert graph −0.006…−0.015 (0/5 folds up). Lucky setting — rejected. |
| pseudo-labels (fold-own test preds ≥0.9) | 0.925 / 0.907 | ±0 | no gain |
| emotion2vec+ large zero-shot (4 of 9 classes) | 0.485 (train) | — | Kaggle CPU kernel dataverse-e2v, 19 min |
| emotion2vec+ large utterance emb (1024) | 0.564 / 0.551 | 0.890 ✗ (0/5 folds up) | does not transfer to Hindi |
| emotion2vec+ 9-class log-scores | 0.566 / 0.565 | 0.896 ✗ (0/5 folds up) | |
| Qwen2.5-Omni-7B thinker zero-shot (typed 4-way logprobs; Kaggle model mount, 2×T4 fp16, 1035 clips in 10 min, ~95% mass on 4 labels) | 0.708 / 0.711 | — | over-predicts neutral (291 vs 217) |
| Omni logprobs → LR (C=1) | 0.739 / 0.751 | base(t3)+1/3: 0.914 (+0.005, 4/5) | |
| Omni audio-token hidden mean, LLM layers 14+28 → LR (C=0.01) | 0.872 / **0.844** | base(t3)+1/3: 0.917 (+0.008, 4/5) | strong alone (≈ w2v-BERT) |
| pre-registered blend: whisper 1/3, w2vbert 1/3, omni_lp 1/6, omni_h 1/6 | — | 0.9167 vs t3-base 0.9089 (+0.008, 4/5) | LR stacker instead: +0.003, 2/5 ✗ |
| **same Omni 1/3 added to the exact SUBMITTED recipe** (probe C=0.03) | — | 0.9168 vs **0.9130** (+0.004, 3/5 up, 2 down) | **below +0.005 bar → not adopted**; only 3 test preds change. File kept: subs/blend_sub+omni.csv |

## Honest CV v2 (src/hcv.py, feats/folds_v2.csv — near-dup groups [same length & whisper cos>0.9, or exact hash] never cross folds; 689 groups)
| model | honest v2 F1 | notes |
|---|---|---|
| whisper_L32 probe (C=0.03) | 0.880 | (v1 all 0.916) |
| w2vbert_L16 probe (C=0.01) | 0.787 | (v1 0.863) — leaned on twins most |
| omni_h 14+28 probe (C=0.003) | 0.809 | |
| omni_lp probe (C=0.3) | 0.740 | |
| **base = whisper + w2vbert (submitted recipe)** | **0.8905** | folds .898 .880 .891 .855 .927; public 0.917 consistent (29% test twins ≈ free) |
| **base 2/3 + omni (h, lp) 1/3** | **0.9027 (+0.012, 4/5 up)** | **passes gate** under honest CV (v1 masked it). File: subs/v2_base_omni.csv |

| C. speaker x-vectors (microsoft/wavlm-base-plus-sv, **centred** — uncentred cosine put everything in 1 cluster) → agglomerative cosine clusters | thr 0.4/0.6/0.8 → 164/54/13 clusters; twins together 0.98 | only 62–75% of mid-threshold clusters span all 4 emotions → clusters partly encode emotion/session |
| per-speaker normalisation (subtract cluster mean, clusters ≥5) | norm: −0.08 alone; blend −0.006…−0.026 | **✗ rejected** at all thresholds; raw+norm best +0.002 (2/5) |

| B. Whisper-large-v3 ASR (feats/asr.csv, 433 s local; needs PYTHONUTF8=1 on Windows) | 656 distinct transcripts / 689 groups; 18 repeated sentences, label purity 0.97 → **content is emotion-specific, not shared across emotions**; 71 test transcripts match a train transcript exactly (60 audio twins + 11) | |
| text char-ngram LR — **bug: regex `[^\w\s]` stripped Devanagari matras** | 0.640 | blend +0.003 (2/5) — rerun with fixed cleaning (src/t1_text2.py) |
| text char-ngram (fixed cleaning: strip Unicode P*/S* only) | 0.692 | audio+omni blend + w0.25: −0.005 ✗ |
| text word 1–2-gram | 0.593 | −0.004 ✗ |
| **text multilingual-e5-large ("query: " + transcript, mean-pool) → LR C=10** | **0.739** | **w0.25 (pre-registered): 0.9109 (+0.008, 3/5)**; w0.15 +0.005 (4/5); w0.35 −0.003 |
| **full: base 2/3 + omni 1/3, then 0.75·audio + 0.25·e5** | **0.9109 vs base 0.8905 (+0.020, 5/5 folds up)** | subs/v2_base_omni_e5.csv — 8 test preds change vs sub1 |

**Tier 3 verdict (v1 CV, superseded by v2 above for Omni):** nothing clears the robustness bar. Omni is the best new signal (marginal). Gains must come from the novel (no-twin) clips via Tier 1/2 ideas.

## Kaggle fine-tune runs (kernels/ser-ft, src/ser_ft.py)
| version | setup | wavlm OOF | w2vbert OOF | notes |
|---|---|---|---|---|
| v1 (2026-10-05) | 2×T4 threads, 5-fold, ≤10 ep, bs 8, lr 2e-5 / head 1e-3, 6 s crops, LS 0.05, layer-weighted mean+std pool, 40-min budget | crashed (meta tensor: concurrent from_pretrained in threads) | fold0 only: 0.881 (probe w2vbert fold0 0.860, whisper probe fold0 0.934) | ~100 s/epoch on T4; ran 21 min |

### Whisper fine-tune (kernels/ser-ft-whisper, src/ser_ft_whisper.py)
| version | setup | OOF | notes |
|---|---|---|---|
| v1 (2026-10-05) | whisper-large-v3 encoder, layers 16–31 + final LN trainable, batch-longest input (no 30-s pad), 5 ep, lr 1e-5 / head 1e-3, bs 8, folds shared over 2 GPUs, load lock, 26-min budget | **0.552** ✗ | ~47 s/epoch/GPU, 823 s total. Two causes: (1) **LayerNorm head** — same head on frozen trimmed feats gets 0.58–0.67 vs 0.82–0.86 with per-feature standardisation (CPU test); (2) no padding degrades top layers (0.833 vs 0.913). |

### Whisper fine-tune v2 — cached trunk (src/wft_cache.py + src/wft_train.py), 2026-10-06
| setup | result | notes |
|---|---|---|
| pad 20 s, cache layer-24 input (1035×1000×1280 fp16, 81 s), train layers 24–31 + LN, LP-FT head (fixed standardisation + linear init from fold LR probe), lr 1e-5/1e-4, 5 ep, bs 8, honest v2 folds | **✗ worse than its own init**: f0 0.898 → 0.874, f1 0.861 → 0.812 | init reproduces the probe exactly (pipeline correct); train loss at its LS floor from epoch 1 → with 660 clips the frozen probe already fits; fine-tuning only distorts. ~73 s/epoch, 3.5 GB. Stopped after 2 folds. |

### Twin lock (src/twinlock.py, 2026-10-06) — post-processing, apply to every sub
- Train evidence: audio near-twins (same length & whisper-L32 cos>0.9, or hash) share label 99.1%; different-recording exact transcript matches (≥3 words, cross-fold) 35/35.
- Test: 78 clips locked (69 audio, 9 text-only). Sub1 unchanged; sub2 (v2_base_omni_e5) had 1 twin flipped by Omni/text (Id 449 neutral → sad). File: subs/v2_base_omni_e5_lock.csv.

### External data (rules: public + free OK; foundational 4b forbids human labels of TEST records → any ext clip matching a test clip must be excluded)
- ✅ swastikatinkhede/multilingual-speech-dataset — CC BY 4.0, in-house recordings, 22 speakers, 1000 clips (Hindi 250 / Marathi 500 / English 250; drop "surprised" → 800). Credit author in write-up.
- ⛔ vishlb/speech-emotion-recognition-hindi — file naming `1/session1/anger/1.1.anger-01.wav` = IITKGP-SEHSC (licence-agreement corpus); likely unauthorised redistribution → not used.
- Overlap check (src/ext_eval.py): ext vs competition whisper-L32 max cosine 0.746 (q99 0.554) → **0 clips near test or train** (rule-safe).
- Add ext to training side of honest folds (whisper C=0.03, w2vbert C=0.01 probes; omni/text unchanged): **pre-registered ALL w1.0: full blend 0.9023 (−0.009, 1/5) ✗**; ALL w0.5 −0.005 (1/5); HINDI-only w1.0 +0.004 (3/5, post-hoc, below gate). **Rejected** — domain mismatch outweighs extra data.

### English translation route (src/translate_en.py + src/en_eval.py, 2026-10-06)
- Whisper-large-v3 task=translate (247 s); quality good ("Get out of my sight." angry, "My vacation has started" happy).
- j-hartmann roberta-large zero-shot (4 of 7 classes): 0.664; LR on log-probs of roberta-large + distilroberta + cardiffnlp sentiment: 0.723; e5 on English: 0.742.
- Text-only: hindi e5 + en_emo = **0.777** (vs 0.739) — better text, BUT full blend PRE-REG 0.25·mean(e5,en_emo): 0.9062 (−0.005, 2/5) ✗; all variants −0.001…−0.005. Text gains are redundant with audio. Rejected.
- LR stacker over 7 members (log-probs, nested over folds): C 0.01/0.03/0.1 → −0.005/−0.009/−0.021 (1/5) ✗. Nested per-class log-bias for macro-F1: −0.011 (0/5) ✗ (preds already balanced).
- Qwen2-Audio-7B-Instruct frozen **audio encoder** (src/qa_enc.py, shard 1 only, 30-s pad): alone L16/L24/L32 = 0.768/0.806/0.768; PRE-REG L32 as 0.2 of audio: −0.003 (2/5) ✗; L24 −0.000. Rejected — weaker than original whisper-large-v3 (0.880).
- Whisper multi-layer (pad-30 feats, replacing whisper_L32 in audio group): PRE-REG L24+L32 concat: alone 0.896 (vs 0.880), full +0.0048 (3/5) on seed-42 folds (oof/v2_whisper_L24L32*). Post-hoc: mean L28–32 +0.006 (4/5); L20+L26+L32 −0.001.
  → **Repeated CV (src/recv.py, fresh near-dup-grouped splits, all members retrained): seed 7 +0.0097 (4/5), seed 11 +0.0120 (4/5) → ADOPTED.** Note blend CV itself varies ±0.01 across splits (0.911 / 0.892 / 0.899).
- whisper-large-v2 (pad 30): alone L24 0.834 / **L32 0.889** (> v3!); PRE-REG L32 as 0.2 of audio: −0.003 (2/5) ✗ (L24 +0.002, 2/5). Redundant with v3.
- Harveenchadha/vakyansh-wav2vec2-hindi-him-4200 (MIT): alone L6 0.800 / L8 0.787; PRE-REG L8 −0.000 ✗.
- ai4bharat/indicwav2vec-hindi: gated on HF (needs login + terms) → skipped. facebook/mms-300m: CC-BY-NC → skipped.
- Repeated CV (src/rcv_members.py + rcv_decide.py; seeds 42/7/11; base = A′ recipe 0.9156/0.9015/0.9108; rule: mean ≥ +0.005 AND >0 on all splits):
  C1 w2vbert L10+L16 (replacing L16): +0.0036/+0.0047/+0.0038, mean +0.0040, folds up 4/5/3 ✗ (consistent but small)
  C2 + whisper-v2 L24+L32 at 0.2 of audio: +0.0012/+0.0074/+0.0061, mean +0.0049 ✗ (by 0.0001)
  C3 = C1+C2 (pre-registered after seeing C1, C2): +0.0013/+0.0063/+0.0062, mean +0.0046 ✗ — not additive. None adopted.
- **Audio side saturated**: every new frozen audio model (Qwen2-Audio enc, whisper-v2, vakyansh, emotion2vec, CLAP, eGeMAPS) is redundant with whisper-v3 in the blend.
- **Plateau: honest v2 0.911** (whisper+w2vbert 2/3, omni 1/3)×0.75 + e5×0.25.

## Known not to work
- Fine-tuning Whisper top layers (even LP-FT, correct padding) on 660 clips — hurts (above). Big-model fine-tuning needs more data.
- LayerNorm (per-sample) normalisation in the classifier head on pooled Whisper features → 0.58–0.67. Use per-feature standardisation / BatchNorm.
- Whisper with batch-longest input (no trailing silence): top-layer features drop 0.913 → 0.833. Pad audio to ≥12 s (20 s = full quality).
- Loading HF models concurrently in threads (meta-tensor crash) — use a lock.

## Submissions
| date | file | CV | public LB | notes |
|---|---|---|---|---|
| 2026-10-05 | subs/probe_whisperL32_w2vbertL16.csv | 0.930 (no-twin 0.913; v2 0.8905) | **0.91737** (#15/40) | first sub (prize eligibility); probe blend; LB ≈ honest no-twin CV, confirms twin-inflation finding |
| 2026-10-05 23:36 IST | subs/v2_base_omni_e5.csv | v2 **0.9109** (+0.020, 5/5) | **0.91973** (+0.0024) | audio (whisper+w2vbert 2/3, Qwen2.5-Omni 1/3) ×0.75 + e5 transcript ×0.25; 8 preds changed; had 1 twin flipped (Id 449) → use _lock version |
| 2026-10-06 ~01:55 IST | subs/v4_wL24L32_lock.csv | seed-42 0.9156; repeated-CV gains +0.0048/+0.0097/+0.0120 (seeds 42/7/11) | **0.94230** (#7/40, +0.012) | **whisper member → L24+L32 concat** (PRE-REG rule passed: mean new-split gain +0.0108, both >0; src/recv.py) + twin lock; 6 preds changed vs A |
| 2026-10-10 ~14:15 IST | subs/v9_a3_full_lock.csv (**A‴ full refit**) | 0.9300/0.9159/0.9183 | **0.94230** (= A″) | differs from A″ only on Id 235 → Id 235 is a PRIVATE clip |
| 2026-10-10 ~14:15 IST | subs/v8_K1_lock.csv (A‴ + Gemma-4 K1) | 0.9250/0.9207/0.9242 | 0.93048 (−1 public clip) | differs on 125/235/431 → one of 125/431 is public and K1 has it wrong |
| 2026-10-06 ~02:20 IST | subs/v5_aug_c1_lock.csv (**A″**) | repeated CV +0.0070/+0.0095/+0.0014 vs A′ (mean +0.0060) | **0.94230** (= A′) | S3 aug-whisper + C1 w2vbert L10+L16; 2 preds differ from A′ |
| 2026-10-06 ~00:45 IST | subs/v2_base_omni_e5_lock.csv | v2 0.911 + twin lock | **0.93060** (#11/40, +0.011) | **final candidate A**: sub2 + twin lock (1 pred changed: Id 449 → sad). Lock confirmed: Id 449 is public; 1 clip ≈ 0.011 public → public ≈ 90 clips. Gap: ~1 clip to 0.942, ~2 to 0.954, ~3 to 0.965. |

## Not tried yet
- emotion2vec_plus_large, Whisper encoder fine-tune, speaker clustering/normalisation, ASR text fusion, pseudo-labels.

## Spikes S1–S3b (PRE-REGISTERED 2026-10-06 ~02:00 IST, before any result)
Error analysis of A′ OOF (seed 42, 0.9156): neutral weakest (31/217 wrong: 14→sad, 11→happy); happy→neutral 11; clips <2 s
err 13.5% vs 3–4 s 4.1%; angry↔sad 15 swaps; twin-group OOF averaging only +0.001 (dropped, incl. test-test twin consistency).
Rule for every spike: repeated CV over the existing splits (seeds 42/7/11, `oof/rcv_s{seed}_fold.npy`), base = A′ recipe;
**adopt if mean gain ≥ +0.005 AND gain > 0 on every split.**
- **S1 group weighting:** every member (whisper_L24L32, w2vbert_L16, omni_h, omni_lp, text_e5) refit with LR sample_weight =
  1/|near-dup group|; same C selection; candidate = A′ blend of the weighted members.
- **S2 multi-pad TTA:** whisper member = mean of L24+L32-concat probes on 30-s (feats/whisper.npy), 20-s and 12-s padded
  features (feats/whisper_pad20/12.npy); other members unchanged.
- **S3 augmentation + TTA:** whisper-large-v3 L24/L32 features for 4 views of every clip: white noise at 20 dB and 30 dB SNR,
  0.25 s and 0.5 s leading silence (each treated as a new waveform: 30-s pad, pool over its valid frames). Probe trained on
  original + 4 views (train side of the fold), C by OOF log-loss; prediction = mean prob over original + 4 views. Replaces whisper member.
- **S3b loop padding:** clip tiled to fill 30 s, pool over all 1500 frames; whisper member = mean of original probe and
  loop-pad probe. Replaces whisper member.
- **S1 result:** gains −0.0000/+0.0035/−0.0026 (seeds 42/7/11), mean +0.0003, folds up 1/3/1 → **✗ rejected.** Weighted members
  alone: whisper 0.900/0.898/0.900 (≈ unweighted), text_e5 drops (0.73/0.72). Twin copies are not hurting the probes.
- **S2 result:** gains −0.0001/+0.0060/+0.0009, mean +0.0023, folds up 1/4/3 → **✗ rejected.** pad12/pad20 L24+L32 alone
  0.866–0.898, weaker than and redundant with the 30-s member.
- **S5 Qwen2-Audio-7B-Instruct full LLM (PRE-REGISTERED now, run 10 Oct on Kaggle):** kernel `kernels/dataverse-qa7b`
  (script src/qa7b_zs.py, mounts qwen-lm/qwen2-audio/transformers/7b-instruct/1, Apache-2.0; smoke-tested locally on a tiny
  random-weight model: template, 25 audio tokens/s, hidden extraction OK). Outputs qa7b_lp [N,2,4], qa7b_h [N,4,4096] (layers 8/16/24/32).
  Members: probe(qa7b_h, 4 layers concat), probe(qa7b_lp, 8 values). Candidate: audio' = 0.8·audio + 0.2·mean(the two), as C2.
  Evaluate: `python src/rcv_spikes.py <seed> s5` for 42/7/11, then `python src/rcv_decide.py`. Same adoption rule.
- **S3 result:** gains +0.0059/+0.0012/+0.0061 (seeds 42/7/11), mean +0.0044, >0 on all splits but folds up 2/2/4 → **✗ not adopted
  (below +0.005).** Member alone is the best whisper member so far: 0.8995/0.8989/0.9075 (C=0.01) vs plain L24+L32. Consistent but small,
  like C1. Extraction 768 s local (feats/whisper_aug.npy); S3 member training ~25 min CPU per split (5× rows).
- **S3b result:** gains −0.0048/+0.0084/+0.0049, mean +0.0029, folds up 1/5/3 → **✗ rejected.** Loop-pad member alone 0.834–0.871.
- Spike verdict (06 Oct): S1, S2, S3, S3b all fail the gate; augmentation (S3) is the only consistently positive one.
- **S3+C1 (PRE-REGISTERED after seeing S3 and C1):** whisper member = whisper_aug_L24L32 (S3) AND w2vbert member =
  w2vbert_L10L16 (C1), rest of A′ unchanged. Same rule: mean gain ≥ +0.005 AND > 0 on every split (seeds 42/7/11).
  Caveat: second look at the same splits; C3 showed C1+C2 was not additive.
- **S3+C1 result:** gains +0.0070/+0.0095/+0.0014, mean **+0.0060**, folds up 5/5/2 → **✓ ADOPTED** (passes the pre-registered rule;
  weakest split seed 11). Base per split 0.9156/0.9015/0.9108 → 0.9226/0.9110/0.9122.
  Sub: subs/v5_aug_c1_lock.csv (seed-42 members, same build as A′ which reproduces exactly; twin lock changed 1 = Id 449).
  Class counts 52/53/53/51. Only **2 test preds change vs A′**: Id 235 happy→neutral, Id 883 angry→happy.
  **Submitted 2026-10-06 ~02:20 IST: public 0.94230 (= A′)** → the 2 changed clips are private, or net zero on public.

## Research round R1–R3 (PRE-REGISTERED 2026-10-06; base is now **A″** = A′ with whisper_aug_L24L32 + w2vbert_L10L16)
Research (web, 06 Oct): X-ARES / ICME-2025 encoder challenge → Dasheng-1.2B strongest general audio encoder (incl. emotion);
VoxEmo (2026) → Qwen2-Audio-7B best open speech LLM for SER, "+A" self-generated acoustic-caption prompt helps on non-English
acted data; Step-Audio-2-mini (Apache-2.0) best open model on paralinguistics (emotion 82%). Rejected: BhavVani (form-gated,
sitcom, 7 classes), Vox-Profile (OpenRAIL non-commercial), Audio Flamingo 3 (NC), MSP-Podcast WavLM baseline (F1 0.31).
Rule for all: repeated CV seeds 42/7/11 vs A″; adopt if mean gain ≥ +0.005 AND > 0 on every split.
- **R1 Dasheng-1.2B (mispeech/dasheng-1.2B, Apache-2.0):** frozen, mean+std pooled block outputs; member = probe on concat of
  the block at 3/4 depth and the last block (same choice as whisper L24+L32); candidate audio' = 0.8·audio + 0.2·member.
- **R2 Step-Audio-2-mini (Kaggle, 10 Oct):** same design as S5: 2 prompts' 4-label log-probs + mean LLM hidden over audio
  tokens at 4 evenly spaced layers; members probe(h concat), probe(lp); candidate extra 0.2 = mean(two members).
- **R3 Qwen2-Audio +A prompt (same kernel as S5, 10 Oct):** third prompt = model first generates an acoustic caption
  (pitch, loudness, rate, pauses, voice quality), then 4-label log-probs after "Emotion:". S5 stays as registered (prompts 0–1);
  R3 = S5 with lp member on all 3 prompts (12 values).
- **R1 result:** Dasheng-1.2B alone 0.820/0.807/0.816 (seeds 42/7/11, C=0.01; ≈ w2v-BERT / omni_h level, far better than CLAP 0.60),
  blend gains vs A″ −0.0047/−0.0012/+0.0124, mean +0.0022, folds up 2/2/5 → **✗ rejected.** Extraction 50 s local (fp16 + autocast;
  needed `pip install einops torchaudio==2.11.0 --no-deps`). Another frozen audio encoder redundant with whisper in the blend.
- **R2, R3:** kernels ready + smoke-tested (kernels/dataverse-sa2, kernels/dataverse-qa7b), NOT pushed — run 10 Oct.

## Local round L1–L5 (PRE-REGISTERED 2026-10-06 ~14:20 IST; base A″; same repeated-CV rule)
Research: attentive pooling on frozen Whisper beats mean pooling on small SER sets (+0.4…+2.3 UA, arXiv 2602.06000); emotion
cues are localized (≈15% of frames carry 80%, arXiv 2506.15754); confident learning finds a few % label errors in common datasets;
augmentation effects are dataset-specific (PMC9415521).
- **L1 w2v-BERT aug + TTA:** w2v-bert-2.0 layers 10 & 16 (mean+std over all frames) for the 4 S3 views (noise 20/30 dB, lead
  silence 0.25/0.5 s, each a new waveform). Member trained on original + 4 views, C by OOF log-loss, prediction = mean over the
  5 versions. Replaces w2vbert_L10L16 in A″.
- **L5 label-noise cleaning (nested):** per outer fold, inner 4-fold StratifiedGroupKFold on the training part with the plain
  A″-weighted blend (whisper_L24L32, w2vbert_L10L16, omni_h, omni_lp, text_e5; each at the C its outer run chose) → drop training
  clips whose inner-OOF prob of their own label < 0.10 → retrain all 5 A″ members (whisper_aug included) on the kept clips at
  the same C → predict the outer fold and test. Evaluation labels untouched.
- **L4 segment pooling:** whisper-large-v3 L24 & L32 valid frames (30-s pad); features = mean+std over the whole clip and over
  each third; probe on the concat; candidate audio' = 0.8·audio + 0.2·member.
- **L3 attentive-statistics pooling head:** same frames, per-frame L24‖L32 (2560) standardised with train-fold stats →
  Linear 256 + GELU + dropout 0.3 → 4-head attentive statistics pooling (weighted mean + std per head) → dropout 0.3 → Linear 4.
  AdamW lr 1e-3, wd 1e-2, 30 epochs, batch 16, label smoothing 0.1, cosine; 3 init seeds averaged per fold.
  Candidate audio' = 0.8·audio + 0.2·member.
- **L2 more whisper views:** 4 extra views: speed 0.9 / 1.1 (resample), pitch ±1 semitone (tempo kept). Member trained on
  original + 8 views at fixed C = 0.01 (S3's C on all splits), prediction = mean over 9 versions. Replaces whisper_aug_L24L32 in A″.
- **L5 result:** drops only 2–5 train clips per fold (mostly neutral/sad labels); gains vs A″ −0.0048/+0.0012/−0.0024, mean −0.0020,
  folds up 1/1/1 → **✗ rejected.** Labels look clean; the low-confidence clips are informative, not noise.
- **L1 result:** w2v-BERT aug member alone **improves** (0.837/0.835/0.825 vs 0.811/0.816/0.808) but blend gains vs A″
  −0.0082/−0.0036/−0.0010, mean −0.0043, folds up 1/1/3 → **✗ rejected.** Same augmentations make w2v-BERT's errors more
  whisper-like → less diversity. Lesson: a stronger member ≠ a better blend; keep w2v-BERT unaugmented.
- **L4 result:** segment-pooled member alone 0.894/0.886/0.900 (≈ plain L24+L32); blend gains vs A″ +0.0001/−0.0047/+0.0025,
  mean −0.0007, folds up 2/1/3 → **✗ rejected.** Within-clip trajectory (thirds) adds nothing beyond global stats.
- **L3 result:** attentive-pooling head alone = **best single member ever**: 0.9085/0.9041/0.9193 (seeds 42/7/11) vs whisper_aug
  0.8995/0.8989/0.9075 and plain L24+L32 0.896/0.895/0.898. As pre-registered extra (0.2): gains vs A″ +0.0037/+0.0025/+0.0074,
  mean **+0.0045**, >0 on all splits, folds up 3/3/4 → **✗ not adopted (below +0.005)** — same pattern as S3. ~5 min/split on GPU.
- **L2 result:** 8-view whisper member alone 0.897/0.890/0.899 (< S3's 0.900/0.899/0.908: speed/pitch views blur emotion cues;
  pitch shift moves L32 most, cos 0.975); blend gains vs A″ −0.0035/−0.0010/+0.0001, mean −0.0015 → **✗ rejected.**
- **Local round verdict (06 Oct):** L1 ✗ (−0.0043), L2 ✗ (−0.0015), L3 ✗ but closest (+0.0045, >0 on all splits, best single
  member), L4 ✗ (−0.0007), L5 ✗ (−0.0020). A″ stays. New files: feats/w2vbert_aug.npy, whisper_frames(_off).npy, whisper_aug2.npy.
- **L3a / L3b (PRE-REGISTERED after seeing L3; both vs A″, same rule):**
  L3a: whisper role = mean(whisper_aug_L24L32, whisper_attn) — the head shares the whisper slot; rest of A″ unchanged.
  L3b: attentive head (same architecture/training as L3) trained on frames of the original + 4 S3 views (noise 20/30 dB,
  0.25/0.5 s lead silence; same RNG as whisper_aug.py), one uniformly random version per clip per step, 30 epochs;
  prediction = mean softmax over the 5 versions → member whisper_attn_aug; candidate: whisper role = mean(whisper_aug_L24L32,
  whisper_attn_aug). Caveat: ~15 candidates tested on these 3 splits this session → treat a bare pass with caution.
- **L3a result:** gains vs A″ −0.0084/−0.0021/+0.0014, mean −0.0030, folds up 1/1/3 → **✗ rejected.** Halving whisper_aug's
  weight costs more than the head adds: the head is useful only as an extra voice (L3, +0.0045), not as a replacement.

## Replication round P1–P3 (PRE-REGISTERED 2026-10-08 ~19:05 IST, replicating similar-competition solutions; vs A″, same rule)
Research: no public competition with this exact data/task (no prior DataVerse edition found). Closest: Interspeech 2025
naturalistic SER challenge (SAILER top-3, arXiv 2505.22133: whisper-large-v3 > WavLM, +text +0.013–0.019, ensembling +0.020,
audio mixing + annotation dropout +0.005; NTUA 1st: DeepSER cross-modal fusion + meta-classifier), Odyssey 2024 (1st: focal loss
+ 7-model vote for imbalance — N/A, our classes balanced; 2nd: w2v-BERT + LLaMA text + LR fusion — we have it, LR stacker ✗),
MER 2025 (pseudo-labels from external LLM agreement — external API forbidden by data rules), EmoSPeech 2024 (Spanish, attention
pooling), Urdu low-resource SSL adaptation (cross-lingual, needs large unlabelled pools — N/A), Wang 2023 kNN interpolation.
- **P1 audio mixing (SAILER):** 2 extra whisper views per clip: the clip + another random clip (any label, fixed RNG) mixed at
  −10 dB and −15 dB relative power. Member trained on original + 4 S3 views + 2 mix views at fixed C = 0.01, prediction = mean over
  the 7 versions. Replaces whisper_aug_L24L32.
- **P2 kNN interpolation (Wang 2023, frozen variant):** whisper member p = 0.7·whisper_aug probe + 0.3·kNN, kNN = 10 nearest
  train-fold clips by cosine on train-fold-standardised L24‖L32 pooled features, distance-weighted votes (w = exp(10·cos)).
  Replaces whisper_aug_L24L32.
- **P3 early audio-text fusion (DeepSER-style):** L3 attentive head + e5 transcript embedding (1024 → Linear 64 + GELU +
  dropout 0.3) concatenated with the pooled audio vector before the output layer; otherwise identical to L3 training.
  Candidate audio' = 0.8·audio + 0.2·member (as L3).
- **P2 result:** kNN alone 0.719/0.707/0.723 (frozen embeddings don't cluster by emotion the way Wang 2023's contrastively
  fine-tuned space does); 0.7·probe + 0.3·kNN gains vs A″ −0.0131/−0.0047/−0.0025, mean −0.0068 → **✗ rejected.**
- **P1 result:** mix-view member alone 0.898/0.894/0.909 (≈ S3); gains vs A″ −0.0035/−0.0012/+0.0012, mean −0.0012 → **✗ rejected.**
- **P3 result:** audio-text fusion head alone = **best single member: 0.911/0.912/0.924** (seeds 42/7/11); as 0.2 extra, gains vs A″
  **+0.0073/+0.0049/+0.0062, mean +0.0062, folds up 3/4/4 → ✓ ADOPTED** (caveat: ~20 candidates tested on these splits).
  Recipe A‴ = 0.75·(0.8·A″-audio + 0.2·whisper_attn_text) + 0.25·text_e5. Sub subs/v6_p3_lock.csv (seed-42 members, A″ build
  reproduced exactly): **0 test preds differ from A″** → identical file, no submission needed (gain is OOF-only at this weight).
- **L3b result:** head trained on S3 views alone 0.913/0.908/0.926 (seeds 42/7/11; aug helps the head too) but in the
  pre-registered slot-sharing structure gains vs A″ −0.0084/−0.0034/+0.0026, mean −0.0031 → **✗ rejected** (same failure as L3a:
  halving whisper_aug's weight hurts). Fixes on the way: framesaug OOM → disk memmaps; np.int64 in .npy headers patched.
- **A‴ seed-averaged test preds (15 fold models over splits 42/7/11), subs/v7_a3_seedavg_lock.csv:** 0 changes vs A″ (identical
  file). Splits disagree on only 2 test clips: Id 235 (42 neutral, 7/11 happy; avg neu 0.45 vs hap 0.44 → neutral; A′ had happy)
  and Id 449 (avg sad 0.46 → sad, also twin-locked to sad). Test predictions are stable across splits; nothing to submit.

## 10-Oct Kaggle round K1–K4 (PRE-REGISTERED 2026-10-10 ~02:20 IST). Base for all 10-Oct candidates = **A‴**
(A″ + P3 head as 0.2 extra). New members enter as a nested extra: audio'' = 0.8·audio' + 0.2·member (audio' = A‴ audio),
same weight as before. S5 / R2 / R3 / omni-zs2 keep their registered member definitions but are now judged vs A‴ the same way.
Rule unchanged: repeated CV seeds 42/7/11, adopt if mean gain ≥ +0.005 AND > 0 on every split.
Research (08–10 Oct): Gemma 4 E2B/E4B (Apache-2.0, Kaggle `google/gemma-4`) take audio via a ~305M USM-style Conformer encoder
(12 layers, 1024-d, 25 tokens/s) — a family new to us; Qwen3-ASR-1.7B (Apache-2.0) uses Qwen3-Omni's AuT encoder (300M, 1024-d,
12.5 Hz, ~40M h pretraining, Hindi supported); OmniVox (2025): richer/contextual prompts help omni-LLM emotion recognition.
- **K1 Gemma 4 E4B-it full audio LLM (Kaggle):** prompts 0–1 as S5 (4-label first-token log-probs) + mean LLM hidden over
  audio tokens at 4 evenly spaced layers (prompt 0). Member = mean(probe(hidden concat), probe(log-probs)).
- **K2 Gemma 4 audio encoder (same kernel):** mean+std over valid frames of the Conformer layers at 2/3 depth and the last
  layer; member = probe on their concat.
- **K3 Qwen3-ASR-1.7B AuT encoder (frozen):** mean+std over valid frames of the encoder layer at 3/4 depth and the last layer;
  member = probe on their concat.
- **K4 Qwen2.5-Omni + transcript (added to kernels/dataverse-omni-zs2 as prompt 3):** audio, then "Transcript: <Whisper ASR
  text>", then prompt-0 question. Member = mean(probe(last-token hidden at LLM layers 7/14/21/28 concat), probe(its 4 log-probs)).
  (Last token, not audio tokens: in a causal LLM only positions after the transcript see both modalities.)
- **K3 result:** Qwen3-ASR-1.7B AuT encoder (24 layers; hook 18 + final, 61 s local extraction, side venv) alone
  0.850/0.847/0.855 — strong for a new family — but gains vs A‴ −0.0023/−0.0012/+0.0025, mean −0.0003
  → **✗ rejected.** Another frozen audio encoder redundant with whisper in the blend. Fix on the way: Qwen3-ASR feature extractor
  returns the frame mask only with return_attention_mask=True.
- **10-Oct Kaggle results (vs A‴):** omni-zs2 (30 min, 1035/1035; zero-shot prompts 0–3 = 0.708/0.664/0.653/**0.724** with transcript):
  omni v2 members −0.0024/−0.0000/+0.0000, mean −0.0008 ✗; **K4** (transcript prompt) −0.0156/+0.0014/+0.0033, mean −0.0036 ✗.
  dataverse-gemma4 (28 min, 1035/1035, bf16, ~100% mass on 4 labels; zero-shot **0.751/0.741** — best zero-shot model so far):
  **K1** LLM hidden alone 0.837/0.811/0.839, log-probs 0.771/0.774/0.780 → gains −0.0050/+0.0048/+0.0058, mean +0.0019 ✗;
  **K2** USM-Conformer encoder alone **0.854/0.851/0.860** → gains −0.0071/+0.0003/−0.0011, mean −0.0027 ✗.
  Pattern holds: strong new members, all redundant with the A‴ blend. Kaggle allows only 2 concurrent GPU sessions.
- **R2 result:** dataverse-sa2 (8 min, 1035/1035, fp16; 75–98% mass on 4 labels) zero-shot only 0.479/0.307 on Hindi (its
  paralinguistic claims are zh/en); members hidden 0.702/0.726/0.711, log-probs 0.52 → gains vs A‴ −0.0121/−0.0012/−0.0024,
  mean −0.0052 → **✗ rejected.**
- **S5 / R3 result:** dataverse-qa7b (79 min, 1035/1035; zero-shot 0.453/0.344, "+A" caption prompt 0.311 — captions are generic
  and often misname the language, e.g. "Mandarin"; label mass after captions 0.02–0.39) → members hidden 0.685/0.692/0.671,
  log-probs ~0.58; **S5** gains vs A‴ −0.0134/−0.0012/−0.0061, mean −0.0069 ✗; **R3** −0.0133/−0.0012/−0.0072, mean −0.0072 ✗.
- **10-Oct round verdict:** all 8 candidates (omni v2, K1, K2, K3, K4, S5, R3, R2) ✗ vs A‴. Strongest new single members:
  Gemma-4 USM encoder 0.85–0.86, Qwen3-ASR AuT 0.85, Omni+transcript last-token hidden 0.85 — all redundant with the blend.
  GPU used ≈ 3 h of 30. **A‴ (= A″ file, subs/v5_aug_c1_lock.csv) stays the final best-CV candidate.**

## Audit + fixes F1–F3 (PRE-REGISTERED 2026-10-10 ~13:15 IST, audit of flaws first; base A‴, same repeated-CV rule)
Audit (measured, seed-42 OOF): (a) **calibration** — LR members are calibrated (|conf−acc| ≤ 0.06) but the P3 head is badly
under-confident (conf 0.708 vs acc 0.910): in a probability average it acts with far less than its nominal 0.2 weight; GOAL.md
asked for per-member calibration before averaging and it was never done. (b) **learning curve** whisper L24+L32 probe: train on
330 / 495 / 660 clips → OOF 0.843 / 0.882 / 0.896, still rising → test predictions from 5-fold models (660 clips each) waste data;
a full-data (826) refit should be better on the 131 non-locked test clips. (c) the text branch (25 % + P3 input) uses generic
Whisper-large-v3 transcripts; Qwen3-ASR-1.7B (Hindi-supported) is now available locally. Also: ~30 candidates tested on the same
3 splits → adopted gains are optimistic by a few thousandths (selection bias).
- **F1 calibrated blend:** every A‴ member gets a temperature T (p ∝ p^(1/T)), fitted by log-loss on the OTHER four folds' OOF
  (leave-one-fold-out) and applied to the held-out fold; same weights as A‴. Candidate = A‴ with all 6 members calibrated.
- **F3 better transcripts:** Qwen3-ASR-1.7B Hindi transcription (greedy) → multilingual-e5-large (same recipe as text_e5) →
  member text_e5q; candidate text role = mean(text_e5, text_e5q).
- **F2 full-data refit (final build only, not CV-gated; justified by the learning curve):** refit every member of the final recipe
  on all 826 train clips with its selected C / hyper-parameters, predict test, same blend + twin lock. Shown as a diff vs A″.
- **F1 result:** temperatures T ≈ 0.98/1.01/0.96/0.91/0.79 for the LR members and **≈ 0.37–0.41 for the P3 head** (calibrated conf
  0.905 vs acc 0.910 — the audit flaw was real); calibrated blend gains vs A‴ −0.0025/+0.0012/+0.0010, mean −0.0001 → **✗ rejected.**
  The under-confident head at nominal 0.2 already sits at the right effective weight; calibration does not change decisions.
- **F3 result:** Qwen3-ASR transcripts are clearly better text: e5 member alone **0.769/0.752/0.757** vs Whisper-ASR 0.739/0.718/0.719
  (+0.03 — audit flaw real), but text = mean(old, new) gains vs A‴ −0.0000/−0.0037/−0.0002, mean −0.0013 → **✗ rejected**
  (better transcripts don't move the blend; audio members already cover what the text adds). feats/asr_qwen3.csv kept.
- **F2 result (final build, not gated):** full-data refit changes per-member test argmax by 1–7 clips (mean |Δp| 0.003–0.018);
  blended + twin-locked **subs/v9_a3_full_lock.csv differs from A″ on 1 test clip: Id 235 neutral → happy** (the 0.45/0.44 coin flip;
  A′ also says happy). Class counts 52/54/52/51.
- **T1 transcript lock v2 (PRE-REGISTERED 14:05 IST):** finding — identical transcripts from different recordings share the label
  (Qwen3-ASR exact ≥3 words: 101 train pairs, 100 %; near-exact ratio ≥ 0.9, ≥4 words: 83 pairs, 100 %; Whisper exact: 54, 98.1 %).
  Rule: a clip without an audio twin takes the label of train clips whose Qwen3 OR Whisper transcript matches exactly (≥3 words)
  or near-exactly (SequenceMatcher ≥ 0.9, ≥4 words, length within 15 %), if all matched labels agree. Evaluated honestly on A‴ OOF:
  matches only against clips in OTHER folds; gain vs A‴ OOF on splits 42/7/11; adopt if mean ≥ +0.005 and > 0 on every split.
- **T1 result:** OOF gains vs A‴ +0.0037/+0.0049/+0.0012 (fixed 10, broke 2 over the 3 splits), mean +0.0033 → **✗ below gate**.
  On TEST it changes **0 clips** (the model already agrees with every transcript match) → no submission effect either way.
- **Audit verdict (14:10 IST):** flaws confirmed (P3 under-confidence; 5-fold models under-trained; weak Whisper transcripts) but only
  F2 changes the test file: subs/v9_a3_full_lock.csv = A″ with Id 235 → happy. Proposed final picks: v9_a3_full_lock (A‴ full refit)
  + v8_K1_lock (best CV mean 0.9233; differs from pick 1 on Ids 125, 235, 431).

## Cheap checks G1–G4 (PRE-REGISTERED 2026-10-12 ~13:20 IST, before any result; deadline extended to 12 Oct 18:30 UTC; base A‴)
Source: reports/Hindi SER competition winning ideas.md (research round 12 Oct). Script src/rcv_cheap.py. Same repeated-CV rule
(seeds 42/7/11: mean gain vs A‴ ≥ +0.005 and > 0 on every split) PLUS ≤ 3 test-clip flips vs A‴ (5-fold test preds, before lock).
~30 candidates already tested on these splits → a pass means "not harmful", not proven.
- **G1 hard majority vote:** the 6 A‴ members (whisper_aug_L24L32, w2vbert_L10L16, omni_h, omni_lp, text_e5, whisper_attn_text)
  each vote their argmax; plurality wins; ties broken by the A‴ soft-blend probabilities restricted to the tied classes.
- **G2 shrinkage-LDA heads:** whisper and w2v roles use StandardScaler + LDA(solver lsqr, shrinkage auto) instead of LR; whisper
  trained on original + 4 S3 aug views, predict mean over the 5 views (as aug_probe); w2vbert L10+L16 original only. No tuning.
- **G3 transductive scaling:** whisper and w2v roles as in A‴ but standardised with mean/std of ALL 1035 clips' original features
  (train + test, no labels), then LR (C by OOF log-loss, same grid), whisper with the same 5-view training/prediction.
- **G4 learned layer weights:** softmax weights over all 33 Whisper-v3 layers (mean+std pooled, per-layer standardised with train-fold
  stats) + linear head; full-batch AdamW lr 1e-3, wd 1e-2, 200 epochs, seed 0; added as nested extra2 (0.2) on A‴.
- **G1–G4 results (12 Oct ~14:30 IST; A‴ per split 0.9300/0.9159/0.9183):**
  G1 hard majority vote −0.0048/+0.0013/−0.0014, mean −0.0016, 2 test flips ✗ · G2 shrinkage-LDA heads (members 0.88/0.82 vs LR)
  −0.0204/−0.0093/−0.0001, mean −0.0099 ✗ · G3 transductive scaling ±0.0000 (0 test flips — scaling stats barely move) ✗ ·
  G4 learned 33-layer weights (member 0.82–0.83) −0.0157/+0.0001/+0.0009, mean −0.0049 ✗. **All rejected; A‴ unchanged.**
- **Literature research round (12 Oct, multi-source search with adversarial claim verification):**
  8 claims survived / 17 refuted; none on Hindi or small data. Its two ideas are already covered: speaker-adversarial head (≈ rejected
  speaker normalisation; needs speaker labels) and early audio-text attention fusion (= adopted P3). Nothing new to test.
