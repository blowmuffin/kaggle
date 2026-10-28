# DataVerse Hindi SER — HANDOFF (resume point; rewritten 2026-10-06 ~02:15 IST before /compact)

## 13 OCT: COMPETITION ENDED, PRIVATE SCORES (from `kaggle competitions submissions`)
| file | public | private |
|---|---|---|
| subs/v8_K1_lock.csv (A‴ + Gemma-4 K1; dropped by rule B) | 0.93048 | **0.95161** (best) |
| subs/v5_aug_c1_lock.csv (A″, **final pick, ticked**) | 0.94230 | 0.94340 |
| subs/v2_base_omni_e5(_lock).csv | 0.91973 / 0.93060 | 0.93558 |
| subs/v9_a3_full_lock.csv (A‴, **final pick, ticked**) | 0.94230 | 0.93492 |
| subs/v8_K2_lock.csv (12-Oct plan, NOT ticked) | 0.94230 | 0.92805 |
| subs/v4_wL24L32_lock.csv (A′) | 0.94230 | 0.92685 |
| subs/probe_whisperL32_w2vbertL16.csv | 0.91737 | 0.88755 |
Public LB at close (reports/leaderboard/): we were #10 at 0.94230 (top 0.96603). Private LB rank not fetched (CLI gives public only).
Ticked finals (user, 13 Oct) = v9 + v5 → counted private = 0.94340 (better of the two); 0.4·pub + 0.6·priv ≈ 0.9430.
Lesson: public and private disagree by several clips both ways; the hedge we dropped (K1) was the best private.

## 12 OCT UPDATE (read first)
- **Deadline extended: 2026-10-12 18:30 UTC = 13 Oct 00:00 IST** (Kaggle competitions list). Rank #8.
- Research round 12 Oct → reports/Hindi SER competition winning ideas.md (notes in research_notes/). Verdict: no idea likely to
  gain > 1 clip; main action = better 2nd final pick.
- Submitted subs/v8_K2_lock.csv (differs from v9 on Ids 235, 542, 927): **public 0.94230 = v9** → by rule B, **FINAL PICKS =
  subs/v9_a3_full_lock.csv + subs/v8_K2_lock.csv** (user ticks them on Kaggle). Ids 542, 927 deduced private.
- Cheap checks G1–G4 (src/rcv_cheap.py): **all ✗** (majority vote, LDA heads, train+test scaling, learned layer weights). Workflow research round: nothing new. **Modelling frozen; finals = v9 + K2.**

Read this first, then `EXPERIMENTS.md` (every experiment + verdict) and `GOAL.md` (goal, rules, Jev assessment).

## Competition facts
- Kaggle `dataverse-detecting-emotions-from-hindi-speech`. 826 train / 209 test WAVs (16 kHz mono, 1–11 s), 4 balanced classes
  angry/happy/neutral/sad. Metric macro-F1. **Final rank = 0.4·public + 0.6·private.** Public ≈ 90 clips (1 clip ≈ 0.011).
- **Deadline 2026-10-10 18:30 UTC = 11 Oct 00:00 IST.** 5 subs/day (UTC day; resets 05:30 IST). Pick 2 finals.
- Submission format = sample_submission: `Id,emotion`, lowercase labels, sample row order.
- Rules that matter: external data/models OK if public + free; **foundational 4b: no human labels of TEST records** (any external
  clip matching a test clip must be excluded); no public code sharing outside Kaggle during the comp.
- Prize: UG/PG students; must have ≥1 submission (done) + register on Unstop (user's job; host's "Registration" forum post).
  Unstop page says you must have joined on Kaggle before 1 Oct — user's team id 17032886 suggests ~4–5 Oct; told user to check / ask host.
  Winners present 15 Oct 10:00 IST at SRCASW Delhi (online possible).
- Leaderboard (06 Oct 01:30 IST): top 0.96544 (3 teams), 0.95387 (2), 0.94230 (Devansh, **us #7**).

## Submissions (all public scores)
| # | file | honest CV | public |
|---|---|---|---|
| 1 | subs/probe_whisperL32_w2vbertL16.csv | v2 0.8905 | 0.91737 |
| 2 | subs/v2_base_omni_e5.csv | v2 0.9109 | 0.91973 |
| 3 | subs/v2_base_omni_e5_lock.csv (= #2 + twin lock) — **candidate A** | 0.9109 + lock | 0.93060 |
| 4 | **subs/v4_wL24L32_lock.csv — candidate A′ (BEST)** | 0.9156 / 0.9015 / 0.9108 (seeds 42/7/11) | **0.94230** |
| 5 | **subs/v5_aug_c1_lock.csv — A″ (best CV)** | 0.9226 / 0.9110 / 0.9122 | 0.94230 |
Final picks unless 10 Oct beats it: **A″ + A′** (differ by only 2 clips → weak hedge; reconsider B on 10 Oct). Rule for B: best public among candidates within 0.005 of A′'s honest CV.

## Current best recipe (A′) — how to rebuild
1. Members = logistic-regression probes on frozen features (src/hcv.py `probe`: StandardScaler+LR, C by OOF log-loss), honest v2 folds:
   - `whisper_L24L32`: concat of whisper-large-v3 encoder layers 24 & 32 (mean+std pooled, 30-s padding) from `feats/whisper.npy`
   - `w2vbert_L16`: w2v-bert-2.0 layer 16 (`feats/w2vbert.npy`)
   - `omni_h`: Qwen2.5-Omni-7B thinker mean hidden over audio tokens, LLM layers 14+28 (`feats/omni_h.npy`, from Kaggle)
   - `omni_lp`: Qwen2.5-Omni zero-shot log-probs for 4 labels (`feats/omni_lp.npy`)
   - `text_e5`: Whisper-large-v3 Hindi ASR (`feats/asr.csv`) → multilingual-e5-large ("query: " prefix, mean pool, `feats/e5.npy`) → LR (no scaler, C by F1)
2. audio = 1/3·whisper_L24L32 + 1/3·w2vbert_L16 + 1/6·omni_h + 1/6·omni_lp ; full = 0.75·audio + 0.25·text_e5 ; argmax.
3. `python src/twinlock.py in.csv out.csv` — test clips with an audio twin in train (exact hash, or same length & whisper-L32
   cos>0.9; 99.1% label agreement) or exact transcript match ≥3 words (35/35) take the train label. Locks 78 test clips.
   OOF/test probs saved in `oof/v2_*.npy` (+`_test`); repeated-CV members in `oof/rcv_s{42,7,11}_*.npy`.

## Method rules (keep!)
- **Honest CV = near-dup-grouped folds** (`src/hcv.py` → `feats/folds_v2.csv`; groups = hash or same length & whisper cos>0.9).
  ~30% of clips are noisy copies; old v1 folds inflated CV (0.930 vs honest 0.891). 60 test clips have train twins.
- Gate: pre-register the variant BEFORE looking; adopt only if ≥ +0.005 and ≥3/5 folds up. Borderline → **repeated CV on fresh
  grouped splits (seeds 7, 11; src/recv.py / src/rcv_members.py + src/rcv_decide.py)**: adopt if mean ≥ +0.005 and >0 on all splits.
  Blend CV varies ±0.01 between splits — single-split gains < 0.005 are noise.
- Sanity per sub: class counts ~45–60 each; twin lock applied; diff vs previous best listed.

## What is exhausted (details in EXPERIMENTS.md)
Fine-tuning (Whisper v1 LayerNorm-head bug; v2 cached-trunk LP-FT worse than init — 660 clips too few); emotion2vec, CLAP,
openSMILE, Qwen2-Audio encoder, whisper-v2, vakyansh, WavLM (all redundant with whisper-v3); speaker normalisation; label
spreading; pseudo-labels; LR stacker; class-bias; external CC-BY Indian SER data (−0.009; IITKGP copy skipped for licence);
English translation + English emotion models (text-only 0.777 but blend −0.005); w2vbert L10+L16 (+0.004 consistent, below gate);
+whisper-v2 L24+L32 (+0.0049); both (+0.0046). **Audio side saturated; local frozen-feature route exhausted.**

## Compute / quotas
- Kaggle GPU: ~6 min left; **resets 10 Oct 00:00 UTC = 05:30 IST** (30 h). TPU: 20 h unused. Local RTX 4050 6 GB, 16 GB RAM.
- **User decision (06 Oct): run the Omni experiment on the KAGGLE quota on 10 Oct — NOT locally** (local 4-bit plan dropped).
- Kaggle model mount path is `/kaggle/input/models/...`, competition data `/kaggle/input/competitions/...` (search with glob).
- Windows: always `PYTHONUTF8=1` (Hindi prints crash cp1252; also needed for `kaggle kernels output` logs). Long jobs → background.

## 10 OCT STATUS (written ~11:40 IST; deadline 11 Oct 00:00 IST)
- Base = **A‴** (A″ + P3 head 0.2; test preds identical to A″ = subs/v5_aug_c1_lock.csv, public 0.94230). Split CV 0.9300/0.9159/0.9183.
- Kaggle GPU (30 h, max **2 concurrent** sessions): pushed 11:31 IST `dataverse-omni-zs2` (omni members v2 + K4 transcript prompt)
  and `dataverse-qa7b` (S5 + R3). A background loop pushes `dataverse-gemma4` (K1 + K2) then `dataverse-sa2` (R2) when a slot frees.
- When a kernel completes: `PYTHONUTF8=1 python -m kaggle kernels output ayushkasare/dataverse-<k> -p runs/<k>` then for seeds
  42/7/11 `python src/rcv_spikes.py <seed> omni2|s5|k1|r2` → `python src/rcv_decide.py` (10-Oct candidates are vs A‴, nested extra2).
- K3 (Qwen3-ASR AuT encoder) local: `../venv5/Scripts/python src/qwen3asr_enc.py` (side venv with transformers 5.19; main env
  stays 4.56.2) → `python src/rcv_spikes.py <seed> k3`.
- Final picks before 11 Oct 00:00 IST: best-CV candidate (A‴ or a 10-Oct winner) + A′/A″ hedge; submissions left today: 5.
- **Result (12:55 IST): every 10-Oct candidate ✗ vs A‴ (see EXPERIMENTS.md). Final picks = subs/v5_aug_c1_lock.csv (A″ = A‴, best CV)
  + subs/v4_wL24L32_lock.csv (A′ hedge, differs on Ids 235, 883); both public 0.94230. Select them manually on Kaggle (Submissions tab).**
- **Audit round (13:00–, user asked "improve all first"): F1 calibration ✗ (−0.0001). F3 (Qwen3-ASR Hindi transcripts → e5 text
  member) + F2 (full-data refit for test, learning curve 330/495/660 clips → 0.843/0.882/0.896) running as ONE chain; resume with:
  `../venv5/Scripts/python src/qwen3asr_asr.py` (resumable, feats/asr_qwen3.part.csv) → `python src/e5_embed.py feats/asr_qwen3.csv
  feats/e5_qwen3.npy` → `python src/rcv_spikes.py <seed> f3` (42/7/11) → `python src/rcv_decide.py` → `python src/attn_pool.py full
  text` → `PYTHONPATH=src python src/final_refit.py subs/v9_a3_full.csv [f3]` → twinlock → diff vs A″.
  Hedge candidates within 0.005 CV built: subs/v8_{K1,K2,K3,K4}_lock.csv (2–3 test clips differ from A″; not submitted).

## NEXT (in order)
1. **Fresh-idea spikes S1–S5 (06 Oct, user-approved; pre-registrations + results in EXPERIMENTS.md "Spikes").** S1 group
   weighting ✗, S2 multi-pad TTA ✗, twin-group averaging ✗ (+0.001). S3 augmentation+TTA mean +0.0044 (>0 all splits, below gate) ✗;
   S3b loop-pad ✗. **S3+C1 (aug whisper + w2vbert L10+L16) ✓ ADOPTED: mean +0.0060 → candidate A″ =
   subs/v5_aug_c1_lock.csv (2 preds differ from A′); submitted: public 0.94230 (= A′).** R1 Dasheng-1.2B ✗ (+0.0022). Local round L1–L5 (06 Oct): all ✗; **L3 attentive-pooling head (src/attn_pool.py) = best single member
   (0.909/0.904/0.919) but +0.0045 as 0.2 extra** — candidate for a follow-up.
   L3a (head shares whisper slot) ✗ −0.0030. L3b (head trained on S3 views) + replication round P1–P3 (08 Oct; similar-competition
   research: Interspeech-2025 SAILER/DeepSER, Odyssey-2024, MER-2025, kNN interpolation) — P2 kNN ✗ −0.0068; L3b, P1 (audio
   mixing) ✗, **P3 early audio-text fusion head (`src/attn_pool.py <seed> text`) ✓ ADOPTED (+0.0062) → recipe A‴, but its
   test predictions are identical to A″ (subs/v6_p3_lock.csv = A″)**. L3b rerun queued — see EXPERIMENTS.md. S5/R3 (Qwen2-Audio + acoustic-caption prompt) and R2 (Step-Audio-2-mini) = kernels for 10 Oct. All candidates: `python src/rcv_decide.py`.
   Not yet tried: Omni with transcript in prompt; LLM retrieval-few-shot text classifier; TPU-based ideas.
2. **10 Oct 05:30 IST:** `PYTHONUTF8=1 python -m kaggle kernels push -p kernels/dataverse-omni-zs2` (ready, NOT pushed; 3 prompts +
   hidden L7/14/21/28; ~30 min; outputs omni2_lp.npy [N,3,4], omni2_h.npy [N,4,3584]; script src/omni_zs2.py). Pre-register:
   omni_lp → mean over 3 prompts' LR probes; omni_h → L7..L28 concat; evaluate with repeated CV.
   **Also push `kernels/dataverse-qa7b`** (S5 + R3: Qwen2-Audio-7B full LLM, 2 prompts + "+A" acoustic-caption prompt with
   generation; smoke-tested, NOT pushed; ~80–100 min, BUDGET_MIN default 150) **and `kernels/dataverse-sa2`** (R2: Step-Audio-2-mini,
   HF download ~16 GB, own forward re-implemented for hidden states; smoke-tested on tiny random weights, NOT pushed; ~30 min).
   If Kaggle refuses a 3rd concurrent GPU session, push it when one finishes. Download each kernel's outputs to `runs/qa7b/`,
   `runs/sa2/` (`kaggle kernels output ayushkasare/<kernel> -p runs/<x>`), then for seeds 42/7/11:
   `python src/rcv_spikes.py <seed> s5` (also writes R3's qa7b_lp3) and `... <seed> r2` → `python src/rcv_decide.py` (vs A″).
3. 10 Oct evening: pick 2 finals (A′ + best alternative) before 11 Oct 00:00 IST.
4. **After 11 Oct 00:00 IST only:** push code to https://github.com/blowmuffin/kaggle (public repo the user gave the organisers):
   src/, kernels/*/*.py, GOAL/EXPERIMENTS/HANDOFF; exclude data/, feats/, oof/, runs/, subs/, ext/, hf_cache. Credit CC-BY dataset author if used (not used).

## Layout
`data/` competition files · `feats/` features+meta+folds · `oof/` member OOF/test probs · `subs/` submissions · `runs/` Kaggle outputs
· `kernels/` Kaggle kernel folders · `ext/indian` CC-BY external data (unused) · `src/` all scripts · HF cache `D:/kaggle/hf_cache`.

## FINAL PICKS (10 Oct ~14:20 IST) — select these two on Kaggle (Submissions tab) before 11 Oct 00:00 IST
1. **subs/v9_a3_full_lock.csv** — A‴ full-data refit (best-CV recipe), public 0.94230.
2. **subs/v5_aug_c1_lock.csv** — A″ (5-fold build), public 0.94230; differs from pick 1 only on Id 235 (private clip, 0.45/0.44 coin
   flip neutral vs happy) → one of the two is guaranteed right on it. K1 (v8_K1_lock, public 0.93048) dropped by rule B.
Deduced from equal public scores: Ids 235 and 883 are private clips. Submissions left today: 3.

## NEXT-STEP PLAN (10 Oct ~21:55 IST, after the Opus + Haiku literature runs)
Research status: the Haiku run was stopped after 7 results and the Opus run after the verify stage. Both runs are stopped; no agents are running. The 13 verified Opus claims are in tasks/w6wscrgl9.output.
Key findings:
- BhavVani (our competition data, 8,734 clips) published SOTA is about 0.45 weighted accuracy under 10-fold leave-speaker-out (CAMuLeNet, Interspeech 2024). Our public 0.94 is far higher. Hypothesis, not yet verified: the gap is near-duplicate and speaker leakage in the competition split, which the twin lock exploits. Our honest folds already account for this.
- Whisper large-v3 encoder is the strongest frozen upstream (EmoBox). Matches our L24/L32 member.
- Zero-shot audio LLMs underperform supervised probes everywhere (VoxEmo, EmoBox). Matches our Gemma-4 and Qwen-Omni results.
- No public benchmark includes Hindi (EmoBox, VoxEmo). Literature pitfalls (leakage, inflated speaker-dependent scores) apply directly to us.
- Unverified lead: Zenodo Feb 2026 Hindi SER paper with behavioral/prosody features (pauses, speaking rate, rhythm). Not read.
Already rejected in EXPERIMENTS.md, do not retry: per-class log-bias for macro-F1, LR stacker, per-speaker normalisation, calibration (F1), audio-LLM zero-shot (K1, K2, K4, R2, R3, S5).
Decision: stop research agents (token cost). Keep the two FINAL PICKS above.
Steps:
1. User: tick v9_a3_full_lock.csv and v5_aug_c1_lock.csv in Kaggle Submissions before 00:00 IST.
2. Optional, local CPU only, only if the user asks: prosody member (pause ratio, speaking rate, energy contour) from the audio files, pre-registered, gate per CLAUDE.md (3 seeds, mean >= +0.005, all splits > 0). Adopt only if the gate passes; otherwise the picks stay.
3. After 11 Oct 00:00 IST: GitHub push (needs user go-ahead).
