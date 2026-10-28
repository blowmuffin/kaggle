# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Kaggle competition `dataverse-detecting-emotions-from-hindi-speech`: 4-class speech emotion recognition (angry/happy/neutral/sad),
826 train / 209 test clips, macro-F1. Not a git repo, no package/test setup: a flat folder of research scripts.
**The competition has ended (13 Oct 00:00 IST).** Private scores of every submission are visible with
`python -m kaggle competitions submissions dataverse-detecting-emotions-from-hindi-speech` (table in `HANDOFF.md`).
Private ≈ 125 clips, so 1 clip ≈ 0.008: score gaps between submissions are mostly a few clips, not evidence for a recipe.

**Read first, in order:** `HANDOFF.md` (resume point: current best recipe, submissions, next steps), `EXPERIMENTS.md`
(every experiment and its verdict; check it before proposing an idea, since most directions are already exhausted),
`GOAL.md` (rules, deadline, robustness policy). Update `EXPERIMENTS.md`/`HANDOFF.md` after each experiment.

## Running things
- Run every script **from the repo root** (paths like `feats/...` are relative): `python src/<script>.py ...`.
  Ad-hoc snippets (`python - <<EOF`) that `import hcv` need `PYTHONPATH=src`.
- Windows: always set `PYTHONUTF8=1` (Hindi text crashes cp1252 output; also needed for `kaggle kernels output` logs).
- HF models cache to `D:/kaggle/hf_cache` (`HF_HOME`). Local GPU is an RTX 4050 6 GB: one heavy job at a time, long jobs in background.
- Extract frozen features: `python src/extract.py whisper|w2vbert|wavlm|...` → `feats/<key>.npy` `[n_clips, n_layers, 2*hidden]` fp16.
- Evaluate a new frozen member against the current blend: `python src/add_member.py <name> feats/<x>.npy [layer ...]`.
- Repeated CV: `python src/rcv_members.py <seed>` (seeds 42, 7, 11), then `python src/rcv_decide.py`. Extra candidates:
  `python src/rcv_spikes.py <seed> <spike>`; candidate blends are the role dicts (`BASE`=A′, `A2`=A″, `A3`=A‴) in `rcv_decide.py`.
- Rebuild the final recipe A‴ (full-data refit): `python src/attn_pool.py full text` → `python src/final_refit.py subs/x.csv`
  → twin lock. Gives `subs/v9_a3_full_lock.csv`.
- Post-process every submission: `python src/twinlock.py subs/in.csv subs/out_lock.csv`.
- Kaggle GPU jobs: `kernels/<name>/` holds `kernel-metadata.json` + a **copy** of the `src/` script. Edit in `src/`, copy over, then
  `PYTHONUTF8=1 python -m kaggle kernels push -p kernels/<name>`. On Kaggle, models mount under `/kaggle/input/models/...` and
  competition data under `/kaggle/input/competitions/...` (scripts find them with glob). Outputs are downloaded into `runs/<name>/`.
- Quota / limits: `python -m kaggle quota` (GPU/TPU hours left, refresh time) and
  `python -m kaggle competitions submission-limits dataverse-detecting-emotions-from-hindi-speech` (submissions left today).

## Architecture / data flow
1. `feats/train_meta.csv` + `feats/test_meta.csv` define the **canonical row order: train rows then test rows** (1035 total).
   Every `feats/*.npy` follows it, so `X[:n]` is train and `X[n:]` is test (`n = 826`).
2. `src/hcv.py` is the shared core: `from hcv import *` gives `y`, `n`, `tr`, `te`, `fold`, `group`, `f1()`, `per_fold()`, and
   `probe(X)` (StandardScaler + LogisticRegression, C picked by OOF log-loss, returns `(C, oof, test)`).
   It builds `feats/folds_v2.csv` on first import.
3. Members (probes on frozen features) save OOF/test probabilities to `oof/v2_<member>.npy` + `_test.npy`
   (repeated-CV versions: `oof/rcv_s{seed}_<member>.npy`). Blends are fixed-weight averages of these probs, then argmax.
4. Submissions go in `subs/` as `Id,emotion` (lowercase labels, `sample_submission.csv` row order), always twin-locked.
   Recipe names (A′, A″, A‴, K1…) and weights are defined in `HANDOFF.md`; `final_refit.py` hard-codes the A‴ blend.

## Method rules (non-obvious, enforced)
- **Honest CV only.** About 30% of clips are noisy near-duplicates. Use the near-dup-grouped folds in `hcv.py`
  (`feats/folds_v2.csv`), never the old `feats/folds.csv` from `src/probe.py` (it inflated CV by about 0.04).
  To re-split with another seed, reassign `hcv.fold` (see `rcv_members.py`).
- **Pre-register** the variant before looking at its score. Adopt only if gain ≥ +0.005 and ≥3/5 folds up; borderline gains need
  repeated CV (seeds 42/7/11), adopted only if the mean gain is ≥ +0.005 and the gain is > 0 on every split. Single-split gains < 0.005 are noise.
- No public-leaderboard tuning. Sanity-check each submission: class counts about 45–60 each, twin lock applied.
- Whisper features need trailing-silence padding (≥12 s, 30 s default); unpadded top layers degrade badly.
- Competition rule: external data must not match test clips (no human labels of test records). Code may now be shared;
  the planned GitHub push (repo + include/exclude list in `HANDOFF.md`) still needs the user's go-ahead.
