# DataVerse Hindi SER: competition goal (draft, 2026-10-05)

## 1. Goal
Finish **top 3 on the final ranking** (0.4 × public + 0.6 × private macro-F1) with a solution we can defend in the
15 Oct presentation. The **private** part carries 60% of the final score, so cross-validation is the judge and the
public leaderboard is only a sanity check.

| Gate | Now | Target |
|---|---|---|
| 5-fold group CV macro-F1 (826 clips) | 0.930 (Whisper + w2v-BERT probe blend) | **≥ 0.945** |
| Public leaderboard | not submitted | ≥ 0.95 (top: 0.965) |
| Private leaderboard | unknown | top 3 |

## 2. Hard facts and constraints (from every competition page, the forum and the leaderboard)
- 826 train / 209 test clips, Hindi, acted and spontaneous speech, 16 kHz mono, 1–11 s (median 2.9 s). 4 nearly balanced classes.
- Metric: macro-F1. Final rank = 0.4 public + 0.6 private. With about 125 private rows, **one wrong clip ≈ 0.8 F1 points**.
- Deadline: **2026-10-10 18:30 UTC = 11 Oct 00:00 IST** (the rules page says 12 Oct; trust Kaggle). 5 submissions/day, 2 final picks.
- Prize eligibility (host, forum): you must **submit at least once** and **register on Unstop** (the link is in the
  "Registration" forum post; you have to do this yourself). Prizes are for UG/PG students only, and winners present on 15 Oct (in person or online).
- External models are allowed only if they are **publicly available, free or low-cost, and equally accessible to all**.
  Competition data **must not be passed to any party outside the competition**.
- Compute: Kaggle GPU ≈ 35 min left until the weekly reset on **10 Oct 05:30 IST** (then 30 h). Local RTX 4050 6 GB, one heavy job at a time.
- Data quirks: 14 duplicate train rows, 1 duplicate pair with conflicting labels, 8 test clips with an exact train twin (all "sad").

## 3. Why not Jev (TypeSafe "System One" model)
Checked against the announcement and the rules. It does **not fit this competition**:
1. **It is text-only.** Emotion here lives in pitch, energy and rhythm, and Jev cannot hear audio. A transcript would also lose
   the signal: acted speech often repeats the same sentence across emotions.
2. **It is early access (waitlist)**, which fails the "equally accessible to all participants" rule.
3. **It is a third-party API with no published data terms.** Sending test clips or transcripts to it conflicts with the data-security clause.
4. **It is zero-shot.** There is no stated way to train it on our 826 labels, and a model trained on those labels will beat it.

**What we do borrow from it** (the ideas, not the API): typed outputs (exactly 4 class probabilities, never free text),
**calibrated confidence** (temperature-scaled, checked with expected calibration error), and a single fast forward pass.
Our model is in effect a "System One" classifier for audio, built from open speech foundation models.

## 4. The solution: a modern open-source stack, best fit for audio
| Layer | Technology (2024–26) | Role | Status |
|---|---|---|---|
| Foundation encoders, frozen | **Whisper-large-v3** encoder, **w2v-BERT 2.0** (multilingual incl. Hindi), WavLM-Large | layer-wise features + calibrated logistic heads | done: 0.916 / 0.869 / 0.860 |
| Emotion foundation model | **emotion2vec+ large** (emotion-specialised self-supervised model; its classes include all 4 of ours) | features + zero-shot prior | to do; **check its licence first** (custom "model-license") |
| Audio-LLM encoder | Qwen2-Audio / Qwen2.5-Omni audio tower (Whisper-based, further trained with emotion tasks) | extra diverse features | to do; keep only if CV improves |
| Parameter-efficient fine-tuning | **Whisper-large-v3 top-half / LoRA fine-tune**, batch-length inputs instead of 30 s padding, 5 folds × 2–3 seeds | the main score driver | script ready (`src/ser_ft_whisper.py`), needs a smoke test |
| Calibrated ensembling | temperature scaling per model, then an equal-weight or regularised logistic stacker on OOF probabilities | robust blend | partial |
| Semi-supervised | **confidence-gated pseudo-labels** on test (calibrated p ≥ 0.95, 2+ backbones agree), retrain the heads | +1–2 pts typical on small sets | to do |
| Not used | text/ASR fusion (unless a quick check shows the words differ by emotion), closed APIs, hand labels | rules and fit | — |

## 5. Robustness rules (protecting the private score)
1. **CV is the only judge.** Accept a change only if OOF macro-F1 rises by **≥ 0.005 and improves on ≥ 3 of 5 folds**
   (or holds F1 and improves log-loss). Confirm final candidates with a second fold seed.
2. **No blend picking by luck.** Fix the blend members before scoring and use equal weights or nested-CV weights.
   (The current 0.930 is "best of many blends", so it is optimistic by about 0.005.)
3. **Average everything:** 5 folds × several seeds for each test prediction, and calibrate before averaging.
4. **Sanity checks on every submission:** classes near balanced (about 45–60 each), and the 8 exact-twin test clips predicted "sad".
5. **Spend submissions sparingly:** at most 2–3 per day, each testing a real CV gain, and never tune to the public leaderboard.
6. **Final 2 picks:** (A) the best-CV ensemble; (B) the best public score among candidates within 0.005 CV of (A).
   Public is 40% of the final score, so (B) is a hedge, not a gamble.

## 6. Milestones
| Date (IST) | Deliverable |
|---|---|
| **5 Oct** | First submission: probe blend (CV 0.930). Makes you prize-eligible. **You** register on Unstop. |
| 6 Oct | emotion2vec+ and audio-LLM encoder probes; calibrated blend v2; smoke-test the Whisper fine-tune locally |
| 6–7 Oct | Whisper fine-tune on the remaining ~35 min of Kaggle GPU (or LoRA locally) |
| 8–9 Oct | Pseudo-labels, multiple seeds, final stacker; **final candidates frozen by 9 Oct night** |
| 10 Oct | Quota resets 05:30: one bonus multi-seed run only if time allows. Pick the 2 finals before **11 Oct 00:00 IST** |
| 15 Oct | Presentation: clean notebook plus the story "foundation models + calibrated System-1 decisions" |
