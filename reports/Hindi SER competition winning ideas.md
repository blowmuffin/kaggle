# Spend the last hours on picks, not models

The five research notes contain no idea likely to add more than one test clip before the 12 Oct 18:30 UTC deadline. Of about 25 proposals, 14 are already in EXPERIMENTS.md, are forbidden, or have licence or gating problems. The rest are cheap checks with a low prior, because every new member since 6 Oct has been redundant with the blend. The highest-value action for the remaining ~10 hours is the choice of the **second final pick**. The current pair (`v9_a3_full_lock` and `v5_aug_c1_lock`) differs on a single private clip (Id 235). A more diverse hedge with equal CV is available among files already built, and choosing it costs one submission. On the modelling side, three untested ideas are worth a pre-registered run in this order: hard majority voting over existing members, a train+test StandardScaler, and SUPERB-style learned weights over all 33 Whisper layers. Each takes under 1.5 h on cached features, and each should be expected to fail the gate. The gate itself is weak evidence at this point. About 30 candidates have now been tested on seeds 42/7/11, and a gain of +0.005 sits roughly at the level the best of 30 pure-noise candidates would reach. A new "pass" therefore needs to change test predictions in a sensible way before it is submitted, not just clear the threshold. The deadline extension makes HANDOFF's "FINAL PICKS" section outdated, but it does not change its main conclusion: the audio side is saturated.

## The arithmetic caps what modelling can still buy

The competition scores 0.4·public + 0.6·private. With about 90 public clips and about 119 private clips, **one public clip is worth about 0.0045 of final score and one private clip about 0.0050** (derived from the 0.011-per-public-clip step recorded in EXPERIMENTS.md). The top public score of 0.965 is two public clips above our 0.942. Of the 209 test clips, 78 are already twin-locked at about 99% label agreement. That leaves about 131 clips that the model decides. At honest CV of about 0.92, that is roughly 10 errors, split between public and private. The recent history shows how little the late ideas moved those clips. The adopted S3+C1 changed 2 test predictions. The adopted P3 head changed 0. The full-data refit (F2) changed 1. The rejected K1–K4 candidates changed 2–4 each. A new idea that passes the gate today will probably flip 0–2 test clips, worth about ±0.005–0.010 of final score. That is the same order as the value of choosing a better hedge pick, which needs no new code.

Selection bias makes the gate unreliable at the margin. Per-split gains for rejected candidates swing by about ±0.007 (for example R1 Dasheng: −0.0047/−0.0012/+0.0124). That puts the standard error of a 3-split mean near 0.004. Under that noise, the best of 30 null candidates would be expected to reach about +0.006 to +0.008, which is about where the two adopted spikes landed (+0.0060 and +0.0062). The project rule (pre-register; mean gain ≥ +0.005; > 0 on every split) is still the right filter. Today, though, a pass should be read as "not harmful", not as a proven improvement. The extra tie-breaker should be whether the test flips make sense: they should mostly fall on low-confidence, non-locked clips and should not reduce neutral predictions, since neutral is the weakest class (31/217 wrong in A′ OOF).

## Most of the researchers' ideas are already spent, forbidden, or gated

The table checks every substantive proposal in the notes against EXPERIMENTS.md and the competition rules. "Evidence" separates numbers the researchers read in a primary source, claims seen only in an abstract, snippet or secondary summary, and the researchers' own speculation.

| Idea (note) | Status vs EXPERIMENTS.md | Evidence quality | Rule / licence risk | Verdict |
|---|---|---|---|---|
| Attentive / QKV pooling on frozen Whisper | Tested: L3 +0.0045 ✗, P3 adopted, L3a/L3b ✗ | Paper tables internally inconsistent | none | **Drop (done)** |
| Hard majority vote over members (Odyssey 2024 1st) | Untested (LR stacker ✗ is a different thing) | Primary paper, but the gain comes from a 7-model imbalanced task | none | **Try, 10 min** |
| SVM / ridge / LDA heads | Untested | No SER source at all; speculation | none | Low; bundle with the vote |
| Learned weights over all 33 Whisper layers (SUPERB) | Partly: hand-picked L24+L32 adopted; L24–32 mean +0.003; L20+26+32 −0.001 | Standard recipe; no gain number read | none | **Try, ~1–1.5 h** |
| Train+test StandardScaler (transductive, no labels) | Untested | Researcher speculation | Allowed: uses no test labels | **Try, ~45 min** |
| Feature-space mixup on pooled vectors | Close cousin P1 audio mixing ✗ (−0.0012) | No source found | none | Drop |
| Pause / speaking-rate prosody member (Zenodo 2026) | eGeMAPS/ComParE ✗; explicit pauses untested | **Abstract only**, no numbers ([Zenodo](https://zenodo.org/records/18797014)); prosody-only probes far below SSL ([Wallbridge 2025](https://arxiv.org/html/2506.02584v1)) | none | Low; only if CPU is idle |
| Pseudo-gender auxiliary (CAMuLeNet) | Speaker x-vector normalisation ✗ | Multitask +0.022 WA in the paper's own ablation ([arXiv 2406.08931](https://arxiv.org/html/2406.08931v2)); a multitask loss does not map onto LR probes | Gender classifier licence unchecked | Drop |
| WhissleAI STT-meta-HI encoder | New checkpoint; Conformer family (K2, K3 strong alone but redundant) | Card: own emotion macro-F1 0.224 ([HF](https://huggingface.co/WhissleAI/STT-meta-HI)) | **Training source of 223K Hindi utterances not stated; could include the BhavVaani sitcom (rule 4b)**; NeMo setup on Windows | **No-go** unless provenance is confirmed |
| IndicConformer / IndicWav2Vec | indicwav2vec skipped as gated | Card returned 401; licence pages conflict ([HF](https://huggingface.co/ai4bharat/IndicConformer)) | Gated; needs the user to accept terms | Drop |
| SenseVoice-Small, Kimi-Audio-7B | Audio-LLM family all ✗ (K1, K4, R2, R3, S5) | Self-reported; no Hindi in released Small | Licence "other" / mixed | Drop |
| whisper-hindi-large-v2 or IndicConformer transcripts | F3: better transcripts +0.03 alone, −0.0013 in blend ✗ | Self-reported WER ([HF](https://huggingface.co/vasista22/whisper-hindi-large-v2)) | Apache-2.0 | Drop |
| IndicBERT v2 / MuRIL / HindiEmotion text member | Text route ✗ three times (char n-gram, English, F3) | Fine-tuning benchmarks only ([IndicXTREME](https://arxiv.org/pdf/2212.05409)); HindiEmotion scored on 47 examples ([HF](https://huggingface.co/vashuag/HindiEmotion)) | MIT for HindiEmotion; others unread | Drop (CPU-idle filler at most) |
| LLM retrieval few-shot on transcripts | Listed as untried in HANDOFF; K4 (Omni + transcript) ✗ −0.0036 | Mixed and weak for low-resource languages ([Unibuc SemEval-2025](https://arxiv.org/pdf/2508.01161v1)) | Retrieval must exclude near-dup twins | Drop for this window |
| ASR-confidence weighting of the text member | Untested | Researcher speculation, no source | none | Drop (text weight already redundant) |
| Transductive batch refinement, kNN, label spreading | Label spreading ✗, P2 kNN ✗, pseudo-labels ±0 | Zero-shot only ([arXiv 2606.17160](https://arxiv.org/pdf/2606.17160), snippet) | none | Drop |
| Seed averaging | Done: 15 fold models, 0 test changes | Seed scatter ±0.003 is snippet-only ([CEUR](https://ceur-ws.org/Vol-4283/paper369.pdf)) | none | Drop (done) |
| New Hindi emotion corpora | External CC-BY ✗ −0.009 | None with a clean licence found | DataCatalyst eval-only; LlamaLens NC | Drop |
| Original BhavVaani labelled release | Never used | Primary paper ([arXiv 2406.08931](https://arxiv.org/html/2406.08931v2)) | **Forbidden**: human labels of test records (rule 4b); form-gated, no licence ([GitHub](https://github.com/arnav10goel/CAMuLeNet)) | **Never** |

Several claims in the notes need correction. The SAILER gains cited in HANDOFF (+0.013–0.019 from text, +0.020 from ensembling, +0.005 from mixing) were **not verified**; only the abstract was readable ([arXiv 2505.22133](https://arxiv.org/abs/2505.22133)). The attentive-pooling paper's gains of +0.5 to +2.3 UA come from tables that contradict each other ([arXiv 2602.06000](https://arxiv.org/html/2602.06000v1)). Our own L3 result is the better evidence for that family. The Hugging Face note recommends testing STT-meta-HI first. This report overrides that: the checkpoint's undisclosed Hindi emotion training data is a live rule-4b risk, and its emotion head is too weak (macro-F1 0.224) to suggest a useful emotion-trained encoder. The one fully read, verified primary source is the BhavVani paper. It reports **0.453 weighted accuracy for the best model under 10-fold leave-speaker-out** ([arXiv 2406.08931](https://arxiv.org/html/2406.08931v2)). That supports the project's view that the competition split is speaker- and near-duplicate-leaky, and that the twin lock, not modelling, explains much of the 0.94.

## Every hedge candidate disagrees on the neutral boundary

A diff of the locked submission files against pick 1 (`v9_a3_full_lock`) shows that every alternative disagrees only on the neutral boundary. Each differing clip is neutral in one file and something else in the other.

| File | Honest CV vs A‴ (mean of 3 splits) | Public | Clips differing from v9 |
|---|---|---|---|
| v5_aug_c1_lock (A″, current pick 2) | ≈ equal (A‴ = A″ build) | 0.94230 | 235 |
| v4_wL24L32_lock (A′) | lower (A″ +0.0060 over it) | 0.94230 | 883 |
| v8_K1_lock | +0.0019 | 0.93048 | 125, 235, 431 |
| v8_K2_lock | −0.0027 | not submitted | 235, 542, 927 |
| v8_K3_lock | −0.0003 | not submitted | 819, 927 |
| v8_K4_lock | −0.0036 | not submitted | 125, 235, 431, 755 |

Kaggle scores each team by the better of its two selected submissions. The value of pick 2 therefore comes from how much it **differs** from pick 1 on uncertain private clips, provided its expected accuracy is equal. A near-copy adds almost nothing. The current v5 hedge pays off only if Id 235 is neutral, which is about a 50/50 call (0.45 vs 0.44). K1 is out: it is a confirmed public clip down (−0.0045 final). Its overlap with v2_base_omni_e5_lock, which also lost a public clip and also says neutral on 431, suggests weakly that 431 is public and that v9's "sad" is correct there. K2 is the most useful untested hedge. It contains v5's Id 235 flip plus two more neutral flips (542, 927), and its CV deficit (−0.0027, about 0.35 of a clip over 131 clips) is well inside split noise. Submitting it once applies HANDOFF's existing rule B (best public among candidates within 0.005 CV). It does not tune to the public board. The [Kaggle Book](https://packtpub.com/en-jp/product/the-kaggle-book-9781801817479/chapter/designing-good-validation-8/section/designing-good-validation-8) and the [DataCamp](https://campus.datacamp.com/courses/winning-a-kaggle-competition-in-python/modeling?ex=12) advice both say the same: one pick on robust CV, one diverse hedge. Neither source quantifies the benefit, so this is a judgment call, not a measured gain.

## Ranked plan for the remaining ten hours

Times assume a start around 08:30 UTC. Freeze modelling by **16:30 UTC**, so that building, twin-locking, sanity-checking (class counts 45–60, lock applied, diff vs v9) and selecting both picks finish with an hour to spare before 18:30 UTC. Pre-register each item in EXPERIMENTS.md before running it. Use the 3-split rule, and add the test-flip sanity check above as a second condition.

| # | Action | Expected gain (final score) | Time | Compute | Go / no-go |
|---|---|---|---|---|---|
| 1 | Submit `v8_K2_lock.csv` once (rule B), then re-pick | 0 to +0.005 as a hedge (Id 235 plus 2 neutral clips) | 15 min | none | **Go** if public ≥ 0.94230: picks = v9 + K2. If it drops: keep v9 + v5. Never pick a file that is a confirmed public clip down |
| 2 | Hard majority vote (and a linear-SVM / LDA swap) on saved `oof/rcv_s*` probabilities | ~0 (prior about 10% to pass) | 30 min | local CPU | Adopt only if mean ≥ +0.005, > 0 on all splits, and ≤ 3 test flips, none of them on locked clips |
| 3 | StandardScaler fit on train+test features for all LR members | 0 to +0.002 | 45–60 min | local CPU (the S3 member is the slow one, about 25 min per split) | Same gate; drop the idea if any split is negative |
| 4 | Softmax-learned weights over the 33 pooled Whisper layers, as a 0.2 extra | 0 to +0.003 (prior about 15%) | 1–1.5 h | local RTX 4050 | Same gate; reject if it replaces rather than adds (the L3a lesson) |
| 5 | Pause-ratio / speech-rate / F0-slope member (~10 features) | ~0 (prior < 10%) | 2 h | local CPU, in parallel with #4 | Only start if #2–#4 are done before 13:00 UTC |
| — | STT-meta-HI, IndicConformer, LLM few-shot, any new text model or audio LLM | ≤ 0 net of risk | 2–3 h each | Kaggle GPU / local | **No-go**: provenance or gating risk, or a family already shown redundant |

If an item passes, build its full-data refit with the F2 procedure, twin-lock it, and diff it against v9. If it changes no test clips (as P3 did), there is nothing to submit. If it changes 1–3 clips, it replaces v9 as pick 1 only when both its CV and its public score are no worse. Otherwise it becomes a hedge candidate under rule B. Submissions are limited to 5 per UTC day. Item 1 plus at most two candidate files leaves a buffer. Run `python -m kaggle competitions submission-limits` before item 1, and run `kaggle quota` only if a Kaggle GPU job is reconsidered (none is recommended).

## Conclusion

The research round confirms what the experiment log already showed: on 826 leaky clips with a twin lock, the remaining score sits in about ten borderline neutral-boundary decisions. More upstream models or better text will not reliably change those decisions. Published SER results on this corpus (0.45 weighted accuracy, speaker-independent) are so far below the leaderboard that the leaders' extra 0.02 is more likely two lucky public clips than a better method. There is no evidence-backed path to 0.965 in ten hours.

The useful lever is the choice of picks. The current pair is nearly identical and wastes the second slot. A CV-equivalent pick that differs on several uncertain private clips is worth about as much as a successful modelling experiment, at almost no cost. Any modelling work in this window should be treated as a low-probability lottery ticket, judged by both the gate and the plausibility of its test flips.
