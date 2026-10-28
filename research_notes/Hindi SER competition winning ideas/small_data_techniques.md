# Small-data SER on frozen SSL features: techniques not yet tried (2024-2026 research)

Scope: 13 tool calls (searches + page reads). No downloads or runs. Items already in EXPERIMENTS.md (LR-only probes, layer concat L24+L32, S3 augmentation, P3 attentive head, P1 mixing, P2 kNN, L1-L5, K1-K4, R1-R3, S5, F1-F3, T1, label spreading, pseudo-labels, LR stacker, calibration, per-speaker norm, external data, fine-tuning) are not re-proposed.

## Key question 1: probe-level improvements

### Takeaway
Classifier-family swaps (linear SVM, ridge, LDA) and learned layer weights are the cheapest untested probe ideas, but I found no 2024-2026 head-to-head paper on frozen Whisper/WavLM SER that shows a reliable gain over L2-LR. Expected gain is small. Treat them as cheap checks under the repeated-CV gate, not as likely wins.

### Cited Findings
- SUPERB-style learnable weighted sum over all layers (one scalar per layer, trained with the head, backbone frozen) is the standard frozen-probe recipe for SER; a benchmarking study reports the last layer is not always best, so all hidden states are collected and weighted. [Speech SSL benchmarking, arXiv 2306.00452](https://arxiv.org/pdf/2306.00452) (also cited in the search summary of [arXiv 2211.01756](https://arxiv.org/pdf/2211.01756)). Gains not quantified in what I read.
- Cross-domain (speech+music) 2024 study compared mean pooling, learnable weighted sum and adapters on frozen SSL models; it reports a learnable weighted-sum variant but I did not read its numbers. [arXiv 2409.17899](https://arxiv.org/pdf/2409.17899)
- VARAN (Aug 2025): layer-specialised heads with data-dependent layer weighting. Gains are reported mainly with LoRA fine-tuning, so it does not directly transfer to a frozen probe. [arXiv 2508.12061](https://arxiv.org/pdf/2508.12061)
- Attentive pooling on frozen Whisper (Shendabadi et al., arXiv 2602.06000, Feb 2026): encoder frozen; Whisper-Small on ShEMO (about 3,000 utterances, 87 speakers, 10-fold speaker-group CV): mean pooling UA 82.41, attentive 82.86, QKV 83.07. On IEMOCAP (4 classes, 5-fold session CV): mean 70.61, attentive 72.64, QKV 72.96 UA. Intermediate layers did better on ShEMO. [arXiv 2602.06000 HTML](https://arxiv.org/html/2602.06000v1)
  - Caveat: the paper's Table 1 numbers do not match its per-layer tables in several cells, and Section 7 mentions "Whisper Large" while the tables are Whisper Small. Treat the gains (+0.5 to +2.3 UA) as unverified.
  - Our status: this is the L3 / P3 family, already tested (L3 alone 0.909 best single member, +0.0045 as 0.2 extra, and P3 adopted). Not new.
- A 2024 CLiCIT cross-lingual study: Whisper best with a linear classifier on the last layer; Wav2Vec 2.0 best on middle/early layers. Supports per-model layer choice, which we already do. [CLiCIT 2024.clicit-1.31](https://aclanthology.org/2024.clicit-1.31.pdf)
- Odyssey 2024 Task 1 1st place (arXiv 2405.20064): 7 models trained with CE and focal loss, uniform and prior-based class weights, on WavLM-large / HuBERT-XL / Whisper-large-v3 features, combined by majority vote; macro-F1 35.69% (challenge metric, 68 submissions). Their focal-loss gain is for class imbalance, which does not apply here (our classes are balanced). The transferable part is hard majority voting across diverse members instead of probability averaging. [arXiv 2405.20064](https://arxiv.org/pdf/2405.20064)
- Classifier comparison: a 2025 master's thesis compared SVM and MLP heads on frozen WavLM/HuBERT/Dasheng embeddings and found embedding choice and data distribution matter more than the head; LLMs underperformed SVM/MLP. No LR vs SVM vs LDA numbers found. [search summary, no primary URL verified]. A NeurIPS 2026 submission reports frozen wav2vec2 + SVM/MLP beats MFCC pipelines by 11-13 pp under speaker-independent GroupKFold. Not a classifier comparison. [search summary, no primary URL verified]

### Inferences
- Linear SVM / ridge / LDA on the same standardised features are cheap (minutes on cached arrays) and give a direct answer. Expected gain is low; the honest-CV gate will probably reject them.
- Learned softmax weights over all 33 Whisper layers (SUPERB-style) is untested for us. We have only hand-picked L24+L32 and L16/L10 choices, and those choices were made on the same splits (selection bias). A learned or uniform-average-over-depth member would be the less-biased version of that search.
- Majority vote over diverse members (Odyssey 1st) is cheap to test on existing OOF/test probabilities.

### Gaps
- No primary-source 2024-2026 SER paper found that reports SVM vs LR vs LDA gains on frozen Whisper features. I did not find a reported gain for feature-space mixup on frozen SSL pooled vectors in SER.
- Did not read the full 2409.17899 weighted-sum numbers.

## Key question 2: prosody / behavioural features on top of SSL

### Takeaway
The Feb 2026 Zenodo paper (Kotian & Singh, Univ. of Mumbai, published in Journal of Tianjin University Science and Technology 59(2), 2026) is abstract-only in what I could read: it gives no dataset size, feature list, protocol or numbers. Its claim (behavioural cues help low-arousal classes) is unverified. Other 2025 evidence says SSL beats prosody for emotion and gives no SSL+prosody fusion gain number.

### Cited Findings
- Zenodo record 18797014 (published 23 Feb 2026): dual-branch model, acoustic encoder + BiLSTM-attention on behavioural-prosodic features (pauses, rhythm, rate, emphasis); claims accuracy and macro-F1 gains from behavioural features, largest for sadness and neutral. Abstract gives no numbers, no feature count, no CV or speaker-independence protocol. [Zenodo 18797014](https://zenodo.org/records/18797014)
- Wallbridge et al., Interspeech 2025 (Masked Prosody Model, arXiv 2506.02584): on RAVDESS (1,440 utterances, 24 speakers, 8 emotions), linear probe on untransformed pitch/energy/VAD = WA 0.10; HuBERT 0.65; wav2vec 0.54; MPM 0.24. Prosody-only representations are far below SSL for emotion. No SSL+prosody fusion experiment. [arXiv 2506.02584](https://arxiv.org/html/2506.02584v1)
- Kakouros, Interspeech 2025 (arXiv 2506.02239): combines acoustic prosodic functionals with SSL representations, gain mainly from restricting features to word-informativeness-selected segments. No fusion-only number in the abstract I read. [search summary, arXiv 2506.02239]
- Our prior: openSMILE eGeMAPS alone 0.578/0.584 and ComParE 0.726/0.673, both ✗ in the blend (EXPERIMENTS.md, t3). eGeMAPS does not contain explicit pause ratio or speech rate.

### Inferences
- The untested part of the Zenodo idea is explicit behavioural features (pause ratio, number and length of pauses, speech rate in syllables/s from the Whisper ASR word timestamps or VAD, energy-contour slope). These are not in eGeMAPS. Expected gain is small: prosody-only is weak in every study I found, and our blend already has three strong audio members.
- Implementation sketch (fits under 3 h): VAD with `webrtcvad` or energy threshold on the existing audio; per clip compute pause count, pause ratio, mean pause length, voiced rate, F0 slope (librosa pyin, about 10 features); standardise; LR member (small C); pre-register as a 0.2-weight extra, gate: mean ≥ +0.005 and > 0 on seeds 42/7/11. Prior probability of passing: low.

### Gaps
- Could not read the Zenodo PDF (only the landing page). Dataset size, features and numbers unknown. Need the PDF before citing any number from it.
- Kakouros fusion number unknown.

## Key question 3: ensembling and selection robust at about 800 clips with near-duplicates

### Takeaway
The strongest robust evidence from the challenge papers is diverse-member ensembling (Odyssey 2024 1st: 7-model majority vote; SAILER: 3-system ensemble top 3). Both are consistent with our existing blend. The gain is unverified on our data and the risk of selection bias on three splits is real (about 30 candidates tested on seeds 42/7/11, per EXPERIMENTS.md).

### Cited Findings
- Odyssey 2024 1st (arXiv 2405.20064): majority vote over 7 diverse models beat single models. [arXiv 2405.20064](https://arxiv.org/pdf/2405.20064)
- SAILER, Interspeech 2025 (arXiv 2505.22133): single system without ensembling beat more than 95% of submissions; 3-system ensemble top 3; macro-F1 above 0.4 (task is naturalistic podcast SER, 4+ classes with heavy imbalance). The full-text numbers for text fusion, mixing, annotation dropout (+0.013-0.019, +0.020, +0.005 as cited in our HANDOFF notes) were not verified by me; the abstract page does not contain them. [arXiv 2505.22133](https://arxiv.org/abs/2505.22133)
- Interspeech 2025 naturalistic challenge: USC team (Lertpetchpun, Feng, Byrd, Narayanan) reports first place on the attribute task and a two-system ensemble giving the top score. Their best model kept "Other"/"No Agreement" samples in training. Self-reported; no official winners page found. [ISCA Archive lertpetchpun25](https://www.isca-archive.org/interspeech_2025/lertpetchpun25_interspeech.html)

### Inferences
- Majority-vote over our existing members (hard votes of the 6 members per fold) is a 10-minute test on saved OOF/test probabilities. Pre-register and apply the same gate.
- Keeping ambiguous/"no agreement" training clips (USC) is the reverse of our L5 label-noise cleaning, which was rejected (−0.0020). Not worth repeating.

### Gaps
- No ablation found that isolates how much ensembling a diverse set of frozen probes gains over the best single member on small SER data.

## Key question 4: what the winners used that transfers

### Takeaway
Most transferable winner components are already in our recipe: Whisper-large-v3 as main frozen encoder (SAILER, Odyssey 2nd/3rd place), w2v-BERT-2.0 as second audio model (Odyssey 2024 2nd place per our HANDOFF notes), text fusion with an LLM/text branch (USC, SAILER), and multi-model ensembling. The remaining untested winner ideas are the hard majority vote and the focal/class-weight variants (N/A for balanced classes).

### Cited Findings
- Odyssey 2024 1st: focal loss + prior class weights for imbalance; 7-model majority vote using WavLM-large, HuBERT-XL and Whisper-large-v3. [arXiv 2405.20064](https://arxiv.org/pdf/2405.20064)
- Odyssey 2024 entry with focal loss and a 3-model ensemble with per-class thresholds (arXiv 2406.10598): not top-ranked. [arXiv 2406.10598](https://arxiv.org/pdf/2406.10598)
- MER 2025 (from our HANDOFF notes): pseudo-labels from an external LLM agreement; not usable (external API forbidden by data rules). Not re-verified here.

## Ranking by expected gain / implementation time (my judgement; no reported gain transfers directly)

| # | idea | source | time | prior of passing gate |
|---|---|---|---|---|
| 1 | Hard majority vote over the 6 existing members vs probability averaging | Odyssey 2024 1st (2405.20064) | ~10 min on saved probs | low-moderate |
| 2 | Linear SVM / ridge / LDA heads on the same standardised whisper L24L32 and w2vbert features (C by OOF) | no direct SER source; generic | ~30 min | low |
| 3 | Learned softmax weights over all 33 Whisper layers (SUPERB-style), CV-trained | SUPERB-style, 2306.00452 | ~1 h | low-moderate (less selection bias than hand picks) |
| 4 | Explicit pause/rate/F0-slope behavioural member (about 10 features) as 0.2 extra | Zenodo 18797014 (unverified), Wallbridge 2506.02584 (prosody alone weak) | ~2 h | low |
| 5 | Feature-space mixup on pooled whisper vectors with LR or small MLP | no SER source found | ~1 h | low (not supported by evidence) |

Not recommended: SetFit or prototype heads (no speech evidence found); attentive pooling variants (already tested as L3/P3); audio mixing, kNN, label cleaning, multi-view TTA (all tested, rejected).

## Overall gaps and honesty notes
- The Zenodo Feb 2026 paper: only the landing page was readable. No dataset, features, protocol or numbers. Do not cite numbers from it.
- Few 2024-2026 papers report frozen-probe ablations on SER with under 1,000 clips. Most gains found are from fine-tuning or from large challenge datasets, which do not transfer.
- Several numbers in the search summaries came from aggregator or mirror pages (2602.06000 benchmark scores, the 2026 NeurIPS paper). Marked as unverified above.
- Every candidate above will be tested on the same three splits used for A-prime-to-A-triple-prime (about 30 candidates already). Adopted gains are optimistic by a few thousandths. Use fresh splits if available.
