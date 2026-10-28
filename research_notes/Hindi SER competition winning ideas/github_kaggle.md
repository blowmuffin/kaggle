# Public code and solutions for Hindi SER (Kaggle DataVerse competition, BhavVaani/BhavVani source)

Scope: public Kaggle notebooks, discussions, GitHub repos and papers for Hindi speech emotion recognition (SER) and the BhavVani dataset. Checked against EXPERIMENTS.md and HANDOFF.md ("exhausted" list). No code was downloaded or run; only pages were read.

Context: the competition clips are the BhavVani set (8,734 clips, about 13 h, sitcom source). Our A‴ recipe has Whisper-L24+L32, w2v-BERT, Omni, e5 text and an attentive audio-text head.

## Kaggle competition pages (Code and Discussion tabs)

### Takeaway
No public competition notebooks or discussion posts could be read. The Kaggle Code and Discussion pages returned only the page title, so I could not verify any public approach or score from them. Treat this as a gap, not evidence that none exist.

### Cited Findings
- The Kaggle Code tab (https://www.kaggle.com/competitions/dataverse-detecting-emotions-from-hindi-speech/code) returned only the title "DataVerse: Detecting Emotions from Hindi Speech | Kaggle" — [Kaggle Code tab](https://www.kaggle.com/competitions/dataverse-detecting-emotions-from-hindi-speech/code)
- The Kaggle Discussion tab returned the same title only, with no posts or login status — [Kaggle Discussion tab](https://www.kaggle.com/competitions/dataverse-detecting-emotions-from-hindi-speech/discussion)
- Web search for the competition and "macro F1 solution" found no writeups — [search, no URL](n/a)

### Inferences
- The public leaderboard figures in HANDOFF.md (0.965 top) cannot be linked to any public method from here. Any "winning recipe" must be inferred, not copied.

### Gaps
- Logged-in Kaggle view of Code and Discussion was not available. The user can check these tabs directly (sort by votes) and paste any notebook link for a read.

---

## BhavVani original paper and CAMuLeNet (Interspeech 2024)

### Takeaway
The BhavVani dataset comes from Goel, Hira and Gupta (Interspeech 2024). Their CAMuLeNet reaches only WA 0.453 / WF1 0.441 on BhavVani under 10-fold leave-speaker-out. The published SOTA on this data is far below our 0.94, which points to near-duplicate and speaker leakage in the competition split, as HANDOFF already suspects.

### Cited Findings
- BhavVani: about 13 h, 8,734 utterances, mean 5.08 s, from the sitcom "Sarabhai vs Sarabhai"; text reused from earlier sarcasm work. Annotation by 18 native Hindi speakers (about 750 clips each), three reviewers per label, Fleiss' kappa 0.637 — [arXiv 2406.08931 HTML](https://arxiv.org/html/2406.08931v2)
- The paper says seven Ekman-based classes, but lists six labels (Neutral, Surprise, Enjoyment, Disgust/Anger combined, Fear, Sadness). Intro says "over 9,000 utterances", which conflicts with 8,734 — [arXiv 2406.08931 HTML](https://arxiv.org/html/2406.08931v2)
- Protocol: 10-fold leave-speaker-out, Adam, up to 20 epochs. Metrics are weighted accuracy (WA) and weighted F1 (WF1), not macro-F1 — [arXiv 2406.08931 HTML](https://arxiv.org/html/2406.08931v2)
- BhavVani results (WA / WF1): Wav2Vec2.0 0.251/0.317; HuBERT 0.253/0.332; WavLM 0.303/0.305; Whisper-Base 0.261/0.265; Whisper-Medium 0.412/0.439; CAMuLeNet 0.453/0.441; CAMuLeNet w/o multitask 0.431/0.429; w/o co-attention and multitask 0.407/0.398 — [arXiv 2406.08931 HTML](https://arxiv.org/html/2406.08931v2)
- CAMuLeNet = co-attention fusion of an AlexNet spectrogram encoder, a Bi-GRU over 40 MFCCs, and Whisper encoder features; plus an auxiliary gender task with BCE loss (loss = 0.4·CE + 0.1·BCE + 0.2). Authors call overall BhavVani performance "sub-optimal" — [arXiv 2406.08931 HTML](https://arxiv.org/html/2406.08931v2)
- Code link: github.com/arnav10goel/CAMuLeNet. Read: the repo says its contents moved to NLP-SBILAB/CAMuLeNet. It has only README.md (no training or eval scripts, no LICENSE), 5 commits, 8 stars. It says the BhavVani data is not included and requires a Google Form access request (forms.gle/9AqxS2oY4XVSeH1UA) — [CAMuLeNet GitHub](https://github.com/arnav10goel/CAMuLeNet)
- Licence: none stated for code or data. The paper says the dataset "will be released to the community" — [arXiv 2406.08931](https://arxiv.org/abs/2406.08931)

### Flag: FORBIDDEN for test-clip labels
- The original labelled BhavVani release is form-gated (forms.gle link on the CAMuLeNet repo) and has no stated licence. Our competition data appears to be drawn from it. Its labels must NOT be used to label or check test clips (competition foundational rule 4b). Using it as external training data also needs the licence and a leakage check, and the HANDOFF "external CC-BY" route already showed domain mismatch hurts (−0.009). Not proposed here.

### Inferences
- The gender auxiliary task (multitask, CAMuLeNet) is not in our EXPERIMENTS list. Gender labels are not in the competition data. A possible route is pseudo-gender labels from a public gender classifier run on train and test audio (no human test labels). Gain is unknown; CAMuLeNet's own multitask gain is +0.022 WA over w/o multitask on BhavVani, and the paper's ablations are on a different protocol.
- The co-attention fusion idea overlaps our P3 audio-text attentive head (covered).

### Gaps
- Full CAMuLeNet code (NLP-SBILAB/CAMuLeNet) was not read. Its licence and training details are unverified.
- No public reproduction of BhavVani scores under our macro-F1 metric exists.

---

## Other Hindi SER GitHub repos

### ankuPRK/Emotion-Recognition-in-Hindi-Speech
- URL: https://github.com/ankuPRK/Emotion-Recognition-in-Hindi-Speech
- What it does: 8-class Hindi SER on 2 speakers (male and female subsets) of IITKGP-SEHSC. Features are MFCC (13) and subband spectral coefficients (26), a 273-dim vector. SVM with RBF kernel is best.
- Reported score: 89.08% accuracy (male), 83.16% (female). Accuracy only, not macro-F1.
- Licence: not stated. 3 commits.
- Covered: yes. Classical spectral features and classifiers are close to our openSMILE and eGeMAPS members, which were too weak (0.58–0.73). Not useful.

### kush-2418/Hindi-Emotion-Analysis-ML
- URL: https://github.com/kush-2418/Hindi-Emotion-Analysis-ML
- Hindi text (not speech), 4 classes angry/happy/sad/neutral. Covered: our e5 text branch (0.739 alone) already does this better. Not useful.

### HF vashuag/HindiEmotion
- URL: https://huggingface.co/vashuag/HindiEmotion
- Indic-BERT text emotion model, reported accuracy about 80.85% on a random test split (per search summary, not read directly). Text only. Covered by our text branch. Not useful.

### alefiury/InterSpeech-SER-2025 (Ferreira et al.)
- URL: https://github.com/alefiury/InterSpeech-SER-2025 ; paper https://arxiv.org/pdf/2506.02088
- Naturalistic SER challenge, not Hindi. Whisper with text (ASR) features, F0 quantisation, graph-attention fusion. Reported test macro-F1 39.79% (val 42.20%); best single model Whisper 0.366 in the paper's table.
- Covered: Whisper + text fusion is our recipe. Score is on a much harder naturalistic task, so it does not transfer. The F0 quantisation idea is untested here (see below). Licence not checked.

### WhissleAI/STT-meta-HI (Hugging Face)
- URL: https://huggingface.co/WhissleAI/STT-meta-HI
- Hindi Conformer-CTC ASR with inline emotion and speaker-attribute tags (angry, fear, happy, neutral, sad, surprise). Reported WER 18.1% (internal validation) and 22.0% (FLEURS Hindi test). The model page lists Kaggle among its notebook options; not verified.
- Possible use: a second Hindi transcript source (F3 showed better transcripts did not move the blend, so low expected gain) or an emotion-tagged zero-shot member. Its training data and licence were NOT verified; it may have been trained on clips overlapping the competition data. Check before any use.
- Covered: partly (Whisper ASR text, Qwen3-ASR transcripts already tried).

### Indic-Mio
- URL: https://soniqo.audio/guides/indic-mio
- Emotion TTS for Indic languages, not a recognizer. Not useful.

---

## Papers with a potentially new idea (not yet tried in our EXPERIMENTS)

### Behavioural and prosodic features for Hindi SER (Zenodo, Feb 2026)
- URL: https://zenodo.org/records/18797014 (title "Evaluating the Impact of Behavioral Features on Hindi Speech Emotion Recognition: A Multimodal Deep Learning Approach", University of Mumbai)
- Claim (abstract only read): adding behavioural descriptors (pauses, rhythm, speaking rate, emphasis) to acoustic and prosodic features improves accuracy and macro-F1, most for sadness and neutral (low arousal). Results tables, corpus and licence not read.
- Covered: NO for pauses and speaking rate. We tried openSMILE eGeMAPS (0.58 alone, no gain) and speaker normalisation, but no pause ratio, rate or rhythm features. Note: HANDOFF lists this as an optional prosody member. The weak point is that our errors concentrate on neutral and sad (neutral 31/217 wrong in A′ OOF, and clips under 2 s err 13.5%), which matches the claim.
- Feasibility: CPU only, a few hours. Gate per CLAUDE.md (3 seeds, mean ≥ +0.005, every split > 0).

### Hindi emotion corpora that could replace the gap (not verified)
- IITG Politician Speech Corpus: 516 Hindi segments labelled Angry, Happy, Neutral, Sad (4 classes, same as ours). Mentioned in search results only; licence and access not checked. Domain (political speeches) differs; external Indian data hurt us before (−0.009 with CC-BY data). Low priority.
- IITKGP-SEHSC: licence-agreement corpus, already excluded in EXPERIMENTS.md.
- MNITJ-SEHSD: 500 utterances, 5 classes, licence not checked. Low priority.
- EmoInHindi and BHAAV: text corpora only. Not useful for audio.

---

## Ranked shortlist (under about 6 h, RTX 4050 or CPU)

1. Pause and speaking-rate prosody member (Zenodo 2026 idea): CPU, pre-register, repeated CV seeds 42/7/11, gate as in CLAUDE.md. Main gap: neutral and sad errors.
2. Auxiliary gender multitask (CAMuLeNet idea) with pseudo-gender labels from a public classifier on train and test audio (no human test labels). Probe-level version: add gender probability as an extra member or a feature. Medium effort; expected gain uncertain.
3. Check WhissleAI STT-meta-HI licence and training provenance before any use (low expected gain).

Not recommended: CAMuLeNet code (no training scripts, licence missing, its own gain small); any use of the gated BhavVani labels; the Interspeech 2025 pipeline (different task, low score).

## Summary of coverage
- Covered already: Whisper and Whisper-text fusion (CAMuLeNet's Whisper branch; Ferreira's text branch), attentive audio-text head (P3), classical spectral SVM (ankuPRK), text emotion (kush-2418, vashuag), Indic transcripts (F3).
- Not covered: pause and rate behavioural features; gender auxiliary multitask; F0 quantisation (Ferreira); emotion-tagged Hindi ASR as a member (WhissleAI, provenance unknown).
- No public Kaggle notebook or discussion for this competition could be read.
