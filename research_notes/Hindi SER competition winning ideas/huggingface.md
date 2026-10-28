# Hugging Face models and datasets for Hindi SER: new candidates

Scope: candidates not already in our stack (Whisper-large-v3, w2v-BERT-2.0, Qwen2.5-Omni, e5, Qwen3-ASR, emotion2vec, CLAP, WavLM, Whisper-v2, vakyansh, Qwen2-Audio, Dasheng, Step-Audio-2-mini, Gemma-4). Checked against EXPERIMENTS.md: Qwen2-Audio encoder, whisper-v2, vakyansh, emotion2vec, CLAP, Dasheng, Gemma-4, Qwen3-ASR AuT, Step-Audio-2 are all already tested and redundant. Not yet tested: STT-meta-HI, IndicConformer, IndicWav2Vec (re-check), SenseVoice, Kimi-Audio, XLS-R Hindi ASR encoders, MuRIL/IndicBERT text models.

Method note: about 11 web searches and 3 page fetches. Only model cards were read; nothing was downloaded. Sizes and licences come from the cards as fetched; where a card was unclear, that is stated.

## Key question 1: SER-specific or Indic speech checkpoints

### Takeaway
No audio SER fine-tune for Hindi exists on HF in a usable, clearly licensed form. The only Hindi speech checkpoint with an emotion output is WhissleAI/STT-meta-HI (Apache-2.0, small, ~458 MB), but its emotion head is weak (macro-F1 0.224). The best frozen-encoder candidates are Indic ASR encoders (IndicConformer 600M, IndicWav2Vec), and both have licence or gating issues to resolve first.

### Ranked shortlist (audio)

**1. WhissleAI/STT-meta-HI** (https://huggingface.co/WhissleAI/STT-meta-HI)
- Size: file size 458.2 MB (parent model STT-meta-1B is 600M params; this card does not state a parameter count).
- Licence: apache-2.0 (license tag on the card).
- Languages: Hindi only.
- Training: fine-tuned from WhissleAI/STT-meta-1B on 223K Hindi utterances with tags for entities, age, gender, emotion, intent. The source of these 223K utterances is not stated on the card, so overlap with the competition clips is unverified.
- SER scores (its own internal eval): emotion accuracy 47.6%, macro-F1 22.4%, weighted-F1 42.2%. Per class: HAPPY F1 0.309, NEUTRAL 0.585, ANGRY 0, SAD 0; FEAR and SURPRISE not reported. Labels: angry, fear, happy, neutral, sad, surprise. (Source: model card, fetched.)
- WER 18.1% on its internal validation set, 22.0% on FLEURS Hindi test.
- Use as a frozen extractor: Conformer-CTC, so encoder-layer mean+std pooling of valid frames, probe on concat (same design as our whisper L24+L32). Expected to fit easily in 6 GB. Card does not say the framework (likely NeMo); check the repo files before extraction.
- Why it might add signal: it is the only Hindi model whose encoder was trained with emotion tags on Hindi speech. Why it may not: its own emotion head is poor. Test this one first if you run anything.

**2. ai4bharat IndicConformer (600M multilingual; Hindi-only checkpoint gated)**
- Main repo https://huggingface.co/ai4bharat/IndicConformer: licence tag cc-by-4.0; gated ("you have to accept the conditions to access its files"). Card text is partly template (placeholder size, English-only dataset text), so language coverage is not confirmed from the card.
- Hindi-only checkpoint https://huggingface.co/ai4bharat/indicconformer_stt_hi_hybrid_ctc_rnnt_large: gated; licence not confirmed in the snippet.
- A third-party listing (mixpeek) gives the 600M multilingual model licence as MIT. This conflicts with cc-by-4.0 on the main repo. Check the exact model page before use.
- Size: ~600M params (per aggregator). Fits 6 GB in fp16 for frozen extraction (about 1.2 GB weights). Needs the IndicConformerASR code (https://github.com/AI4Bharat/IndicConformerASR).
- Action needed by the user: accepting the gate is a user action (we cannot accept terms). Not a candidate until the user does so and the licence is confirmed.
- SER scores: none reported (ASR model).

**3. ai4bharat IndicWav2Vec (Hindi)**
- Project page https://indicnlp.ai4bharat.org/indicwav2vec/ says IndicWav2Vec models are MIT; the pretraining data and YouTube videos are CC BY 4.0. AIKosh listings for other languages also say MIT.
- Conflict: our EXPERIMENTS.md records ai4bharat/indicwav2vec-hindi as gated on HF (login + terms), so it was skipped. The MIT claim is from AI4Bharat's own page, so the gate may be a separate HF setting. Re-check the HF page before deciding.
- Size: about 300M-class wav2vec2 (not verified on the card). Fits 6 GB.
- Why it is weaker than the rest: it is a wav2vec2 ASR-style Hindi encoder, which is the same family as w2v-BERT and WavLM (both already redundant). Our vakyansh Hindi wav2vec2 was redundant too. Expected value: low.

**4. Hindi XLS-R ASR fine-tunes** (shivam/xls-r-300m-hindi, LegolasTheElf/Wav2Vec2_xls_r_300m_hi_final, shiwangi27/wave2vec2-large-xlsr-hindi)
- https://huggingface.co/shivam/xls-r-300m-hindi (trained on Common Voice 8.0 Hindi), https://huggingface.co/LegolasTheElf/Wav2Vec2_xls_r_300m_hi_final (Common Voice 7.0), https://huggingface.co/shiwangi27/wave2vec2-large-xlsr-hindi.
- All three are ASR, not emotion. Licences not read. Size ~300M each.
- Value: low. Same family as vakyansh and w2v-BERT. Only worth a run if the IndicWav2Vec/IndicConformer routes fail.

**5. SenseVoice-Small** (https://huggingface.co/FunAudioLLM/SenseVoiceSmall; mirror listed at https://huggingface.co/jaman21/SenseVoiceSmall)
- Licence: "other", linking to the FunASR model licence. Not a standard licence; read it before any use.
- Languages: the released Small model targets Mandarin, Cantonese, English, Japanese, Korean (release note, per the search snippet). The 50+ language claim refers to the wider model. Hindi is not in the released Small set, so Hindi SER should be weak.
- SER: the project claims SER performance at or above prior best without target finetuning (self-reported, no Hindi number). Treat as unverified.
- Size: ~230M-class (not verified on the card). Easy to fit on GPU.
- Value: low for Hindi; the tag-based emotion output is not in our blend design.

**6. Kimi-Audio-7B-Instruct** (https://huggingface.co/moonshotai/Kimi-Audio-7B-Instruct)
- Licence: mixed. The card says it is derived from Qwen2.5-7B. Mirrors describe Apache-2.0 and MIT parts. Confirm the LICENSE file on the official repo.
- Size: 7B. fp16 is about 14 GB, so it needs Kaggle (T4 16 GB) or 4-bit locally. Base model (Kimi-Audio-7B) is not usable out of the box; use Instruct.
- Languages: broad pretraining (13M h of audio, per the card), no Hindi-specific coverage list found.
- SER: listed as a task, no Hindi score found.
- Value: low. It is another 7B audio LLM. Our Omni, Qwen2-Audio, Gemma-4, Step-Audio-2 zero-shot and hidden-state members were all redundant with the blend.

### Other audio foundations checked
- EmoBox (Interspeech 2024, https://huggingface.co/papers/2406.07162): 10 pretrained models, 32 emotion datasets, 14 languages. A third-party summary says Whisper-large-v3 was best; not confirmed against the paper. No 2025 or 2026 multilingual SER foundation model was found in this search, so this gap is unresolved.
- MMS (facebook/mms-300m) is CC-BY-NC (already skipped).

### Inferences
- The only audio candidate with a Hindi emotion objective and a permissive licence is STT-meta-HI. It is the first to test, and it is cheap.
- IndicConformer's encoder is the strongest untested Hindi-trained frozen encoder by size, but gating needs the user to accept terms and the licence conflicts across pages.
- A new encoder in an already-covered family (wav2vec2/Conformer ASR) is likely redundant. The strongest lead is an encoder trained with emotion or Hindi supervision, not a new pretraining family.

### Gaps
- No verified 2025–2026 multilingual SER foundation model with Hindi scores was found.
- IndicWav2Vec gating versus MIT licence is unresolved.
- The source data of STT-meta-HI's 223K utterances is not stated; overlap with the competition is unknown.
- Parameter counts and framework for STT-meta-HI and IndicConformer are not on the cards.

## Key question 2: Paralinguistic and emotion foundation models 2025–2026 with SOTA on multilingual SER

### Takeaway
No 2025–2026 model with a verified multilingual SER SOTA that covers Hindi was found. EmoBox (2024) points to Whisper-large-v3 as the best frozen encoder in a third-party summary, which matches our finding.

### Cited Findings
- EmoBox: 10 pretrained models, 32 emotion datasets, 14 languages, intra- and cross-corpus benchmark — [EmoBox paper](https://huggingface.co/papers/2406.07162)
- Whisper large v3 reported best on both settings — third-party summary only: [themoonlight review](https://www.themoonlight.io/tw/review/emobox-multilingual-multi-corpus-speech-emotion-recognition-toolkit-and-benchmark); not verified against the paper.
- SenseVoice claims SER without target finetuning at or above prior best (self-reported, no Hindi figure) — [SenseVoice model card](https://huggingface.co/FunAudioLLM/SenseVoiceSmall)

### Inferences
- The literature direction is consistent with our results: Whisper frozen features are the strongest upstream, and zero-shot audio LLMs trail supervised probes.

### Gaps
- No 2026 multilingual SER benchmark with Hindi scores found. Searches for EmoBox updates returned only the 2024 paper.

## Key question 3: Hindi text emotion or sentiment models (vs our e5 text probe)

### Takeaway
One Hindi text emotion model exists: vashuag/HindiEmotion (MIT). It is small and trained on translated GoEmotions, so domain mismatch with our transcripts is likely. A MuRIL sentiment model is also available, but it is sentiment, not emotion. Given that text gains were redundant with audio in our blend (text-only +0.038 but blend ≤ +0.02), expect a small effect.

### Ranked shortlist (text)

**1. vashuag/HindiEmotion** (https://huggingface.co/vashuag/HindiEmotion)
- Base: ai4bharat/indic-bert (IndicBERT). Parameters: 33.4M. Licence: MIT.
- Training data: utkarsharora100/google_go_emotions_hindi_translated (about 211k rows; train share not stated). Domain: machine-translated Reddit text, not speech transcripts.
- Labels: seven in the widget (anger, disgust, fear, joy, neutral, sadness, surprise); metrics cover six.
- Scores: main metrics block (test) accuracy 0.8085, weighted F1 0.7983, on 47 test examples. Summary block (epoch 5) accuracy 0.6997, F1 0.6750. The two blocks disagree; the test set is very small. Treat as rough.
- Use: extract the CLS or mean embedding of our Qwen3-ASR or Whisper transcripts and probe, or use its logits as features. Fits CPU easily.
- Expected value: low to moderate. Our e5 text member is already 0.739 F1 on transcripts. Qwen3 transcripts raised e5 to 0.769 alone but did not move the blend (F3 ✗). Expect redundancy.

**2. airzipm/sentiment-analysis-muril-v2** (https://huggingface.co/airzipm/sentiment-analysis-muril-v2)
- Base: google/muril-base-cased. Handles Devanagari Hindi and romanized Hinglish. Task: sentiment (negative/neutral/positive), not emotion. Self-reported accuracy ~0.86 on its test set. Licence not read.
- Value: low for four emotion classes; a sentiment polarity may partly match angry/sad vs happy/neutral. Could be a feature.

**3. Aayushi137/hindi-sentiment-muril** (https://huggingface.co/Aayushi137/hindi-sentiment-muril)
- MuRIL fine-tune, Hindi sentiment on indic_sentiment, self-reported 0.910 validation accuracy. Licence not read. Value: same as 2.

### Other text references (not on HF as ready emotion models)
- IndicBERT v2 fine-tuned on BRIGHTER for multilabel emotion: Hindi F1 87.37 (SemEval-2025 Task 11 system), reported in [aclanthology 2025.semeval-1.262](https://aclanthology.org/2025.semeval-1.262/). This is the strongest published Hindi emotion-text number found. The checkpoint was not confirmed as public on HF, so the route is to fine-tune IndicBERT v2 ourselves on BRIGHTER if the user wants it, which needs its licence check.
- Hinglish: a study notes IndicBERT saw Devanagari only, so romanized text is a gap (https://arxiv.org/pdf/2402.03137).

### Inferences
- Text is the only route where HF has a ready Hindi emotion model, but our text branch has already shown the blend ignores its gains. Only worth a run if it is cheap (CPU, <10 min).

### Gaps
- Licences of the MuRIL models and the BRIGHTER data were not verified.
- No public IndicBERT v2 emotion checkpoint confirmed on HF.

## Key question 4: Public Hindi/Indic emotional speech datasets and licences

### Takeaway
No public, permissively licensed Hindi acted-emotion speech corpus on HF was found. The closest options are not usable: a licence-restricted sample, a rule-labelled TTS set, or non-emotion corpora. BhavVaani is the only Hindi SER corpus found, and it is forbidden as a test-label source and not confirmed to be on HF.

### Cited Findings
- DataCatalyst Multilingual TTS Sample: emotion/style annotations, but only 6 utterances per language (~10 s each); DataCatalyst Evaluation License (DEL-1.0): internal evaluation only, no commercial use, no training of production models, no redistribution — [HF dataset README](https://huggingface.co/datasets/DataCatalystAI/DataCatalyst_Multilingual_TTS_Sample/blob/main/README.md?code=true). Not usable for training.
- Itsharshi/indian-tts-dataset: emotion labels come from a rule-based classifier on transcript and style, not human annotation; licence cut off in the results — [HF dataset](https://huggingface.co/datasets/Itsharshi/indian-tts-dataset). Not used.
- IndicVoices: CC BY 4.0, 23.7K hours, 22 Indian languages, 51K speakers. Natural/spontaneous speech, not emotion-labelled — [AIKosh](https://aikosh.indiaai.gov.in/home/datasets/details/indicvoices.html); [ACL 2024 Findings](https://preview.aclanthology.org/dashboard/2024.findings-acl.639). Possible for self-supervised or pretraining, not as SER labels.
- BhavVaani (Hindi SER, 8,734 utterances, 13 h, from the sitcom Sarabhai vs Sarabhai): licence not stated, release status unclear ("will be released to the community") — found via the search summary. **Flagged: this is our competition's source corpus (BhavVaani/SRCASW). It must not be used as a test-label source under the competition rules.** HANDOFF already rejected it (form-gated, 7 classes).
- Indian EmoSpeech Command Dataset (arXiv 1910.13801): 4 emotions (calm, happy, angry, fearful), licence and HF release not confirmed — [arXiv](https://arxiv.org/pdf/1910.13801).
- Rasa (Assamese, Bengali, Tamil; 6 Ekman emotions): not Hindi, licence not given — [arXiv 2407.14056](https://arxiv.org/pdf/2407.14056).
- QCRI LlamaLens-Hindi: text, cc-by-nc-sa-4.0 (non-commercial, share-alike) — [HF](https://huggingface.co/datasets/QCRI/LlamaLens-Hindi/tree/main).
- Already handled in EXPERIMENTS.md: swastikatinkhede/multilingual-speech-dataset (CC BY 4.0, Hindi 250 clips) — tested as external data and rejected (−0.009); vishlb/speech-emotion-recognition-hindi (IITKGP-SEHSC, licence-restricted) — skipped.

### Inferences
- No new Hindi acted-emotion corpus with a clear permissive licence exists on HF in this search, so external data gains look capped. The earlier external-data test (−0.009) already reflects this.

### Gaps
- Licence for Itsharshi and Indian EmoSpeech unresolved.
- IITKGP-SEHSC and MNITJ-SEHSD licences are restrictive (academic corpora); not on HF in the results.
- Did not check the HF dataset search directly for "hindi emotion" with licence filter (search tool only).

## Compliance and risk flags
- Gated items (IndicConformer Hindi checkpoint, indicwav2vec-hindi per EXPERIMENTS.md) need the user to accept terms; not done by us.
- BhavVaani: do not use any BhavVaani/SRCASW clip matched to test as a label source.
- Non-commercial items (MMS, LlamaLens-Hindi, DataCatalyst sample) are excluded from any competition submission that depends on them.

## Recommended order if time allows (deadline passed 11 Oct 00:00 IST per HANDOFF; this is for a later round)
1. STT-meta-HI encoder, layer-pooled probe (Apache-2.0, cheap, Hindi emotion-tagged encoder) — pre-register, gate ≥ +0.005 on 3 splits.
2. Confirm IndicConformer / IndicWav2Vec licences and gating on the HF pages, then decide.
3. vashuag/HindiEmotion as a text member on existing ASR transcripts (CPU, minutes).
