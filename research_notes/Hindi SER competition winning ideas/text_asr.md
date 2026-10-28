# Text-side upgrades for the Hindi SER transcript member (ASR + text classifiers)

Scope: public, free models only. Pages were read, nothing was downloaded or run. Numbers marked "unverified" were seen only in search snippets or are not stated on the page. Our baseline for context: Whisper-large-v3 ASR → multilingual-e5-large → LR gives text-only 0.739 (honest v2 CV); Qwen3-ASR transcripts give 0.769/0.752/0.757 on text alone (F3), but blend gain was −0.0013 (EXPERIMENTS.md).

## Key Question 1: Which Hindi ASR models have the best WER, and would better transcripts help emotion?

### Takeaway
Public Hindi WER numbers are mostly self-reported and use different test sets, so they do not rank models reliably. The strongest open candidate with a clean licence is vasista22/whisper-hindi-large-v2 (Apache-2.0). AI4Bharat IndicConformer is gated (HF returns 401 without login and terms acceptance), which is the same blocker that already ruled out indicwav2vec. Our own experiment suggests better transcripts will not move the blend much.

### Cited Findings
- AI4Bharat IndicConformer-600M-multilingual: model card at https://huggingface.co/ai4bharat/IndicConformer-600M-multilingual returned HTTP 401 to WebFetch, so licence, size and WER tables could not be read. Gated access is an inference from the 401, not confirmed.
- Community GGUF conversion of IndicConformer reports Hindi WER 13.5% (Kathbath test) and 15.2% (FLEURS test) — converter's own measurement, not an AI4Bharat figure — [Source](https://huggingface.co/Singla0009/IndicConformer-GGUF)
- Vistaar (2023): on the Hindi subset, average WER was 23.9 (Google STT), 20.0 (Azure), 18.6 (Nvidia large), 13.6 (IndicWhisper, the lowest); IndicWhisper had the lowest WER on 39 of 59 benchmarks overall — [Source](https://arxiv.org/pdf/2305.15386)
- vasista22/whisper-hindi-large-v2: licence Apache-2.0; base openai/whisper-large-v2; trained on GramVaani, ULCA, Shrutilipi, FLEURS train+dev. Self-reported Hindi WER: FLEURS test 6.80, Common Voice 11.0 test 10.98. No Kathbath or MUCS WER on the card — [Source](https://huggingface.co/vasista22/whisper-hindi-large-v2). Parameter count not stated; it is a large-v2 fine-tune (~1.5B, inferred).
- Qwen3-ASR-1.7B (model card): Apache-2.0; Hindi is in the 30-language list; on Fleurs† (8 extra languages incl. hi) it scores 6.62 WER and 12.80 WER on News-Multilingual (aggregates, no per-language Hindi on the card). The card lists about 2B params (BF16). Source: [Source](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)
- Qwen3-ASR Technical Report FLEURS table, Hindi row (text extraction garbled, so treat as approximate): Qwen3-ASR-0.6B ≈ 19.12%, 1.7B ≈ 17.15%, Flash-1208 ≈ 13.77% — [Source](https://arxiv.org/pdf/2601.21337). The model-card aggregate (6.62 on a subset that includes hi) and this per-language figure disagree in scale; check the PDF table before citing.
- Tara (Trelis) Hindi/Hinglish model card, vendor-reported: 12.06 mean WER over the 7 Vistaar sets, vs Sarvam Saaras-v3 12.32. Unverified, developer's own numbers — [Source](https://huggingface.co/Trelis/tara)
- Sarvam Saaras V4 / V3: closed API, no open weights; Sarvam reports LLM-WER alongside WER, so numbers are not comparable to plain WER. No Kathbath Hindi number found — [Source](https://www.sarvam.ai/blogs/introducing-saaras-v4). Excluded by the "public, free models" constraint.
- Vaani Benchmark V1.0 (Hindi, 102 districts, three transcriptions per segment): argues single-reference WER overstates errors. No system WERs in the abstract; dataset licence unclear (a CC BY 4.0 badge appears on the arXiv page, which may cover the paper only) — [Source](https://arxiv.org/abs/2606.21408)
- Hindi-emotion link: ASR-SER joint training (Interspeech 2023) reports 63.4% weighted accuracy on IEMOCAP with hierarchical co-attention over ASR hidden states and text; English data — [Source](https://ar5iv.labs.arxiv.org/html/2110.15684)

### Inferences
- Whisper-hindi-large-v2 is the most usable public Hindi-tuned ASR with a clean licence. Its FLEURS and CV numbers are likely no better than our Whisper-large-v3 + Qwen3 transcripts, so it may not change the transcript quality much. Expected gain is small.
- Our local F3 result already shows better transcripts (+0.03 text-alone) do not improve the blend. The text branch is redundant with the audio members. Another ASR is unlikely to beat that ceiling.
- IndicConformer would likely be the strongest Hindi WER model, but it is gated. Following the earlier rule (no gated models), skip it unless the user accepts the terms.

### Gaps
- No verified IndicConformer Hindi WER from AI4Bharat itself (card returned 401).
- No verified Kathbath or Vistaar WER for Whisper-large-v3 on Hindi in this search.
- Vaani system WER tables were not reached (abstract only).
- Qwen3-ASR Hindi per-language figure conflicts in scale with the model-card aggregate; needs the PDF table.

## Key Question 2: Which Hindi text embedding or classification models beat multilingual-e5 on Hindi classification?

### Takeaway
No head-to-head result against multilingual-e5-large on Hindi emotion was found. The best evidence favours Indic-specific encoders (IndicBERT v2, MuRIL) over general multilingual ones on Indic classification benchmarks, but these are mostly fine-tuning results, not frozen-embedding-plus-LR results like ours.

### Cited Findings
- IndicXNLI: MuRIL generally beat IndicBERT and mBERT on Indic languages; IndicBERT was worse than mBERT, which the authors attribute to size — [Source](https://arxiv.org/pdf/2204.08776)
- IndicXTREME (IndicBERT v2 paper): IndicBERT v2 outperforms XLM-R and MuRIL; appendix averages IndicBERT-SS 60.3, IndicBERT 57.8, MuRIL 53.3 (averages; check the table before quoting per-task numbers) — [Source](https://deepai.org/publication/indicxtreme-a-multi-task-benchmark-for-evaluating-indic-languages). The related corpus/benchmark paper is [Source](https://arxiv.org/pdf/2212.05409).
- IndicBERT v2 MLM checkpoint on HF: [Source](https://huggingface.co/ai4bharat/IndicBERTv2-MLM-only) (licence and size not read)
- SemEval-2025 Task 11 (Hindi, multilabel BRIGHTER): an IndicBERT v2 fine-tune reports F1 87.37 for Hindi; averaging method not stated in the snippet — [Source](https://aclanthology.org/2025.semeval-1.262/)
- WASSA 2021 Hindi-English code-mixed tweets: BERT accuracy 71.43% (accuracy, not macro-F1) — [Source](https://aclanthology.org/2021.wassa-1.21)
- WILDRE 2022 gold code-mixed Hindi-English tweets: SVM accuracy 75.17% beat BERT in that paper — [Source](https://aclanthology.org/2022.wildre-1.7.pdf)
- L3Cube: HindSBERT / L3Cube-MahaSBERT sentence-BERT benchmarks exist (Hindi and Marathi), evaluated on classification tasks (e.g. BBC news). No Hindi emotion dataset from L3Cube was found. The L3Cube emotion dataset found is Marathi (MahaEmotions, 11 labels, GPT-4 annotated) — [Source](https://arxiv.org/pdf/2211.11187), [Source](https://arxiv.org/abs/2506.00863)
- BGE-M3 (MIT, 1024-d, 8192-token input, "more than 100 languages"): Hindi coverage not named on the card — [Source](https://huggingface.co/BAAI/bge-m3)
- multilingual-e5-large (MIT, 24 layers, 1024-d, 100 languages per the card; low-resource languages degrade): [Source](https://huggingface.co/intfloat/multilingual-e5-large). Our current encoder, so no gain expected from it.
- Hindi emotion model page found by search, not read: [Source](https://huggingface.co/vashuag/HindiEmotion)

### Inferences
- The most plausible cheap upgrade is an IndicBERT v2 (or MuRIL) mean-pooled embedding + LR on our transcripts, replacing or averaging with e5. Evidence is indirect; the text member is already redundant with audio, so expect a small gain at best. It would need the pre-registered gate (≥ +0.005, all splits up).
- Fine-tuning IndicBERT v2 on 660 clips is likely to overfit, following our fine-tuning history.

### Gaps
- No frozen-embedding Hindi emotion benchmark comparing e5, MuRIL, IndicBERT v2 and BGE-M3.
- MuRIL and IndicBERT v2 licence and size were not read.
- IndicSentiment benchmark: no results returned.

## Key Question 3: Is an open LLM (7–9B, 4-bit) used as a few-shot or retrieval-augmented classifier on transcripts feasible and effective, with evidence on Hindi?

### Takeaway
Feasible: a 7–9B instruct model in 4-bit fits 6 GB and a Kaggle T4 handles 1,035 short transcripts in well under two hours with logprob scoring. Evidence is weak for Hindi. Retrieval-augmented few-shot did well in SemEval-2025 Task 11 (multilingual, no Hindi figure found), but a 2024–25 review says few-shot prompting often trails fine-tuning and dictionary methods on emotion in low-resource settings.

### Cited Findings
- EmoRAG (SemEval-2025 Task 11): retriever pulls labelled examples into the LLM prompt; claimed strong performance across high- and low-resource languages without training — [Source](https://preview.aclanthology.org/setup/2025.semeval-1.90) (exact Hindi score not read)
- Unibuc SemEval-2025: few-shot ICL ranked first on Emakhuwa (macro-F1 0.325), but English 0.755 and Portuguese 0.173; models over-predict fear and under-predict joy — [Source](https://arxiv.org/pdf/2508.01161v1)
- Barnes (WASSA 2023): in low-resource emotion settings, rule- and dictionary-based methods matched or beat few-shot and prompting — [Source](https://virtual2023.aclweb.org/paper_WASSA_39.html)
- Few-shot text-based emotion detection (2025): [Source](https://arxiv.org/pdf/2507.05918) (not read in full)
- Another SemEval-2025 Task 11 paper (URL not captured in the search): LoRA fine-tuning of a multilingual LLM per language is reported as the most effective approach. Unverified, so verify before citing.
- Our own evidence: transcripts are emotion-specific (label purity 0.97 across repeated transcripts), and 71 test transcripts match a train transcript exactly; English translation plus English emotion models reached text-only 0.777 but cost −0.005 in the blend (EXPERIMENTS.md).

### Inferences
- Plausible design on Kaggle GPU: Qwen2.5-7B-Instruct or similar, 4-bit (~5 GB), prompt = k retrieved train transcripts with labels (k≈10–20, bge-m3 or e5 similarity), then 4-way first-token logprobs as features into LR. Roughly 1–2 s per clip on a T4, so 1,035 clips in ~20–35 min, fitting the budget.
- Honesty requirement: retrieval must draw only from training folds, and must exclude near-duplicate twins (about 30% of clips), otherwise CV is inflated. The same goes for the test-time pool.
- Expected value: the audio branch already covers most of what a text classifier adds (see the blend history). Treat as a pre-registered text member with the repeated-CV gate; expect ≤ +0.003 at most.

### Gaps
- No Hindi-specific retrieval-augmented emotion result with numbers found.
- No check of the EmoRAG or Unibuc per-language Hindi numbers (none in scope).

## Key Question 4: Does any work fuse ASR confidence or word timing with acoustic features for SER?

### Takeaway
Searches found work on fusing ASR hidden states or N-best lists with SER, but none that uses per-word ASR confidence or word timing as SER inputs. The gap is real, not a search failure of one query.

### Cited Findings
- Fusing ASR outputs in joint training for SER (ASR hidden + text outputs with hierarchical co-attention; IEMOCAP 63.4% WA) — [Source](https://ar5iv.labs.arxiv.org/html/2110.15684)
- Post-ASR emotion with LLMs (Edinburgh, GenSEC): transcript selection metric matters (chrF, WER, length heuristics); best submission beats baseline by 20% absolute accuracy — [Source](https://www.themoonlight.io/tw/review/context-and-system-fusion-in-post-asr-emotion-recognition-with-large-language-models) (secondary summary; original paper not read)
- 2025–2026 thesis on ASR integration into SER (LLM refines N-best hypotheses, LLM confidence for pseudo-labelling): no timing or per-word confidence — [Source](https://arxiv.org/pdf/2601.17901) (abstract only read)

### Inferences
- Word timing could give pause ratio and speaking-rate features. These are cheap to compute from a Whisper word-timestamp pass and are an acoustic-side idea rather than a text one. The local plan already lists a prosody member as optional.
- ASR confidence (mean token logprob) is free from Whisper generation and could serve as a weight or gate on the text member for low-confidence transcripts. Untested.

### Gaps
- No paper found that uses word-level ASR confidence or timing in an LLM-based SER system.

## Recommended order (cheap to expensive, all public)
1. Check licences and sizes of IndicBERT v2 and MuRIL (HF pages not read). Then embed transcripts with them and fit LR as a new text member. Gate: pre-registered, repeated CV, ≥ +0.005 mean and > 0 on every split.
2. Whisper word-timestamp pass plus Whisper mean token logprob on the 1,035 clips (CPU or local GPU, minutes). Add pause ratio and speaking rate, and use token logprob as a text-member weight. Same gate.
3. LLM retrieval few-shot on Kaggle (Qwen2.5-7B-Instruct 4-bit or similar): retrieval from training folds only, twin exclusion, 4-way logprobs into LR. Same gate.
4. Whisper-hindi-large-v2 as a second transcript: low expected value given F3. Only if cheap.
5. Skip: IndicConformer (gated, needs terms acceptance), Saaras and Scribe (closed APIs), L3Cube Hindi emotion (no such dataset found).

## Source notes
- Licences for IndicConformer, IndicBERT v2, MuRIL, bge-m3 model size, and Vaani dataset are not confirmed.
- Vendor WER figures (Tara, Sarvam, ElevenLabs Scribe) are not independently verified.
