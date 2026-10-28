# Presentation speech: Hearing emotion in Hindi speech

13 slides · 756 words ≈ 6 min at a calm pace. Same text is in the deck's speaker notes.

## Slide 1 · Hearing emotion in Hindi speech  (0:40)

Good morning. I'm Ayush Kasare, presenting our 6th-place solution: a private macro-F1 of 0.943 among 51 teams.

Consider one sentence: "Main theek hoon." Delivered four ways, it is anger, happiness, sadness or neutrality. The words are identical; the emotion lives entirely in prosody and timing. Add Hindi, clips under three seconds and only 826 labelled examples, and the real risk is not underfitting. It is fooling yourself. [pause] So our thesis: the strongest solution is the one whose validation you can trust.

## Slide 2 · The solution in one slide  (0:35)

Four ideas define the solution. We diagnosed a leakage structure before training any model. We used large speech foundation models as frozen feature extractors with lightweight, regularised heads, because at this scale that beats fine-tuning. We built a six-member ensemble selected for error diversity, not individual accuracy. And we turned the dataset's duplication into a validated, rule-compliant post-processing step. Of roughly fifty pre-registered hypotheses, four were adopted.

## Slide 3 · The hidden structure: one in three clips is a copy  (0:40)

Exploratory analysis revealed many clips with identical durations: the same recordings re-encoded with added noise. Only 12 percent were byte-identical, so we detected the rest with duration matching plus cosine similarity in Whisper's embedding space.

About 30 percent of clips have a twin, 826 clips collapse into 689 recordings, and twins share their label 99.1 percent of the time. With roughly 125 private test clips, where one prediction is worth almost a point, that structure decides everything.

## Slide 4 · Honest validation: grouped folds or nothing  (0:35)

Under random K-fold, a recording's twin leaks into validation and the model is rewarded for memorisation. The same model scores 0.930 under random folds and 0.891 under StratifiedGroupKFold, where each near-duplicate group stays in one fold: four points of pure leakage. Our first public score, 0.917, confirmed the grouped estimate. Every number from here on is leakage-free.

## Slide 5 · Decision protocol: pre-register, repeat, then adopt  (0:30)

Grouped folds remove leakage, but not variance: CV moves about a point between reshuffles. So, much like a clinical trial, every experiment was specified before scoring, evaluated on three independent grouped splits, and adopted only if the mean gain exceeded 0.005 and was positive on every split. Four of fifty passed.

## Slide 6 · Architecture: six views of every clip  (0:45)

The result is a six-member ensemble. Whisper large-v3 is the acoustic backbone. w2v-BERT 2.0 adds a multilingual view from its mid-network layers. Qwen2.5-Omni, a seven-billion-parameter audio LLM, contributes both hidden states and zero-shot label probabilities. A text member embeds Hindi transcripts, because repeated sentences keep their emotion 97 percent of the time. And a fusion head combines audio and text.

We average them with fixed weights: every learned combiner we tested overfit 826 samples. Here, simplicity is a statistical choice.

## Slide 7 · Member 1: Whisper-large-v3 encoder, the backbone  (0:30)

Whisper's frozen encoder gives layers 24 and 32, pooled over valid frames, into a regularised probe. A non-obvious finding: Whisper expects 30-second windows, and short unpadded clips degrade its top layers from 0.916 to 0.833. Silence padding restores them. Augmented training views take this member to 0.902 alone.

## Slide 8 · Member 6: attentive audio-text fusion head  (0:30)

Our most significant modelling contribution addresses a simple fact: emotional cues are temporally sparse, and mean pooling dilutes them. This head uses four-head attentive statistics pooling over Whisper frames, fused early with the transcript embedding. It became the strongest single model, at 0.916.

## Slide 9 · Twin lock: turn the duplicate problem into a feature  (0:30)

The duplication that threatened validation becomes an asset at inference. 78 test clips have a verified training twin, so they inherit its label, given 99.1 percent agreement in training. It uses only competition training labels, never external labels of test records, and was validated on training data first.

## Slide 10 · How the score was built, step by step  (0:35)

The progression, all leakage-free: a two-encoder baseline at 0.891; the audio LLM and text member, 0.911; multi-layer Whisper, 0.916; augmentation, 0.923; the fusion head, 0.930. Each step pre-registered and confirmed on three splits. The public board plateaued at 0.942: with 90 clips it could not resolve these gains.

## Slide 11 · What did not work: about 50 ideas  (0:35)

Negative results matter as much. Fine-tuning Whisper on 660 clips degraded performance. English-trained emotion models did not transfer to Hindi. Zero-shot audio LLMs trailed supervised probes. And the central lesson: Gemma-4's audio encoder scored 0.855 alone yet weakened the ensemble, because it encoded what Whisper already knew. Diversity dominates strength.

## Slide 12 · Results: 6th of 51 on the private leaderboard  (0:30)

The outcome: 0.943 private, 6th of 51. Candidly, a candidate we dropped for its public score reached 0.952 privately. With 90 and 125 clips, the leaderboards disagree by a few predictions either way, and first place is about two clips away: exactly the variance our protocol was built around.

## Slide 13 · Takeaways  (0:35)

Back to "Main theek hoon." Telling those four deliveries apart did not require the largest model. It required diagnosing the data, validating without leakage, selecting for diversity, and letting pre-registered evidence make every decision. [pause] Thank you. The code and full experiment log, including every failed idea, are public on GitHub. I'm happy to take questions.

---

# Q&A preparation

**Q: Why did you finish 6th and not 1st?**

First place is about 0.017 above us, roughly two private clips. At that scale, ranks 1 to 6 are within the noise of 125 clips; our own submissions disagreed by more than that. We optimised for a score that generalises, judged by honest grouped validation, not for the public leaderboard.

**Q: Isn't the twin lock cheating?**

No. It uses only the competition's own training labels and no external labels of test clips, which is what the rules forbid. It is a post-processing rule that we validated on training data (99.1% agreement) before using it. We are transparent that it exists, and we report our honest validation separately from it.

**Q: Why not fine-tune the models? Isn't that standard?**

We tried. With 660 clips per fold, fine-tuning Whisper scored 0.55, and even careful partial fine-tuning fell below the frozen model it started from. The frozen model plus a small classifier already fits the data; fine-tuning only distorts it. With thousands of clips, fine-tuning would be the right call.

**Q: How do you know your improvements aren't just luck?**

Every idea was written down before it was scored, tested on three different grouped splits, and adopted only if the average gain was at least 0.005 and positive on every split. About 50 ideas went in and 4 came out. With that many tries, even the adopted gains may be slightly optimistic, which is why we preferred few, clear changes.

**Q: Would this work on new speakers in the real world?**

Honestly, we don't know yet, and that is our first next step. The original research corpus reports about 0.45 accuracy when test speakers are completely unseen. The competition split shares recordings between train and test, which our twin lock uses, so real-world performance would be lower.

**Q: Why fixed blend weights instead of learning them?**

We tried learning them: a stacker lost between 0.005 and 0.021, per-class bias tuning lost 0.011. With 826 clips, a learned combiner fits noise in the out-of-fold predictions. Fixed weights are slightly less optimal in theory and much more robust in practice.

**Q: Did you use external data?**

We tested a public, CC-BY Indian emotion dataset after checking that none of its clips matched competition clips. It lowered validation by 0.009 because of domain mismatch, so it was not used. No external labels of test records were ever used.

**Q: How much compute did this need?**

A laptop RTX 4050 with 6 GB, plus about three GPU-hours on Kaggle's free T4s for the 7-billion-parameter audio models. Caching features once is what kept it cheap.

**Q: What would you do differently?**

Two things. Choose the second final submission for diversity rather than public score: the dropped one scored 0.952 private. And measure speaker-independent performance earlier, since that is what matters outside the competition.
