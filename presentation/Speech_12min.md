# 12-minute speech: Hearing emotion in Hindi speech

18 slides · 1339 words ≈ 10 min at a calm pace, leaving ~2 min of buffer. Same text is in the deck's speaker notes.

## Slide 1 · Hearing emotion in Hindi speech  (0:50)

Good morning. I'm Ayush Kasare, and this is our 6th-place solution, with a private macro-F1 of 0.943 out of 51 teams.

Let me start with one sentence: "Main theek hoon." Delivered four ways, it is anger, happiness, sadness or neutrality. The lexical content is identical; the emotion lives entirely in prosody, timbre and timing. Here that came with three constraints: Hindi, clips under three seconds, and only 826 labelled examples. [pause]

In that regime, the risk is not underfitting. It is fooling yourself. So the thesis of this talk is simple: the strongest solution is the one whose validation you can trust, and every design decision we made follows from that.

## Slide 2 · The solution in one slide  (0:40)

Four ideas define the solution.

First, we diagnosed a leakage structure in the data before training a single model. Second, we treated large speech foundation models as frozen feature extractors and trained lightweight, well-regularised heads on top, because at this sample size that beats fine-tuning. Third, we built a six-member ensemble selected for error diversity, not individual accuracy. And fourth, we converted the dataset's duplication into a validated, rule-compliant post-processing step. Of roughly fifty pre-registered hypotheses, four were adopted.

## Slide 3 · Data at a glance  (0:30)

826 training clips and 209 test clips, median duration 2.9 seconds, with four near-balanced classes, so macro-F1 is a fair metric.

The number that shaped our strategy is 0.008. The private leaderboard holds roughly 125 clips, so a single prediction moves the score by almost a full point. At this variance, leaderboard feedback is mostly noise.

## Slide 4 · The hidden structure: one in three clips is a copy  (0:45)

The first real finding came from exploratory analysis. A large number of clips shared identical durations. On inspection, they were the same recordings re-encoded with added noise, at a residual SNR of 18 to 30 dB.

Only 12 percent were byte-identical, so hashing alone misses them. We detected near-duplicates by combining duration matching with cosine similarity in Whisper's embedding space. The result: about 30 percent of clips have a twin, the 826 clips collapse into 689 independent recordings, and twins share their label 99.1 percent of the time.

## Slide 5 · Honest validation: grouped folds or nothing  (0:50)

Here is why it matters. Under random K-fold, one copy of a recording sits in training while its near-duplicate sits in validation, so the model is rewarded for memorisation, not generalisation.

The same model scores 0.930 under random folds and 0.891 under StratifiedGroupKFold, where every near-duplicate group stays within one fold. That is roughly four points of optimism caused purely by leakage. Our first public score, 0.917, confirmed the grouped estimate, not the inflated one. Every number from here on is leakage-free.

## Slide 6 · Decision protocol: pre-register, repeat, then adopt  (0:40)

Honest folds solve leakage, but not variance. Our blend's CV moves by about one point simply by re-shuffling the folds, so a single-split gain of half a point is indistinguishable from noise.

We therefore adopted a protocol closer to clinical trials than to typical Kaggle practice: every experiment was specified before it was scored, evaluated on three independent grouped splits with all members retrained, and accepted only if the mean gain exceeded 0.005 and was positive on every split. Of roughly fifty candidates, four passed.

## Slide 7 · Architecture: six views of every clip  (0:35)

The final model is a six-member ensemble, each member observing the clip from a different angle. Four operate on the acoustic signal, one on the transcribed text, and one fuses audio and text at the representation level.

Each member outputs class probabilities; we combine them with fixed weights specified in advance, shown here as effective weights. A twin-lock step then resolves verified duplicates.

## Slide 8 · Member 1: Whisper-large-v3 encoder, the backbone  (0:50)

The backbone is the Whisper large-v3 encoder, kept frozen. We extract layers 24 and 32, mean- and standard-deviation-pooled over valid frames, and fit a regularised logistic-regression probe.

One non-obvious finding: Whisper's encoder is trained on 30-second windows. Feeding it short, unpadded clips degrades its top layers from 0.916 to 0.833. Restoring trailing-silence padding recovers the full representation quality. Training on four augmented views per clip, with noise and leading silence, takes it to 0.902 standalone.

## Slide 9 · Members 2–4: deliberately different second opinions  (0:45)

Members two to four were chosen to be different, not just strong. w2v-BERT 2.0 is a multilingual self-supervised encoder with Hindi in pretraining; layer sweeps showed its emotional information peaks mid-network, so we use layers 10 and 16. Qwen2.5-Omni, a seven-billion-parameter audio language model, contributes twice: through its internal hidden states, and through zero-shot label probabilities from a constrained prompt.

A revealing ablation: augmenting w2v-BERT improved its standalone score but reduced the ensemble, because its errors became correlated with Whisper's. Ensemble value comes from decorrelated errors, not individual accuracy.

## Slide 10 · Member 5: what is said, not just how  (0:30)

The fifth member is linguistic. Whisper produces Hindi transcripts, multilingual-e5 embeds them, and a linear classifier predicts emotion from meaning alone.

This was a data-driven choice: sentences that recur in the corpus carry the same emotion 97 percent of the time, so lexical content is genuinely informative here. Alone it scores 0.74, but it is weakest precisely where acoustics are strongest, which is exactly what an ensemble needs.

## Slide 11 · Member 6: attentive audio-text fusion head  (0:45)

The sixth member was our most significant modelling contribution. Emotional cues are temporally sparse: a pitch excursion, a stressed syllable. Mean pooling dilutes them across the whole clip.

So we trained an attentive statistics pooling head on frame-level Whisper features, four attention heads learning which frames matter, and fused it early with the transcript embedding before the classification layer. It became the strongest single model in the solution at 0.916.

## Slide 12 · The blend: fixed weights, chosen for robustness  (0:30)

For combination, we deliberately chose fixed-weight probability averaging. A learned stacker lost up to two points; bias tuning, voting and calibration added nothing.

With 826 samples, any learned combiner overfits the out-of-fold predictions. A fixed average carries no such variance. Here, simplicity is a statistical choice, not a shortcut.

## Slide 13 · Twin lock: turn the duplicate problem into a feature  (0:40)

The duplication that threatened our validation becomes an asset at inference. 78 of the 209 test clips have a verified training twin, 69 matched acoustically and 9 by identical transcripts. Given 99.1 percent label agreement among twins in training, those test clips inherit their twin's label.

Crucially, this uses only competition training labels and no external labels of test records, and the rule was validated on training data first.

## Slide 14 · How the score was built, step by step  (0:45)

This is the progression, all in leakage-free cross-validation. A two-encoder baseline at 0.891. Adding the audio language model and the text member: 0.911. Multi-layer Whisper features: 0.916. Augmentation and refined w2v-BERT layers: 0.923. The fusion head: 0.930.

Four steps, each pre-registered and confirmed across three splits. Note the public leaderboard plateaued at 0.942: with about 90 clips it lacked the resolution to detect these improvements.

## Slide 15 · Where the model still fails  (0:30)

A strong solution should also be clear about its failure modes. Neutral is the hardest class, with an F1 of 0.886, confused mainly with happy and sad, which is consistent with neutral occupying the centre of the arousal-valence space. And clips under two seconds show a 14 percent error rate against 4 percent for three-to-four-second clips: below a certain duration, there is simply too little prosodic evidence.

## Slide 16 · What did not work: about 50 ideas  (0:45)

The negative results are as important as the positive ones. Fine-tuning Whisper on 660 clips degraded performance; even linear-probe-then-fine-tune fell below its own initialisation. Emotion-specific models pretrained on English did not transfer to Hindi. Zero-shot audio LLMs trailed supervised probes by a wide margin.

The central lesson: Google's Gemma-4 audio encoder scored 0.855 on its own, yet reduced the ensemble, because it encoded the same information as Whisper. In ensemble design, diversity dominates strength.

## Slide 17 · Results: 6th of 51 on the private leaderboard  (0:45)

The outcome: 0.943 private macro-F1, 6th of 51 teams.

One candid observation: a candidate we dropped for its lower public score reached 0.952 privately. With 90 public and 125 private clips, the two leaderboards disagree by a few predictions in both directions; the gap to first place is roughly two clips. That is exactly the variance our protocol was designed around.

## Slide 18 · Takeaways  (0:40)

To close, back to "Main theek hoon." Teaching a machine to tell those four deliveries apart did not require the largest model. It required diagnosing the data, validating without leakage, building an ensemble for diversity, and letting pre-registered evidence, not leaderboard noise, make every decision.

Four principles: validate honestly, use foundation models as frozen experts at small scale, select for diversity, and decide before you measure. [pause] Thank you. The full code and experiment log, including every failed idea, are public on GitHub. I'm happy to take questions.

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
