# Presentation speech: Hearing emotion in Hindi speech

Team Ayush Kasare · DataVerse 2026 · about 15 minutes at a calm pace. The same text is in the deck's speaker notes (Presenter View).

**Delivery cues:** [pause] = stop for one breath. Speak to the room, not the slide. On the twin-lock and results slides, slow down: those are where questions come from.

## Slide 1 · Hearing emotion in Hindi speech  (0:40)

Good morning. My name is Ayush Kasare. Before I show you a single number, I'd like you to imagine one sentence.

"Main theek hoon." I'm fine.

Say it through clenched teeth, and it's anger. Say it with a smile, and it's happiness. Say it slowly, looking down, and it's sadness. Say it flat, and it's neutral. Same words, four different emotions. [pause]

That was the challenge in this competition: teach a machine to hear that difference, in Hindi, from clips that are often under three seconds long.

Our solution finished 6th out of 51 teams, with a private macro-F1 of 0.943. But today I want to convince you of something more important than a rank: that this is a solution whose number you can trust.

## Slide 2 · The solution in one slide  (0:45)

Here is the whole story in four lines. Everything after this slide just opens them up.

One: we found a leak in the data before we built any model.
Two: instead of training giant models, we used powerful pre-trained speech models as they are, frozen, and trained small, careful classifiers on top of them.
Three: we combined several models, and we chose them for diversity, not for individual strength.
And four: we turned the data's biggest flaw into an advantage, using only the rules of the competition.

We tested about fifty ideas this way. Only four earned a place in the final model.

## Slide 3 · 1 · The data and its hidden structure  (0:05)

Every good data science story starts the same way: by looking at the data properly.

## Slide 4 · Data at a glance  (0:40)

We had 826 training clips and 209 test clips. The median clip is under three seconds, barely one sentence. The four emotions are balanced, so macro-F1 treats every class fairly.

But I want you to remember one number on this slide: 0.008. The private leaderboard has only about 125 clips, so one single clip moves the score by almost a full point. In a competition this small, luck is very loud. Our entire method is designed to stop luck from fooling us.

## Slide 5 · The hidden structure: one in three clips is a copy  (0:55)

On the first night, I noticed something strange: many clips had exactly the same length. When I compared them, they were the same recording, saved again with a little noise added.

Only 12 percent were exact, byte-for-byte copies. The rest you can only catch by comparing how they sound, which we did using Whisper's internal representation.

The result: about one in three clips has a twin. Our 826 clips are really only 689 different recordings. And twins carry the same label 99 percent of the time.

This one discovery shaped everything that follows. It is a trap, and it is also a gift. Let me show you the trap first.

## Slide 6 · Honest validation: grouped folds or nothing  (1:00)

If you split this data randomly for validation, one copy of a recording lands in training and its twin lands in the validation set. The model is not predicting. It is remembering.

Look at the bars. With random splits, our first model scored 0.930. With grouped splits, where all copies of a recording always stay on the same side, the same model scores 0.891.

That gap, 0.04, is fake skill.

And here is the proof: our first submission scored 0.917 on the public leaderboard. That matches the honest number, not the flattering one. So from this slide onward, every number I show you is the honest one.

## Slide 7 · Decision protocol: pre-register, repeat, then adopt  (0:50)

Honest data needs honest decisions. With so few clips, the validation score itself moves by about one point just from how you shuffle the data. So a small improvement can be pure luck.

We borrowed a rule from clinical trials. Step one: write down the experiment before you see its result. Step two: test it on three different splits of the data. Step three: adopt it only if it improves the average by at least half a point and never makes any split worse.

It is strict. About fifty ideas went in, and four came out. But it means every improvement you see in this talk is real.

## Slide 8 · 2 · The model  (0:05)

So what did we actually build?

## Slide 9 · Architecture: six views of every clip  (0:45)

Think of our model as a panel of six judges listening to every clip.

Four judges listen to the sound: the tone, the pitch, the energy of the voice. One judge reads the transcript, the actual words. And one judge does both at once.

Each judge gives a probability for each emotion. We average them with fixed weights that we decided in advance, which you can see as percentages on the slide. Finally, a last safety check, the twin lock, which I'll explain shortly.

## Slide 10 · Member 1: Whisper-large-v3 encoder, the backbone  (0:55)

Our lead judge is Whisper large-v3, a speech model trained on millions of hours of audio. We don't retrain it. We freeze it and read the information inside it.

One discovery here surprised me. Whisper was built to listen to 30-second windows. If you give it a three-second clip without padding, its deepest layers get confused, and accuracy drops from 0.916 to 0.833. Simply padding the clip with silence brings it back.

We also show the model every training clip five ways: clean, with light background noise, and with a short pause at the start. That teaches it to recognise emotion, not recording quirks. On its own, this judge scores 0.902.

## Slide 11 · Where emotion lives inside each encoder  (0:40)

Where inside a neural network does emotion live? We measured it, layer by layer.

In Whisper, deeper is better, all the way to the last layer. In w2v-BERT, emotion peaks in the middle of the network and collapses at the top, because the top layers specialise in something else.

Most people simply take the last layer. For w2v-BERT, that would have thrown away a quarter of its accuracy. Five minutes of measurement made a big difference.

## Slide 12 · Members 2–4: deliberately different second opinions  (0:55)

The next three judges are deliberately different from Whisper.

w2v-BERT was trained on many languages, including Hindi. Qwen2.5-Omni is a seven-billion-parameter model that understands audio. We use it in two ways: we read its internal state, and we simply ask it, "which emotion is this?"

Here is a counter-intuitive lesson. When we made w2v-BERT stronger with the same augmentation, the whole panel got worse. Why? Because it started making the same mistakes as Whisper. A judge who always agrees with the others adds nothing.

## Slide 13 · Member 5: what is said, not just how  (0:40)

Our fifth judge never hears the voice. It reads the words.

Whisper writes down the Hindi sentence, and a multilingual language model turns that sentence into meaning. Why would words help? In this data, when a sentence repeats, it almost always carries the same emotion: 97 percent of the time.

On its own this judge only scores 0.74. But inside the panel, it catches exactly the cases where the voice is ambiguous.

## Slide 14 · Member 6: attentive audio-text fusion head  (0:50)

The sixth judge was the breakthrough.

Emotion is not spread evenly across a clip. It lives in a few moments: a rising pitch, a sharp word, a break in the voice. If you average the whole clip, those moments get diluted.

So this small network learns where to pay attention in the clip, and it combines that with the meaning of the transcript. It became our best single model, at 0.916. Added to the panel as an extra voice, it gave us the final lift.

## Slide 15 · The blend: fixed weights, chosen for robustness  (0:35)

How do you combine six judges? We tried the clever ways: a model that learns the weights, per-class tuning, calibration, majority voting. Every single one did worse.

With only 826 clips, any combiner that learns from the data ends up fitting noise. A fixed, simple average won. In small data, simplicity is not laziness. It's robustness.

## Slide 16 · Twin lock: turn the duplicate problem into a feature  (0:45)

Now, the gift I promised.

78 of the 209 test clips have a twin in the training data: either the same recording, or exactly the same sentence. Since twins agree 99 percent of the time, a test clip takes the label of its training twin.

This is important: it uses only the competition's own training labels. Nothing external, no outside labels of test clips. And we checked the rule on the training data before we ever used it.

## Slide 17 · Final build: refit every member on all 826 clips  (0:30)

One last step before submitting. During validation, each model only trains on 660 clips. But look at this learning curve: it is still climbing. More data still helps.

So for the final predictions, we retrained every judge on all 826 clips.

## Slide 18 · 3 · Evidence and results  (0:05)

Now: does it actually work, and where does it fail?

## Slide 19 · How the score was built, step by step  (0:50)

This is the journey, in honest numbers.

We started at 0.891. Adding the audio language model and the transcript took us to 0.911. Using two Whisper layers instead of one: 0.916. Augmentation: 0.923. And the attention judge: 0.930.

Four steps, each one written down in advance and confirmed on three different splits.

Notice the table on the right: the public leaderboard stopped moving at 0.942. With only 90 clips, it simply could not see these improvements. That is exactly why we trusted our validation, not the leaderboard.

## Slide 20 · Where the model still fails  (0:40)

A good data scientist also shows where the model fails.

Neutral is the hardest emotion. "Calm" sits right between happy and sad, and that's where most confusions are. And very short clips, under two seconds, fail four times more often than three-second clips. There simply isn't enough voice to judge.

These two pockets are where the next improvements will come from.

## Slide 21 · What did not work: about 50 ideas  (0:50)

And this is what didn't work. I think this slide matters as much as the successes.

Fine-tuning large models on 660 clips: worse. Emotion models trained on English: they didn't transfer to Hindi. Asking audio chatbots directly: far below our trained judges. Pseudo-labels, stacking, speaker normalisation: no reliable gain.

Every one was tested under the same strict rule. So when I say this solution is close to the ceiling of what this data allows, that is not a guess. It's fifty measurements.

## Slide 22 · The ceiling: a stronger member is not a better blend  (0:40)

The biggest lesson from those failures: a stronger model is not always a better team member.

Google's Gemma-4 audio encoder scored 0.855 on its own, better than two of our judges, and it still made the panel worse, because it hears the same things Whisper hears. The only addition that helped was the one that brought different information: the words, fused with the voice.

Diversity beats strength.

## Slide 23 · Results: 6th of 51 on the private leaderboard  (0:55)

The result: 0.943 on the private leaderboard, 6th out of 51 teams.

Let me be candid, because this slide teaches something. Look at the last pair of bars. One model that I dropped because its public score was lower scored 0.952 on the private leaderboard. Public and private disagreed by a few clips in both directions, and with 90 and 125 clips, that is simply noise.

The gap to first place is about two private clips. What I can promise is that every one of our points was earned honestly, which is the whole philosophy of this talk: trust honest validation over a small leaderboard.

## Slide 24 · Engineering: small hardware, careful caching  (0:35)

All of this ran on a laptop GPU with 6 GB of memory, plus free Kaggle GPUs for the large audio models: about three GPU-hours in total.

The trick is caching. We extract each model's features once, so testing a new idea takes minutes instead of hours. That's what made fifty careful experiments possible in about a week. And we caught several subtle bugs along the way, precisely because we measured everything.

## Slide 25 · With more time and data  (0:30)

With more time, four things. First, test on speakers the model has never heard: that is the real-world test. Second, target short clips and the neutral class. Third, add prosody features like pauses and speaking rate. And fourth, distil our six judges into one fast model that can actually be deployed.

## Slide 26 · Takeaways  (0:45)

So, back to "Main theek hoon." One sentence, four emotions.

To teach a machine the difference, we didn't need the biggest model. We needed to be honest about our data, to listen through the right layers, to build a diverse panel of judges, and to trust measurement over luck.

Validate honestly. Use foundation models wisely. Choose diversity. Decide in advance.

Thank you. All the code, and every experiment, including the failures, is public on GitHub. I'd be happy to take your questions.

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
