# Our winning solution, explained step by step (with code)

DataVerse 2026: Detecting Emotions from Hindi Speech · Team Ayush Kasare · 6th of 51 · private macro-F1 0.9434

This guide walks through the exact pipeline that produced our final submissions, in the order the data flows. Every step shows the **real code** from `src/`, the **data-science term** for what it does, and what that term means in plain words. A glossary at the end collects every term.

---

## 0. The big picture

```
 826 train + 209 test WAV clips
        │
        ▼
 [1] Find near-duplicates ──► groups of copies (689 groups)
        │
        ▼
 [2] Honest validation: StratifiedGroupKFold (copies never split across folds)
        │
        ▼
 [3] Frozen foundation models ──► features (numbers describing each clip)
        │   Whisper-large-v3 · w2v-BERT 2.0 · Qwen2.5-Omni-7B · Whisper ASR → e5 text
        ▼
 [4] Small classifiers ("probes") on each feature set ──► 4 probabilities per clip
        │   + one small neural network: the attentive audio-text fusion head
        ▼
 [5] Blend: fixed-weight average of the 6 members' probabilities
        │
        ▼
 [6] Twin lock: test clips with a training twin take the twin's label
        │
        ▼
 [7] Refit everything on all 826 clips ──► submission.csv
```

Running through all of it is a **decision protocol**: every idea was written down before testing, tested on 3 different splits, and kept only if it clearly helped.

---

## 1. The problem and the metric

**Task:** each audio clip must be labelled as one of 4 emotions: angry, happy, neutral or sad. That makes it a **multi-class classification** problem, meaning we predict one category out of several.

**Metric: macro-F1.** Three definitions first:

- **Precision** for a class is the share of clips we called "angry" that really were angry. It measures how often we're right when we say angry.
- **Recall** for a class is the share of truly angry clips that we found.
- **F1** combines the two as their **harmonic mean**, a kind of average that punishes imbalance: F1 = 2·P·R / (P + R). It's high only when both precision and recall are high.

**Macro-F1** computes F1 for each of the 4 classes separately, then takes the plain average, so every emotion counts equally.

**Public and private leaderboards.** Kaggle splits the 209 test clips into a **public** part (about 90 clips, whose score you see during the competition) and a **private** part (about 125 clips, revealed only at the end). The final ranking here used 0.4 × public + 0.6 × private. With about 125 private clips, **one clip ≈ 0.008 F1**, so tiny differences are mostly luck.

---

## 2. Step 1: discovering the hidden structure (near-duplicates)

**What we found:** about 30% of clips are the *same recording* saved again with some added noise. Only 12% are byte-identical, so a file **hash** misses most of them. (A hash is a fingerprint of a file's bytes: identical files give identical hashes.)

**How we found them**, in three ideas:

- **Same duration:** copies have exactly the same length.
- **Embedding similarity:** an **embedding** is a vector (a list of numbers) that a neural network produces to describe an input. Similar sounds give similar vectors.
- **Cosine similarity** measures how closely two vectors point in the same direction: 1 means identical direction, 0 means unrelated. We used Whisper's layer-32 embedding, and treated cosine > 0.9 together with the same duration as "same recording".

`src/hcv.py`:

```python
def _groups():
    W = np.load('feats/whisper.npy', mmap_mode='r')[:n, 32].astype(np.float32)   # Whisper layer-32 embeddings, train clips
    W = (W - W.mean(0)) / (W.std(0) + 1e-6)                                         # standardise each feature
    W /= np.linalg.norm(W, axis=1, keepdims=True)                                   # unit length → dot product = cosine
    S = W @ W.T                                                                     # cosine similarity of every pair
    same_len = np.abs(tr.dur.values[:, None] - tr.dur.values[None, :]) < 1e-3
    A = ((S > 0.9) & same_len) | (tr.h.values[:, None] == tr.h.values[None, :])     # near-copy OR identical hash
    parent = list(range(n))                                                         # union-find: merge linked clips into groups
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for i, j in zip(*np.nonzero(np.triu(A, 1))):
        parent[find(i)] = find(j)
    return np.array([find(i) for i in range(n)])
```

**Union-find**, also called **connected components**, means that if A is a copy of B and B is a copy of C, all three end up in one **group**. Result: 826 clips → **689 groups**. Copies share the same label **99.1%** of the time.

---

## 3. Step 2: honest validation (the most important decision)

**Validation** means estimating how well the model will do on unseen data, using only the training data.

**K-fold cross-validation (CV):** split the training data into K parts called **folds** (we use 5). Train on 4 folds, predict the 5th, and rotate until every clip has been predicted once by a model that never saw it. Those predictions are **out-of-fold (OOF)** predictions, and scoring them gives the **CV score**.

**Data leakage:** information from the evaluation data sneaking into training, which makes scores look better than they really are. Here, if one copy of a recording is in training and its twin is in validation, the model just *remembers* it.

**The fix is StratifiedGroupKFold.** "Group" means every group of copies stays entirely inside one fold. "Stratified" means each fold keeps roughly the same class balance.

`src/hcv.py`:

```python
g = _groups()
f = np.zeros(n, int)
for k, (_, v) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(tr, y, g)):
    f[v] = k                                   # f[i] = which fold clip i belongs to
```

**Effect:** the same first model scored **0.930** with random folds and **0.891** with grouped folds, so about 0.04 was fake skill. Our first public score (0.917) matched the honest number. From then on, every decision used grouped CV.

---

## 4. Step 3: frozen foundation models turn audio into features

**Foundation model:** a very large neural network pre-trained on massive data (Whisper: hundreds of thousands of hours of speech), whose knowledge can be reused for new tasks.

**Encoder:** the part of a model that turns an input (audio) into internal representations.

**Frozen:** we do **not** change the model's weights. We only read its outputs. With just 826 clips, re-training (**fine-tuning**) a billion-parameter model overfits. **Overfitting** means memorising training examples instead of learning general patterns.

**Hidden states / layers:** an encoder is a stack of layers (Whisper-large-v3 has 32). Each layer outputs one vector per **frame**, a short time slice (20 ms for Whisper, so 50 frames per second). Different layers capture different things; deeper layers are more abstract.

**Pooling:** a clip has a different number of frames depending on its length, but a classifier needs one fixed-size vector. **Mean pooling** averages over time; adding the **standard deviation** (how much the values vary) keeps information about variation, such as how much the pitch moves. So for each layer we store `[mean, std]`.

`src/extract.py`:

```python
@torch.inference_mode()                                          # no training, just reading the model
def embed(x):                                                    # x = raw waveform, 16 kHz
    inp = fe(x, sampling_rate=16000, return_tensors='pt').input_features   # log-Mel spectrogram, padded to 30 s
    hs = model(inp, output_hidden_states=True).hidden_states                # 33 tensors: input embedding + 32 layers
    n = min(hs[0].shape[1], int(np.ceil(len(x) / 320)))         # number of REAL frames (50 per s); ignore padding
    h = torch.stack(hs)[:, 0, :n].float()                       # [layers, frames, 1280]
    return torch.cat([h.mean(1), h.std(1)], -1)                 # [layers, 2560] = mean + std per layer
```

- **Log-Mel spectrogram:** the audio turned into an image-like grid of energy per frequency band over time, on a scale similar to human hearing. It's Whisper's input format.
- **Padding:** Whisper always receives a 30-second window, so short clips are padded with silence. **Key finding:** Whisper *needs* that silence padding. Without it, its top layers dropped from 0.913 to 0.833, because they were trained on 30-s windows. We pad, but pool only over the real frames (`n`).

The output is `feats/whisper.npy` with shape `[1035 clips, 33 layers, 2560]`, stored once (**caching**) so every later experiment costs seconds instead of GPU-hours.

---

## 5. Step 4: probes, small classifiers on frozen features

**Probe (linear probe):** a simple linear classifier trained on frozen features. If a probe can classify emotion well, the emotion information is already in the features.

`src/hcv.py`:

```python
def probe(X, Cs=(0.003, 0.01, 0.03, 0.1, 0.3)):
    best = None
    for C in Cs:                                                     # try several regularisation strengths
        oof = np.zeros((n, 4)); test = np.zeros((len(X) - n, 4))
        for k in range(5):                                           # 5 grouped folds
            m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=4000))
            m.fit(X[:n][fold != k], y[fold != k])                    # train on 4 folds
            oof[fold == k] = m.predict_proba(X[:n][fold == k])       # predict the held-out fold
            test += m.predict_proba(X[n:]) / 5                       # test = average of the 5 fold models
        ll = log_loss(y, oof)
        if best is None or ll < best[0]:
            best = (ll, C, oof, test)
    return best[1:]                                                  # (chosen C, OOF probs, test probs)
```

- **StandardScaler (standardisation):** rescales every feature to mean 0 and standard deviation 1, so no feature dominates just because its numbers are bigger.
- **Logistic regression:** a linear model that outputs a **probability** for each class (via the **softmax** function, which turns scores into probabilities that sum to 1).
- **Regularisation (C):** a penalty that keeps the model's weights small, so it can't memorise noise. In scikit-learn, **small C means strong regularisation**. With 5,120 features and only 660 training clips per fold, strong regularisation is essential.
- **Hyper-parameter:** a setting chosen *before* training, like C. We choose it by **OOF log-loss**.
- **Log-loss (cross-entropy):** measures how good the predicted *probabilities* are, not just the final label. It punishes confident mistakes heavily, which makes it a smoother, more reliable way to pick C than F1.
- **`predict_proba`:** returns the 4 class probabilities. We keep **probabilities**, not just labels, because blending later needs them.

---

## 6. Step 5: the six ensemble members

An **ensemble** combines several models. It works best when members make **different (decorrelated) mistakes**, so one member's error is outvoted by the others.

### Member 1: Whisper-large-v3, layers 24 + 32, with augmentation (20% of the final blend)

- **Layer choice:** we tested layers one by one (a **layer sweep**). Whisper gets better with depth; concatenating layers 24 and 32 (5,120 features) beat any single layer.
- **Data augmentation:** creating modified copies of training data so the model learns what matters (emotion) and ignores what doesn't (noise, timing).

`src/whisper_aug.py`, with four augmented **views** per clip:

```python
def views(a, rng):
    noise = lambda db: a + rng.standard_normal(len(a)) * np.sqrt(np.mean(a ** 2) / 10 ** (db / 10))  # white noise at given SNR
    lead = lambda s: np.concatenate([np.zeros(int(s * 16000)), a])                                   # s seconds of silence before
    return [noise(20), noise(30), lead(0.25), lead(0.5), ...]
```

**SNR (signal-to-noise ratio)** in **dB (decibels)** says how loud the signal is compared with the noise: 20 dB is mild noise, 30 dB is very light.

Training uses all 5 versions (original plus 4 views) of each training clip. Prediction averages over the 5 versions of each test clip. Averaging over augmented versions at prediction time is called **test-time augmentation (TTA)**. From `src/rcv_spikes.py`:

```python
m.fit(np.vstack([X[tri] for X in views]), np.tile(y[tri], len(views)))     # train on original + 4 views
pr = lambda idx: np.mean([m.predict_proba(X[idx]) for X in views], 0)      # predict = mean over the 5 versions
```

Alone it scores **0.902** (grouped CV, mean of 3 splits).

### Member 2: w2v-BERT 2.0, layers 10 + 16 (20%)

A multilingual **self-supervised** speech model. Self-supervised means it learned from unlabelled audio by predicting hidden parts of it, and its training audio included Hindi. The layer sweep showed emotion information **peaks in the middle layers** and collapses at the top (layer 24 scores only 0.59), so we use layers 10 and 16. Same `probe()` as above. Alone it scores 0.812.

An important lesson: when we *augmented* this member, its own score went up (0.81 → 0.83), but the ensemble got worse, because its errors became too similar to Whisper's. **A stronger member is not always a better ensemble member.**

### Members 3 and 4: Qwen2.5-Omni-7B, an audio language model (10% + 10%)

An **LLM (large language model)** that can also listen to audio. "7B" means 7 billion **parameters** (learned weights). We ran it on Kaggle's free GPUs. It contributes in two ways.

`src/omni_zs.py`:

```python
Q = 'What emotion does the speaker express in this Hindi audio? Choose one: angry, happy, neutral, sad. Answer with one word.'
out = model(**inp, output_hidden_states=True)
logp = torch.log_softmax(out.logits[0, -1].float(), -1)                          # model's next-word probabilities
lp[i] = [torch.logsumexp(logp[cs], 0).item() for cs in CAND]                      # log-prob of each emotion word
m = (inp['input_ids'][0] == AUDIO)                                                # positions of the audio tokens
hid[i] = torch.stack([hs[14][0][m].mean(0), hs[-1][0][m].mean(0)])                # mean hidden state, layers 14 and 28
```

- **Zero-shot:** asking the model to classify without any training on our data. Its answer comes as **log-probabilities** (logarithms of probabilities) for the 4 emotion words. A small probe on these 4 numbers corrects the model's bias (it over-predicted neutral). Alone: 0.738.
- **Hidden states over audio tokens:** the LLM turns audio into **tokens** (units it processes). We average its internal vectors over those audio tokens at layers 14 and 28, then probe them. Alone: 0.820.

### Member 5: transcript text (25%)

Emotion is partly in the *words*. Two steps:

1. **ASR (automatic speech recognition):** Whisper-large-v3 writes down the Hindi sentence.
2. **Text embedding:** `multilingual-e5-large` turns the sentence into a 1,024-number vector that captures meaning.

`src/e5_embed.py`:

```python
b = tok(['query: ' + s for s in texts], padding=True, truncation=True, max_length=128, return_tensors='pt')
h = m(**b).last_hidden_state; mk = b['attention_mask'][..., None]
e = (h * mk).sum(1) / mk.sum(1)                          # mean pooling over real tokens (ignore padding)
e = torch.nn.functional.normalize(e, dim=-1)            # L2 normalisation: vector length 1
```

**L2 normalisation** scales a vector to length 1, so only its direction (meaning) matters. Then logistic regression on these vectors, with C chosen by F1. Why it works: sentences that repeat in this dataset keep the same emotion **97%** of the time. Alone it scores only 0.725, but it's weakest exactly where audio is strongest, which is ideal diversity.

### Member 6: attentive audio-text fusion head (our key contribution, 15%)

**Problem:** emotion cues are **temporally sparse**, meaning they appear in a few moments (a rising pitch, a stressed word). Mean pooling averages them away. **Solution:** a small neural network that *learns which frames matter*.

`src/attn_pool.py`:

```python
class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(2560, 256), nn.GELU(), nn.Dropout(0.3))   # compress each frame 2560 → 256
        self.att = nn.Linear(256, 4)                                                  # 4 attention heads: a score per frame
        self.txt = nn.Sequential(nn.Linear(1024, 64), nn.GELU(), nn.Dropout(0.3))     # compress transcript embedding
        self.out = nn.Sequential(nn.Dropout(0.3), nn.Linear(2 * 4 * 256 + 64, 4))     # final layer → 4 emotions

    def forward(self, x, mask, idx):
        h = self.proj(x)                                                   # [batch, frames, 256]
        w = self.att(h).masked_fill(~mask[..., None], -1e4).softmax(1)     # attention weights over frames (ignore padding)
        mu = torch.einsum('bth,btd->bhd', w, h)                            # attention-weighted mean
        sd = (torch.einsum('bth,btd->bhd', w, h * h) - mu ** 2).clamp_min(1e-6).sqrt()   # attention-weighted std
        z = torch.cat([mu, sd], -1).flatten(1)                             # attentive statistics pooling
        z = torch.cat([z, self.txt(E[idx])], 1)                            # EARLY FUSION with the transcript
        return self.out(z)
```

The terms in that code:

- **Linear layer:** multiplies by a learned weight matrix, the basic neural-network building block.
- **GELU:** an **activation function**, a non-linear "bend" that lets networks learn complex patterns.
- **Dropout (0.3):** during training, randomly switches off 30% of values, so the network can't rely on any single one. It's a regularisation against overfitting.
- **Attention:** the network gives each frame a weight (via softmax, so the weights sum to 1), learning to focus on the emotional moments. **4 heads** means 4 independent sets of weights, each free to focus on different cues.
- **Attentive statistics pooling:** the weighted mean and weighted standard deviation, using those attention weights instead of treating all frames equally.
- **Early fusion:** combining audio and text *inside* the model, before the final decision, so the model learns how they interact. (**Late fusion** would average separate audio and text predictions instead.)

Training settings:

```python
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
loss = F.cross_entropy(model(...), Y[idx], label_smoothing=0.1)      # 30 epochs, batch 16, 3 seeds averaged
```

- **AdamW:** a standard optimiser (the algorithm that updates the weights), with **weight decay**, a regularisation that shrinks weights.
- **Learning rate (lr):** the step size of each update.
- **Cosine schedule:** the learning rate decreases smoothly along a cosine curve, from big steps at the start to tiny steps at the end.
- **Label smoothing (0.1):** instead of "100% angry", the training target is "90% angry, the rest spread over the others". This stops over-confidence.
- **Epoch:** one full pass over the training data. **Batch:** the number of clips processed per update.
- **Seeds:** the random starting points. We train 3 seeds and average them to reduce randomness.
- Frames are standardised using **train-fold statistics only**, so nothing from the validation fold leaks in.

It's the **best single model: 0.916**.

---

## 7. Step 6: the blend, a fixed-weight ensemble

Each member gives 4 probabilities per clip. We average them with **fixed weights**, chosen in advance.

From `src/rcv_decide.py`:

```python
def full(s, **kw):
    r = {**BASE, **kw}
    audio = 1/3 * (L(s, r['whisper']) + L(s, r['w2v'])) + 1/6 * (L(s, r['omni_h']) + L(s, r['omni_lp']))
    if r['extra']: audio = 0.8 * audio + 0.2 * L(s, r['extra'])      # the fusion head joins as a 20% voice
    return 0.75 * audio + 0.25 * L(s, r['text'])                     # 75% audio, 25% transcript text
# final prediction = argmax of these blended probabilities (the most probable emotion)
```

That gives effective weights of Whisper 20%, w2v-BERT 20%, Omni hidden 10%, Omni log-probs 10%, fusion head 15% and text 25%. **argmax** means picking the class with the highest probability.

**Why not learn the weights?** We tried **stacking**: training a second model on the members' OOF predictions to learn the best combination. It *lost* 0.005–0.021. With 826 clips, any learned combiner overfits the OOF predictions. A fixed average has no parameters to overfit. **Simplicity here is a statistical choice.**

---

## 8. Step 7: the decision protocol (why our gains are real)

Even honest CV has **variance**: the blend's score moves by about ±0.01 just from how the folds are shuffled. So a +0.003 gain on one split may be pure luck. Our rules:

1. **Pre-registration:** write down the exact experiment *before* seeing its result. That way we can't quietly tweak it until it looks good.
2. **Repeated CV:** retrain everything on **3 different grouped splits** (random seeds 42, 7 and 11).
3. **Adoption gate:** keep an idea only if the **mean gain is at least +0.005** and the gain is **positive on every split**.

From `src/rcv_decide.py`:

```python
g = [F(full(s, **candidate)) - F(full(s, **base)) for s in (42, 7, 11)]   # gain on each of the 3 splits
ok = np.mean(g) >= 0.005 and min(g) > 0                                  # the gate
```

About 50 ideas went in and 4 passed:

1. Omni and text members together
2. Whisper layers 24 + 32
3. Augmentation, plus w2v-BERT layers 10 + 16
4. The fusion head

**Selection bias:** if you test many ideas, some look good by chance. Our strict gate keeps that small, and that's why our "didn't work" list is trustworthy too.

---

## 9. Step 8: the twin lock (post-processing)

78 of the 209 test clips have a twin in training. Since twins share their label 99.1% of the time, we copy the training label onto them. **Post-processing** means rules applied after the model predicts.

`src/twinlock.py`:

```python
S = W[n:] @ W[:n].T                                                    # cosine: every test clip vs every train clip
S[np.abs(te.dur.values[:, None] - tr.dur.values[None, :]) >= 1e-3] = -1   # must have the same duration
S[te.h.values[:, None] == tr.h.values[None, :]] = 2                    # identical file = certain twin
for i in range(len(te)):
    if S[i].max() > 0.9:                                               # audio twin found
        label = tr.emotion.values[S[i].argmax()]
    elif len(tte[i].split()) >= 3 and (ttr == tte[i]).any():           # else: same transcript (3+ words)
        labs = tr.emotion.values[ttr == tte[i]]
        if len(set(labs)) == 1: label = labs[0]                        # only if all matches agree
```

It locked 69 clips by audio and 9 by text. It's legitimate: it uses **only competition training labels**, never outside labels of test clips (which the rules forbid), and we validated the rule on training data first.

---

## 10. Step 9: the final refit and submission

During CV, each model trains on only 4/5 of the data (660 clips). The **learning curve** (score against training-set size) was still rising: 0.843 at 330 clips, 0.882 at 495, 0.896 at 660. So for the final test predictions we **refit** every member on **all 826 clips**, keeping the hyper-parameters (like C) that CV chose.

`src/final_refit.py`:

```python
def full_fit(X, C, scale=True):
    return lr(C, scale).fit(X[:n], y).predict_proba(X[n:])     # train on all 826, predict the 209 test clips
```

Then: blend → argmax → write `Id,emotion` → twin lock → `subs/v9_a3_full_lock.csv`.

**Sanity checks** on every submission:

- Each class predicted about 45–60 times. The test set is balanced, so a big skew signals a bug.
- The twin lock was applied.
- We listed which test predictions changed compared with the previous best.

**Our two final picks:**

- `v5_aug_c1_lock.csv` (recipe A″: the blend without the fusion head). This is the one counted: **private 0.9434**.
- `v9_a3_full_lock.csv` (recipe A‴: plus the fusion head, refit on all data). Private 0.935.

They differ on only one test clip. Kaggle counts the better of your two picks.

---

## 11. Why this strategy wins

1. **We fixed validation before modelling.** Teams using random folds were optimising against leaked copies. Our numbers were honest from day 1, and they matched the leaderboard.
2. **We used frozen foundation models with small heads**, which is the right tool for 826 clips. Fine-tuning Whisper scored 0.55; the frozen probe scored 0.90.
3. **We chose ensemble members for diversity**, measured on the blend, not alone. Gemma-4's encoder (0.855 alone) made the blend *worse*, so it was rejected.
4. **We turned the flaw into a feature** with a validated twin lock.
5. **Every decision was pre-registered and repeated.** That's why we ignored the public board when it couldn't see real gains (it plateaued at 0.942 while CV rose from 0.916 to 0.930).

Compared with the other presenting teams' **private** scores: ours **0.943 (6th)**, parishim 0.928 (12th), Pikachu 0.904 (23rd), TechNova 0.880 (29th, despite a public 0.966: a classic public-leaderboard overfit).

---

## 12. Glossary

| Term | Meaning |
|---|---|
| Classification | Predicting a category (here, one of 4 emotions) |
| Precision / Recall | Precision: of what we called X, how much was X. Recall: of all real X, how much we found |
| F1 / Macro-F1 | Balance of precision and recall / the average F1 over all classes equally |
| Public / Private LB | Test-score portions shown during / after the competition |
| Hash | A fingerprint of a file's bytes; identical files give identical hashes |
| Embedding | A vector of numbers a model produces to describe an input |
| Cosine similarity | How closely two vectors point the same way (1 = identical) |
| Union-find / connected components | Merging linked items into groups |
| Cross-validation (CV) / fold | Rotating train/validation splits / one part of the split |
| Out-of-fold (OOF) | Predictions made by a model that never saw that clip |
| Data leakage | Evaluation information sneaking into training, inflating scores |
| StratifiedGroupKFold | Folds that keep groups together and class balance equal |
| Foundation model | Huge pre-trained model reused for new tasks |
| Encoder / layer / frame | Model part that builds representations / one stage in it / a 20 ms time slice |
| Frozen / fine-tuning | Model weights unchanged / weights re-trained on our data |
| Overfitting | Memorising training data instead of learning general patterns |
| Mean + std pooling | Turning many frames into one vector (average and spread) |
| Log-Mel spectrogram | Audio as an energy-by-frequency-over-time image |
| Padding | Adding silence to reach a fixed length |
| Probe | A simple classifier trained on frozen features |
| Standardisation | Rescaling each feature to mean 0, std 1 |
| Logistic regression | A linear model that outputs class probabilities |
| Softmax | Turns scores into probabilities summing to 1 |
| Regularisation / C | A penalty against overfitting / its strength (small C = strong) |
| Hyper-parameter | A setting chosen before training |
| Log-loss (cross-entropy) | Scores probability quality; punishes confident mistakes |
| Ensemble / member | A combination of models / one model in it |
| Decorrelated errors | Members wrong on different clips, so errors cancel out |
| Data augmentation / view | Modified copies of training data / one such copy |
| SNR / dB | Signal-to-noise ratio / its unit (higher = cleaner) |
| TTA | Averaging predictions over augmented versions at test time |
| Self-supervised | Learned from unlabelled data by predicting hidden parts |
| LLM / parameters / tokens | Large language model / learned weights / units it processes |
| Zero-shot | Using a model with no training on our task |
| Log-probability | The logarithm of a probability |
| ASR | Automatic speech recognition (speech → text) |
| L2 normalisation | Scaling a vector to length 1 |
| Temporally sparse | Appearing only in a few moments of time |
| Linear layer / GELU | Learned matrix multiply / smooth non-linear activation |
| Dropout | Randomly zeroing values during training (regularisation) |
| Attention / heads | Learned weights saying which frames matter / independent sets of them |
| Attentive statistics pooling | Attention-weighted mean and std over frames |
| Early vs late fusion | Combining modalities inside the model vs averaging separate predictions |
| AdamW / weight decay | Optimiser / weight-shrinking regularisation |
| Learning rate / cosine schedule | Update step size / smooth decrease of it over training |
| Label smoothing | Softened training targets to prevent over-confidence |
| Epoch / batch / seed | One pass over the data / clips per update / random starting point |
| Fixed-weight blend / stacking | Pre-set average / a learned combiner model |
| argmax | Pick the class with the highest probability |
| Variance (of CV) | How much the score changes from shuffling alone |
| Pre-registration | Writing down an experiment before seeing its result |
| Repeated CV / adoption gate | CV on several splits / the rule for keeping an idea |
| Selection bias | Ideas looking good by chance when many are tried |
| Post-processing | Rules applied after the model predicts |
| Learning curve / refit | Score vs training size / retraining on all data |

---

## 13. How to reproduce it (commands)

Run every command from the repo root, with `PYTHONUTF8=1` set on Windows.

```bash
python src/extract.py whisper                 # Whisper-large-v3 features → feats/whisper.npy
python src/extract.py w2vbert                 # w2v-BERT 2.0 features → feats/w2vbert.npy
python src/whisper_aug.py                     # augmented Whisper views → feats/whisper_aug.npy
python src/e5_embed.py feats/asr.csv feats/e5.npy        # transcript embeddings (ASR from Whisper)
# Qwen2.5-Omni features: Kaggle kernel kernels/dataverse-omni-zs → feats/omni_h.npy, feats/omni_lp.npy
python src/rcv_members.py 42                  # train all probes on grouped folds (repeat with 7 and 11)
python src/rcv_spikes.py 42 s3                # augmented Whisper member
python src/attn_pool.py 42 text               # attentive audio-text fusion head
python src/rcv_decide.py                      # apply the adoption gate to every candidate
python src/attn_pool.py full text             # fusion head trained on all 826 clips
python src/final_refit.py subs/v9_a3_full.csv # refit all members, blend → submission
python src/twinlock.py subs/v9_a3_full.csv subs/v9_a3_full_lock.csv   # twin lock → final file
```
