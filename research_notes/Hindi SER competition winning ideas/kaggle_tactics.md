# Kaggle tactics for small audio / speech competitions (leakage, near-duplicates, public/private split)

Scope note: this search found very few primary winning write-ups for speech-emotion or small-data audio competitions. Most fetches of PDFs and some GitHub/Medium pages failed (binary PDF dumps, HTTP 403). Only the BirdCLEF 2024 1st-place write-up (Zenn, Japanese blog) gave a readable score table. Everything else below is either search-snippet level or a secondary summary, and is marked that way. No source reported a transductive or test-cluster trick on a speech competition.

## Q1. How did winners exploit near-duplicate or same-speaker structure legitimately?

### Takeaway
No retrievable write-up described a winner exploiting test-clip near-duplicates or speaker clusters in an audio competition. The closest documented material is about removing duplicates from training data and choosing folds by signal statistics (BirdCLEF 2024), which is a training-side hygiene step, not a test-side trick.

### Cited Findings
- BirdCLEF 2024 1st place: "There are some duplicate audio files in te train data, so we discard them." No count given. — [Zenn, BirdCLEF2024 1st Solution explained](https://zenn.dev/yuto_mo/articles/ad43c630729073)
- Same write-up: fold0 plus the 0.8 quantile of a signal-statistic score T = std + var + rms + pwr was used as the ensemble training subset, and the author says it "worked well"; fold0 was described as giving better results than other folds. — [Zenn](https://zenn.dev/yuto_mo/articles/ad43c630729073)
- Freesound 2019: public and private scores came from different test slices (secondary search summary; no primary source read, so no URL can be cited for this claim). Shows only that the public/private split is structural, not a tactic.
- Transductive batch refinement for zero-shot audio classification (not Kaggle, not supervised): +4.6 to +9.2 points at full-fold batch; the gain reverses below a batch size because cluster means are re-estimated from their own errors. — [arXiv 2606.17160](https://arxiv.org/pdf/2606.17160) (search snippet; full text not read)

### Inferences
- Our twin lock is the same family as "discard/group duplicates" and is already in place. The Zenn evidence supports duplicate handling in training, which our honest near-dup-grouped folds already do.
- The transductive literature (2606.17160) targets zero-shot classifiers whose class means are re-estimated from test batches. Our members are supervised LR probes, so the transferable part is weak. A cluster-mean refinement of our probabilities would behave like label spreading or kNN interpolation, both already rejected in EXPERIMENTS.md (label spreading ✗ not robust; P2 kNN ✗ −0.0068). Not recommended.

### Gaps
- No speech-emotion or speaker-structure winner write-up was retrieved. The Interspeech-2025 and Odyssey-2024 solutions named in EXPERIMENTS.md were not re-read here.
- The BirdCLEF 2025 top-3 review deck (lamsade.dauphine.fr) and the CEUR paper 2107.07728 (BirdCLEF 2021, Henkel/Pfeiffer/Singer) returned binary PDF data; their body text was not readable. Search snippets only: the 2025 1st place appears to rely on a stronger self-training loop and an engineered inference stack; the 2025 1st and 2nd place both used pseudo-labelling. Not verified.

## Q2. What do top tiny-data audio competitions do for ensembling and seed averaging?

### Takeaway
The one concrete table found (BirdCLEF 2024 1st) shows that ensembling several models and aggregating chunk predictions gave the largest measured gains. It does not mention seed averaging. The strongest seed-variance evidence is a secondary CEUR paper, not a winner's write-up.

### Cited Findings
- BirdCLEF 2024 1st place score table (private / public): baseline 0.544 / 0.600; sigmoid at inference 0.588 / 0.629; horizontal cutmix 0.615 / 0.670; 10-second input 0.670 / 0.716; ensemble of 5 efficientnet_b0 with min over chunks 0.689 / 0.735; mean[min over 3 efficientnet, min over 3 regnety] 0.692 / 0.734 (highest private row). Ensembling gave about +0.019 private over the best single-model row (10-second input, 0.670 to 0.689); the two-architecture mean added only about +0.003 over the 5-model ensemble. — [Zenn](https://zenn.dev/yuto_mo/articles/ad43c630729073)
- Same article states the public and private scores correlate at 0.96 across its runs. — [Zenn](https://zenn.dev/yuto_mo/articles/ad43c630729073)
- Seed scatter: a 2026 CEUR paper from a 9th-place team (public 2nd) reports a ±0.003 run-to-run seed scatter on the public board, comparable to many gains, and responded with weight averaging and ensemble diversity. — [CEUR Vol-4283 paper369](https://ceur-ws.org/Vol-4283/paper369.pdf) (search snippet only; PDF fetch failed, so the numbers are unverified)
- Ensemble-min over chunks and two-architecture mean are the inference-time choices named in the 2024 write-up. Seeds and TTA are not stated there. — [Zenn](https://zenn.dev/yuto_mo/articles/ad43c630729073)

### Inferences
- Ensembling beat the best single model by about +0.019 private in the 2024 table, but the second architecture added only +0.003. This is one competition, and our blend (whisper, w2v-BERT, Omni, e5) already ensembles across model families. The marginal value of a further member is small, as our own tests show.
- Seed averaging is cheap for us: our A‴ seed-averaged test predictions (15 fold models) changed 0 clips vs A″ (HANDOFF/EXPERIMENTS). So seed averaging is already done and is not a source of gain.
- The 2024 min-over-chunks reduction is not directly applicable: our clips are 1–11 s single utterances, not 10-minute soundscapes split into chunks.

### Gaps
- No winner write-up for a speech-emotion or small-audio competition with a full seed/TTA description was retrieved.
- Freesound 2019 9th-place (TheoViel/kaggle_freesound on GitHub) returned HTTP 403. Its write-up was not read. A secondary summary page said the team placed 9th of 880 with lwlrap and no pretrained models allowed, but gave no ensemble details.
- Cornell Birdcall 2020 1st place: repository found (github.com/ryanwongsa/kaggle-birdsong-recognition) and the Kaggle kernel taggatle/cornell-birdcall-identification-1st-place-solution named, but neither body text was retrieved.

## Q3. What do experienced Kagglers recommend for choosing two final submissions with a tiny public split?

### Takeaway
Common advice: pick one submission that is best on robust local CV and one that is best on the public board as a hedge, or prefer a diverse ensemble over near-duplicates. Public scores on small test sets are weak evidence. Our two current picks would not satisfy the diversity advice because they differ on only one clip.

### Cited Findings
- DataCamp "Winning a Kaggle competition in Python": pick one submission best on local validation and another best on the public leaderboard, as a hedge. — [DataCamp course](https://campus.datacamp.com/courses/winning-a-kaggle-competition-in-python/modeling?ex=12) (search snippet)
- Public slice can be small: one paper's setup had about 34% public and 66% private; the selected primary submission was, after reveal, the best private score among 40 scored submissions. Diversity-aware selection is argued to protect against shake-up. Single case study. — [CEUR paper369](https://ceur-ws.org/Vol-4283/paper369.pdf) (search snippet; PDF fetch failed)
- Packt "The Kaggle Book": trusting the public part of the test data "is not advisable because it is quite often misleading." — [Packt, Designing good validation](https://packtpub.com/en-jp/product/the-kaggle-book-9781801817479/chapter/designing-good-validation-8/section/designing-good-validation-8) (search snippet)
- AICrowd simulation of shake-ups (single model): public vs private differences up to about 10%; better-public models tended to be worse-private. — [AICrowd forum](https://discourse.aicrowd.com/t/simulating-leaderboard-shakeups-to-give-some-context/5917) (search snippet; one model, anecdotal)
- Data Science Bowl 2019: many teams that trusted local CV over the public board climbed roughly 500 to 1000 places on private (secondary, search snippet). — [AICrowd forum summary](https://discourse.aicrowd.com/t/simulating-leaderboard-shakeups-to-give-some-context/5917)
- A slide deck on Kaggle tips suggests CV beats the public LB about 90% of the time and proposes a 0.5-weighted blend of CV and LB as a heuristic (not validated). — [search snippet, lec-3 slides](https://www.lamsade.dauphine.fr/~ebenhamou/Becoming_a_Kaggle_Master/static/slides/lec-3.pdf)
- Few submissions reduce the chance of overfitting the public board. — [search snippet, same deck](https://www.lamsade.dauphine.fr/~ebenhamou/Becoming_a_Kaggle_Master/static/slides/lec-3.pdf)

### Inferences
- Our rule (honest CV first, public as tie-break) matches the consensus.
- Our current picks differ on one clip (Id 235). That is not a hedge in the diversity sense. If the goal is protection against shake-up, the second pick should be an honest-CV-competitive candidate that differs on more private-relevant clips. Candidates within the 0.005 CV rule from HANDOFF: v8_K1 (public 0.93048, lower; dropped by rule B), v8_K2/K3/K4 (2–3 test clips differ, not submitted). The trade-off is public-only evidence, so this is a judgment call, not a proven gain.
- Context check: the user's brief says the deadline is 12 Oct 2026 18:30 UTC with about 11 h left. HANDOFF.md records the deadline as 10 Oct 18:30 UTC (= 11 Oct 00:00 IST) and says final picks were already selected on 10 Oct. These cannot both be right. Verify the deadline on the Kaggle competition page before acting on any pick change.

### Gaps
- No quantitative study shows a gain from choosing a diverse vs CV-best pick on a tiny public split. The only quantitative example is one anecdotal case (CEUR paper369) that could not be verified.
- DrivenData-style platforms may auto-select the best public score (single forum comment). Check this competition's rules tab, which is not something this search could confirm.

## Q4. Which allowed uses of test-set structure without labels (clustering, consistency) gave gains?

### Takeaway
No source retrieved reports a gain from unlabelled test-set clustering or consistency in a supervised audio competition. The only transductive gain documented (2606.17160) is for zero-shot models and depends on batch size. Our own tests already covered test-side smoothing (label spreading, pseudo-labels, kNN, twin-group averaging), all rejected or near zero.

### Cited Findings
- Batch-level transductive refinement for zero-shot audio classification: +4.6 to +9.2 points on full-fold batches; reverses at small batch sizes. — [arXiv 2606.17160](https://arxiv.org/pdf/2606.17160) (search snippet)
- BirdCLEF 2025 1st/2nd: both reported as relying on pseudo-labelling (search snippet, not verified). — [BirdCLEF 2025 review slides](https://www.lamsade.dauphine.fr/~ebenhamou/Becoming_a_Kaggle_Master/static/slides/Birdclef_2025_top3_solution_review.pdf) (fetch returned binary; snippet only)

### Inferences
- Already tested in EXPERIMENTS.md with the same family of method: label spreading (+0.014 at one setting, not robust), pseudo-labels (±0), kNN interpolation (−0.0068), twin-group averaging (+0.001), test-test twin consistency. The pattern in our data is that test-side smoothing adds nothing beyond the twin lock.
- Untested, low-cost, rule-compliant idea (inference, not from a source): transductive feature standardisation, fitting the StandardScaler on train+test features. It uses no labels. Expected effect is small; the probe is fitted on the same scaled features. Would need a pre-registered repeated-CV test.

### Gaps
- No speech or audio competition write-up found that documents transductive test-cluster gains.
- The 2025 BirdCLEF pseudo-label claim is unverified (PDF unreadable).

## Ranked tactics for our setting (from the evidence above)

| # | tactic | source | reported gain | fit for us |
|---|---|---|---|---|
| 1 | Multi-architecture ensemble (mean across models) | BirdCLEF 2024 1st, Zenn | ~+0.02 private | Already in place |
| 2 | Duplicate removal in training | BirdCLEF 2024 1st, Zenn | not quantified | Our near-dup grouping covers it. Only remaining item: the 1 conflicting pair (Id 636 sad / Id 75 neutral) |
| 3 | Pick final = best honest CV + public hedge | DataCamp, Packt (snippets) | none measured | Current picks differ on 1 clip; a hedge would need more diversity |
| 4 | Seed averaging | CEUR paper369 (snippet) | seed scatter ±0.003 | Done (A‴ seed-avg: 0 clips changed) |
| 5 | Batch transductive refinement | arXiv 2606.17160 (zero-shot) | +4.6 to +9.2 (zero-shot only) | Not applicable to supervised probes; similar to rejected methods |
| 6 | Pseudo-labels, label spreading, kNN | BirdCLEF 2025 (snippet) and our tests | mixed / rejected | Already rejected |

## Sources actually read in full (readable text)
- [Zenn, BirdCLEF2024 1st Solution explained](https://zenn.dev/yuto_mo/articles/ad43c630729073) (Japanese, the only table-level source)

## Sources that failed to load (binary or 403)
- lamsade BirdCLEF 2025 top-3 review PDF; CEUR paper369 PDF; arXiv 2107.07728 PDF (BirdCLEF 2021 Henkel et al.); TheoViel/kaggle_freesound (HTTP 403); Medium "Kaggle Handbook: Tips & Tricks To Survive a Kaggle Shake-up" (HTTP 403 or no text).
