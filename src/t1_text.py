"""B: what do the transcripts say? Sentence repetition vs emotion, text-only classifier, blend test (honest v2 folds)."""
import re, numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from hcv import *

A = pd.read_csv('feats/asr.csv', encoding='utf-8')
norm = lambda s: re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', ' ', str(s).lower())).strip()
A['t'] = A.text.fillna('').map(norm)
ttr, tte = A.t.values[:n], A.t.values[n:]
print('examples:'); [print('  ', tr.emotion[i], '|', ttr[i][:80]) for i in range(0, 40, 5)]
# sentence repetition, de-duplicated at near-dup group level
g = pd.DataFrame({'t': ttr, 'emo': tr.emotion, 'grp': group}).drop_duplicates('grp')
vc = g.t.value_counts()
rep = g[g.t.isin(vc[vc >= 2].index)]
pur = rep.groupby('t').emo.agg(lambda s: s.value_counts().iloc[0] / len(s))
print(f'\nunique groups {len(g)} | distinct transcripts {g.t.nunique()} | transcripts used by >=2 groups: {len(vc[vc >= 2])} covering {len(rep)} groups')
print(f'label purity of repeated transcripts: mean {pur.mean():.3f} | fully pure {np.mean(pur == 1):.2f}')
print('emotions per repeated sentence:', rep.groupby('t').emo.nunique().value_counts().sort_index().to_dict())
nov = ~te.h.isin(set(tr.h)).values
print(f'test transcripts that exactly match a train transcript: {np.isin(tte, ttr).sum()} / {len(tte)}')
# text-only classifier (char n-grams) on honest folds
X = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), min_df=2, sublinear_tf=True).fit(A.t).transform(A.t)
best = None
for C in (0.3, 1, 3, 10):
    oof = np.zeros((n, 4)); test = np.zeros((len(tte), 4))
    for k in range(5):
        m = LogisticRegression(C=C, max_iter=4000).fit(X[:n][fold != k], y[fold != k])
        oof[fold == k] = m.predict_proba(X[:n][fold == k]); test += m.predict_proba(X[n:]) / 5
    if best is None or f1(oof) > best[0]:
        best = (f1(oof), C, oof, test)
s, C, oof, test = best
np.save('oof/v2_text.npy', oof); np.save('oof/v2_text_test.npy', test)
print(f'\nTEXT-ONLY char-ngram LR C={C}: honest F1 {s:.4f}')
O = {m: np.load(f'oof/v2_{m}.npy') for m in ['whisper_L32', 'w2vbert_L16', 'omni_h', 'omni_lp']}
cur = 1 / 3 * (O['whisper_L32'] + O['w2vbert_L16']) + 1 / 6 * (O['omni_h'] + O['omni_lp'])
for w in (0.15, 0.25):
    b = (1 - w) * cur + w * oof; d = per_fold(b, cur)
    print(f'base+omni + text w={w}: {f1(b):.4f} ({f1(b) - f1(cur):+.4f}) folds up {sum(x > 0 for x in d)}/5 ' + ' '.join(f'{x:+.3f}' for x in d))
