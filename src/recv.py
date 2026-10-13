"""Repeated honest CV on fresh near-dup-grouped fold splits (seeds 7, 11): re-train every blend member, compare
current blend vs whisper L24+L32-concat replacement. PRE-REGISTERED: adopt if mean gain >= +0.005 and > 0 on both seeds."""
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
import hcv
from hcv import y, n, group, tr, f1, probe

W = np.load('feats/whisper.npy', mmap_mode='r'); V = np.load('feats/w2vbert.npy', mmap_mode='r')
X = {'whisper_L32': W[:, 32].astype(np.float32), 'w2vbert_L16': V[:, 16].astype(np.float32),
     'omni_h': np.load('feats/omni_h.npy').astype(np.float32).reshape(1035, -1), 'omni_lp': np.load('feats/omni_lp.npy'),
     'whisper_L24L32': np.hstack([W[:, 24], W[:, 32]]).astype(np.float32)}
E5 = np.load('feats/e5.npy')
gains = []
for seed in (7, 11):
    f = np.zeros(n, int)
    for k, (_, v) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(tr, y, group)):
        f[v] = k
    hcv.fold = f
    O = {m: probe(Z)[1] for m, Z in X.items()}
    best = None
    for C in (0.3, 1, 3, 10, 30):  # e5 text probe exactly as in t1_text2 (no scaler, C by F1)
        o = np.zeros((n, 4))
        for k in range(5):
            o[f == k] = LogisticRegression(C=C, max_iter=4000).fit(E5[:n][f != k], y[f != k]).predict_proba(E5[:n][f == k])
        if best is None or f1(o) > best[0]: best = (f1(o), o)
    O['text_e5'] = best[1]
    rest = 1 / 3 * O['w2vbert_L16'] + 1 / 6 * (O['omni_h'] + O['omni_lp'])
    cur = 0.75 * (1 / 3 * O['whisper_L32'] + rest) + 0.25 * O['text_e5']
    new = 0.75 * (1 / 3 * O['whisper_L24L32'] + rest) + 0.25 * O['text_e5']
    d = hcv.per_fold(new, cur); g = f1(new) - f1(cur); gains.append(g)
    print(f'seed {seed}: current {f1(cur):.4f} | L24+L32 {f1(new):.4f} ({g:+.4f}) folds up {sum(x > 0 for x in d)}/5 ' + ' '.join(f'{x:+.3f}' for x in d)
          + f' | members: ' + ' '.join(f'{m} {f1(o):.3f}' for m, o in O.items()), flush=True)
print(f'\nmean gain {np.mean(gains):+.4f} (+ original split +0.0048) | ADOPT: {np.mean(gains) >= 0.005 and min(gains) > 0}')
