"""Tier-3 evaluation: probe every feature set, report all / no-twin OOF macro-F1, try blends, label spreading, pseudo-labels.
Honest metric = no-twin F1 (clips without a near-duplicate in another fold). Saves oof/t3_<name>.npy (+ _test)."""
import os, itertools, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score, log_loss

C4 = ['angry', 'happy', 'neutral', 'sad']
tr = pd.read_csv('feats/train_meta.csv'); n = len(tr)
y = tr.emotion.map({c: i for i, c in enumerate(C4)}).to_numpy()
fold = pd.read_csv('feats/folds.csv').fold.to_numpy()

# no-twin mask: whisper-L32 cosine > 0.95 to a clip in another fold
W = np.load('feats/whisper.npy', mmap_mode='r')[:, 32].astype(np.float32)
Wn = (W - W[:n].mean(0)) / (W[:n].std(0) + 1e-6); Wn /= np.linalg.norm(Wn, axis=1, keepdims=True)
S = Wn[:n] @ Wn[:n].T; S[fold[:, None] == fold[None, :]] = -9
NT = S.max(1) <= 0.95
np.save('feats/notwin_mask.npy', NT)


def score(p, m=None):
    m = np.ones(n, bool) if m is None else m
    ok = m & ~np.isnan(p).any(1)
    return (f1_score(y[ok], p[ok].argmax(1), average='macro'), f1_score(y[ok & NT], p[ok & NT].argmax(1), average='macro'),
            log_loss(y[ok], np.clip(p[ok], 1e-6, 1), labels=range(4)), ok.sum())


def probe(X, Cs=(0.003, 0.01, 0.03, 0.1, 0.3), extra=None):
    """LR probe; rows with NaN features are skipped (OOF stays NaN). extra(k) -> (Xpl, ypl) pseudo-label rows for fold k."""
    have = ~np.isnan(X).any(1); best = None
    for C in Cs:
        oof = np.full((n, 4), np.nan); test = np.zeros((len(X) - n, 4)); tests = []
        for k in range(5):
            trn = (fold != k) & have[:n]; val = (fold == k) & have[:n]
            Xt, yt = X[:n][trn], y[trn]
            if extra is not None:
                Xp, yp = extra(k); Xt, yt = np.vstack([Xt, Xp]), np.r_[yt, yp]
            m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=4000)).fit(Xt, yt)
            oof[val] = m.predict_proba(X[:n][val])
            tk = np.full((len(X) - n, 4), np.nan); tk[have[n:]] = m.predict_proba(X[n:][have[n:]]); tests.append(tk)
        ll = score(oof)[2]
        if best is None or ll < best[0]:
            best = (ll, C, oof, np.nanmean(tests, 0), tests)
    return best[1:]


def feat(name):
    f = f'feats/{name}.npy'
    return np.load(f).astype(np.float32) if os.path.exists(f) else None


sets = {}
Wh = np.load('feats/whisper.npy', mmap_mode='r')
sets['whisper_L32'] = Wh[:, 32].astype(np.float32)
sets['whisper_L24-32'] = Wh[:, 24:33].astype(np.float32).mean(1)
sets['w2vbert_L16'] = np.load('feats/w2vbert.npy', mmap_mode='r')[:, 16].astype(np.float32)
for nm in ['egemaps', 'compare', 'clap', 'e2v_feats', 'e2v_scores', 'omni_lp']:
    if (X := feat(nm)) is not None:
        sets[nm] = np.log(np.clip(X, 1e-6, None)) if nm == 'e2v_scores' else X
if (H := feat('omni_h')) is not None:
    miss = np.isnan(feat('omni_lp')).any(1) if feat('omni_lp') is not None else np.zeros(len(H), bool)
    for j, L in enumerate([14, 28]):
        X = H[:, j].copy(); X[miss] = np.nan; sets[f'omni_h{L}'] = X

print(f'no-twin clips: {NT.sum()} / {n}')
zs = {'clap_zs': feat('clap_zs'), 'e2v_zs': None if feat('e2v_scores') is None else feat('e2v_scores')[:, [0, 3, 4, 6]],
      'omni_zs': feat('omni_lp')}
for nm, Z in zs.items():
    if Z is not None:
        f, fn, _, k = score(np.exp(Z[:n]) if nm == 'omni_zs' else Z[:n])
        print(f'ZERO-SHOT {nm:12s} F1 all {f:.4f} | no-twin {fn:.4f} | rows {k}')

P = {}
for nm, X in sets.items():
    C, oof, test, tests = probe(X)
    P[nm] = (oof, test, tests)
    np.save(f'oof/t3_{nm}.npy', oof); np.save(f'oof/t3_{nm}_test.npy', test)
    f, fn, ll, k = score(oof)
    print(f'PROBE {nm:15s} C={C:<6} F1 all {f:.4f} | no-twin {fn:.4f} | logloss {ll:.3f} | rows {k}', flush=True)

# blends (equal-weight mean of probs) on rows every member covers
base = ['whisper_L32', 'w2vbert_L16']
def blend(names):
    return np.mean([P[m][0] for m in names], 0)
f, fn, ll, k = score(blend(base)); print(f'\nBLEND base {base}: all {f:.4f} | no-twin {fn:.4f} | ll {ll:.3f}')
for nm in P:
    if nm in base: continue
    f, fn, ll, k = score(blend(base + [nm]))
    print(f'  + {nm:15s}: all {f:.4f} | no-twin {fn:.4f} | ll {ll:.3f} | rows {k}')

# label spreading on train+test kNN graph (whisper L32), honest per fold
from sklearn.semi_supervised import LabelSpreading
Xs = StandardScaler().fit(sets['whisper_L32'][:n]).transform(sets['whisper_L32'])
oof = np.zeros((n, 4))
for k in range(5):
    lab = np.r_[np.where(fold != k, y, -1), -np.ones(len(Xs) - n, int)]
    ls = LabelSpreading(kernel='knn', n_neighbors=10, alpha=0.2, max_iter=100).fit(Xs, lab)
    oof[fold == k] = ls.label_distributions_[:n][fold == k]
f, fn, ll, k = score(oof); print(f'\nLABEL SPREADING whisper_L32 kNN10: all {f:.4f} | no-twin {fn:.4f}')
f, fn, ll, k = score(blend(base) * 0.75 + oof * 0.25); print(f'  base blend + 0.25 spreading: all {f:.4f} | no-twin {fn:.4f}')

# pseudo-labels: fold-k models' own test predictions (base blend), confident >= 0.9, added to fold-k training
def pl_extra_factory(Xfeat):
    def extra(k):
        pt = np.mean([P[m][2][k] for m in base], 0)
        keep = pt.max(1) >= 0.9
        return Xfeat[n:][keep], pt[keep].argmax(1)
    return extra
pl = []
for nm in base:
    C, oof, test, tests = probe(sets[nm], extra=pl_extra_factory(sets[nm])); pl.append(oof)
    f, fn, ll, k = score(oof); print(f'PSEUDO-LABEL {nm:12s}: all {f:.4f} | no-twin {fn:.4f}')
f, fn, ll, k = score(np.mean(pl, 0)); print(f'PSEUDO-LABEL base blend: all {f:.4f} | no-twin {fn:.4f} | ll {ll:.3f}')
