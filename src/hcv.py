"""Shared honest CV (v2): near-duplicate groups never cross folds. Import: from hcv import *
Builds feats/folds_v2.csv on first use (groups = exact hash OR [same length AND whisper-L32 cosine > 0.9])."""
import os, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold

C4 = np.array(['angry', 'happy', 'neutral', 'sad'])
tr = pd.read_csv('feats/train_meta.csv'); te = pd.read_csv('feats/test_meta.csv'); n = len(tr)
y = tr.emotion.map({c: i for i, c in enumerate(C4)}).to_numpy()


def _groups():
    W = np.load('feats/whisper.npy', mmap_mode='r')[:n, 32].astype(np.float32)
    W = (W - W.mean(0)) / (W.std(0) + 1e-6); W /= np.linalg.norm(W, axis=1, keepdims=True)
    S = W @ W.T
    same_len = np.abs(tr.dur.values[:, None] - tr.dur.values[None, :]) < 1e-3
    A = ((S > 0.9) & same_len) | (tr.h.values[:, None] == tr.h.values[None, :])
    parent = list(range(n))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for i, j in zip(*np.nonzero(np.triu(A, 1))):
        parent[find(i)] = find(j)
    return np.array([find(i) for i in range(n)])


if not os.path.exists('feats/folds_v2.csv'):
    g = _groups()
    f = np.zeros(n, int)
    for k, (_, v) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(tr, y, g)):
        f[v] = k
    pd.DataFrame({'Id': tr.Id, 'group': g, 'fold': f}).to_csv('feats/folds_v2.csv', index=False)
_fv = pd.read_csv('feats/folds_v2.csv')
fold, group = _fv.fold.to_numpy(), _fv.group.to_numpy()


def f1(p, m=None):
    m = np.ones(n, bool) if m is None else m
    return f1_score(y[m], p[m].argmax(1), average='macro')


def per_fold(p, ref):
    return [f1(p, fold == k) - f1(ref, fold == k) for k in range(5)]


def probe(X, Cs=(0.003, 0.01, 0.03, 0.1, 0.3), w=None):
    """Standardise + LR, C by OOF log-loss. w = optional per-train-clip LR sample weights. Returns (C, oof, test)."""
    best = None
    for C in Cs:
        oof = np.zeros((n, 4)); test = np.zeros((len(X) - n, 4))
        for k in range(5):
            kw = {} if w is None else {'logisticregression__sample_weight': w[fold != k]}
            m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=4000)).fit(X[:n][fold != k], y[fold != k], **kw)
            oof[fold == k] = m.predict_proba(X[:n][fold == k]); test += m.predict_proba(X[n:]) / 5
        ll = log_loss(y, oof)
        if best is None or ll < best[0]:
            best = (ll, C, oof, test)
    return best[1:]
