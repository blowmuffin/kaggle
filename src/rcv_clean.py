"""Spike L5 (pre-registered): nested label-noise cleaning. Usage: python src/rcv_clean.py <seed>
Per outer fold: inner 4-fold grouped CV on the training part with the plain A″-weighted blend scores every training clip;
clips whose inner-OOF prob of their own label < 0.10 are dropped; all 5 A″ members are refit on the kept clips (same C as the
uncleaned run) and predict the outer fold + test. -> oof/rcv_s{seed}_{member}_cl.npy (+ _test). Evaluation labels untouched."""
import sys, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedGroupKFold
import hcv
from hcv import y, n, group, f1

seed = int(sys.argv[1]); THR = 0.10; N = 1035
hcv.fold = f = np.load(f'oof/rcv_s{seed}_fold.npy')
W = np.load('feats/whisper.npy', mmap_mode='r'); V = np.load('feats/w2vbert.npy', mmap_mode='r'); A = np.load('feats/whisper_aug.npy', mmap_mode='r')
cat = lambda *a: np.hstack(a).astype(np.float32)
X = {'whisper_L24L32': cat(W[:, 24], W[:, 32]), 'w2vbert_L10L16': cat(V[:, 10], V[:, 16]),
     'omni_h': np.load('feats/omni_h.npy').reshape(N, -1).astype(np.float32), 'omni_lp': np.load('feats/omni_lp.npy'),
     'text_e5': np.load('feats/e5.npy')}
XA = [cat(A[:, v, 0], A[:, v, 1]) for v in range(4)]  # whisper aug views for the whisper_aug member
C = {'whisper_L24L32': 0.1, 'whisper_aug_L24L32': 0.01, 'w2vbert_L10L16': 0.01, 'omni_lp': 0.3,
     'omni_h': {42: 0.003, 7: 0.01, 11: 0.003}[seed]}
best = None  # text C exactly as rcv_members: grid by OOF F1 on this split
for c in (0.3, 1, 3, 10, 30):
    o = np.zeros((n, 4))
    for k in range(5):
        o[f == k] = LogisticRegression(C=c, max_iter=4000).fit(X['text_e5'][:n][f != k], y[f != k]).predict_proba(X['text_e5'][:n][f == k])
    if best is None or f1(o) > best[0]: best = (f1(o), c)
C['text_e5'] = best[1]


def fit(m, idx):
    """Fit member m on train rows idx; return predict(rows) -> probs."""
    if m == 'text_e5':
        lr = LogisticRegression(C=C[m], max_iter=4000).fit(X[m][idx], y[idx]); return lambda r: lr.predict_proba(X[m][r])
    if m == 'whisper_aug_L24L32':
        V5 = [X['whisper_L24L32']] + XA
        p = make_pipeline(StandardScaler(), LogisticRegression(C=C[m], max_iter=4000)).fit(np.vstack([Z[idx] for Z in V5]), np.tile(y[idx], 5))
        return lambda r: np.mean([p.predict_proba(Z[r]) for Z in V5], 0)
    p = make_pipeline(StandardScaler(), LogisticRegression(C=C[m], max_iter=4000)).fit(X[m][idx], y[idx])
    return lambda r: p.predict_proba(X[m][r])


blend = lambda P: 0.75 * (1 / 3 * (P['whisper_L24L32'] + P['w2vbert_L10L16']) + 1 / 6 * (P['omni_h'] + P['omni_lp'])) + 0.25 * P['text_e5']
MEMBERS = ['whisper_aug_L24L32', 'w2vbert_L10L16', 'omni_h', 'omni_lp', 'text_e5']
oof = {m: np.zeros((n, 4)) for m in MEMBERS}; test = {m: np.zeros((N - n, 4)) for m in MEMBERS}
for k in range(5):
    T, Va = np.where(f != k)[0], np.where(f == k)[0]
    inner = {m: np.zeros((len(T), 4)) for m in ['whisper_L24L32', 'w2vbert_L10L16', 'omni_h', 'omni_lp', 'text_e5']}
    for a, b in StratifiedGroupKFold(4, shuffle=True, random_state=seed).split(T, y[T], group[T]):
        for m in inner:
            inner[m][b] = fit(m, T[a])(T[b])
    keep = blend(inner)[np.arange(len(T)), y[T]] >= THR
    drop = T[~keep]
    print(f'seed {seed} fold {k}: drop {len(drop)}/{len(T)} | by label {np.bincount(y[drop], minlength=4)}', flush=True)
    for m in MEMBERS:
        pr = fit(m, T[keep]); oof[m][Va] = pr(Va); test[m] += pr(np.arange(n, N)) / 5
for m in MEMBERS:
    np.save(f'oof/rcv_s{seed}_{m}_cl.npy', oof[m]); np.save(f'oof/rcv_s{seed}_{m}_cl_test.npy', test[m])
    print(f'seed {seed} {m}_cl {f1(oof[m]):.4f}', flush=True)
