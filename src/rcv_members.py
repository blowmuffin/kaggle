"""Repeated-CV member training for one fold split. Usage: python src/rcv_members.py <seed>   (42 = hcv v2 folds)
Trains every member probe on near-dup-grouped folds of that seed; saves oof/rcv_s{seed}_{member}.npy (+ _test)."""
import sys, numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
import hcv
from hcv import y, n, group, tr, f1, probe

seed = int(sys.argv[1])
if seed != 42:
    f = np.zeros(n, int)
    for k, (_, v) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(tr, y, group)):
        f[v] = k
    hcv.fold = f
f = hcv.fold
W = np.load('feats/whisper.npy', mmap_mode='r'); V = np.load('feats/w2vbert.npy', mmap_mode='r'); W2 = np.load('feats/whisperv2.npy', mmap_mode='r')
X = {'whisper_L24L32': np.hstack([W[:, 24], W[:, 32]]), 'w2vbert_L16': V[:, 16], 'w2vbert_L10L16': np.hstack([V[:, 10], V[:, 16]]),
     'omni_h': np.load('feats/omni_h.npy').reshape(1035, -1), 'omni_lp': np.load('feats/omni_lp.npy'),
     'whisperv2_L24L32': np.hstack([W2[:, 24], W2[:, 32]])}
for m, Z in X.items():
    C, oof, test = probe(np.asarray(Z, dtype=np.float32))
    np.save(f'oof/rcv_s{seed}_{m}.npy', oof); np.save(f'oof/rcv_s{seed}_{m}_test.npy', test)
    print(f'seed {seed} {m:18s} C={C:<6} {f1(oof):.4f}', flush=True)
E5 = np.load('feats/e5.npy'); best = None
for C in (0.3, 1, 3, 10, 30):
    o = np.zeros((n, 4)); t = np.zeros((1035 - n, 4))
    for k in range(5):
        lr = LogisticRegression(C=C, max_iter=4000).fit(E5[:n][f != k], y[f != k])
        o[f == k] = lr.predict_proba(E5[:n][f == k]); t += lr.predict_proba(E5[n:]) / 5
    if best is None or f1(o) > best[0]: best = (f1(o), o, t)
np.save(f'oof/rcv_s{seed}_text_e5.npy', best[1]); np.save(f'oof/rcv_s{seed}_text_e5_test.npy', best[2])
np.save(f'oof/rcv_s{seed}_fold.npy', f)
print(f'seed {seed} text_e5 {best[0]:.4f} | DONE', flush=True)
