"""Fix F2 (pre-registered): full-data refit for the final test predictions. Every member of the recipe is refit on ALL 826 train
clips with the hyper-parameters its seed-42 run selected; the P3 head comes from `python src/attn_pool.py full text`.
Blend = A‴ weights (text role = mean(text_e5, text_e5q) if 'f3' is passed). Usage: python src/final_refit.py out.csv [f3]
Prints per-member argmax changes vs the 5-fold (seed-42) test predictions as a sanity check."""
import sys, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import hcv
from hcv import y, n, f1

out_csv, F3 = sys.argv[1], 'f3' in sys.argv[2:]
N = 1035; f = hcv.fold  # seed-42 folds (only used to re-select the text C exactly as rcv_members did)
cat = lambda *a: np.hstack(a).astype(np.float32)
W, V, A = (np.load(p, mmap_mode='r') for p in ('feats/whisper.npy', 'feats/w2vbert.npy', 'feats/whisper_aug.npy'))
lr = lambda C, scale=True: make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=4000)) if scale else LogisticRegression(C=C, max_iter=4000)


def full_fit(X, C, scale=True):
    return lr(C, scale).fit(X[:n], y).predict_proba(X[n:])


def text_full(E):
    best = None
    for C in (0.3, 1, 3, 10, 30):  # C by OOF F1 on the seed-42 folds, as rcv_members
        o = np.zeros((n, 4))
        for k in range(5):
            o[f == k] = lr(C, False).fit(E[:n][f != k], y[f != k]).predict_proba(E[:n][f == k])
        if best is None or f1(o) > best[0]: best = (f1(o), C)
    return full_fit(E, best[1], scale=False), best[1]


P = {}
views = [cat(W[:, 24], W[:, 32])] + [cat(A[:, v, 0], A[:, v, 1]) for v in range(4)]
m = lr(0.01).fit(np.vstack([X[:n] for X in views]), np.tile(y, 5))
P['whisper_aug_L24L32'] = np.mean([m.predict_proba(X[n:]) for X in views], 0)
P['w2vbert_L10L16'] = full_fit(cat(V[:, 10], V[:, 16]), 0.01)
P['omni_h'] = full_fit(np.load('feats/omni_h.npy').reshape(N, -1).astype(np.float32), 0.003)
P['omni_lp'] = full_fit(np.load('feats/omni_lp.npy'), 0.3)
P['text_e5'], c1 = text_full(np.load('feats/e5.npy'))
if F3: P['text_e5q'], c2 = text_full(np.load('feats/e5_qwen3.npy'))
P['whisper_attn_text'] = np.load('oof/full_whisper_attn_text_test.npy')
for k, p in P.items():
    ref = np.load(f'oof/rcv_s42_{k}_test.npy')
    print(f'{k:20s} full-fit vs 5-fold test argmax changes: {(p.argmax(1) != ref.argmax(1)).sum():3d} | mean |dp| {np.abs(p - ref).mean():.3f}')
audio = 1 / 3 * (P['whisper_aug_L24L32'] + P['w2vbert_L10L16']) + 1 / 6 * (P['omni_h'] + P['omni_lp'])
audio = 0.8 * audio + 0.2 * P['whisper_attn_text']
text = (P['text_e5'] + P['text_e5q']) / 2 if F3 else P['text_e5']
p = 0.75 * audio + 0.25 * text
C4 = np.array(['angry', 'happy', 'neutral', 'sad'])
te = pd.read_csv('feats/test_meta.csv'); ss = pd.read_csv('data/SRCASW-BhavVaani/sample_submission.csv')
ss[['Id']].merge(pd.DataFrame({'Id': te.Id, 'emotion': C4[p.argmax(1)]}), on='Id').to_csv(out_csv, index=False)
print('wrote', out_csv, '| text C', c1, *([c2] if F3 else []))
