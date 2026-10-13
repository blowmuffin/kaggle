"""Cheap checks G1-G4 (pre-registered in EXPERIMENTS.md, 12 Oct) vs A‴ on the repeated-CV splits 42/7/11.
Builds the G2/G3/G4 members (skips existing files), then prints gains, folds up and test flips vs A‴.
Usage: python src/rcv_cheap.py"""
import os, numpy as np, torch
import torch.nn.functional as TF
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import log_loss, f1_score
from hcv import y, n
from rcv_decide import S, L, F, full, BASE, A3

N = 1035
W, A, V = (np.load(p, mmap_mode='r') for p in ('feats/whisper.npy', 'feats/whisper_aug.npy', 'feats/w2vbert.npy'))
XW = [np.hstack([W[:, 24], W[:, 32]]).astype(np.float32)] + [np.hstack([A[:, v, 0], A[:, v, 1]]).astype(np.float32) for v in range(4)]
XV = [np.hstack([V[:, 10], V[:, 16]]).astype(np.float32)]
CS = (0.003, 0.01, 0.03, 0.1, 0.3)


def views_fit(views, f, make, Cs=(None,)):
    """Train on all views of the train fold, predict the mean over views; param by OOF log-loss (as rcv_spikes.aug_probe)."""
    best = None
    for C in Cs:
        o = np.zeros((n, 4)); t = np.zeros((N - n, 4))
        for k in range(5):
            tri, vai = np.where(f != k)[0], np.where(f == k)[0]
            m = make(C).fit(np.vstack([X[tri] for X in views]), np.tile(y[tri], len(views)))
            pr = lambda idx: np.mean([m.predict_proba(X[idx]) for X in views], 0)
            o[vai] = pr(vai); t += pr(np.arange(n, N)) / 5
        if best is None or log_loss(y, o) < best[0]: best = (log_loss(y, o), o, t)
    return best[1:]


def global_std(views):  # G3: label-free standardisation with all 1035 clips' original features
    mu, sd = views[0].mean(0), views[0].std(0) + 1e-6
    return [(X - mu) / sd for X in views]


WL = torch.tensor(np.asarray(W, np.float32))  # [N, 33, 2560]


def layer_weights(f):  # G4: softmax layer weights + linear head, full-batch AdamW
    o = np.zeros((n, 4)); t = np.zeros((N - n, 4))
    for k in range(5):
        tri, vai = np.where(f != k)[0], np.where(f == k)[0]
        mu, sd = WL[tri].mean(0), WL[tri].std(0) + 1e-6
        Z = lambda idx: (WL[idx] - mu) / sd
        torch.manual_seed(0)
        w = torch.zeros(33, requires_grad=True); head = torch.nn.Linear(2560, 4)
        opt = torch.optim.AdamW([w, *head.parameters()], lr=1e-3, weight_decay=1e-2)
        fwd = lambda X: head((torch.softmax(w, 0)[None, :, None] * X).sum(1))
        Xt, yt = Z(tri), torch.tensor(y[tri])
        for _ in range(200):
            opt.zero_grad(); TF.cross_entropy(fwd(Xt), yt).backward(); opt.step()
        with torch.no_grad():
            pr = lambda idx: torch.softmax(fwd(Z(idx)), 1).numpy()
            o[vai] = pr(vai); t += pr(np.arange(n, N)) / 5
    return o, t


lda = lambda C: make_pipeline(StandardScaler(), LDA(solver='lsqr', shrinkage='auto'))
lr = lambda C: LogisticRegression(C=C, max_iter=4000)
BUILD = {'whisper_aug_lda': lambda f: views_fit(XW, f, lda), 'w2vbert_L10L16_lda': lambda f: views_fit(XV, f, lda),
         'whisper_aug_gs': lambda f: views_fit(global_std(XW), f, lr, CS), 'w2vbert_L10L16_gs': lambda f: views_fit(global_std(XV), f, lr, CS),
         'whisper_lw': layer_weights}
for s in S:
    f = np.load(f'oof/rcv_s{s}_fold.npy')
    for m, build in BUILD.items():
        if os.path.exists(f'oof/rcv_s{s}_{m}_test.npy'): continue
        o, t = build(f)
        np.save(f'oof/rcv_s{s}_{m}.npy', o); np.save(f'oof/rcv_s{s}_{m}_test.npy', t)
        print(f'seed {s} {m:22s} {F(o):.4f}', flush=True)

ROLES = ('whisper', 'w2v', 'omni_h', 'omni_lp', 'text', 'extra')


def blend(s, test=False, **kw):
    sfx = '_test' if test else ''
    return full(s, **{k: (v + sfx if v else None) for k, v in {**BASE, **kw}.items()})


def vote(s, test=False):  # G1: plurality of the 6 A‴ members; ties broken by the A‴ soft blend (soft < 1 never beats a full vote)
    sfx = '_test' if test else ''
    r = {**BASE, **A3}
    P = [L(s, r[k] + sfx) for k in ROLES]
    return sum(np.eye(4)[p.argmax(1)] for p in P) + blend(s, test, **A3)


CANDS = [('G1 hard majority vote', vote), ('G2 shrinkage-LDA heads', dict(whisper='whisper_aug_lda', w2v='w2vbert_L10L16_lda')),
         ('G3 transductive scaling', dict(whisper='whisper_aug_gs', w2v='w2vbert_L10L16_gs')), ('G4 + learned layer weights (0.2)', dict(extra2='whisper_lw'))]
print('A‴ per split ' + ' '.join(f'{F(blend(s, **A3)):.4f}' for s in S))
for name, c in CANDS:
    get = (lambda s, test=False: c(s, test)) if callable(c) else (lambda s, test=False: blend(s, test, **{**A3, **c}))
    g, folds = [], []
    for s in S:
        a, b, fo = get(s), blend(s, **A3), np.load(f'oof/rcv_s{s}_fold.npy')
        g.append(F(a) - F(b))
        fk = lambda p, k: f1_score(y[fo == k], p[fo == k].argmax(1), average='macro')
        folds.append(sum(fk(a, k) > fk(b, k) for k in range(5)))
    flips = int((get(42, True).argmax(1) != blend(42, True, **A3).argmax(1)).sum())
    ok = np.mean(g) >= 0.005 and min(g) > 0 and flips <= 3
    print(f'{name:34s} gains ' + ' '.join(f'{x:+.4f}' for x in g) + f' | mean {np.mean(g):+.4f} | folds up {folds} | test flips {flips} | ADOPT: {ok}')
