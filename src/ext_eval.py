"""External-data evaluation. 1) overlap check vs competition clips (exclude any ext clip near a TEST clip — rule 4b).
2) add ext clips to the TRAINING side of each honest v2 fold; score on competition folds only.
Pre-registered decision variant: all languages, weight 1.0. Others = robustness only."""
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import log_loss
from hcv import *

M = pd.read_csv('feats/ext_meta.csv')
EW, EV = np.load('feats/ext_whisper_L32.npy'), np.load('feats/ext_w2vbert_L16.npy')
CW = np.load('feats/whisper.npy', mmap_mode='r')[:, 32].astype(np.float32)
CV_ = np.load('feats/w2vbert.npy', mmap_mode='r')[:, 16].astype(np.float32)
mu, sd = CW[:n].mean(0), CW[:n].std(0) + 1e-6
nz = lambda X: (lambda Z: Z / np.linalg.norm(Z, axis=1, keepdims=True))((X - mu) / sd)
S = nz(EW) @ nz(CW).T
dur_c = np.r_[tr.dur.values, te.dur.values]
same = np.abs(M.dur.values[:, None] - dur_c[None, :]) < 0.02
near_te = ((S[:, n:] > 0.9) | ((S[:, n:] > 0.8) & same[:, n:])).any(1)
near_tr = ((S[:, :n] > 0.9) | ((S[:, :n] > 0.8) & same[:, :n])).any(1)
print(f'ext clips: {len(M)} | max cosine to any competition clip: q50 {np.median(S.max(1)):.3f} q99 {np.quantile(S.max(1), .99):.3f} max {S.max():.3f}')
print(f'ext clips near a competition TEST clip: {near_te.sum()} | near a TRAIN clip: {near_tr.sum()}')
keep = ~near_te & ~near_tr
ye = pd.Series(M.emotion).map({c: i for i, c in enumerate(C4)}).to_numpy()


def probe_ext(Xc, Xe, C, sel, w):
    oof = np.zeros((n, 4)); test = np.zeros((len(Xc) - n, 4))
    for k in range(5):
        Xt = np.vstack([Xc[:n][fold != k], Xe[sel]]); yt = np.r_[y[fold != k], ye[sel]]
        sw = np.r_[np.ones((fold != k).sum()), np.full(sel.sum(), w)]
        m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=4000)).fit(Xt, yt, logisticregression__sample_weight=sw)
        oof[fold == k] = m.predict_proba(Xc[:n][fold == k]); test += m.predict_proba(Xc[n:]) / 5
    return oof, test


L = lambda m, t='': np.load(f'oof/v2_{m}{t}.npy')
cur = 0.75 * (1 / 3 * (L('whisper_L32') + L('w2vbert_L16')) + 1 / 6 * (L('omni_h') + L('omni_lp'))) + 0.25 * L('text_e5')
print(f'\ncurrent best full blend: {f1(cur):.4f} | whisper probe {f1(L("whisper_L32")):.4f} | w2vbert probe {f1(L("w2vbert_L16")):.4f}')
variants = [('ALL w1.0 (pre-registered)', keep, 1.0), ('ALL w0.5', keep, 0.5), ('HINDI w1.0', keep & (M.lang == 'hindi').values, 1.0)]
for name, sel, w in variants:
    ow, tw = probe_ext(CW, EW, 0.03, sel, w); ov, tv = probe_ext(CV_, EV, 0.01, sel, w)
    full = 0.75 * (1 / 3 * (ow + ov) + 1 / 6 * (L('omni_h') + L('omni_lp'))) + 0.25 * L('text_e5')
    d = per_fold(full, cur)
    print(f'{name:26s} n_ext {sel.sum():3d} | whisper {f1(ow):.4f} ({f1(ow) - f1(L("whisper_L32")):+.4f}) w2vbert {f1(ov):.4f} ({f1(ov) - f1(L("w2vbert_L16")):+.4f}) '
          f'| full blend {f1(full):.4f} ({f1(full) - f1(cur):+.4f}) folds up {sum(x > 0 for x in d)}/5 ' + ' '.join(f'{x:+.3f}' for x in d), flush=True)
    if 'pre-registered' in name:
        np.save('oof/v2_ext_whisper.npy', ow); np.save('oof/v2_ext_whisper_test.npy', tw)
        np.save('oof/v2_ext_w2vbert.npy', ov); np.save('oof/v2_ext_w2vbert_test.npy', tv)
