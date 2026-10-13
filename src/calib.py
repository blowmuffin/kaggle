"""Fix F1 (pre-registered): per-member temperature calibration before blending. For every A‴ member and split seed:
T fitted by log-loss on the OTHER four folds' OOF (leave-one-fold-out) and applied to the held-out fold; test probs use T fitted
on all OOF. p_cal ∝ p^(1/T). -> oof/rcv_s{seed}_{member}_cal.npy (+ _test). Usage: python src/calib.py"""
import numpy as np
from scipy.optimize import minimize_scalar
from hcv import y

MEMBERS = ['whisper_aug_L24L32', 'w2vbert_L10L16', 'omni_h', 'omni_lp', 'text_e5', 'whisper_attn_text']


def temp(P, Y):
    lp = np.log(np.clip(P, 1e-9, 1))
    def nll(logT):
        z = lp / np.exp(logT); z -= z.max(1, keepdims=True)
        return -(z[np.arange(len(Y)), Y] - np.log(np.exp(z).sum(1))).mean()
    return np.exp(minimize_scalar(nll, bounds=(-3, 3), method='bounded').x)


def apply(P, T):
    z = np.log(np.clip(P, 1e-9, 1)) / T; z = np.exp(z - z.max(1, keepdims=True))
    return z / z.sum(1, keepdims=True)


for s in (42, 7, 11):
    f = np.load(f'oof/rcv_s{s}_fold.npy')
    for m in MEMBERS:
        P, Pt = np.load(f'oof/rcv_s{s}_{m}.npy'), np.load(f'oof/rcv_s{s}_{m}_test.npy')
        out = np.zeros_like(P)
        for k in range(5):
            out[f == k] = apply(P[f == k], temp(P[f != k], y[f != k]))
        T = temp(P, y)
        np.save(f'oof/rcv_s{s}_{m}_cal.npy', out); np.save(f'oof/rcv_s{s}_{m}_cal_test.npy', apply(Pt, T))
        print(f'seed {s} {m:20s} T(all OOF) {T:.3f}  conf {out.max(1).mean():.3f} vs acc {(out.argmax(1) == y).mean():.3f}', flush=True)
