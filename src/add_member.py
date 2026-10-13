"""Evaluate a new frozen-feature member. Usage: python src/add_member.py <name> <feats.npy> [layer_idx ...]
Probe each listed layer (or the whole array if 2-D). PRE-REGISTERED decision: the LAST listed layer, added as
audio' = 0.8·audio + 0.2·X inside full = 0.75·audio' + 0.25·text_e5. Gate: ≥ +0.005 and ≥3/5 folds up."""
import sys, numpy as np
from hcv import *

name, path, layers = sys.argv[1], sys.argv[2], [int(a) for a in sys.argv[3:]]
F = np.load(path, mmap_mode='r')
L = lambda m, t='': np.load(f'oof/v2_{m}{t}.npy')
audio = lambda t='': 1 / 3 * (L('whisper_L32', t) + L('w2vbert_L16', t)) + 1 / 6 * (L('omni_h', t) + L('omni_lp', t))
cur = 0.75 * audio() + 0.25 * L('text_e5')
print(f'current full blend {f1(cur):.4f}')
for li in (layers or [None]):
    X = np.asarray(F if li is None else F[:, li], dtype=np.float32)
    C, oof, test = probe(X)
    tag = f'{name}' + ('' if li is None else f'_{li}')
    np.save(f'oof/v2_{tag}.npy', oof); np.save(f'oof/v2_{tag}_test.npy', test)
    full = 0.75 * (0.8 * audio() + 0.2 * oof) + 0.25 * L('text_e5'); d = per_fold(full, cur)
    pre = ' <- PRE-REGISTERED' if li == (layers[-1] if layers else None) else ''
    print(f'{tag:16s} C={C:<6} alone {f1(oof):.4f} | full blend {f1(full):.4f} ({f1(full) - f1(cur):+.4f}) folds up {sum(x > 0 for x in d)}/5 '
          + ' '.join(f'{x:+.3f}' for x in d) + pre, flush=True)
