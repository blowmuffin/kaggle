"""Save probe OOF/test probs for chosen layers, then score simple blends."""
import itertools, numpy as np
from sklearn.metrics import f1_score
from probe import cv, y, ntr

PICK = {'whisper': [32, 29, (28, 32)], 'w2vbert': [16, 10], 'wavlm': [13, 15]}  # int = layer, tuple = mean of layer range
res = {}
for key, layers in PICK.items():
    F = np.load(f'feats/{key}.npy')
    for l in layers:
        X = F[:, l[0]:l[1] + 1].astype(np.float32).mean(1) if isinstance(l, tuple) else F[:, l].astype(np.float32)
        s, oof, test = cv(X, C=0.03)
        name = f'{key}_L{l if isinstance(l, int) else "%d-%d" % l}'
        np.save(f'oof/probe_{name}.npy', oof); np.save(f'oof/probe_{name}_test.npy', test)
        res[name] = oof
        print(f'{name}: {s:.4f}', flush=True)
names = list(res)
for r in (2, 3):
    best = sorted(((f1_score(y, np.mean([res[n] for n in c], 0).argmax(1), average='macro'), c)
                   for c in itertools.combinations(names, r)), reverse=True)[:4]
    for s, c in best:
        print(f'blend {r}: {s:.4f} {c}')
