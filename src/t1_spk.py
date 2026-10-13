"""C: speaker clusters from WavLM-SV x-vectors; per-speaker feature normalisation (honest v2 folds)."""
import numpy as np, pandas as pd
from sklearn.cluster import AgglomerativeClustering
from hcv import *

E = np.load('feats/spk.npy'); E = E - E.mean(0); E = E / np.linalg.norm(E, axis=1, keepdims=True)  # centre first
N = len(E)
for thr in (0.4, 0.5, 0.6, 0.7, 0.8):
    lab = AgglomerativeClustering(n_clusters=None, distance_threshold=thr, metric='cosine', linkage='average').fit_predict(E)
    same_twin = np.mean([lab[i] == lab[j] for i, j in zip(*np.nonzero(np.triu(group[:, None] == group[None, :], 1)))])
    sizes = np.bincount(lab)
    emo_cov = pd.crosstab(lab[:n], tr.emotion).gt(0).sum(1)
    print(f'thr {thr}: {lab.max() + 1} clusters | sizes>=10: {(sizes >= 10).sum()} | twins same cluster {same_twin:.2f} | '
          f'clusters(>=10 train clips) covering all 4 emotions {(emo_cov[np.bincount(lab[:n], minlength=lab.max() + 1)[emo_cov.index] >= 10] == 4).mean():.2f}')
    np.save(f'feats/spkc_{thr}.npy', lab)

O = {m: np.load(f'oof/v2_{m}.npy') for m in ['whisper_L32', 'w2vbert_L16', 'omni_h', 'omni_lp']}
cur = 1 / 3 * (O['whisper_L32'] + O['w2vbert_L16']) + 1 / 6 * (O['omni_h'] + O['omni_lp'])
W = np.load('feats/whisper.npy', mmap_mode='r')[:, 32].astype(np.float32)
V = np.load('feats/w2vbert.npy', mmap_mode='r')[:, 16].astype(np.float32)
print(f'\ncurrent best (base+omni) honest F1 {f1(cur):.4f}; whisper alone {f1(O["whisper_L32"]):.4f}, w2vbert alone {f1(O["w2vbert_L16"]):.4f}')
for thr in (0.4, 0.6, 0.8):
    lab = np.load(f'feats/spkc_{thr}.npy')
    def spk_norm(X):  # subtract the speaker-cluster mean (label-free, uses train+test clips)
        Z = (X - X.mean(0)) / (X.std(0) + 1e-6); M = np.zeros_like(Z)
        for c in np.unique(lab):
            if (lab == c).sum() >= 5:  # tiny clusters: a singleton's mean is itself → would zero the clip
                M[lab == c] = Z[lab == c].mean(0)
        return Z - M
    res = {}
    for nm, X in [('whisper_L32', W), ('w2vbert_L16', V)]:
        for mode, Xn in [('norm', spk_norm(X)), ('raw+norm', np.hstack([X, spk_norm(X)]))]:
            C, oof, test = probe(Xn, Cs=(0.003, 0.01, 0.03, 0.1)); res[(nm, mode)] = (oof, test)
            d = per_fold(oof, O[nm])
            print(f'thr {thr} {nm:12s} {mode:9s} C={C:<6} F1 {f1(oof):.4f} ({f1(oof) - f1(O[nm]):+.4f} vs raw) folds up {sum(x > 0 for x in d)}/5', flush=True)
    for mode in ('norm', 'raw+norm'):
        b = 1 / 3 * (res[('whisper_L32', mode)][0] + res[('w2vbert_L16', mode)][0]) + 1 / 6 * (O['omni_h'] + O['omni_lp'])
        d = per_fold(b, cur)
        print(f'  -> blend with spk-{mode}: {f1(b):.4f} ({f1(b) - f1(cur):+.4f}) folds up {sum(x > 0 for x in d)}/5 ' + ' '.join(f'{x:+.3f}' for x in d))
        np.save(f'oof/v2_spk{thr}_{mode}_blend.npy', b)
        np.save(f'oof/v2_spk{thr}_{mode}_blend_test.npy', 1 / 3 * (res[('whisper_L32', mode)][1] + res[('w2vbert_L16', mode)][1]) + 1 / 6 * (np.load('oof/v2_omni_h_test.npy') + np.load('oof/v2_omni_lp_test.npy')))
