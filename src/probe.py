"""Linear probe per layer on frozen features, shared 5-fold group CV.
Usage: python src/probe.py wavlm [whisper ...]  -> prints per-layer OOF macro-F1, saves best-layer OOF/test probs."""
import sys, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

CLASSES = ['angry', 'happy', 'neutral', 'sad']
tr = pd.read_csv('feats/train_meta.csv')
y = tr.emotion.map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
ntr = len(tr)
try:
    fold = pd.read_csv('feats/folds.csv').fold.to_numpy()
except FileNotFoundError:  # duplicates (same audio hash) share a fold
    fold = np.zeros(ntr, int)
    for k, (_, v) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(tr, y, tr.h)):
        fold[v] = k
    pd.DataFrame({'Id': tr.Id, 'fold': fold}).to_csv('feats/folds.csv', index=False)


def cv(X, C=0.5):
    oof = np.zeros((ntr, 4)); test = np.zeros((len(X) - ntr, 4))
    for k in range(5):
        m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))
        m.fit(X[:ntr][fold != k], y[fold != k])
        oof[fold == k] = m.predict_proba(X[:ntr][fold == k])
        test += m.predict_proba(X[ntr:]) / 5
    return f1_score(y, oof.argmax(1), average='macro'), oof, test


if __name__ == '__main__':
    for key in sys.argv[1:]:
        F = np.load(f'feats/{key}.npy').astype(np.float32)
        scores = [cv(F[:, l], C=0.01)[0] for l in range(F.shape[1])]
        print(key, 'per-layer F1:', ' '.join(f'{l}:{s:.3f}' for l, s in enumerate(scores)))
        best = int(np.argmax(scores))
        for C in [0.003, 0.01, 0.03, 0.1]:
            print(f'  layer {best} C={C}: {cv(F[:, best], C)[0]:.4f}')
