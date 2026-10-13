"""emotion2vec+ large (FunASR model licence: free use with attribution) utterance embeddings + 9-class scores.
Kaggle CPU kernel (no GPU quota). Outputs e2v_feats.npy [N,1024], e2v_scores.npy [N,9]; rows = train.csv then test.csv.
Classes: 0 angry 1 disgusted 2 fearful 3 happy 4 neutral 5 other 6 sad 7 surprised 8 unknown."""
import os, sys, subprocess, time, glob
T0 = time.time()
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'funasr'], check=False)
import numpy as np, pandas as pd, torch
from funasr import AutoModel

root = os.path.dirname(sorted(glob.glob('/kaggle/input/**/train.csv', recursive=True))[0])
tr = pd.read_csv(f'{root}/train.csv'); te = pd.read_csv(f'{root}/test.csv')
paths = [f'{root}/train/{f}' for f in tr.filename] + [f'{root}/test/{f}' for f in te.filename]
torch.set_num_threads(os.cpu_count())
model = AutoModel(model='emotion2vec/emotion2vec_plus_large', hub='hf', disable_update=True, device='cpu')
feats = np.zeros((len(paths), 1024), np.float32); scores = np.zeros((len(paths), 9), np.float32)
for i, p in enumerate(paths):
    r = model.generate(p, granularity='utterance', extract_embedding=True)[0]
    feats[i] = np.asarray(r['feats']).reshape(-1)[:1024]; scores[i] = r['scores']
    if i % 100 == 0:
        print(i, len(paths), f'{time.time() - T0:.0f}s', r['labels'][int(np.argmax(r['scores']))], flush=True)
np.save('/kaggle/working/e2v_feats.npy', feats); np.save('/kaggle/working/e2v_scores.npy', scores)
C = ['angry', 'happy', 'neutral', 'sad']
y = tr.emotion.map({c: k for k, c in enumerate(C)}).to_numpy()
from sklearn.metrics import f1_score
zs = scores[:len(tr)][:, [0, 3, 4, 6]].argmax(1)
print(f'zero-shot 4-class macro-F1 (train): {f1_score(y, zs, average="macro"):.4f} | total {time.time() - T0:.0f}s', flush=True)
