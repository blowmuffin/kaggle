"""Twin lock: a test clip whose audio has a train twin (exact hash, or same length & whisper-L32 cosine > 0.9; 99.1%
label agreement in train) or whose transcript (>=3 words) exactly matches a train transcript (35/35 in train) takes
that train label. Audio twin wins over transcript. Usage: python src/twinlock.py in.csv out.csv"""
import sys, unicodedata, numpy as np, pandas as pd

tr = pd.read_csv('feats/train_meta.csv'); te = pd.read_csv('feats/test_meta.csv'); n = len(tr)
W = np.load('feats/whisper.npy', mmap_mode='r')[:, 32].astype(np.float32)
W = (W - W[:n].mean(0)) / (W[:n].std(0) + 1e-6); W /= np.linalg.norm(W, axis=1, keepdims=True)
S = W[n:] @ W[:n].T
S[np.abs(te.dur.values[:, None] - tr.dur.values[None, :]) >= 1e-3] = -1
S[te.h.values[:, None] == tr.h.values[None, :]] = 2  # exact copy
A = pd.read_csv('feats/asr.csv', encoding='utf-8')
clean = lambda s: ' '.join(''.join(' ' if unicodedata.category(ch)[0] in 'PS' else ch for ch in str(s)).split())
t = A.text.fillna('').map(clean).values; ttr, tte = t[:n], t[n:]


def lock_labels():
    lab, src = {}, {}
    for i in range(len(te)):
        if S[i].max() > 0.9:
            lab[te.Id[i]], src[te.Id[i]] = tr.emotion.values[S[i].argmax()], 'audio'
        elif len(tte[i].split()) >= 3 and (ttr == tte[i]).any():
            labs = tr.emotion.values[ttr == tte[i]]
            if len(set(labs)) == 1:
                lab[te.Id[i]], src[te.Id[i]] = labs[0], 'text'
    return lab, src


if __name__ == '__main__':
    sub = pd.read_csv(sys.argv[1]); lab, src = lock_labels()
    new = sub.emotion.copy()
    for k, (i, e) in enumerate(zip(sub.Id, sub.emotion)):
        if i in lab and lab[i] != e:
            print(f'  Id {i}: {e} -> {lab[i]} ({src[i]} twin)'); new[k] = lab[i]
    print(f'locked {len(lab)} test clips ({sum(v == "audio" for v in src.values())} audio, {sum(v == "text" for v in src.values())} text); changed {(new != sub.emotion).sum()}')
    sub.assign(emotion=new).to_csv(sys.argv[2], index=False)
