"""Spike members (S1-S3b, pre-registered in EXPERIMENTS.md) on the saved repeated-CV splits.
Usage: python src/rcv_spikes.py <seed> s1|s2|s3|s5|r1|r2|l1|l2|l4|p1|p2|omni2|k1|k3|f3   -> oof/rcv_s{seed}_{member}.npy (+ _test); decide with src/rcv_decide.py"""
import sys, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import log_loss
import hcv
from hcv import y, n, group, f1, probe

seed, spike = int(sys.argv[1]), sys.argv[2]
hcv.fold = f = np.load(f'oof/rcv_s{seed}_fold.npy')
N = 1035


def save(m, C, o, t):
    np.save(f'oof/rcv_s{seed}_{m}.npy', o); np.save(f'oof/rcv_s{seed}_{m}_test.npy', t)
    print(f'seed {seed} {m:24s} C={C:<6} {f1(o):.4f}', flush=True)


cat = lambda A, i, j: np.hstack([A[:, i], A[:, j]]).astype(np.float32)

def aug_probe(views, Cs=(0.003, 0.01, 0.03, 0.1, 0.3)):
    """views[0] = original features, rest = augmented views [N, d]. Train on all views of the train fold, predict the mean
    over views; C by OOF log-loss. Returns (C, oof, test)."""
    best = None
    for C in Cs:
        o = np.zeros((n, 4)); t = np.zeros((N - n, 4))
        for k in range(5):
            tri, vai = np.where(f != k)[0], np.where(f == k)[0]
            m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=4000))
            m.fit(np.vstack([X[tri] for X in views]), np.tile(y[tri], len(views)))
            pr = lambda idx: np.mean([m.predict_proba(X[idx]) for X in views], 0)
            o[vai] = pr(vai); t += pr(np.arange(n, N)) / 5
        if best is None or log_loss(y, o) < best[0]: best = (log_loss(y, o), C, o, t)
    return best[1:]

W = np.load('feats/whisper.npy', mmap_mode='r')

if spike == 's1':  # weight 1/|near-dup group|: every recording counts once
    sw = 1 / np.bincount(group)[group]
    V = np.load('feats/w2vbert.npy', mmap_mode='r')
    X = {'whisper_L24L32': cat(W, 24, 32), 'w2vbert_L16': np.asarray(V[:, 16], np.float32),
         'omni_h': np.load('feats/omni_h.npy').reshape(N, -1).astype(np.float32), 'omni_lp': np.load('feats/omni_lp.npy')}
    for m, Z in X.items():
        save(m + '_gw', *probe(Z, w=sw))
    E5 = np.load('feats/e5.npy'); best = None
    for C in (0.3, 1, 3, 10, 30):  # text probe as in rcv_members (no scaler, C by F1)
        o = np.zeros((n, 4)); t = np.zeros((N - n, 4))
        for k in range(5):
            lr = LogisticRegression(C=C, max_iter=4000).fit(E5[:n][f != k], y[f != k], sample_weight=sw[f != k])
            o[f == k] = lr.predict_proba(E5[:n][f == k]); t += lr.predict_proba(E5[n:]) / 5
        if best is None or f1(o) > best[1]: best = (C, f1(o), o, t)
    save('text_e5_gw', best[0], best[2], best[3])

elif spike == 's2':  # padding-length TTA members (30-s member already exists)
    for p in (12, 20):
        P = np.load(f'feats/whisper_pad{p}.npy', mmap_mode='r')  # layers 16, 24, 32
        save(f'whisper_pad{p}_L24L32', *probe(cat(P, 1, 2)))

elif spike == 's3':  # S3: train on original + 4 aug views, predict mean over 5; S3b: loop-pad probe
    A = np.load('feats/whisper_aug.npy', mmap_mode='r')  # [N, views(noise20, noise30, shift25, shift50, loop), layers(24, 32), 2560]
    X0 = cat(W, 24, 32)
    XA = [np.hstack([A[:, v, 0], A[:, v, 1]]).astype(np.float32) for v in range(4)]
    save('whisper_loop_L24L32', *probe(np.hstack([A[:, 4, 0], A[:, 4, 1]]).astype(np.float32)))
    save('whisper_aug_L24L32', *aug_probe([X0] + XA))

elif spike == 's5':  # Qwen2-Audio-7B full LLM (kernels/dataverse-qa7b outputs downloaded to runs/qa7b/)
    H, P = np.load('runs/qa7b/qa7b_h.npy'), np.load('runs/qa7b/qa7b_lp.npy')  # kernel dataverse-qa7b
    assert not np.isnan(P).any(), 'kernel did not process every clip'
    save('qa7b_h', *probe(H.reshape(N, -1).astype(np.float32))); save('qa7b_lp', *probe(P[:, :2].reshape(N, -1)))  # S5: prompts 0-1
    save('qa7b_lp3', *probe(P.reshape(N, -1)))  # R3: + acoustic-caption prompt

elif spike == 'r1':  # Dasheng-1.2B: concat of block 30/40 and last block (feats/dasheng.npy from src/dasheng_extract.py)
    D = np.load('feats/dasheng.npy')
    save('dasheng', *probe(D.reshape(N, -1).astype(np.float32)))

elif spike == 'r2':  # Step-Audio-2-mini (kernels/dataverse-sa2 outputs downloaded to runs/sa2/)
    H, P = np.load('runs/sa2/sa2_h.npy'), np.load('runs/sa2/sa2_lp.npy')
    assert not np.isnan(P).any(), 'kernel did not process every clip'
    save('sa2_h', *probe(H.reshape(N, -1).astype(np.float32))); save('sa2_lp', *probe(P.reshape(N, -1)))

elif spike == 'l1':  # w2v-BERT L10+L16 trained on original + 4 S3 views, mean over 5 at prediction
    V = np.load('feats/w2vbert.npy', mmap_mode='r'); VA = np.load('feats/w2vbert_aug.npy', mmap_mode='r')
    save('w2vbert_aug_L10L16', *aug_probe([cat(V, 10, 16)] + [np.hstack([VA[:, v, 0], VA[:, v, 1]]).astype(np.float32) for v in range(4)]))

elif spike == 'l2':  # whisper L24+L32 on original + 4 S3 views + 4 speed/pitch views, fixed C=0.01
    A, A2 = np.load('feats/whisper_aug.npy', mmap_mode='r'), np.load('feats/whisper_aug2.npy', mmap_mode='r')
    vw = [cat(W, 24, 32)] + [np.hstack([B[:, v, 0], B[:, v, 1]]).astype(np.float32) for B in (A, A2) for v in range(4)]
    save('whisper_aug8_L24L32', *aug_probe(vw, Cs=(0.01,)))

elif spike == 'l4':  # segment pooling: mean+std over the whole clip and each third, L24 and L32
    Fr, off = np.load('feats/whisper_frames.npy', mmap_mode='r'), np.load('feats/whisper_frames_off.npy')
    def seg(i):
        h = np.asarray(Fr[off[i]:off[i + 1]], np.float32).reshape(off[i + 1] - off[i], -1)  # [T, 2*1280]
        parts = [h] + [p if len(p) else h for p in np.array_split(h, 3)]
        return np.concatenate([np.concatenate([p.mean(0), p.std(0)]) for p in parts])
    save('whisper_seg', *probe(np.stack([seg(i) for i in range(N)]).astype(np.float32)))

elif spike == 'p2':  # kNN interpolation (Wang 2023, frozen): 0.7 * whisper_aug probe + 0.3 * distance-weighted 10-NN vote
    Z = cat(W, 24, 32); kn = np.zeros((n, 4)); kt = np.zeros((N - n, 4))
    def knn(Q, idx):  # cosine on train-fold-standardised features, votes w = exp(10 cos)
        mu, sd = Z[idx].mean(0), Z[idx].std(0) + 1e-6
        nz = lambda A: (lambda B: B / np.linalg.norm(B, axis=1, keepdims=True))((A - mu) / sd)
        S = nz(Q) @ nz(Z[idx]).T; top = np.argsort(-S, 1)[:, :10]; P = np.zeros((len(Q), 4))
        for j in range(10):
            np.add.at(P, (np.arange(len(Q)), y[idx][top[:, j]]), np.exp(10 * S[np.arange(len(Q)), top[:, j]]))
        return P / P.sum(1, keepdims=True)
    for k in range(5):
        tri = np.where(f != k)[0]; kn[f == k] = knn(Z[:n][f == k], tri); kt += knn(Z[n:], tri) / 5
    save('whisper_knn', 0, kn, kt)
    a, at = np.load(f'oof/rcv_s{seed}_whisper_aug_L24L32.npy'), np.load(f'oof/rcv_s{seed}_whisper_aug_L24L32_test.npy')
    save('whisper_aug_knn', 0, 0.7 * a + 0.3 * kn, 0.7 * at + 0.3 * kt)

elif spike == 'p1':  # SAILER audio mixing: original + 4 S3 views + 2 mix views, fixed C=0.01, mean over 7 at prediction
    A, M = np.load('feats/whisper_aug.npy', mmap_mode='r'), np.load('feats/whisper_mix.npy', mmap_mode='r')
    vw = [cat(W, 24, 32)] + [np.hstack([B[:, v, 0], B[:, v, 1]]).astype(np.float32) for B, nv in ((A, 4), (M, 2)) for v in range(nv)]
    save('whisper_mix_L24L32', *aug_probe(vw, Cs=(0.01,)))

elif spike == 'omni2':  # kernels/dataverse-omni-zs2 -> runs/omni-zs2/: omni members v2 + K4 (transcript prompt)
    P, Hh, Ht = (np.load(f'runs/omni-zs2/omni2_{x}.npy') for x in ('lp', 'h', 'th'))
    assert not np.isnan(P).any(), 'kernel did not process every clip'
    save('omni2_h', *probe(Hh.reshape(N, -1).astype(np.float32)))
    r = [probe(P[:, q]) for q in range(3)]  # pre-registered: mean over the 3 prompts' LR probes
    save('omni2_lp', r[0][0], np.mean([x[1] for x in r], 0), np.mean([x[2] for x in r], 0))
    save('omni_th', *probe(Ht.reshape(N, -1).astype(np.float32))); save('omni_tlp', *probe(P[:, 3]))

elif spike == 'k1':  # kernels/dataverse-gemma4 -> runs/gemma4/: K1 LLM hidden + log-probs, K2 audio-encoder stats
    H, P, Ee = (np.load(f'runs/gemma4/gemma4_{x}.npy') for x in ('h', 'lp', 'enc'))
    assert not np.isnan(P).any(), 'kernel did not process every clip'
    save('gemma4_h', *probe(H.reshape(N, -1).astype(np.float32))); save('gemma4_lp', *probe(P.reshape(N, -1)))
    save('gemma4_enc', *probe(Ee.reshape(N, -1).astype(np.float32)))

elif spike == 'k3':  # Qwen3-ASR AuT encoder (src/qwen3asr_enc.py -> feats/qasr_enc.npy)
    save('qasr_enc', *probe(np.load('feats/qasr_enc.npy').reshape(N, -1).astype(np.float32)))

elif spike == 'f3':  # F3: e5 on Qwen3-ASR transcripts (feats/e5_qwen3.npy), text probe exactly as rcv_members (no scaler, C by F1)
    E = np.load('feats/e5_qwen3.npy'); best = None
    for C in (0.3, 1, 3, 10, 30):
        o = np.zeros((n, 4)); t = np.zeros((N - n, 4))
        for k in range(5):
            lr = LogisticRegression(C=C, max_iter=4000).fit(E[:n][f != k], y[f != k])
            o[f == k] = lr.predict_proba(E[:n][f == k]); t += lr.predict_proba(E[n:]) / 5
        if best is None or f1(o) > best[1]: best = (C, f1(o), o, t)
    save('text_e5q', best[0], best[2], best[3])
