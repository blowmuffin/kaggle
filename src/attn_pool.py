"""Spike L3 (pre-registered): attentive-statistics pooling head on frozen whisper-large-v3 L24||L32 frames.
Usage: python src/attn_pool.py <seed>|full [aug] [text]  -> oof/rcv_s{seed}_whisper_attn[_aug].npy (+ _test). Needs feats/whisper_frames*.npy.
aug (L3b): also the 4 S3-view frame caches; one random version per clip per step, prediction = mean over the 5 versions.
Per fold: frames standardised with train-fold stats -> Linear 256 + GELU + dropout 0.3 -> 4-head attentive statistics pooling
-> dropout 0.3 -> Linear 4. AdamW lr 1e-3 wd 1e-2, 30 epochs, batch 16, label smoothing 0.1, cosine; 3 init seeds averaged."""
import sys, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import hcv
from hcv import y, n, f1

FULL = sys.argv[1] == 'full'  # F2: train on all 826 train clips, test predictions only
seed = 0 if FULL else int(sys.argv[1]); AUG = 'aug' in sys.argv[2:]; TEXT = 'text' in sys.argv[2:]; N = 1035; H = 4; D = 256
f = np.full(n, -1) if FULL else np.load(f'oof/rcv_s{seed}_fold.npy')
KS = [99] if FULL else range(5)
off = np.load('feats/whisper_frames_off.npy')
X = torch.from_numpy(np.load('feats/whisper_frames.npy').reshape(off[-1], -1)).cuda()  # [frames, 2560] fp16
Y = torch.from_numpy(y).cuda()
E = torch.from_numpy(np.load('feats/e5.npy')).float().cuda()  # P3: e5 transcript embedding (used only with 'text')
VF = [(np.load(f'feats/whisper_frames_v{j}.npy', mmap_mode='r'), np.load(f'feats/whisper_frames_v{j}_off.npy')) for j in range(4)] if AUG else []
NV = 1 + len(VF)  # versions: 0 = original (on GPU), 1..4 = S3 views (CPU memmap)


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(2560, D), nn.GELU(), nn.Dropout(0.3))
        self.att = nn.Linear(D, H)
        self.txt = nn.Sequential(nn.Linear(1024, 64), nn.GELU(), nn.Dropout(0.3)) if TEXT else None
        self.out = nn.Sequential(nn.Dropout(0.3), nn.Linear(2 * H * D + (64 if TEXT else 0), 4))

    def forward(self, x, mask, idx=None):  # x [B,T,2560], mask [B,T] True = valid; idx -> transcript rows (TEXT)
        h = self.proj(x)
        w = self.att(h).masked_fill(~mask[..., None], -1e4).softmax(1)  # [B,T,H]
        mu = torch.einsum('bth,btd->bhd', w, h)
        sd = (torch.einsum('bth,btd->bhd', w, h * h) - mu ** 2).clamp_min(1e-6).sqrt()
        z = torch.cat([mu, sd], -1).flatten(1)
        if TEXT: z = torch.cat([z, self.txt(E[torch.as_tensor(idx, device='cuda')])], 1)  # early audio-text fusion (P3)
        return self.out(z)


def frames(i, v):
    if v == 0: return X[off[i]:off[i + 1]]
    Fr, o = VF[v - 1]
    return torch.from_numpy(np.asarray(Fr[o[i]:o[i + 1]]).reshape(o[i + 1] - o[i], -1)).cuda()


def batch(idx, mu, sd, ver=None):
    ver = np.zeros(len(idx), int) if ver is None else ver
    hs = [frames(i, v) for i, v in zip(idx, ver)]
    lens = [len(h) for h in hs]; T = max(lens)
    x = torch.zeros(len(idx), T, 2560, device='cuda'); mask = torch.zeros(len(idx), T, dtype=torch.bool, device='cuda')
    for j, h in enumerate(hs):
        x[j, :lens[j]] = (h.float() - mu) / sd; mask[j, :lens[j]] = True
    return x, mask


def predict(model, idx, mu, sd):  # mean softmax over all versions (only the original unless AUG)
    model.eval(); P = 0
    with torch.no_grad():
        for v in range(NV):
            P = P + torch.cat([model(*batch(idx[b:b + 64], mu, sd, np.full(len(idx[b:b + 64]), v)), idx[b:b + 64]).softmax(-1).cpu()
                               for b in range(0, len(idx), 64)]).numpy() / NV
    return P


oof = np.zeros((n, 4)); test = np.zeros((N - n, 4)); te_idx = np.arange(n, N)
for k in KS:
    tri, vai = np.where(f != k)[0], np.where(f == k)[0]
    s1 = torch.zeros(2560, device='cuda', dtype=torch.float64); s2 = torch.zeros_like(s1); cnt = 0
    for i in tri:  # train-fold frame stats, chunked per clip to fit next to another GPU job
        h = X[off[i]:off[i + 1]].double(); s1 += h.sum(0); s2 += (h * h).sum(0); cnt += len(h)
    mu = (s1 / cnt).float(); sd = (s2 / cnt - (s1 / cnt) ** 2).clamp_min(0).sqrt().float() + 1e-5
    for s in range(3):
        torch.manual_seed(100 * seed + 10 * k + s); rng = np.random.default_rng(100 * seed + 10 * k + s)
        model = Head().cuda(); opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        steps = 30 * int(np.ceil(len(tri) / 16)); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
        for ep in range(30):
            model.train(); perm = rng.permutation(tri)
            for b in range(0, len(perm), 16):
                idx = perm[b:b + 16]
                ver = rng.integers(0, NV, size=len(idx)) if AUG else None
                loss = F.cross_entropy(model(*batch(idx, mu, sd, ver), idx), Y[idx], label_smoothing=0.1)
                opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if len(vai): oof[vai] += predict(model, vai, mu, sd) / 3
        test += predict(model, te_idx, mu, sd) / (3 * len(KS))
    if not FULL: print(f'seed {seed} fold {k}: F1 {f1(oof, f == k):.4f}', flush=True)
name = 'whisper_attn' + ('_aug' if AUG else '') + ('_text' if TEXT else '')
if FULL:
    np.save(f'oof/full_{name}_test.npy', test); print('saved', f'oof/full_{name}_test.npy', test.shape, flush=True)
else:
    np.save(f'oof/rcv_s{seed}_{name}.npy', oof); np.save(f'oof/rcv_s{seed}_{name}_test.npy', test)
    print(f'seed {seed} {name} {f1(oof):.4f}', flush=True)
