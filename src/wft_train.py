"""Fine-tune Whisper-large-v3 layers 24-31 + final LN on the cached trunk (pad 20 s), LP-FT: head = fixed per-feature
standardisation + linear initialised from the fold's logistic-regression probe. Honest v2 folds.
Env: SEED, EPOCHS (5), LR_ENC (1e-5), FOLDS ("0,1,2,3,4"), SMOKE=1 (fold 0, 1 epoch, 5 batches).
Per-fold outputs: oof/wft_s{SEED}_f{k}.npy (val probs, last epoch) and oof/wft_s{SEED}_f{k}_test.npy."""
import os, copy, math, time
os.environ.setdefault('HF_HOME', 'D:/kaggle/hf_cache')
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from sklearn.linear_model import LogisticRegression
from transformers import WhisperForAudioClassification, get_cosine_schedule_with_warmup
from hcv import *

SEED, EPOCHS = int(os.environ.get('SEED', 0)), int(os.environ.get('EPOCHS', 5))
LR_ENC, LR_HEAD, BS = float(os.environ.get('LR_ENC', 1e-5)), 1e-4, 8
SMOKE = os.environ.get('SMOKE') == '1'
FOLDS = [0] if SMOKE else [int(k) for k in os.environ.get('FOLDS', '0,1,2,3,4').split(',')]
cache = np.load('feats/wcache_L24_p20.npy', mmap_mode='r'); NV = np.load('feats/wcache_nvalid.npy')
P32 = np.load('feats/whisper_pad20.npy')[:, 2].astype(np.float32)  # frozen L32 mean+std at pad 20 (same pipeline)
enc0 = WhisperForAudioClassification.from_pretrained('openai/whisper-large-v3').encoder
dev = 'cuda'


class Top(nn.Module):
    def __init__(self, mu, sd, W, b):
        super().__init__()
        self.layers = copy.deepcopy(enc0.layers[24:]); self.ln = copy.deepcopy(enc0.layer_norm)
        self.register_buffer('mu', torch.tensor(mu)); self.register_buffer('sd', torch.tensor(sd))
        self.drop = nn.Dropout(0.1); self.fc = nn.Linear(len(mu), 4)
        with torch.no_grad():
            self.fc.weight.copy_(torch.tensor(W)); self.fc.bias.copy_(torch.tensor(b))

    def forward(self, x, nv):
        for l in self.layers:
            x = (checkpoint(l, x, None, None, use_reentrant=False) if self.training else l(x, None, None))[0]
        h = self.ln(x).float()
        m = (torch.arange(h.shape[1], device=h.device)[None] < nv[:, None]).float()[..., None]
        c = m.sum(1)
        mu = (h * m).sum(1) / c
        sd = (((h - mu[:, None]) ** 2 * m).sum(1) / (c - 1).clamp(min=1)).clamp(min=1e-6).sqrt()
        return self.fc(self.drop((torch.cat([mu, sd], -1) - self.mu) / self.sd))


def batch(idx):
    s = np.sort(idx)
    return torch.from_numpy(np.ascontiguousarray(cache[s])).to(dev), torch.from_numpy(NV[s]).to(dev), s


@torch.inference_mode()
def predict(model, idx):
    model.eval(); out = {}
    for i in range(0, len(idx), 16):
        x, nv, s = batch(idx[i:i + 16])
        with torch.autocast('cuda', dtype=torch.float16):
            p = model(x, nv).float().softmax(-1).cpu().numpy()
        out.update(zip(s, p))
    return np.stack([out[i] for i in idx])


te_idx = np.arange(n, len(cache))
for k in FOLDS:
    t0 = time.time(); torch.manual_seed(SEED * 100 + k); np.random.seed(SEED * 100 + k)
    tri, vai = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
    mu, sd = P32[tri].mean(0), P32[tri].std(0) + 1e-6
    lr = LogisticRegression(C=0.03, max_iter=4000).fit((P32[tri] - mu) / sd, y[tri])
    model = Top(mu, sd, lr.coef_.astype(np.float32), lr.intercept_.astype(np.float32)).to(dev)
    pv = predict(model, vai)
    print(f'[s{SEED} f{k}] init (= LR probe) valF1 {f1_score(y[vai], pv.argmax(1), average="macro"):.4f}', flush=True)
    opt = torch.optim.AdamW([{'params': [*model.layers.parameters(), *model.ln.parameters()], 'lr': LR_ENC},
                             {'params': model.fc.parameters(), 'lr': LR_HEAD}], weight_decay=0.01)
    epochs = 1 if SMOKE else EPOCHS
    nb = 5 if SMOKE else math.ceil(len(tri) / BS)
    sch = get_cosine_schedule_with_warmup(opt, int(0.1 * epochs * nb), epochs * nb)
    scaler = torch.amp.GradScaler()
    for ep in range(epochs):
        model.train(); perm = np.random.permutation(tri); tl = 0; t = time.time()
        for b in range(nb):
            idx = perm[b * BS:(b + 1) * BS]
            x, nv, s = batch(idx)
            with torch.autocast('cuda', dtype=torch.float16):
                loss = F.cross_entropy(model(x, nv).float(), torch.as_tensor(y[s], device=dev), label_smoothing=0.05)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); sch.step()
            tl += loss.item()
        pv = predict(model, vai)
        print(f'[s{SEED} f{k}] ep {ep + 1}/{epochs} loss {tl / nb:.3f} valF1 {f1_score(y[vai], pv.argmax(1), average="macro"):.4f} '
              f'{time.time() - t:.0f}s mem {torch.cuda.max_memory_allocated() / 2**30:.1f}G', flush=True)
    if not SMOKE:
        np.save(f'oof/wft_s{SEED}_f{k}.npy', pv); np.save(f'oof/wft_s{SEED}_f{k}_test.npy', predict(model, te_idx))
    print(f'[s{SEED} f{k}] done {time.time() - t0:.0f}s', flush=True)
    del model, opt; torch.cuda.empty_cache()
