"""Fine-tune Whisper-large-v3 encoder (top half) for Hindi SER. 5 folds shared across GPUs (threads), hard time budget.
Inputs padded to batch-longest (not 30 s); positional embeddings sliced. SMOKE=1 -> tiny local check.
Outputs (/kaggle/working or ./out_ft): oof_whisper.npy, test_whisper.npy, log_whisper.json, submission.csv"""
import os, sys, subprocess, time
T0 = time.time()
SMOKE = os.environ.get('SMOKE') == '1'
KAGGLE = os.path.isdir('/kaggle/working')
if KAGGLE:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'transformers==4.56.2'], check=False)
BUDGET_S = float(os.environ.get('BUDGET_MIN', '26')) * 60
import glob, json, math, random, threading, traceback
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd, soundfile as sf, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from sklearn.metrics import f1_score
from transformers import WhisperFeatureExtractor, WhisperForAudioClassification, get_cosine_schedule_with_warmup

FOLDS = '__FOLDS__'
CLASSES = ['angry', 'happy', 'neutral', 'sad']
NAME, LR, EPOCHS, BS = 'openai/whisper-large-v3', 1e-5, 5, 8
FREEZE = 31 if SMOKE else 16  # layers below this index stay frozen
OUT = '/kaggle/working' if KAGGLE else 'out_ft'
os.makedirs(OUT, exist_ok=True)
root = os.path.dirname(sorted(glob.glob('/kaggle/input/**/train.csv', recursive=True) + glob.glob('data/**/train.csv', recursive=True))[0])
tr = pd.read_csv(f'{root}/train.csv'); te = pd.read_csv(f'{root}/test.csv'); ss = pd.read_csv(f'{root}/sample_submission.csv')
assert len(FOLDS) == len(tr), (len(FOLDS), len(tr))
fold = np.array([int(c) for c in FOLDS])
y = tr.emotion.map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
with ThreadPoolExecutor(8) as ex:
    wav_tr = list(ex.map(lambda f: sf.read(f'{root}/train/{f}', dtype='float32')[0], tr.filename))
    wav_te = list(ex.map(lambda f: sf.read(f'{root}/test/{f}', dtype='float32')[0], te.filename))
print(f'data {root} train={len(tr)} test={len(te)} gpus={torch.cuda.device_count()} load {time.time() - T0:.0f}s', flush=True)
if SMOKE:
    keep = np.r_[np.flatnonzero(fold != 0)[:24], np.flatnonzero(fold == 0)[:8]]
    tr, y, fold = tr.iloc[keep].reset_index(drop=True), y[keep], fold[keep]
    wav_tr = [wav_tr[i] for i in keep]; wav_te = wav_te[:8]; te = te.iloc[:8]
MAX_TRAIN = 6 * 16000
fe = WhisperFeatureExtractor.from_pretrained(NAME)
LOAD_LOCK = threading.Lock()  # HF from_pretrained is not thread-safe (v1 crash: meta tensors)


def augment(x):
    if len(x) > MAX_TRAIN:
        s = random.randint(0, len(x) - MAX_TRAIN); x = x[s:s + MAX_TRAIN]
    x = x * np.float32(10 ** (random.uniform(-3, 3) / 20))
    if random.random() < 0.3:
        rms = np.sqrt((x ** 2).mean() + 1e-12)
        x = x + np.random.randn(len(x)).astype(np.float32) * rms / 10 ** (random.uniform(20, 40) / 20)
    return np.clip(x, -1, 1).astype(np.float32)


def batches(lengths, bs, shuffle):
    noise = np.random.rand(len(lengths)) * 8000 if shuffle else 0
    order = np.argsort(np.asarray(lengths) + noise)
    bl = [order[i:i + bs] for i in range(0, len(order), bs)]
    if shuffle:
        random.shuffle(bl)
    return bl


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = WhisperForAudioClassification.from_pretrained(NAME).encoder
        for n, p in self.enc.named_parameters():
            p.requires_grad_(n.startswith('layer_norm') or (n.startswith('layers.') and int(n.split('.')[1]) >= FREEZE))
        H = self.enc.config.d_model
        self.w = nn.Parameter(torch.zeros(len(self.enc.layers) + 1))
        self.head = nn.Sequential(nn.LayerNorm(2 * H), nn.Dropout(0.2), nn.Linear(2 * H, 4))

    def forward(self, feats, mel_mask):
        e = self.enc
        with torch.no_grad():
            x = F.gelu(e.conv2(F.gelu(e.conv1(feats)))).permute(0, 2, 1)
            x = x + e.embed_positions.weight[:x.shape[1]]
            hs = [x]
            for layer in e.layers[:FREEZE]:
                x = layer(x, None, layer_head_mask=None)[0]; hs.append(x)
        for layer in e.layers[FREEZE:]:
            x = (checkpoint(layer, x, None, None, use_reentrant=False) if self.training else layer(x, None, None))[0]
            hs.append(x)
        hs[-1] = e.layer_norm(x)
        h = sum(wi * hi for wi, hi in zip(self.w.softmax(0), hs)).float()
        m = mel_mask[:, ::2][:, :h.shape[1]][..., None].float()
        n = m.sum(1).clamp(min=1)
        mu = (h * m).sum(1) / n
        sd = (((h - mu[:, None]) ** 2 * m).sum(1) / n).clamp(min=1e-6).sqrt()
        return self.head(torch.cat([mu, sd], -1))


oof = np.full((len(y), 4), np.nan, np.float32); test_sum = np.zeros((len(wav_te), 4), np.float32)
log, done, queue, ep_time = [], [], list(range(1 if SMOKE else 5)), [None]
STATE_LOCK = threading.Lock()
deadline = T0 + BUDGET_S


def worker(dev):
    def encode(xs):
        e = fe(xs, sampling_rate=16000, padding='longest', return_attention_mask=True, return_tensors='pt')
        return e['input_features'].to(dev), e['attention_mask'].to(dev)

    @torch.inference_mode()
    def predict(model, wavs):
        model.eval(); out = np.zeros((len(wavs), 4), np.float32)
        for b in batches([len(w) for w in wavs], BS, False):
            with torch.autocast('cuda', dtype=torch.float16):
                out[b] = model(*encode([wavs[i] for i in b])).float().softmax(-1).cpu().numpy()
        return out

    global test_sum
    while True:
        with STATE_LOCK:
            if not queue:
                return
            need = (ep_time[0] or 0) * EPOCHS * 0.7
            if time.time() + need > deadline:
                print(f'[{dev}] stop: {deadline - time.time():.0f}s left, folds left {queue}', flush=True); return
            k = queue.pop(0)
        torch.manual_seed(k); random.seed(k); np.random.seed(k)
        with LOAD_LOCK:
            model = Net().to(dev)
        opt = torch.optim.AdamW([{'params': [p for p in model.enc.parameters() if p.requires_grad], 'lr': LR},
                                 {'params': [model.w, *model.head.parameters()], 'lr': 1e-3}], weight_decay=0.01)
        tri, vai = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
        epochs = 1 if SMOKE else EPOCHS
        steps = epochs * math.ceil(len(tri) / BS)
        sch = get_cosine_schedule_with_warmup(opt, int(0.1 * steps), steps)
        scaler = torch.amp.GradScaler()
        best, pv_best, pt = (-1, 9e9), None, None
        for ep in range(epochs):
            t = time.time(); model.train(); tl = 0
            for b in batches([min(len(wav_tr[i]), MAX_TRAIN) for i in tri], BS, True):
                idx = tri[b]
                x, m = encode([augment(wav_tr[i]) for i in idx])
                with torch.autocast('cuda', dtype=torch.float16):
                    loss = F.cross_entropy(model(x, m).float(), torch.as_tensor(y[idx], device=dev), label_smoothing=0.05)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward(); scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); sch.step()
                tl += loss.item() * len(idx)
            pv = predict(model, [wav_tr[i] for i in vai])
            f1 = f1_score(y[vai], pv.argmax(1), average='macro')
            vl = float(-np.log(pv[np.arange(len(vai)), y[vai]] + 1e-9).mean())
            if (f1, -vl) > (best[0], -best[1]):
                best, pv_best, pt = (f1, vl), pv, predict(model, wav_te)
            dt = time.time() - t
            with STATE_LOCK:
                ep_time[0] = dt
                log.append(dict(fold=k, ep=ep, loss=tl / len(tri), f1=f1, vl=vl, s=dt))
            print(f'[whisper {dev}] fold {k} ep {ep + 1}/{epochs} loss {tl / len(tri):.3f} valF1 {f1:.4f} vloss {vl:.3f} {dt:.0f}s', flush=True)
            if time.time() > deadline:
                break
        with STATE_LOCK:
            oof[vai] = pv_best; test_sum += pt; done.append(k)
            f = np.isin(fold, done)
            print(f'[whisper] FOLD {k} best {best[0]:.4f} | OOF so far {f1_score(y[f], oof[f].argmax(1), average="macro"):.4f} '
                  f'folds {sorted(done)} | {time.time() - T0:.0f}s', flush=True)
            np.save(f'{OUT}/oof_whisper.npy', oof); np.save(f'{OUT}/test_whisper.npy', test_sum / len(done))
            json.dump(dict(done=done, log=log), open(f'{OUT}/log_whisper.json', 'w', encoding='utf-8'))
        del model, opt; torch.cuda.empty_cache()


def safe(dev):
    try:
        worker(dev)
    except Exception:
        traceback.print_exc()


devs = [torch.device(f'cuda:{i}') for i in range(max(1, torch.cuda.device_count()))]
th = [threading.Thread(target=safe, args=(d,)) for d in devs]
[t.start() for t in th]; [t.join() for t in th]
if done:
    f = np.isin(fold, done)
    print(f'whisper: OOF macro-F1 {f1_score(y[f], oof[f].argmax(1), average="macro"):.4f} on {f.sum()} rows, folds {sorted(done)}', flush=True)
    sub = pd.DataFrame({'Id': te.Id, 'emotion': np.array(CLASSES)[test_sum.argmax(1)]})
    sub = sub if SMOKE else ss[['Id']].merge(sub, on='Id', how='left')
    sub.to_csv(f'{OUT}/submission.csv', index=False)
    print(sub.emotion.value_counts().to_dict())
print(f'total {time.time() - T0:.0f}s', flush=True)
