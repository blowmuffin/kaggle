"""Fine-tune speech encoders for Hindi SER: 5-fold CV, one model per GPU (threads), hard time budget.
Kaggle: runs wavlm on cuda:0 and w2vbert on cuda:1 in parallel. SMOKE=1 -> tiny local check.
Outputs (/kaggle/working or ./out_ft): oof_<key>.npy, test_<key>.npy, log_<key>.json, submission.csv"""
import os, sys, subprocess, time
T0 = time.time()
SMOKE = os.environ.get('SMOKE') == '1'
KAGGLE = os.path.isdir('/kaggle/working')
if KAGGLE:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'transformers==4.56.2'], check=False)
BUDGET_S = float(os.environ.get('BUDGET_MIN', '40')) * 60
import glob, json, math, random, threading, traceback
import numpy as np, pandas as pd, soundfile as sf, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import f1_score
from transformers import AutoFeatureExtractor, AutoModel, get_cosine_schedule_with_warmup

FOLDS = '3122001330220100040201120342324101234042113014000000342130442113124300041343030212422241030430402143414221000100440441420430214231300034443132104244034321120210024021332442432341023012433313023401032213104002223001003231304443331441401222113221401441121123423004233211202111121043114221120423213103111113442111240140032040121314340244111332433440123424214102003000221340320043400100320342110332144323313324333224412330024112230133402430424424340143310230131331220142321033421114132113010431400211023102043103124023110444424240023232430042221202300122433431304231333411441100432114432441331412423444424300021404340313303221020100422101304404401443444301030311232102200433220240101402142244314041214434224221023311320301022222201040304434024031233043104332144232034113012131223202100324211121104123442331342132334430130303434320'
CLASSES = ['angry', 'happy', 'neutral', 'sad']
MODELS = {  # key: (hf id, backbone lr, max epochs)
    'wavlm': ('microsoft/wavlm-large', 2e-5, 10),
    'w2vbert': ('facebook/w2v-bert-2.0', 2e-5, 10),
}
OUT = '/kaggle/working' if KAGGLE else 'out_ft'
os.makedirs(OUT, exist_ok=True)
root = os.path.dirname(sorted(glob.glob('/kaggle/input/**/train.csv', recursive=True) + glob.glob('data/**/train.csv', recursive=True))[0])
tr = pd.read_csv(f'{root}/train.csv'); te = pd.read_csv(f'{root}/test.csv'); ss = pd.read_csv(f'{root}/sample_submission.csv')
assert len(FOLDS) == len(tr), (len(FOLDS), len(tr))
fold = np.array([int(c) for c in FOLDS])
y = tr.emotion.map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
wav_tr = [sf.read(f'{root}/train/{f}', dtype='float32')[0] for f in tr.filename]
wav_te = [sf.read(f'{root}/test/{f}', dtype='float32')[0] for f in te.filename]
print(f'data {root} train={len(tr)} test={len(te)} gpus={torch.cuda.device_count()} load {time.time() - T0:.0f}s', flush=True)
if SMOKE:
    keep = np.r_[np.flatnonzero(fold != 0)[:24], np.flatnonzero(fold == 0)[:8]]
    tr, y, fold = tr.iloc[keep].reset_index(drop=True), y[keep], fold[keep]
    wav_tr = [wav_tr[i] for i in keep]; wav_te = wav_te[:8]; te = te.iloc[:8]
MAX_TRAIN = 6 * 16000


def augment(x):
    if len(x) > MAX_TRAIN:
        s = random.randint(0, len(x) - MAX_TRAIN); x = x[s:s + MAX_TRAIN]
    x = x * np.float32(10 ** (random.uniform(-3, 3) / 20))
    if random.random() < 0.3:
        rms = np.sqrt((x ** 2).mean() + 1e-12)
        x = x + np.random.randn(len(x)).astype(np.float32) * rms / 10 ** (random.uniform(20, 40) / 20)
    return np.clip(x, -1, 1).astype(np.float32)


def batches(lengths, bs, shuffle):
    """Length-bucketed batches to cut padding."""
    noise = np.random.rand(len(lengths)) * 8000 if shuffle else 0
    order = np.argsort(np.asarray(lengths) + noise)
    bl = [order[i:i + bs] for i in range(0, len(order), bs)]
    if shuffle:
        random.shuffle(bl)
    return bl


class Net(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.enc = AutoModel.from_pretrained(name, layerdrop=0.0)
        if hasattr(self.enc, 'feature_extractor'):
            self.enc.feature_extractor._freeze_parameters()
        self.enc.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        c = self.enc.config
        self.w = nn.Parameter(torch.zeros(c.num_hidden_layers + 1))
        self.head = nn.Sequential(nn.LayerNorm(2 * c.hidden_size), nn.Dropout(0.2), nn.Linear(2 * c.hidden_size, 4))

    def forward(self, x, m):
        hs = self.enc(x, attention_mask=m, output_hidden_states=True).hidden_states
        h = sum(wi * hi for wi, hi in zip(self.w.softmax(0), hs)).float()
        if m.shape[1] != h.shape[1]:
            m = self.enc._get_feature_vector_attention_mask(h.shape[1], m)
        m = m[..., None].float()
        n = m.sum(1).clamp(min=1)
        mu = (h * m).sum(1) / n
        sd = (((h - mu[:, None]) ** 2 * m).sum(1) / n).clamp(min=1e-6).sqrt()
        return self.head(torch.cat([mu, sd], -1))


def run(key, dev, deadline):
    name, lr, max_ep = MODELS[key]
    fe = AutoFeatureExtractor.from_pretrained(name)

    def encode(xs):
        e = fe(xs, sampling_rate=16000, padding=True, return_attention_mask=True, return_tensors='pt')
        x = e['input_values'] if 'input_values' in e else e['input_features']
        return x.to(dev), e['attention_mask'].to(dev)

    @torch.inference_mode()
    def predict(model, wavs):
        model.eval(); out = np.zeros((len(wavs), 4), np.float32)
        for b in batches([len(w) for w in wavs], 8, False):
            with torch.autocast('cuda', dtype=torch.float16):
                out[b] = model(*encode([wavs[i] for i in b])).float().softmax(-1).cpu().numpy()
        return out

    oof = np.full((len(y), 4), np.nan, np.float32); test = np.zeros((len(wav_te), 4), np.float32)
    log, done, ep_time = [], [], None
    for k in range(1 if SMOKE else 5):
        left = deadline - time.time()
        epochs = 1 if SMOKE else max_ep
        if ep_time:  # fit remaining folds into the budget
            epochs = min(max_ep, int(left / (5 - k) / ep_time))
        if epochs < (1 if SMOKE else 3):
            print(f'[{key}] stop before fold {k}: {left:.0f}s left', flush=True); break
        torch.manual_seed(k); random.seed(k); np.random.seed(k)
        model = Net(name).to(dev)
        if SMOKE:
            for p in model.enc.parameters(): p.requires_grad_(False)
        enc_p = [p for p in model.enc.parameters() if p.requires_grad]
        opt = torch.optim.AdamW([{'params': enc_p, 'lr': lr}, {'params': [model.w, *model.head.parameters()], 'lr': 1e-3}], weight_decay=0.01)
        tri, vai = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
        steps = epochs * math.ceil(len(tri) / 8)
        sch = get_cosine_schedule_with_warmup(opt, int(0.1 * steps), steps)
        scaler = torch.amp.GradScaler()
        best = (-1, 9e9)
        for ep in range(epochs):
            t = time.time(); model.train(); tl = 0
            for b in batches([min(len(wav_tr[i]), MAX_TRAIN) for i in tri], 8, True):
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
                best = (f1, vl); oof[vai] = pv; pt = predict(model, wav_te)
            ep_time = time.time() - t
            log.append(dict(fold=k, ep=ep, loss=tl / len(tri), f1=f1, vl=vl, s=ep_time))
            print(f'[{key}] fold {k} ep {ep + 1}/{epochs} loss {tl / len(tri):.3f} valF1 {f1:.4f} vloss {vl:.3f} {ep_time:.0f}s', flush=True)
            if time.time() > deadline:
                break
        test += pt; done.append(k)
        del model, opt; torch.cuda.empty_cache()
        f = np.isin(fold, done)
        print(f'[{key}] FOLD {k} best {best[0]:.4f} | OOF so far {f1_score(y[f], oof[f].argmax(1), average="macro"):.4f} | {time.time() - T0:.0f}s', flush=True)
        np.save(f'{OUT}/oof_{key}.npy', oof); np.save(f'{OUT}/test_{key}.npy', test / len(done))
        json.dump(dict(done=done, log=log), open(f'{OUT}/log_{key}.json', 'w', encoding='utf-8'))


def safe(key, dev, deadline):
    try:
        run(key, dev, deadline)
    except Exception:
        traceback.print_exc()


keys = list(MODELS)
deadline = T0 + BUDGET_S
if torch.cuda.device_count() >= 2:
    th = [threading.Thread(target=safe, args=(k, torch.device(f'cuda:{i}'), deadline)) for i, k in enumerate(keys)]
    [t.start() for t in th]; [t.join() for t in th]
else:  # one GPU: run models back to back, splitting the remaining time
    for i, k in enumerate(keys):
        safe(k, torch.device('cuda:0'), time.time() + (deadline - time.time()) / (len(keys) - i))

# report + ensemble submission from whatever finished
probs = []
for k in keys:
    if os.path.exists(f'{OUT}/oof_{k}.npy'):
        o = np.load(f'{OUT}/oof_{k}.npy'); f = ~np.isnan(o).any(1)
        print(f'{k}: OOF macro-F1 {f1_score(y[f], o[f].argmax(1), average="macro"):.4f} on {f.sum()} rows', flush=True)
        probs.append(np.load(f'{OUT}/test_{k}.npy'))
if probs:
    sub = pd.DataFrame({'Id': te.Id, 'emotion': np.array(CLASSES)[np.mean(probs, 0).argmax(1)]})
    sub = ss[['Id']].merge(sub, on='Id', how='left') if not SMOKE else sub
    sub.to_csv(f'{OUT}/submission.csv', index=False)
    print(sub.emotion.value_counts().to_dict())
print(f'total {time.time() - T0:.0f}s', flush=True)
