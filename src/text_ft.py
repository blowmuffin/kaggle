"""Fine-tune MuRIL (google/muril-base-cased, Indian-language BERT) on Whisper transcripts. Honest v2 folds.
Env: SEED, EPOCHS (8), MODEL, SMOKE=1. Output: oof/v2_text_ft_s{SEED}.npy (+ _test)."""
import os, math, time, unicodedata
os.environ.setdefault('HF_HOME', 'D:/kaggle/hf_cache')
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from hcv import *

SEED, EPOCHS = int(os.environ.get('SEED', 0)), int(os.environ.get('EPOCHS', 8))
NAME = os.environ.get('MODEL', 'google/muril-base-cased'); SMOKE = os.environ.get('SMOKE') == '1'
A = pd.read_csv('feats/asr.csv', encoding='utf-8')
clean = lambda s: ' '.join(''.join(' ' if unicodedata.category(ch)[0] in 'PS' else ch for ch in str(s)).split())
texts = A.text.fillna('').map(clean).tolist()
tok = AutoTokenizer.from_pretrained(NAME)
enc = tok(texts, padding='max_length', truncation=True, max_length=64, return_tensors='pt')
dev = 'cuda'


@torch.inference_mode()
def predict(m, idx):
    m.eval(); out = []
    for i in range(0, len(idx), 64):
        b = idx[i:i + 64]
        with torch.autocast('cuda', dtype=torch.float16):
            out.append(m(input_ids=enc.input_ids[b].to(dev), attention_mask=enc.attention_mask[b].to(dev)).logits.float().softmax(-1).cpu().numpy())
    return np.vstack(out)


oof = np.zeros((n, 4)); test = np.zeros((len(texts) - n, 4)); te_idx = np.arange(n, len(texts))
for k in ([0] if SMOKE else range(5)):
    t = time.time(); torch.manual_seed(SEED * 100 + k); np.random.seed(SEED * 100 + k)
    tri, vai = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
    m = AutoModelForSequenceClassification.from_pretrained(NAME, num_labels=4).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-5, weight_decay=0.01)
    ep_n = 1 if SMOKE else EPOCHS; nb = math.ceil(len(tri) / 16)
    sch = get_linear_schedule_with_warmup(opt, int(0.1 * ep_n * nb), ep_n * nb); scaler = torch.amp.GradScaler()
    for ep in range(ep_n):
        m.train(); perm = np.random.permutation(tri)
        for i in range(nb):
            b = perm[i * 16:(i + 1) * 16]
            with torch.autocast('cuda', dtype=torch.float16):
                logits = m(input_ids=enc.input_ids[b].to(dev), attention_mask=enc.attention_mask[b].to(dev)).logits
            loss = F.cross_entropy(logits.float(), torch.as_tensor(y[b], device=dev), label_smoothing=0.05)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); scaler.step(opt); scaler.update(); sch.step()
    oof[vai] = predict(m, vai); test += predict(m, te_idx) / 5
    print(f'fold {k} valF1 {f1_score(y[vai], oof[vai].argmax(1), average="macro"):.4f} {time.time() - t:.0f}s', flush=True)
    del m, opt; torch.cuda.empty_cache()
if not SMOKE:
    np.save(f'oof/v2_text_ft_s{SEED}.npy', oof); np.save(f'oof/v2_text_ft_s{SEED}_test.npy', test)
    print(f'TEXT FT {NAME} seed {SEED}: honest F1 {f1(oof):.4f}', flush=True)
