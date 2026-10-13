"""multilingual-e5-large transcript embeddings, same recipe as t1_text2.py (strip Unicode P*/S*, 'query: ' prefix, max 128 tokens,
mean pool, L2 normalise). Usage: python src/e5_embed.py feats/asr_qwen3.csv feats/e5_qwen3.npy"""
import os, sys, unicodedata
os.environ.setdefault('HF_HOME', 'D:/kaggle/hf_cache')
import numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModel

src, dst = sys.argv[1], sys.argv[2]
clean = lambda s: ' '.join(''.join(' ' if unicodedata.category(ch)[0] in 'PS' else ch for ch in str(s)).split())
t = pd.read_csv(src, encoding='utf-8').text.fillna('').map(clean).tolist(); assert len(t) == 1035
tok = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large')
m = AutoModel.from_pretrained('intfloat/multilingual-e5-large', torch_dtype=torch.float16).cuda().eval()
out = []
with torch.inference_mode():
    for i in range(0, len(t), 32):
        b = tok(['query: ' + s for s in t[i:i + 32]], padding=True, truncation=True, max_length=128, return_tensors='pt').to('cuda')
        h = m(**b).last_hidden_state; mk = b['attention_mask'][..., None]
        e = (h * mk).sum(1) / mk.sum(1); out.append(torch.nn.functional.normalize(e, dim=-1).float().cpu().numpy())
np.save(dst, np.vstack(out).astype(np.float32)); print('saved', dst, np.vstack(out).shape)
