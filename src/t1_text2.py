"""B v2: transcript features with matras kept (strip only Unicode punctuation/symbols) + multilingual-e5-large
sentence embeddings. Text-only and blend tests on honest v2 folds."""
import os, unicodedata, numpy as np, pandas as pd
os.environ.setdefault('HF_HOME', 'D:/kaggle/hf_cache')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from hcv import *

A = pd.read_csv('feats/asr.csv', encoding='utf-8')
clean = lambda s: ' '.join(''.join(' ' if unicodedata.category(ch)[0] in 'PS' else ch for ch in str(s)).split())
A['t'] = A.text.fillna('').map(clean)
print('examples:'); [print('  ', tr.emotion[i], '|', A.t[i][:70]) for i in range(0, 30, 6)]

if not os.path.exists('feats/e5.npy'):
    import torch
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large')
    m = AutoModel.from_pretrained('intfloat/multilingual-e5-large', torch_dtype=torch.float16).cuda().eval()
    out = []
    with torch.inference_mode():
        for i in range(0, len(A), 32):
            b = tok(['query: ' + s for s in A.t[i:i + 32]], padding=True, truncation=True, max_length=128, return_tensors='pt').to('cuda')
            h = m(**b).last_hidden_state.float(); mk = b['attention_mask'][..., None].float()
            e = (h * mk).sum(1) / mk.sum(1); out.append(torch.nn.functional.normalize(e, dim=-1).cpu().numpy())
    np.save('feats/e5.npy', np.vstack(out).astype(np.float32))
E5 = np.load('feats/e5.npy')


def lr_probe(X, Cs):
    best = None
    for C in Cs:
        oof = np.zeros((n, 4)); test = np.zeros((X.shape[0] - n, 4))
        for k in range(5):
            mdl = LogisticRegression(C=C, max_iter=4000).fit(X[:n][fold != k], y[fold != k])
            oof[fold == k] = mdl.predict_proba(X[:n][fold == k]); test += mdl.predict_proba(X[n:]) / 5
        if best is None or f1(oof) > best[0]:
            best = (f1(oof), C, oof, test)
    return best


O = {m: np.load(f'oof/v2_{m}.npy') for m in ['whisper_L32', 'w2vbert_L16', 'omni_h', 'omni_lp']}
cur = 1 / 3 * (O['whisper_L32'] + O['w2vbert_L16']) + 1 / 6 * (O['omni_h'] + O['omni_lp'])
print(f'\ncurrent best audio blend (base+omni): {f1(cur):.4f}')
Xc = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), min_df=2, sublinear_tf=True).fit(A.t).transform(A.t)
Xw = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=1, sublinear_tf=True).fit(A.t).transform(A.t)
for nm, X, Cs in [('text_char', Xc, (0.3, 1, 3, 10)), ('text_word', Xw, (0.3, 1, 3, 10)), ('text_e5', E5, (0.3, 1, 3, 10, 30))]:
    s, C, oof, test = lr_probe(X, Cs)
    np.save(f'oof/v2_{nm}.npy', oof); np.save(f'oof/v2_{nm}_test.npy', test)
    print(f'{nm:10s} C={C:<4} text-only honest F1 {s:.4f}')
    for w in (0.15, 0.25, 0.35):
        b = (1 - w) * cur + w * oof; d = per_fold(b, cur)
        print(f'   audio blend + {nm} w={w}: {f1(b):.4f} ({f1(b) - f1(cur):+.4f}) folds up {sum(x > 0 for x in d)}/5 ' + ' '.join(f'{x:+.3f}' for x in d))
from sklearn.metrics import confusion_matrix
print('\naudio-blend confusion (rows true a/h/n/s):\n', confusion_matrix(y, cur.argmax(1)))
print('text_e5 confusion:\n', confusion_matrix(y, np.load('oof/v2_text_e5.npy').argmax(1)))
