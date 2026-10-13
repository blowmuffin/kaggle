"""Whisper-large-v3 speech→English translation, then English text emotion models (trained on large English corpora).
-> feats/asr_en.csv, feats/en_emo_rl.npy [N,7] (j-hartmann roberta-large: anger disgust fear joy neutral sadness surprise),
   feats/en_emo_db.npy [N,7] (distilroberta), feats/en_sent.npy [N,3] (cardiffnlp neg/neu/pos), feats/en_e5.npy [N,1024]"""
import os, time
os.environ.setdefault('HF_HOME', 'D:/kaggle/hf_cache')
import numpy as np, pandas as pd, soundfile as sf, torch

R = 'data/SRCASW-BhavVaani/'
tr = pd.read_csv('feats/train_meta.csv'); te = pd.read_csv('feats/test_meta.csv')
paths = [R + 'train/' + f for f in tr.filename] + [R + 'test/' + f for f in te.filename]
t = time.time()
if not os.path.exists('feats/asr_en.csv'):
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    wavs = [sf.read(p, dtype='float32')[0] for p in paths]
    proc = WhisperProcessor.from_pretrained('openai/whisper-large-v3')
    m = WhisperForConditionalGeneration.from_pretrained('openai/whisper-large-v3', torch_dtype=torch.float16).cuda().eval()
    out = []
    with torch.inference_mode():
        for i in range(0, len(wavs), 8):
            f = proc(wavs[i:i + 8], sampling_rate=16000, return_tensors='pt').input_features.cuda().half()
            out += proc.batch_decode(m.generate(f, language='hi', task='translate', max_new_tokens=128), skip_special_tokens=True)
    pd.DataFrame({'row': range(len(out)), 'Id': list(tr.Id) + list(te.Id), 'text_en': [s.strip() for s in out]}).to_csv('feats/asr_en.csv', index=False, encoding='utf-8')
    del m; torch.cuda.empty_cache(); print('translate done', f'{time.time() - t:.0f}s', flush=True)
E = pd.read_csv('feats/asr_en.csv', encoding='utf-8').text_en.fillna('').astype(str).tolist()
print('examples:', *[f'{tr.emotion[i]} | {E[i]}' for i in range(0, 40, 8)], sep='\n  ', flush=True)
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel


@torch.inference_mode()
def clf(name, out):
    tok = AutoTokenizer.from_pretrained(name); m = AutoModelForSequenceClassification.from_pretrained(name).cuda().eval()
    P = np.vstack([m(**tok(E[i:i + 64], padding=True, truncation=True, max_length=128, return_tensors='pt').to('cuda')).logits.float().softmax(-1).cpu().numpy()
                   for i in range(0, len(E), 64)])
    np.save(out, P.astype(np.float32)); print(name, m.config.id2label, flush=True)


clf('j-hartmann/emotion-english-roberta-large', 'feats/en_emo_rl.npy')
clf('j-hartmann/emotion-english-distilroberta-base', 'feats/en_emo_db.npy')
clf('cardiffnlp/twitter-roberta-base-sentiment-latest', 'feats/en_sent.npy')
with torch.inference_mode():
    tok = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large')
    m = AutoModel.from_pretrained('intfloat/multilingual-e5-large', torch_dtype=torch.float16).cuda().eval(); out = []
    for i in range(0, len(E), 32):
        b = tok(['query: ' + s for s in E[i:i + 32]], padding=True, truncation=True, max_length=128, return_tensors='pt').to('cuda')
        h = m(**b).last_hidden_state.float(); mk = b['attention_mask'][..., None].float()
        out.append(torch.nn.functional.normalize((h * mk).sum(1) / mk.sum(1), dim=-1).cpu().numpy())
    np.save('feats/en_e5.npy', np.vstack(out).astype(np.float32))
print('all done', f'{time.time() - t:.0f}s', flush=True)
