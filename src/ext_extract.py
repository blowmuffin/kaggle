"""External data (CC BY 4.0, swastikatinkhede/multilingual-speech-dataset, 22 speakers, Hindi/Marathi/English).
Drops 'Surprised'. Resample to 16 kHz mono. Features match feats/whisper.npy (30-s pad, L32) and feats/w2vbert.npy (L16).
-> feats/ext_meta.csv, feats/ext_whisper_L32.npy [N,2560], feats/ext_w2vbert_L16.npy [N,2048]"""
import os, glob, re, time
os.environ.setdefault('HF_HOME', 'D:/kaggle/hf_cache')
import numpy as np, pandas as pd, soundfile as sf, torch
from math import gcd
from scipy.signal import resample_poly

rows = []
for p in glob.glob('ext/indian/**/*.wav', recursive=True):
    parts = p.replace('\\', '/').split('/')
    emo = parts[-2].lower()
    if emo not in ('angry', 'happy', 'neutral', 'sad'):
        continue
    rows.append(dict(path=p, lang=parts[-3].lower(), emotion=emo, speaker=re.match(r'(S\d+)_', parts[-1]).group(1)))
M = pd.DataFrame(rows).sort_values('path').reset_index(drop=True)
print(M.groupby(['lang', 'emotion']).size().unstack(), flush=True)


def load16(p):
    x, sr = sf.read(p, dtype='float32', always_2d=True); x = x.mean(1)
    if sr != 16000:
        g = gcd(sr, 16000); x = resample_poly(x, 16000 // g, sr // g).astype(np.float32)
    return x


wavs = [load16(p) for p in M.path]
M['dur'] = [len(x) / 16000 for x in wavs]
M.to_csv('feats/ext_meta.csv', index=False)
from transformers import WhisperFeatureExtractor, WhisperModel, AutoFeatureExtractor, Wav2Vec2BertModel
t = time.time()
with torch.inference_mode():
    fe = WhisperFeatureExtractor.from_pretrained('openai/whisper-large-v3')
    enc = WhisperModel.from_pretrained('openai/whisper-large-v3', torch_dtype=torch.float16).encoder.cuda().eval()
    W = []
    for x in wavs:
        x = x[:30 * 16000]
        hs = enc(fe(x, sampling_rate=16000, return_tensors='pt').input_features.cuda().half(), output_hidden_states=True).hidden_states
        h = hs[32][0, :int(np.ceil(len(x) / 320))].float()
        W.append(torch.cat([h.mean(0), h.std(0)]).cpu().numpy())
    np.save('feats/ext_whisper_L32.npy', np.stack(W).astype(np.float32)); del enc; torch.cuda.empty_cache()
    print('whisper done', f'{time.time() - t:.0f}s', flush=True)
    fe = AutoFeatureExtractor.from_pretrained('facebook/w2v-bert-2.0')
    m = Wav2Vec2BertModel.from_pretrained('facebook/w2v-bert-2.0', torch_dtype=torch.float16).cuda().eval()
    V = []
    for x in wavs:
        h = m(fe(x, sampling_rate=16000, return_tensors='pt').input_features.cuda().half(), output_hidden_states=True).hidden_states[16][0].float()
        V.append(torch.cat([h.mean(0), h.std(0)]).cpu().numpy())
    np.save('feats/ext_w2vbert_L16.npy', np.stack(V).astype(np.float32))
print('done', len(M), f'{time.time() - t:.0f}s', flush=True)
