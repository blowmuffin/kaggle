"""Frozen-encoder features: per-layer mean+std pooled over valid frames.
Usage: python src/extract.py wavlm|whisper|w2vbert  -> feats/<key>.npy  [n_clips, n_layers, 2*hidden] fp16
Row order = train_meta.csv rows, then test_meta.csv rows."""
import os, sys, time
os.environ.setdefault('HF_HOME', 'D:/kaggle/hf_cache')
import numpy as np, pandas as pd, soundfile as sf, torch

R = 'data/SRCASW-BhavVaani/'
MODELS = {
    'wavlm': 'microsoft/wavlm-large',
    'whisper': 'openai/whisper-large-v3',
    'w2vbert': 'facebook/w2v-bert-2.0',
    'whispermed': 'openai/whisper-medium',
    'hubert': 'facebook/hubert-large-ll60k',
    'mms': 'facebook/mms-300m',
    'whisperv2': 'openai/whisper-large-v2',
    'indicw2v': 'ai4bharat/indicwav2vec-hindi',
    'vakyansh': 'Harveenchadha/vakyansh-wav2vec2-hindi-him-4200',
}
key = sys.argv[1]
name = MODELS[key]
dev = 'cuda'
tr = pd.read_csv('feats/train_meta.csv'); te = pd.read_csv('feats/test_meta.csv')
paths = [R + 'train/' + f for f in tr.filename] + [R + 'test/' + f for f in te.filename]

if key.startswith('whisper'):
    from transformers import WhisperFeatureExtractor, WhisperModel
    fe = WhisperFeatureExtractor.from_pretrained(name)
    model = WhisperModel.from_pretrained(name, torch_dtype=torch.float16).encoder
elif key == 'w2vbert':
    from transformers import AutoFeatureExtractor, Wav2Vec2BertModel
    fe = AutoFeatureExtractor.from_pretrained(name)
    model = Wav2Vec2BertModel.from_pretrained(name, torch_dtype=torch.float16)
else:
    from transformers import AutoFeatureExtractor, AutoModel
    fe = AutoFeatureExtractor.from_pretrained(name)
    model = AutoModel.from_pretrained(name, torch_dtype=torch.float16)
model.to(dev).eval()


@torch.inference_mode()
def embed(x):
    if key.startswith('whisper'):
        inp = fe(x, sampling_rate=16000, return_tensors='pt').input_features.to(dev, torch.float16)
        hs = model(inp, output_hidden_states=True).hidden_states
        n = min(hs[0].shape[1], int(np.ceil(len(x) / 320)))  # 50 frames/s; drop 30-s padding
    elif key == 'w2vbert':
        inp = fe(x, sampling_rate=16000, return_tensors='pt').input_features.to(dev, torch.float16)
        hs = model(inp, output_hidden_states=True).hidden_states
        n = hs[0].shape[1]
    else:
        inp = fe(x, sampling_rate=16000, return_tensors='pt').input_values.to(dev, torch.float16)
        hs = model(inp, output_hidden_states=True).hidden_states
        n = hs[0].shape[1]
    h = torch.stack(hs)[:, 0, :n].float()  # [L, T, H]
    return torch.cat([h.mean(1), h.std(1)], -1).cpu().numpy().astype(np.float16)


t = time.time()
out = None
for i, p in enumerate(paths):
    x, sr = sf.read(p, dtype='float32')
    assert sr == 16000 and x.ndim == 1
    e = embed(x)
    if out is None:
        out = np.zeros((len(paths),) + e.shape, np.float16)
    out[i] = e
    if i % 200 == 0:
        print(key, i, len(paths), f'{time.time() - t:.0f}s', flush=True)
assert np.isfinite(out.astype(np.float32)).all()
np.save(f'feats/{key}.npy', out)
print('done', key, out.shape, f'{time.time() - t:.0f}s')
