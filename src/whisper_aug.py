"""Spike S3/S3b: whisper-large-v3 L24/L32 mean+std features for augmented views of every clip (custom forward = check_trim.py).
Views: noise 20 dB, noise 30 dB, 0.25 s lead silence, 0.5 s lead silence, loop-pad to 30 s. Each view is a new waveform:
30-s pad, pool over its valid frames. -> feats/whisper_aug.npy [1035, 5 views, 2 layers (24, 32), 2560] fp16"""
import os, time
os.environ.setdefault('HF_HOME', 'D:/kaggle/hf_cache')
import numpy as np, pandas as pd, soundfile as sf, torch, torch.nn.functional as F
from transformers import WhisperFeatureExtractor, WhisperForAudioClassification

R = 'data/SRCASW-BhavVaani/'
tr = pd.read_csv('feats/train_meta.csv'); te = pd.read_csv('feats/test_meta.csv')
paths = [R + 'train/' + f for f in tr.filename] + [R + 'test/' + f for f in te.filename]
fe = WhisperFeatureExtractor.from_pretrained('openai/whisper-large-v3')
e = WhisperForAudioClassification.from_pretrained('openai/whisper-large-v3', torch_dtype=torch.float16).encoder.cuda().eval()
VIEWS = ['noise20', 'noise30', 'shift25', 'shift50', 'loop']


def views(a, rng):
    noise = lambda db: a + rng.standard_normal(len(a)).astype(np.float32) * np.sqrt(np.mean(a ** 2) / 10 ** (db / 10))
    lead = lambda s: np.concatenate([np.zeros(int(s * 16000), np.float32), a])
    return [noise(20), noise(30), lead(0.25), lead(0.5), np.tile(a, int(np.ceil(480000 / len(a))))[:480000]]


out = np.zeros((len(paths), len(VIEWS), 2, 2 * e.config.d_model), np.float16)
t = time.time()
with torch.inference_mode():
    for i, p in enumerate(paths):
        vs = views(sf.read(p, dtype='float32')[0], np.random.default_rng(i))
        h = fe(vs, sampling_rate=16000, return_tensors='pt').input_features.cuda().half()
        h = F.gelu(e.conv2(F.gelu(e.conv1(h)))).permute(0, 2, 1)
        h = h + e.embed_positions.weight[:h.shape[1]]
        for li, layer in enumerate(e.layers, 1):
            h = layer(h, None, None)[0]
            if li == 24: h24 = h
        hs = {24: h24, 32: e.layer_norm(h)}
        for j, v in enumerate(vs):
            n = int(np.ceil(len(v) / 320))
            out[i, j] = torch.stack([torch.cat([hs[l][j, :n].float().mean(0), hs[l][j, :n].float().std(0)]) for l in (24, 32)]).cpu().numpy()
        if i % 100 == 0:
            print(i, len(paths), f'{time.time() - t:.0f}s', flush=True)
assert np.isfinite(out.astype(np.float32)).all()
np.save('feats/whisper_aug.npy', out)
print('done', out.shape, f'{time.time() - t:.0f}s')
