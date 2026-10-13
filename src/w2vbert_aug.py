"""Spike L1: w2v-bert-2.0 layers 10 & 16 mean+std (all frames, as extract.py) for the 4 S3 views of every clip
(noise 20/30 dB, 0.25/0.5 s lead silence; same RNG as whisper_aug.py). -> feats/w2vbert_aug.npy [1035, 4, 2, 2048] fp16"""
import os, time
os.environ.setdefault('HF_HOME', 'D:/kaggle/hf_cache')
import numpy as np, pandas as pd, soundfile as sf, torch
from transformers import AutoFeatureExtractor, Wav2Vec2BertModel

R = 'data/SRCASW-BhavVaani/'
tr = pd.read_csv('feats/train_meta.csv'); te = pd.read_csv('feats/test_meta.csv')
paths = [R + 'train/' + f for f in tr.filename] + [R + 'test/' + f for f in te.filename]
fe = AutoFeatureExtractor.from_pretrained('facebook/w2v-bert-2.0')
model = Wav2Vec2BertModel.from_pretrained('facebook/w2v-bert-2.0', torch_dtype=torch.float16).cuda().eval()


def views(a, rng):
    noise = lambda db: a + rng.standard_normal(len(a)).astype(np.float32) * np.sqrt(np.mean(a ** 2) / 10 ** (db / 10))
    lead = lambda s: np.concatenate([np.zeros(int(s * 16000), np.float32), a])
    return [noise(20), noise(30), lead(0.25), lead(0.5)]


out = np.zeros((len(paths), 4, 2, 2048), np.float16); t = time.time()
with torch.inference_mode():
    for i, p in enumerate(paths):
        for j, v in enumerate(views(sf.read(p, dtype='float32')[0], np.random.default_rng(i))):
            inp = fe(v, sampling_rate=16000, return_tensors='pt').input_features.cuda().half()
            hs = model(inp, output_hidden_states=True).hidden_states
            out[i, j] = torch.stack([torch.cat([hs[l][0].float().mean(0), hs[l][0].float().std(0)]) for l in (10, 16)]).cpu().numpy()
        if i % 200 == 0:
            print(i, len(paths), f'{time.time() - t:.0f}s', flush=True)
assert np.isfinite(out.astype(np.float32)).all()
np.save('feats/w2vbert_aug.npy', out)
print('done', out.shape, f'{time.time() - t:.0f}s')
