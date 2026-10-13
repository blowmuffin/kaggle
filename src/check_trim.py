"""Does feeding Whisper batch-length input (no 30-s padding) keep its emotion features? Frozen probe on trimmed features."""
import os
os.environ.setdefault('HF_HOME', 'D:/kaggle/hf_cache')
import numpy as np, pandas as pd, soundfile as sf, torch, torch.nn.functional as F
from transformers import WhisperFeatureExtractor, WhisperForAudioClassification

R = 'data/SRCASW-BhavVaani/'
tr = pd.read_csv('feats/train_meta.csv'); te = pd.read_csv('feats/test_meta.csv')
paths = [R + 'train/' + f for f in tr.filename] + [R + 'test/' + f for f in te.filename]
fe = WhisperFeatureExtractor.from_pretrained('openai/whisper-large-v3')
e = WhisperForAudioClassification.from_pretrained('openai/whisper-large-v3', torch_dtype=torch.float16).encoder.cuda().eval()
LAYERS = [16, 24, 32]
PAD = float(os.environ.get('PAD_S', '0'))  # 0 = batch-longest, else pad audio to PAD seconds
out = np.zeros((len(paths), len(LAYERS), 2 * e.config.d_model), np.float16)
with torch.inference_mode():
    for i, p in enumerate(paths):
        a = sf.read(p, dtype='float32')[0]
        kw = dict(padding='max_length', max_length=int(PAD * 16000)) if PAD else dict(padding='longest')
        x = fe([a], sampling_rate=16000, return_tensors='pt', **kw).input_features.cuda().half()
        n = int(np.ceil(len(a) / 320))  # valid frames
        h = F.gelu(e.conv2(F.gelu(e.conv1(x)))).permute(0, 2, 1)
        h = h + e.embed_positions.weight[:h.shape[1]]
        hs = [h]
        for layer in e.layers:
            h = layer(h, None, None)[0]; hs.append(h)
        hs[-1] = e.layer_norm(h)
        out[i] = torch.stack([torch.cat([hs[l][0, :n].float().mean(0), hs[l][0, :n].float().std(0)]) for l in LAYERS]).cpu().numpy()
np.save(f'feats/whisper_pad{int(PAD)}.npy', out)
from probe import cv
for j, l in enumerate(LAYERS):
    print(f'pad {PAD:.0f}s whisper L{l}: {cv(out[:, j].astype(np.float32), C=0.03)[0]:.4f}  (30-s padded L32 was 0.913)', flush=True)
