"""Cache the frozen Whisper-large-v3 trunk once: input to encoder layer 24 for every clip, audio zero-padded to 20 s.
-> feats/wcache_L24_p20.npy [N, 1000, 1280] fp16 (memmap), feats/wcache_nvalid.npy [N] valid frames (50/s)."""
import os, time
os.environ.setdefault('HF_HOME', 'D:/kaggle/hf_cache')
import numpy as np, pandas as pd, soundfile as sf, torch, torch.nn.functional as F
from transformers import WhisperFeatureExtractor, WhisperForAudioClassification

R, PAD, CUT = 'data/SRCASW-BhavVaani/', 20, 24
tr = pd.read_csv('feats/train_meta.csv'); te = pd.read_csv('feats/test_meta.csv')
paths = [R + 'train/' + f for f in tr.filename] + [R + 'test/' + f for f in te.filename]
fe = WhisperFeatureExtractor.from_pretrained('openai/whisper-large-v3')
e = WhisperForAudioClassification.from_pretrained('openai/whisper-large-v3', torch_dtype=torch.float16).encoder.cuda().eval()
T = PAD * 50
out = np.lib.format.open_memmap('feats/wcache_L24_p20.npy', mode='w+', dtype=np.float16, shape=(len(paths), T, e.config.d_model))
nv = np.zeros(len(paths), np.int32); t = time.time()
with torch.inference_mode():
    for i, p in enumerate(paths):
        a = sf.read(p, dtype='float32')[0]
        x = fe([a], sampling_rate=16000, padding='max_length', max_length=PAD * 16000, return_tensors='pt').input_features.cuda().half()
        h = F.gelu(e.conv2(F.gelu(e.conv1(x)))).permute(0, 2, 1)
        h = h + e.embed_positions.weight[:h.shape[1]]
        for layer in e.layers[:CUT]:
            h = layer(h, None, None)[0]
        out[i] = h[0].cpu().numpy(); nv[i] = int(np.ceil(len(a) / 320))
out.flush(); np.save('feats/wcache_nvalid.npy', nv)
print('cache', out.shape, f'{time.time() - t:.0f}s', flush=True)
