"""Frozen audio encoder of Qwen2-Audio-7B-Instruct (Whisper-large-v3 init, further trained in audio-LLM pretraining
incl. emotion tasks). Loads only 'audio_tower.*' from shard 1. 30-s padding (as trained), pool valid frames.
-> feats/qa_enc.npy [N, 3, 2560] for hidden states 16, 24, 32 (32 = after avg-pool + LN)."""
import os, time
os.environ.setdefault('HF_HOME', 'D:/kaggle/hf_cache')
import numpy as np, pandas as pd, soundfile as sf, torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import Qwen2AudioConfig, WhisperFeatureExtractor
from transformers.models.qwen2_audio.modeling_qwen2_audio import Qwen2AudioEncoder

REPO = 'Qwen/Qwen2-Audio-7B-Instruct'
cfg = Qwen2AudioConfig.from_pretrained(REPO)
enc = Qwen2AudioEncoder(cfg.audio_config)
sd = {}
with safe_open(hf_hub_download(REPO, 'model-00001-of-00005.safetensors'), 'pt') as f:
    for k in f.keys():
        if k.startswith('audio_tower.'):
            sd[k[len('audio_tower.'):]] = f.get_tensor(k)
miss, unexp = enc.load_state_dict(sd, strict=False)
print('missing', miss, '| unexpected', unexp, flush=True)
assert not miss
enc = enc.half().cuda().eval()
fe = WhisperFeatureExtractor.from_pretrained(REPO)
R = 'data/SRCASW-BhavVaani/'
tr = pd.read_csv('feats/train_meta.csv'); te = pd.read_csv('feats/test_meta.csv')
paths = [R + 'train/' + f for f in tr.filename] + [R + 'test/' + f for f in te.filename]
LAYERS = [16, 24, 32]; out = np.zeros((len(paths), len(LAYERS), 2 * cfg.audio_config.d_model), np.float32); t = time.time()
with torch.inference_mode():
    for i, p in enumerate(paths):
        x = sf.read(p, dtype='float32')[0][:30 * 16000]
        hs = enc(fe(x, sampling_rate=16000, return_tensors='pt').input_features.cuda().half(), output_hidden_states=True).hidden_states
        for j, l in enumerate(LAYERS):
            h = hs[l][0].float(); nv = max(1, int(np.ceil(len(x) / 320 * h.shape[0] / 1500)))
            out[i, j] = torch.cat([h[:nv].mean(0), h[:nv].std(0)]).cpu().numpy()
np.save('feats/qa_enc.npy', out); print('qa_enc', out.shape, f'{time.time() - t:.0f}s', flush=True)
