"""Spike K3 (pre-registered): Qwen3-ASR-1.7B (Apache-2.0) AuT audio encoder, frozen. Loads only the audio-tower weights.
Mean+std over all (valid, packed) tokens of the encoder layer at 3/4 depth (forward hook) and the final output (after ln_post).
Needs transformers >= 5 -> run with the side venv:  ../venv5/Scripts/python src/qwen3asr_enc.py  -> feats/qasr_enc.npy [N, 2, 2d] fp16"""
import os, time
os.environ.setdefault('HF_HOME', 'D:/kaggle/hf_cache')
import numpy as np, pandas as pd, soundfile as sf, torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoConfig, AutoModel, AutoFeatureExtractor

REPO = 'Qwen/Qwen3-ASR-1.7B-hf'
cfg = AutoConfig.from_pretrained(REPO)
enc = AutoModel.from_config(cfg.audio_config)
with safe_open(hf_hub_download(REPO, 'model.safetensors'), 'pt') as fh:
    keys = [k for k in fh.keys() if 'audio_tower.' in k]
    pre = keys[0][:keys[0].index('audio_tower.') + len('audio_tower.')]
    sd = {k[len(pre):]: fh.get_tensor(k) for k in keys}
miss, unexp = enc.load_state_dict(sd, strict=False)
print('prefix', pre, '| tensors', len(sd), '| missing', miss, '| unexpected', unexp, flush=True)
assert not miss and not unexp
enc = enc.to('cuda', torch.bfloat16).eval()
fe = AutoFeatureExtractor.from_pretrained(REPO)
NL = len(enc.layers); LI = 3 * NL // 4 - 1
grab = {}
enc.layers[LI].register_forward_hook(lambda m, i, o: grab.__setitem__('h', o[0] if isinstance(o, tuple) else o))

R = 'data/SRCASW-BhavVaani/'
tr = pd.read_csv('feats/train_meta.csv'); te = pd.read_csv('feats/test_meta.csv')
paths = [R + 'train/' + f for f in tr.filename] + [R + 'test/' + f for f in te.filename]
d = enc.config.d_model
out = np.zeros((len(paths), 2, 2 * d), np.float16); t = time.time()
pool = lambda h: torch.cat([h.float().mean(0), h.float().std(0)])
with torch.inference_mode():
    for i, p in enumerate(paths):
        f = fe(sf.read(p, dtype='float32')[0], sampling_rate=16000, return_tensors='pt', return_attention_mask=True)  # mask = valid mel frames
        if i == 0: print('feature keys', {k: tuple(v.shape) for k, v in f.items()}, '| layers', NL, 'hook layer', LI + 1, flush=True)
        last = enc(f['input_features'].to('cuda', torch.bfloat16), f['attention_mask'].cuda()).last_hidden_state
        h = grab['h']; h = h.reshape(-1, h.shape[-1]); last = last.reshape(-1, last.shape[-1])
        assert len(h) == len(last), (h.shape, last.shape)  # packed valid tokens only
        out[i] = torch.stack([pool(h), pool(last)]).cpu().numpy()
        if i % 200 == 0: print(i, len(paths), f'{time.time() - t:.0f}s tokens {len(last)}', flush=True)
assert np.isfinite(out.astype(np.float32)).all()
np.save('feats/qasr_enc.npy', out); print('done', out.shape, f'{time.time() - t:.0f}s')
