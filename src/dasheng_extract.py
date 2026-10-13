"""R1: Dasheng-1.2B (mispeech/dasheng-1.2B, Apache-2.0) frozen features. Mean+std pooled over valid tokens (25/s) of
block 30 of 40 (3/4 depth, forward hook) and the final normed output. -> feats/dasheng.npy [N, 2, 3072] fp16"""
import os, time
os.environ.setdefault('HF_HOME', 'D:/kaggle/hf_cache')
import numpy as np, pandas as pd, soundfile as sf, torch
from transformers import AutoModel, AutoFeatureExtractor

REPO = 'mispeech/dasheng-1.2B'
R = 'data/SRCASW-BhavVaani/'
tr = pd.read_csv('feats/train_meta.csv'); te = pd.read_csv('feats/test_meta.csv')
paths = [R + 'train/' + f for f in tr.filename] + [R + 'test/' + f for f in te.filename]
fe = AutoFeatureExtractor.from_pretrained(REPO, trust_remote_code=True)
model = AutoModel.from_pretrained(REPO, outputdim=None, trust_remote_code=True).half().cuda().eval()
hooked = {}
model.encoder.blocks[29].register_forward_hook(lambda m, i, o: hooked.__setitem__('h', o))
pool = lambda h: torch.cat([h.float().mean(0), h.float().std(0)])

out = np.zeros((len(paths), 2, 2 * 1536), np.float16); t = time.time()
with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):  # remote code feeds a float32 tensor into a conv
    for i, p in enumerate(paths):
        x = fe(torch.from_numpy(sf.read(p, dtype='float32')[0])[None], sampling_rate=16000, return_tensors='pt').input_values.cuda().half()
        last = model(x).hidden_states[0]
        nv = x.shape[-1] // 4  # valid tokens (padded tail of a >10 s clip's last split is cut)
        mid = hooked['h'].reshape(-1, 1536)[:nv]
        assert last.shape[0] == nv or last.shape[0] == x.shape[-1] // 4, (last.shape, nv)
        out[i] = torch.stack([pool(mid), pool(last[:nv])]).cpu().numpy()
        if i % 200 == 0:
            print(i, len(paths), f'{time.time() - t:.0f}s tokens {nv}', flush=True)
assert np.isfinite(out.astype(np.float32)).all()
np.save('feats/dasheng.npy', out)
print('done', out.shape, f'{time.time() - t:.0f}s')
