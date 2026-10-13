"""L3/L4 frame cache + L2 views, whisper-large-v3 (custom forward = whisper_aug.py, 30-s pad).
python src/whisper_frames.py frames -> feats/whisper_frames.npy [sum valid frames, 2 (L24, L32), 1280] fp16 + feats/whisper_frames_off.npy
python src/whisper_frames.py aug2   -> feats/whisper_aug2.npy [1035, 4 views (speed 0.9, 1.1, pitch -1, +1 st), 2, 2560] fp16
python src/whisper_frames.py framesaug -> feats/whisper_frames_v{0..3}.npy (+ _off) for the 4 S3 views (same RNG as whisper_aug.py)
python src/whisper_frames.py mix    -> feats/whisper_mix.npy [1035, 2 views (random other clip mixed at -10 / -15 dB), 2, 2560] (P1)"""
import os, sys, time
os.environ.setdefault('HF_HOME', 'D:/kaggle/hf_cache')
import numpy as np, pandas as pd, soundfile as sf, torch, torch.nn.functional as F
from transformers import WhisperFeatureExtractor, WhisperForAudioClassification

mode = sys.argv[1]
R = 'data/SRCASW-BhavVaani/'
tr = pd.read_csv('feats/train_meta.csv'); te = pd.read_csv('feats/test_meta.csv')
paths = [R + 'train/' + f for f in tr.filename] + [R + 'test/' + f for f in te.filename]
fe = WhisperFeatureExtractor.from_pretrained('openai/whisper-large-v3')
e = WhisperForAudioClassification.from_pretrained('openai/whisper-large-v3', torch_dtype=torch.float16).encoder.cuda().eval()


def encode(vs):
    """List of waveforms -> {24: [B,1500,1280], 32: [B,1500,1280]} (32 after final LayerNorm, as HF hidden_states[-1])."""
    h = fe(vs, sampling_rate=16000, return_tensors='pt').input_features.cuda().half()
    h = F.gelu(e.conv2(F.gelu(e.conv1(h)))).permute(0, 2, 1)
    h = h + e.embed_positions.weight[:h.shape[1]]
    for li, layer in enumerate(e.layers, 1):
        h = layer(h, None, None)[0]
        if li == 24: h24 = h
    return {24: h24, 32: e.layer_norm(h)}


def views2(a):
    import librosa
    speed = lambda s: librosa.resample(a, orig_sr=int(16000 * s), target_sr=16000)
    pitch = lambda st: librosa.effects.pitch_shift(a, sr=16000, n_steps=st)
    return [speed(0.9), speed(1.1), pitch(-1), pitch(1)]


def views1(a, rng):  # = whisper_aug.views without the loop view (same RNG order -> identical noise)
    noise = lambda db: a + rng.standard_normal(len(a)).astype(np.float32) * np.sqrt(np.mean(a ** 2) / 10 ** (db / 10))
    lead = lambda s: np.concatenate([np.zeros(int(s * 16000), np.float32), a])
    return [noise(20), noise(30), lead(0.25), lead(0.5)]


wavs = lambda: (sf.read(p, dtype='float32')[0] for p in paths)
t = time.time()
with torch.inference_mode():
    if mode == 'frames':
        nv = [int(np.ceil(sf.info(p).frames / 320)) for p in paths]
        off = np.concatenate([[0], np.cumsum(nv)])
        out = np.zeros((off[-1], 2, 1280), np.float16)
        for i, a in enumerate(wavs()):
            hs = encode([a])
            out[off[i]:off[i + 1]] = torch.stack([hs[24][0, :nv[i]], hs[32][0, :nv[i]]], 1).cpu().numpy()
            if i % 200 == 0: print(i, len(paths), f'{time.time() - t:.0f}s', flush=True)
        np.save('feats/whisper_frames_off.npy', off)
    elif mode == 'mix':  # P1 (SAILER audio mixing): clip + random other clip (labels unused) at -10 / -15 dB relative power
        A = [sf.read(p, dtype='float32')[0] for p in paths]
        out = np.zeros((len(paths), 2, 2, 2560), np.float16)
        for i, a in enumerate(A):
            rng = np.random.default_rng(10_000 + i); vs = []
            for db in (10, 15):
                j = rng.choice([q for q in range(len(A)) if q != i]); b = np.resize(A[j], len(a))
                vs.append(a + b * np.sqrt(np.mean(a ** 2) / (np.mean(b ** 2) + 1e-12) / 10 ** (db / 10)))
            hs = encode(vs); nv = int(np.ceil(len(a) / 320))
            for v in range(2):
                out[i, v] = torch.stack([torch.cat([hs[l][v, :nv].float().mean(0), hs[l][v, :nv].float().std(0)]) for l in (24, 32)]).cpu().numpy()
            if i % 200 == 0: print(i, len(paths), f'{time.time() - t:.0f}s', flush=True)
        assert np.isfinite(out.astype(np.float32)).all(); np.save('feats/whisper_mix.npy', out); print('done mix', out.shape); sys.exit()
    elif mode == 'framesaug':
        L = [sf.info(p).frames for p in paths]
        nv = np.array([[int(np.ceil(l / 320))] * 2 + [int(np.ceil((l + int(s * 16000)) / 320)) for s in (0.25, 0.5)] for l in L])
        offs = [np.concatenate([[0], np.cumsum(nv[:, j])]) for j in range(4)]
        outs = [np.lib.format.open_memmap(f'feats/whisper_frames_v{j}.npy', mode='w+', dtype=np.float16, shape=(int(o[-1]), 2, 1280))
                for j, o in enumerate(offs)]  # on disk: 4 x 0.85 GB in RAM ran out next to other jobs
        for i, a in enumerate(wavs()):
            hs = encode(views1(a, np.random.default_rng(i)))
            for j in range(4):
                outs[j][offs[j][i]:offs[j][i + 1]] = torch.stack([hs[24][j, :nv[i, j]], hs[32][j, :nv[i, j]]], 1).cpu().numpy()
            if i % 200 == 0: print(i, len(paths), f'{time.time() - t:.0f}s', flush=True)
        for j in range(4):
            outs[j].flush(); assert all(np.isfinite(outs[j][r:r + 20000]).all() for r in range(0, len(outs[j]), 20000))
            np.save(f'feats/whisper_frames_v{j}_off.npy', offs[j])
        print('done framesaug', [o.shape for o in outs], f'{time.time() - t:.0f}s'); sys.exit()
    else:
        out = np.zeros((len(paths), 4, 2, 2560), np.float16)
        for i, a in enumerate(wavs()):
            vs = views2(a); hs = encode(vs)
            for j, v in enumerate(vs):
                n = int(np.ceil(len(v) / 320))
                out[i, j] = torch.stack([torch.cat([hs[l][j, :n].float().mean(0), hs[l][j, :n].float().std(0)]) for l in (24, 32)]).cpu().numpy()
            if i % 200 == 0: print(i, len(paths), f'{time.time() - t:.0f}s', flush=True)
assert np.isfinite(out.astype(np.float32)).all()
np.save('feats/whisper_frames.npy' if mode == 'frames' else 'feats/whisper_aug2.npy', out)
print('done', mode, out.shape, f'{time.time() - t:.0f}s')
