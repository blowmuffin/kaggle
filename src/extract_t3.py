"""Tier-3 local features. Usage: python src/extract_t3.py smile|clap
smile -> feats/egemaps.npy [N,88], feats/compare.npy [N,6373] (openSMILE functionals, CPU)
clap  -> feats/clap.npy [N,512] audio embeddings, feats/clap_zs.npy [N,4] prompt similarities (laion/larger_clap_general)
Rows = train_meta then test_meta."""
import os, sys, time
os.environ.setdefault('HF_HOME', 'D:/kaggle/hf_cache')
import numpy as np, pandas as pd, soundfile as sf

R = 'data/SRCASW-BhavVaani/'
tr = pd.read_csv('feats/train_meta.csv'); te = pd.read_csv('feats/test_meta.csv')
paths = [R + 'train/' + f for f in tr.filename] + [R + 'test/' + f for f in te.filename]
t = time.time()
if sys.argv[1] == 'smile':
    import opensmile
    for name, fs in [('egemaps', opensmile.FeatureSet.eGeMAPSv02), ('compare', opensmile.FeatureSet.ComParE_2016)]:
        sm = opensmile.Smile(feature_set=fs, feature_level=opensmile.FeatureLevel.Functionals)
        X = np.stack([sm.process_signal(sf.read(p, dtype='float32')[0], 16000).to_numpy()[0] for p in paths]).astype(np.float32)
        np.save(f'feats/{name}.npy', np.nan_to_num(X)); print(name, X.shape, f'{time.time() - t:.0f}s', flush=True)
else:
    import torch
    from scipy.signal import resample_poly
    from transformers import ClapModel, ClapProcessor
    name = 'laion/larger_clap_general'
    proc = ClapProcessor.from_pretrained(name); model = ClapModel.from_pretrained(name).cuda().eval()
    prompts = ['a person speaking angrily', 'a person speaking happily', 'a person speaking in a calm neutral tone', 'a person speaking sadly']
    with torch.inference_mode():
        tx = model.get_text_features(**proc(text=prompts, return_tensors='pt', padding=True).to('cuda'))
        tx = tx / tx.norm(dim=-1, keepdim=True)
        E = []
        for p in paths:
            x = resample_poly(sf.read(p, dtype='float32')[0], 3, 1).astype(np.float32)  # CLAP expects 48 kHz
            a = model.get_audio_features(**proc(audios=[x], sampling_rate=48000, return_tensors='pt').to('cuda'))
            E.append((a / a.norm(dim=-1, keepdim=True))[0].cpu().numpy())
    E = np.stack(E).astype(np.float32)
    np.save('feats/clap.npy', E); np.save('feats/clap_zs.npy', E @ tx.cpu().numpy().T)
    print('clap', E.shape, f'{time.time() - t:.0f}s', flush=True)
