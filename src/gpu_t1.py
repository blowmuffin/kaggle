"""Tier-1 GPU extraction. Usage: python src/gpu_t1.py spk|asr
spk -> feats/spk.npy [N,512] WavLM-base-plus-sv x-vectors (speaker identity)
asr -> feats/asr.csv (row, Id, split, text) Whisper-large-v3 Hindi transcripts. Rows = train_meta then test_meta."""
import os, sys, time
os.environ.setdefault('HF_HOME', 'D:/kaggle/hf_cache')
import numpy as np, pandas as pd, soundfile as sf, torch

R = 'data/SRCASW-BhavVaani/'
tr = pd.read_csv('feats/train_meta.csv'); te = pd.read_csv('feats/test_meta.csv')
paths = [R + 'train/' + f for f in tr.filename] + [R + 'test/' + f for f in te.filename]
wavs = [sf.read(p, dtype='float32')[0] for p in paths]
t = time.time()
with torch.inference_mode():
    if sys.argv[1] == 'spk':
        from transformers import AutoFeatureExtractor, WavLMForXVector
        fe = AutoFeatureExtractor.from_pretrained('microsoft/wavlm-base-plus-sv')
        m = WavLMForXVector.from_pretrained('microsoft/wavlm-base-plus-sv').cuda().eval()
        E = np.stack([m(**fe([x], sampling_rate=16000, return_tensors='pt').to('cuda')).embeddings[0].cpu().numpy() for x in wavs])
        np.save('feats/spk.npy', E.astype(np.float32)); print('spk', E.shape, f'{time.time() - t:.0f}s', flush=True)
    else:
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        proc = WhisperProcessor.from_pretrained('openai/whisper-large-v3')
        m = WhisperForConditionalGeneration.from_pretrained('openai/whisper-large-v3', torch_dtype=torch.float16).cuda().eval()
        out = []
        for i in range(0, len(wavs), 8):
            feats = proc(wavs[i:i + 8], sampling_rate=16000, return_tensors='pt').input_features.cuda().half()
            ids = m.generate(feats, language='hi', task='transcribe', max_new_tokens=128)
            out += proc.batch_decode(ids, skip_special_tokens=True)
            if i % 200 == 0:
                print(i, f'{time.time() - t:.0f}s', out[-1][:60], flush=True)
        ids_ = list(tr.Id) + list(te.Id)
        pd.DataFrame({'row': range(len(out)), 'Id': ids_, 'split': ['train'] * len(tr) + ['test'] * len(te),
                      'text': [s.strip() for s in out]}).to_csv('feats/asr.csv', index=False, encoding='utf-8')
        print('asr done', len(out), f'{time.time() - t:.0f}s', flush=True)
