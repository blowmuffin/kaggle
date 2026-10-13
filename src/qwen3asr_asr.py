"""Fix F3 (pre-registered): Hindi transcripts with Qwen3-ASR-1.7B (Apache-2.0), language forced to Hindi, greedy decoding.
Needs transformers >= 5 -> ../venv5/Scripts/python src/qwen3asr_asr.py  -> feats/asr_qwen3.csv (row, Id, split, text; canonical order)"""
import os, time
os.environ.setdefault('HF_HOME', 'D:/kaggle/hf_cache')
import pandas as pd, soundfile as sf, torch
from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

REPO = 'Qwen/Qwen3-ASR-1.7B-hf'
proc = AutoProcessor.from_pretrained(REPO)
model = Qwen3ASRForConditionalGeneration.from_pretrained(REPO, dtype=torch.bfloat16).cuda().eval()
R = 'data/SRCASW-BhavVaani/'
tr = pd.read_csv('feats/train_meta.csv'); te = pd.read_csv('feats/test_meta.csv')
rows = [(i, r.Id, 'train', R + 'train/' + r.filename) for i, r in enumerate(tr.itertuples())] + \
       [(len(tr) + i, r.Id, 'test', R + 'test/' + r.filename) for i, r in enumerate(te.itertuples())]
PART = 'feats/asr_qwen3.part.csv'  # resume point: the session was cut off mid-run before
out = [tuple(r) for r in pd.read_csv(PART, encoding='utf-8').fillna('').itertuples(index=False)] if os.path.exists(PART) else []; t = time.time()
print('resuming after', len(out), 'clips', flush=True)
with torch.inference_mode():
    for k, (row, Id, split, p) in enumerate(rows):
        if k < len(out): continue
        inp = proc.apply_transcription_request(sf.read(p, dtype='float32')[0], language='Hindi').to('cuda', torch.bfloat16)
        g = model.generate(**inp, max_new_tokens=128, do_sample=False)
        text = proc.batch_decode(g[:, inp['input_ids'].shape[1]:], skip_special_tokens=True)[0].strip()
        out.append((row, Id, split, text))
        if k % 100 == 0:
            print(k, len(rows), f'{time.time() - t:.0f}s', repr(text[:60]), flush=True)
            pd.DataFrame(out, columns=['row', 'Id', 'split', 'text']).to_csv(PART, index=False, encoding='utf-8')
pd.DataFrame(out, columns=['row', 'Id', 'split', 'text']).to_csv('feats/asr_qwen3.csv', index=False, encoding='utf-8')
print('done', len(out), f'{time.time() - t:.0f}s')
