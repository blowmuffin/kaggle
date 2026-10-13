"""Qwen2.5-Omni-7B (thinker) as a typed 4-way emotion decision + audio-LLM features. Kaggle 2xT4, mounted model, hard budget.
Outputs: omni_lp.npy [N,4] log-probs (angry,happy,neutral,sad; NaN = not processed), omni_h.npy [N,2,3584] fp16
mean hidden state over audio tokens at LLM layers 14 and 28. Row order = train.csv rows then test.csv rows."""
import os, sys, subprocess, time, glob
T0 = time.time()
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'transformers==4.56.2'], check=False)
BUDGET_S = float(os.environ.get('BUDGET_MIN', '13')) * 60
import numpy as np, pandas as pd, soundfile as sf, torch
from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration

root = os.path.dirname(sorted(glob.glob('/kaggle/input/**/train.csv', recursive=True))[0])
cfgs = [c for c in glob.glob('/kaggle/input/**/config.json', recursive=True) if 'omni' in c.lower()]
if not cfgs:
    for d, _, fs in os.walk('/kaggle/input'):
        if d.count(os.sep) <= 6: print('DIR', d, fs[:5])
    raise SystemExit('Qwen2.5-Omni model not mounted')
mpath = os.path.dirname(sorted(cfgs, key=len)[0])
tr = pd.read_csv(f'{root}/train.csv'); te = pd.read_csv(f'{root}/test.csv')
paths = [f'{root}/train/{f}' for f in tr.filename] + [f'{root}/test/{f}' for f in te.filename]
N, ntr = len(paths), len(tr)
order = list(range(ntr, N)) + list(np.random.RandomState(0).permutation(ntr))  # test first, then train shuffled
print(f'data {root} model {mpath} gpus={torch.cuda.device_count()} {time.time() - T0:.0f}s', flush=True)

proc = Qwen2_5OmniProcessor.from_pretrained(mpath)
model, info = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
    mpath, torch_dtype=torch.float16, device_map='auto', max_memory={0: '14GiB', 1: '14GiB'}, output_loading_info=True)
model.eval()
print('missing keys:', len(info['missing_keys']), info['missing_keys'][:3], f'| loaded {time.time() - T0:.0f}s', flush=True)
AUDIO = model.config.audio_token_index
SYS = ('You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and '
       'visual inputs, as well as generating text and speech.')
Q = 'What emotion does the speaker express in this Hindi audio? Choose one: angry, happy, neutral, sad. Answer with one word.'
conv = [{'role': 'system', 'content': [{'type': 'text', 'text': SYS}]},
        {'role': 'user', 'content': [{'type': 'audio', 'audio': 'x.wav'}, {'type': 'text', 'text': Q}]}]
text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
tok = proc.tokenizer
CAND = [sorted({tok.encode(v, add_special_tokens=False)[0] for v in (l, l.capitalize(), ' ' + l)})
        for l in ['angry', 'happy', 'neutral', 'sad']]
assert len({t for c in CAND for t in c}) == sum(len(c) for c in CAND), CAND

lp = np.full((N, 4), np.nan, np.float32); hid = np.zeros((N, 2, model.config.text_config.hidden_size), np.float16)
dev0 = next(model.parameters()).device


def save():
    np.save('/kaggle/working/omni_lp.npy', lp); np.save('/kaggle/working/omni_h.npy', hid)


done = 0
with torch.inference_mode():
    for c, i in enumerate(order):
        if time.time() - T0 > BUDGET_S:
            print(f'budget hit after {done} clips', flush=True); break
        x = sf.read(paths[i], dtype='float32')[0]
        inp = proc(text=text, audio=[x], return_tensors='pt', padding=True).to(dev0)
        inp['input_features'] = inp['input_features'].half()
        out = model(**inp, output_hidden_states=True, use_cache=False)
        logp = torch.log_softmax(out.logits[0, -1].float(), -1)
        lp[i] = [torch.logsumexp(logp[cs], 0).item() for cs in CAND]
        m = (inp['input_ids'][0] == AUDIO)
        hs = out.hidden_states
        hid[i] = torch.stack([hs[14][0][m.to(hs[14].device)].float().mean(0).cpu(), hs[-1][0][m.to(hs[-1].device)].float().mean(0).cpu()]).numpy()
        done += 1
        if c % 100 == 0:
            p = np.exp(lp[i]); print(f'{c} clips {time.time() - T0:.0f}s | mass on 4 labels {p.sum():.2f}', flush=True); save()
save()
C = ['angry', 'happy', 'neutral', 'sad']
y = tr.emotion.map({c: k for k, c in enumerate(C)}).to_numpy()
ok = ~np.isnan(lp[:ntr]).any(1)
if ok.any():
    from sklearn.metrics import f1_score
    print(f'zero-shot macro-F1 on {ok.sum()} train clips: {f1_score(y[ok], lp[:ntr][ok].argmax(1), average="macro"):.4f}', flush=True)
print(f'processed {done}/{N}, total {time.time() - T0:.0f}s', flush=True)
