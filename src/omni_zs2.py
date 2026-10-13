"""v2 (for 10 Oct): Qwen2.5-Omni-7B thinker, 3 prompt phrasings (log-probs each) + hidden means over audio tokens at LLM
layers 7/14/21/28 (prompt 0), plus K4: audio + "Transcript: <Whisper ASR>" + prompt-0 question -> its log-probs (prompt 3) and
the LAST-token hidden at layers 7/14/21/28 (only positions after the transcript see both modalities in a causal LLM).
Kaggle 2xT4, mounted model, hard budget. Outputs omni2_lp.npy [N,4,4] (0-2 = omni-zs2, 3 = K4), omni2_h.npy [N,4,3584],
omni2_th.npy [N,4,3584] (K4). Row order = train.csv rows then test.csv rows. Transcripts: ASR_B64 (injected into the kernel
copy) or feats/asr.csv locally. Local smoke test (tiny random weights): SMOKE=1"""
import os, sys, subprocess, time, glob, json, gzip, base64
T0 = time.time()
SMOKE = bool(os.environ.get('SMOKE'))
if not SMOKE:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'transformers==4.56.2'], check=False)
BUDGET_S = float(os.environ.get('BUDGET_MIN', '75')) * 60
import numpy as np, pandas as pd, soundfile as sf, torch
from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration

ASR_B64 = None  # replaced by a gzip+base64 JSON list of transcripts in kernels/dataverse-omni-zs2
if SMOKE:
    os.environ.setdefault('HF_HOME', 'D:/kaggle/hf_cache')
    root, OUT, mpath = 'data/SRCASW-BhavVaani', os.environ.get('SMOKE_OUT', '.'), 'Qwen/Qwen2.5-Omni-7B'
    from transformers import Qwen2_5OmniConfig
    cfg = Qwen2_5OmniConfig.from_pretrained(mpath).thinker_config
    cfg.audio_config.update(dict(encoder_layers=2, d_model=64, encoder_attention_heads=2, encoder_ffn_dim=128, output_dim=64))
    cfg.text_config.update(dict(hidden_size=64, intermediate_size=128, num_hidden_layers=28, num_attention_heads=2, num_key_value_heads=2))
    cfg.text_config.rope_scaling = {**cfg.text_config.rope_scaling, 'mrope_section': [4, 6, 6]}  # head_dim 32 -> 16 rotary pairs
    cfg.vision_config.update(dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64, fullatt_block_indexes=[0]))
    model = Qwen2_5OmniThinkerForConditionalGeneration(cfg).eval()
else:
    root = os.path.dirname(sorted(glob.glob('/kaggle/input/**/train.csv', recursive=True))[0]); OUT = '/kaggle/working'
    cfgs = [c for c in glob.glob('/kaggle/input/**/config.json', recursive=True) if 'omni' in c.lower()]
    if not cfgs:
        for d, _, fs in os.walk('/kaggle/input'):
            if d.count(os.sep) <= 6: print('DIR', d, fs[:5])
        raise SystemExit('Qwen2.5-Omni model not mounted')
    mpath = os.path.dirname(sorted(cfgs, key=len)[0])
    model, info = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        mpath, torch_dtype=torch.float16, device_map='auto', max_memory={0: '14GiB', 1: '14GiB'}, output_loading_info=True)
    model.eval()
    print('missing keys:', len(info['missing_keys']), info['missing_keys'][:3], flush=True)
tr = pd.read_csv(f'{root}/train.csv'); te = pd.read_csv(f'{root}/test.csv')
paths = [f'{root}/train/{f}' for f in tr.filename] + [f'{root}/test/{f}' for f in te.filename]
N, ntr = len(paths), len(tr)
asr = json.loads(gzip.decompress(base64.b64decode(ASR_B64))) if ASR_B64 else pd.read_csv('feats/asr.csv').text.fillna('').tolist()
assert len(asr) == N, (len(asr), N)
order = list(range(ntr, N)) + list(np.random.RandomState(0).permutation(ntr))  # test first, then train shuffled
if SMOKE: order = order[:2] + [order[-1]]
print(f'data {root} model {mpath} gpus={torch.cuda.device_count()} loaded {time.time() - T0:.0f}s', flush=True)

proc = Qwen2_5OmniProcessor.from_pretrained(mpath)
AUDIO = model.config.audio_token_index
SYS = ('You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and '
       'visual inputs, as well as generating text and speech.')
QS = ['What emotion does the speaker express in this Hindi audio? Choose one: angry, happy, neutral, sad. Answer with one word.',
      "Listen to the speaker's tone of voice, not just the words. Is the speaker angry, happy, neutral, or sad? Answer with one word.",
      'Classify the emotion of this Hindi speech clip as angry, happy, neutral or sad, considering pitch, loudness, speaking rate and the meaning of the words. Answer with one word.']
chat = lambda q: proc.apply_chat_template([{'role': 'system', 'content': [{'type': 'text', 'text': SYS}]},
          {'role': 'user', 'content': [{'type': 'audio', 'audio': 'x.wav'}, {'type': 'text', 'text': q}]}],
          add_generation_prompt=True, tokenize=False)
texts = [chat(q) for q in QS]
HL = [7, 14, 21, 28]
tok = proc.tokenizer
CAND = [sorted({tok.encode(v, add_special_tokens=False)[0] for v in (l, l.capitalize(), ' ' + l)})
        for l in ['angry', 'happy', 'neutral', 'sad']]
assert len({t for c in CAND for t in c}) == sum(len(c) for c in CAND), CAND
print('K4 prompt example:', repr(chat(f'Transcript: {asr[0]}\n{QS[0]}')[-260:]), flush=True)

H = model.config.text_config.hidden_size
lp = np.full((N, len(QS) + 1, 4), np.nan, np.float32); hid = np.zeros((N, len(HL), H), np.float16); thid = np.zeros((N, len(HL), H), np.float16)
dev0 = next(model.parameters()).device
save = lambda: (np.save(f'{OUT}/omni2_lp.npy', lp), np.save(f'{OUT}/omni2_h.npy', hid), np.save(f'{OUT}/omni2_th.npy', thid))
done = 0
with torch.inference_mode():
    for c, i in enumerate(order):
        if time.time() - T0 > BUDGET_S:
            print(f'budget hit after {done} clips', flush=True); break
        x = sf.read(paths[i], dtype='float32')[0]
        for qi, text in enumerate(texts + [chat(f'Transcript: {asr[i]}\n{QS[0]}')]):
            inp = proc(text=text, audio=[x], return_tensors='pt', padding=True).to(dev0)
            inp['input_features'] = inp['input_features'].to(model.dtype)
            out = model(**inp, output_hidden_states=qi in (0, 3), use_cache=False)
            logp = torch.log_softmax(out.logits[0, -1].float(), -1)
            lp[i, qi] = [torch.logsumexp(logp[cs], 0).item() for cs in CAND]
            if qi == 0:
                m = (inp['input_ids'][0] == AUDIO); hs = out.hidden_states
                hid[i] = torch.stack([hs[L][0][m.to(hs[L].device)].float().mean(0).cpu() for L in HL]).numpy()
            if qi == 3:  # K4: last-token hidden sees audio + transcript + question
                hs = out.hidden_states; thid[i] = torch.stack([hs[L][0, -1].float().cpu() for L in HL]).numpy()
        done += 1
        if c % 100 == 0:
            p = np.exp(lp[i]); print(f'{c} clips {time.time() - T0:.0f}s | audio tokens {int(m.sum())} | mass on 4 labels per prompt {p.sum(1).round(2)}', flush=True); save()
save()
C = ['angry', 'happy', 'neutral', 'sad']
y = tr.emotion.map({c: k for k, c in enumerate(C)}).to_numpy()
ok = ~np.isnan(lp[:ntr]).any((1, 2))
if ok.sum() > 10:
    from sklearn.metrics import f1_score
    for qi in range(lp.shape[1]):
        print(f'prompt {qi} zero-shot macro-F1 on {ok.sum()} train clips: {f1_score(y[ok], lp[:ntr][ok, qi].argmax(1), average="macro"):.4f}', flush=True)
print(f'processed {done}/{N}, total {time.time() - T0:.0f}s', flush=True)
