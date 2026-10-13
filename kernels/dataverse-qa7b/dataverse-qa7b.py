"""Spikes S5 + R3 (for 10 Oct, Kaggle 2xT4): full Qwen2-Audio-7B-Instruct LLM (Apache-2.0). Per clip: 2 prompts ->
log-probs of 4 labels (first token, summed over case/space variants); prompt 0 -> mean LLM hidden over audio tokens at 4
evenly spaced layers; R3 "+A" prompt (VoxEmo 2026): greedy one-sentence acoustic caption, then label log-probs after "Emotion:".
Outputs /kaggle/working/qa7b_lp.npy [N,3,4] (S5 = [:, :2]), qa7b_h.npy [N,4,4096] fp16, qa7b_cap.csv. Row order = train then test.
Local smoke test (tiny random weights, CPU): SMOKE=1 python src/qa7b_zs.py"""
import os, sys, subprocess, time, glob
T0 = time.time()
SMOKE = bool(os.environ.get('SMOKE'))
if not SMOKE:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'transformers==4.56.2'], check=False)
BUDGET_S = float(os.environ.get('BUDGET_MIN', '150')) * 60
import numpy as np, pandas as pd, soundfile as sf, torch
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration, Qwen2AudioConfig

REPO = 'Qwen/Qwen2-Audio-7B-Instruct'
if SMOKE:
    os.environ.setdefault('HF_HOME', 'D:/kaggle/hf_cache')
    root, OUT = 'data/SRCASW-BhavVaani', os.environ.get('SMOKE_OUT', '.')
    proc = AutoProcessor.from_pretrained(REPO)
    cfg = Qwen2AudioConfig.from_pretrained(REPO)
    cfg.audio_config.update(dict(encoder_layers=2, d_model=64, encoder_attention_heads=2, encoder_ffn_dim=128))
    cfg.text_config.update(dict(num_hidden_layers=4, hidden_size=64, intermediate_size=128, num_attention_heads=2, num_key_value_heads=2))
    model = Qwen2AudioForConditionalGeneration(cfg).eval()
else:
    root, OUT = os.path.dirname(sorted(glob.glob('/kaggle/input/**/train.csv', recursive=True))[0]), '/kaggle/working'
    cfgs = [c for c in glob.glob('/kaggle/input/**/config.json', recursive=True) if 'qwen2-audio' in c.lower()]
    if cfgs:  # Kaggle model mount qwen-lm/qwen2-audio/transformers/7b-instruct/1
        mpath = os.path.dirname(sorted(cfgs, key=len)[0])
    else:
        from huggingface_hub import snapshot_download
        mpath = snapshot_download(REPO, cache_dir='/tmp/hf')
    print('model', mpath, flush=True)
    proc = AutoProcessor.from_pretrained(mpath)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        mpath, torch_dtype=torch.float16, device_map='auto', max_memory={0: '14GiB', 1: '14GiB'}).eval()
tr = pd.read_csv(f'{root}/train.csv'); te = pd.read_csv(f'{root}/test.csv')
paths = [f'{root}/train/{f}' for f in tr.filename] + [f'{root}/test/{f}' for f in te.filename]
N, ntr = len(paths), len(tr)
order = list(range(ntr, N)) + list(np.random.RandomState(0).permutation(ntr))  # test first, then train shuffled
if SMOKE: order = order[:2] + [order[-1]]
print(f'data {root} gpus={torch.cuda.device_count()} loaded {time.time() - T0:.0f}s', flush=True)

AUDIO = model.config.audio_token_index
QS = ['What emotion does the speaker express in this Hindi audio? Choose one: angry, happy, neutral, sad. Answer with one word.',
      "Listen to the speaker's tone of voice, not just the words. Is the speaker angry, happy, neutral, or sad? Answer with one word."]
texts = [proc.apply_chat_template([{'role': 'system', 'content': 'You are a helpful assistant.'},
          {'role': 'user', 'content': [{'type': 'audio', 'audio_url': 'x.wav'}, {'type': 'text', 'text': q}]}],
          add_generation_prompt=True, tokenize=False) for q in QS]
QA = ('First describe the acoustic and paralinguistic cues you hear (pitch, loudness, speaking rate, pauses, voice quality, '
      'non-speech events) in one sentence, then say whether the speaker is angry, happy, neutral or sad.')
text_a = proc.apply_chat_template([{'role': 'system', 'content': 'You are a helpful assistant.'},
          {'role': 'user', 'content': [{'type': 'audio', 'audio_url': 'x.wav'}, {'type': 'text', 'text': QA}]}],
          add_generation_prompt=True, tokenize=False) + 'Acoustic cues:'
NL = model.config.text_config.num_hidden_layers
HL = [NL // 4, NL // 2, 3 * NL // 4, NL]
tok = proc.tokenizer
CAND = [sorted({tok.encode(v, add_special_tokens=False)[0] for v in (l, l.capitalize(), ' ' + l, ' ' + l.capitalize())})
        for l in ['angry', 'happy', 'neutral', 'sad']]
assert len({t for c in CAND for t in c}) == sum(len(c) for c in CAND), CAND
print('prompt 0:', repr(texts[0]), '| label tokens', CAND, '| hidden layers', HL, flush=True)

lp = np.full((N, len(QS) + 1, 4), np.nan, np.float32); caps = [''] * N; hid = np.zeros((N, len(HL), model.config.text_config.hidden_size), np.float16)
dev0 = next(model.parameters()).device


def save():
    np.save(f'{OUT}/qa7b_lp.npy', lp); np.save(f'{OUT}/qa7b_h.npy', hid)
    pd.DataFrame({'caption': caps}).to_csv(f'{OUT}/qa7b_cap.csv', index=False)


done = 0
with torch.inference_mode():
    for c, i in enumerate(order):
        if time.time() - T0 > BUDGET_S:
            print(f'budget hit after {done} clips', flush=True); break
        x = sf.read(paths[i], dtype='float32')[0]
        for qi, text in enumerate(texts):
            inp = proc(text=text, audio=[x], sampling_rate=16000, return_tensors='pt', padding=True).to(dev0)
            inp['input_features'] = inp['input_features'].to(model.dtype)
            out = model(**inp, output_hidden_states=(qi == 0), use_cache=False)
            logp = torch.log_softmax(out.logits[0, -1].float(), -1)
            lp[i, qi] = [torch.logsumexp(logp[cs], 0).item() for cs in CAND]
            if qi == 0:
                m = (inp['input_ids'][0] == AUDIO); hs = out.hidden_states
                assert m.sum() > 0 and len(hs) == NL + 1
                hid[i] = torch.stack([hs[L][0][m.to(hs[L].device)].float().mean(0).cpu() for L in HL]).numpy()
        inp = proc(text=text_a, audio=[x], sampling_rate=16000, return_tensors='pt', padding=True).to(dev0)  # R3: caption, then label
        inp['input_features'] = inp['input_features'].to(model.dtype)
        g = model.generate(**inp, max_new_tokens=48, do_sample=False)
        caps[i] = tok.decode(g[0, inp['input_ids'].shape[1]:], skip_special_tokens=True).split('\n')[0].strip()
        inp = proc(text=f'{text_a} {caps[i]}\nEmotion:', audio=[x], sampling_rate=16000, return_tensors='pt', padding=True).to(dev0)
        inp['input_features'] = inp['input_features'].to(model.dtype)
        logp = torch.log_softmax(model(**inp, use_cache=False).logits[0, -1].float(), -1)
        lp[i, len(QS)] = [torch.logsumexp(logp[cs], 0).item() for cs in CAND]
        done += 1
        if c % 100 == 0:
            p = np.exp(lp[i]); print(f'{c} clips {time.time() - T0:.0f}s | audio tokens {int(m.sum())} | mass on 4 labels per prompt {p.sum(1).round(2)} | caption: {caps[i][:120]}', flush=True); save()
save()
C = ['angry', 'happy', 'neutral', 'sad']
y = tr.emotion.map({c: k for k, c in enumerate(C)}).to_numpy()
ok = ~np.isnan(lp[:ntr]).any((1, 2))
if ok.sum() > 10:
    from sklearn.metrics import f1_score
    for qi in range(lp.shape[1]):
        print(f'prompt {qi} zero-shot macro-F1 on {ok.sum()} train clips: {f1_score(y[ok], lp[:ntr][ok, qi].argmax(1), average="macro"):.4f}', flush=True)
print(f'processed {done}/{N}, total {time.time() - T0:.0f}s', flush=True)
