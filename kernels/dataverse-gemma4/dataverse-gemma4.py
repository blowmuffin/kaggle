"""Spikes K1 + K2 (for 10 Oct, Kaggle 2xT4): Gemma 4 E4B-it (Apache-2.0, Kaggle google/gemma-4/transformers/gemma-4-e4b-it/1),
transformers 5.19, bf16 (Gemma overflows in fp16). Per clip: 2 prompts (as S5) -> 4-label first-token log-probs; prompt 0 ->
K1 mean LLM hidden over audio tokens at layers 10/21/31/42, K2 mean+std of the USM-Conformer audio encoder at layer 8 (2/3)
and 12 (last, before output_proj) over valid frames. Outputs /kaggle/working/gemma4_lp.npy [N,2,4], gemma4_h.npy [N,4,2560],
gemma4_enc.npy [N,2,2048] (fp16). Row order = train then test. Local smoke test (tiny random weights): SMOKE=1"""
import os, sys, subprocess, time, glob
T0 = time.time()
SMOKE = bool(os.environ.get('SMOKE'))
if not SMOKE:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'transformers==5.19.0'], check=False)
BUDGET_S = float(os.environ.get('BUDGET_MIN', '110')) * 60
import numpy as np, pandas as pd, soundfile as sf, torch
from transformers import AutoConfig, AutoProcessor, Gemma4ForConditionalGeneration

REPO = 'google/gemma-4-E4B-it'
if SMOKE:
    os.environ.setdefault('HF_HOME', 'D:/kaggle/hf_cache')
    root, OUT, mpath = 'data/SRCASW-BhavVaani', os.environ.get('SMOKE_OUT', '.'), REPO
    cfg = AutoConfig.from_pretrained(REPO)
    t = cfg.text_config
    t.update(dict(hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1,
                  head_dim=32, global_head_dim=32, hidden_size_per_layer_input=16, num_kv_shared_layers=0, layer_types=t.layer_types[:4]))
    cfg.audio_config.update(dict(num_hidden_layers=12, hidden_size=64, num_attention_heads=2))
    model = Gemma4ForConditionalGeneration(cfg).float().eval()  # config says bf16; keep the tiny model in one dtype
else:
    root, OUT = os.path.dirname(sorted(glob.glob('/kaggle/input/**/train.csv', recursive=True))[0]), '/kaggle/working'
    mpath = os.path.dirname(sorted([c for c in glob.glob('/kaggle/input/**/config.json', recursive=True) if 'gemma-4' in c.lower()], key=len)[0])
    model = Gemma4ForConditionalGeneration.from_pretrained(mpath, dtype=torch.bfloat16, device_map='auto',
                                                           max_memory={0: '14GiB', 1: '14GiB'}).eval()
proc = AutoProcessor.from_pretrained(mpath)
tr = pd.read_csv(f'{root}/train.csv'); te = pd.read_csv(f'{root}/test.csv')
paths = [f'{root}/train/{f}' for f in tr.filename] + [f'{root}/test/{f}' for f in te.filename]
N, ntr = len(paths), len(tr)
order = list(range(ntr, N)) + list(np.random.RandomState(0).permutation(ntr))  # test first, then train shuffled
if SMOKE: order = order[:2] + [order[-1]]
print(f'model {mpath} gpus={torch.cuda.device_count()} loaded {time.time() - T0:.0f}s', flush=True)

QS = ['What emotion does the speaker express in this Hindi audio? Choose one: angry, happy, neutral, sad. Answer with one word.',
      "Listen to the speaker's tone of voice, not just the words. Is the speaker angry, happy, neutral, or sad? Answer with one word."]
texts = [proc.apply_chat_template([{'role': 'user', 'content': [{'type': 'audio', 'audio': 'x.wav'}, {'type': 'text', 'text': q}]}],
                                  add_generation_prompt=True, tokenize=False) for q in QS]
AUDIO = model.config.audio_token_id
NL = model.config.text_config.num_hidden_layers
HL = [NL // 4, NL // 2, 3 * NL // 4, NL]
tower = model.model.audio_tower; NA = len(tower.layers); EL = [2 * NA // 3 - 1, NA - 1]  # layer 8 and 12 (0-based 7, 11)
grab = {}
for li in EL:
    tower.layers[li].register_forward_hook(lambda m, i, o, li=li: grab.__setitem__(li, o[0] if isinstance(o, tuple) else o))
tower.register_forward_hook(lambda m, i, o: grab.__setitem__('mask', o.attention_mask))
tok = proc.tokenizer
CAND = [sorted({tok.encode(v, add_special_tokens=False)[0] for v in (l, l.capitalize(), ' ' + l, ' ' + l.capitalize())})
        for l in ['angry', 'happy', 'neutral', 'sad']]
assert len({t for c in CAND for t in c}) == sum(len(c) for c in CAND), CAND
print('prompt 0:', repr(texts[0][:300]), '| label tokens', CAND, '| LLM layers', HL, '| encoder layers', [l + 1 for l in EL], flush=True)

dev0 = model.model.language_model.embed_tokens.weight.device
lp = np.full((N, len(QS), 4), np.nan, np.float32)
hid = np.zeros((N, len(HL), model.config.text_config.hidden_size), np.float16)
enc = np.zeros((N, 2, 2 * tower.config.hidden_size), np.float16)
save = lambda: (np.save(f'{OUT}/gemma4_lp.npy', lp), np.save(f'{OUT}/gemma4_h.npy', hid), np.save(f'{OUT}/gemma4_enc.npy', enc))
done = 0
with torch.inference_mode():
    for c, i in enumerate(order):
        if time.time() - T0 > BUDGET_S:
            print(f'budget hit after {done} clips', flush=True); break
        x = sf.read(paths[i], dtype='float32')[0]
        for qi, text in enumerate(texts):
            inp = proc(text=text, audio=[x], return_tensors='pt').to(dev0)
            if 'input_features' in inp: inp['input_features'] = inp['input_features'].to(model.dtype)
            out = model(**inp, output_hidden_states=(qi == 0), use_cache=False)
            logp = torch.log_softmax(out.logits[0, -1].float(), -1)
            lp[i, qi] = [torch.logsumexp(logp[cs], 0).item() for cs in CAND]
            if qi == 0:
                m = (inp['input_ids'][0] == AUDIO); hs = out.hidden_states
                assert m.sum() > 0 and len(hs) == NL + 1, (int(m.sum()), len(hs))
                hid[i] = torch.stack([hs[L][0][m.to(hs[L].device)].float().mean(0).cpu() for L in HL]).numpy()
                v = grab['mask'][0].bool()
                enc[i] = torch.stack([torch.cat([grab[l][0][v].float().mean(0), grab[l][0][v].float().std(0)]).cpu() for l in EL]).numpy()
        done += 1
        if c % 100 == 0:
            print(f'{c} clips {time.time() - T0:.0f}s | audio tokens {int(m.sum())} enc frames {int(v.sum())} | '
                  f'mass on 4 labels per prompt {np.exp(lp[i]).sum(1).round(2)}', flush=True); save()
save()
C = ['angry', 'happy', 'neutral', 'sad']
y = tr.emotion.map({c: k for k, c in enumerate(C)}).to_numpy()
ok = ~np.isnan(lp[:ntr]).any((1, 2))
if ok.sum() > 10:
    from sklearn.metrics import f1_score
    for qi in range(len(QS)):
        print(f'prompt {qi} zero-shot macro-F1 on {ok.sum()} train clips: {f1_score(y[ok], lp[:ntr][ok, qi].argmax(1), average="macro"):.4f}', flush=True)
print(f'processed {done}/{N}, total {time.time() - T0:.0f}s', flush=True)
