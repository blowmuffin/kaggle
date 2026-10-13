"""Spike R2 (for 10 Oct, Kaggle 2xT4): Step-Audio-2-mini (stepfun-ai, Apache-2.0, 8B, HF download, trust_remote_code).
Prompt format mirrors stepaudio2.py (github stepfun-ai/Step-Audio2). Its forward() drops hidden states, so we re-run it:
embed -> write audio embeddings after <audio_start> -> Qwen2 LLM (hidden states) -> lm_head. fp16 (T4 has no fast bf16).
Per clip: 2 prompts -> 4-label first-token log-probs; prompt 0 -> mean hidden over audio patches at layers 7/14/21/28.
Outputs /kaggle/working/sa2_lp.npy [N,2,4], sa2_h.npy [N,4,3584] fp16. Local smoke test (tiny random weights): SMOKE=1"""
import os, sys, time, glob, importlib
T0 = time.time()
SMOKE = bool(os.environ.get('SMOKE'))
if not SMOKE:  # same version as the local smoke test
    import subprocess; subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'transformers==4.56.2'], check=False)
BUDGET_S = float(os.environ.get('BUDGET_MIN', '60')) * 60
import numpy as np, pandas as pd, soundfile as sf, torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

REPO = 'stepfun-ai/Step-Audio-2-mini'
if SMOKE:
    os.environ.setdefault('HF_HOME', 'D:/kaggle/hf_cache')
    root, OUT, mpath = 'data/SRCASW-BhavVaani', os.environ.get('SMOKE_OUT', '.'), REPO
    cfg = AutoConfig.from_pretrained(REPO, trust_remote_code=True)
    cfg.text_config.update(dict(hidden_size=64, intermediate_size=128, num_attention_heads=2, num_key_value_heads=2, num_hidden_layers=4))
    a = cfg.audio_encoder_config
    a.update(dict(n_audio_state=64, n_audio_head=2, n_audio_layer=2, llm_dim=64)) if hasattr(a, 'update') else [setattr(a, k, v) for k, v in dict(n_audio_state=64, n_audio_head=2, n_audio_layer=2, llm_dim=64).items()]
    model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True).eval()
else:
    from huggingface_hub import snapshot_download
    root, OUT = os.path.dirname(sorted(glob.glob('/kaggle/input/**/train.csv', recursive=True))[0]), '/kaggle/working'
    mpath = snapshot_download(REPO, cache_dir='/tmp/hf', ignore_patterns=['token2wav/*', 'assets/*'])
    n = AutoConfig.from_pretrained(mpath, trust_remote_code=True).text_config.num_hidden_layers
    dmap = {'encoder': 0, 'adapter': 0, 'model.embed_tokens': 0, 'model.rotary_emb': 0, 'model.norm': 1, 'lm_head': 1}
    dmap.update({f'model.layers.{i}': int(i >= n // 2) for i in range(n)})
    model = AutoModelForCausalLM.from_pretrained(mpath, trust_remote_code=True, torch_dtype=torch.float16, device_map=dmap).eval()
model.bf16 = False
mod = importlib.import_module(type(model).__module__)  # remote modeling file: log_mel_spectrogram, compute_token_num
tok = AutoTokenizer.from_pretrained(mpath, trust_remote_code=True)
tr = pd.read_csv(f'{root}/train.csv'); te = pd.read_csv(f'{root}/test.csv')
paths = [f'{root}/train/{f}' for f in tr.filename] + [f'{root}/test/{f}' for f in te.filename]
N, ntr = len(paths), len(tr)
order = list(range(ntr, N)) + list(np.random.RandomState(0).permutation(ntr))  # test first, then train shuffled
if SMOKE: order = order[:2] + [order[-1]]
print(f'model {mpath} dtype {next(model.parameters()).dtype} gpus={torch.cuda.device_count()} loaded {time.time() - T0:.0f}s', flush=True)

AUDIO_START, AUDIO_PATCH = tok.convert_tokens_to_ids('<audio_start>'), tok.convert_tokens_to_ids('<audio_patch>')
SYS = 'You are an expert in audio analysis, please analyze the audio content and answer the questions accurately.'
QS = ['What emotion does the speaker express in this Hindi audio? Choose one: angry, happy, neutral, sad. Answer with one word.',
      "Listen to the speaker's tone of voice, not just the words. Is the speaker angry, happy, neutral, or sad? Answer with one word."]
NL = model.config.text_config.num_hidden_layers
HL = [NL // 4, NL // 2, 3 * NL // 4, NL]
CAND = [sorted({tok.encode(v, add_special_tokens=False)[0] for v in (l, l.capitalize(), ' ' + l, ' ' + l.capitalize())})
        for l in ['angry', 'happy', 'neutral', 'sad']]
assert len({t for c in CAND for t in c}) == sum(len(c) for c in CAND), CAND
ids = lambda s: tok(s, add_special_tokens=False)['input_ids']
dev0 = model.model.embed_tokens.weight.device
print('label tokens', CAND, '| hidden layers', HL, flush=True)


def run(x, q, want_hidden):
    mel = mod.log_mel_spectrogram(torch.from_numpy(x), n_mels=128, padding=479)  # same as stepaudio2.apply_chat_template
    k = mod.compute_token_num(mel.shape[1])
    seq = (ids(f'<|BOT|>system\n{SYS}<|EOT|>') + ids('<|BOT|>human\n') + [AUDIO_START] + [AUDIO_PATCH] * k
           + ids('<audio_end>') + ids(q) + ids('<|EOT|>') + ids('<|BOT|>assistant\n'))
    input_ids = torch.tensor([seq], device=dev0)
    h = model.model.embed_tokens(input_ids)
    a, lens = model.encoder(mel[None].to(dev0, h.dtype), torch.tensor([mel.shape[1] - 2], device=dev0))  # [1, 128, T] as padding_mels
    a = model.adapter(a); L = int((lens[0] - 1) // 2 + 1)
    assert L == k, (L, k)
    s = seq.index(AUDIO_START) + 1
    h[0, s:s + L] = a[0, :L].to(h.dtype)
    out = model.model(inputs_embeds=h, attention_mask=torch.ones_like(input_ids), output_hidden_states=want_hidden)
    logits = model.lm_head(out.last_hidden_state[:, -1])
    m = (input_ids[0] == AUDIO_PATCH)
    hid = torch.stack([out.hidden_states[l][0][m.to(out.hidden_states[l].device)].float().mean(0).cpu() for l in HL]) if want_hidden else None
    return torch.log_softmax(logits[0].float(), -1), hid, k


lp = np.full((N, len(QS), 4), np.nan, np.float32); hid = np.zeros((N, len(HL), model.config.text_config.hidden_size), np.float16)
save = lambda: (np.save(f'{OUT}/sa2_lp.npy', lp), np.save(f'{OUT}/sa2_h.npy', hid))
done = 0
with torch.inference_mode():
    for c, i in enumerate(order):
        if time.time() - T0 > BUDGET_S:
            print(f'budget hit after {done} clips', flush=True); break
        x = sf.read(paths[i], dtype='float32')[0]
        assert len(x) <= 25 * 16000  # stepaudio2 splits >25 s; ours are <= 11.2 s
        for qi, q in enumerate(QS):
            logp, h, k = run(x, q, qi == 0)
            lp[i, qi] = [torch.logsumexp(logp[cs], 0).item() for cs in CAND]
            if qi == 0: hid[i] = h.numpy()
        done += 1
        if c % 100 == 0:
            print(f'{c} clips {time.time() - T0:.0f}s | audio tokens {k} | mass on 4 labels per prompt {np.exp(lp[i]).sum(1).round(2)}', flush=True); save()
save()
C = ['angry', 'happy', 'neutral', 'sad']
y = tr.emotion.map({c: j for j, c in enumerate(C)}).to_numpy()
ok = ~np.isnan(lp[:ntr]).any((1, 2))
if ok.sum() > 10:
    from sklearn.metrics import f1_score
    for qi in range(len(QS)):
        print(f'prompt {qi} zero-shot macro-F1 on {ok.sum()} train clips: {f1_score(y[ok], lp[:ntr][ok, qi].argmax(1), average="macro"):.4f}', flush=True)
print(f'processed {done}/{N}, total {time.time() - T0:.0f}s', flush=True)
