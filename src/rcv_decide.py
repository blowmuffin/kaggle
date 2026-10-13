"""Combine repeated-CV member OOFs over seeds 42/7/11 and apply the PRE-REGISTERED rule per candidate:
adopt if mean gain over the 3 splits >= +0.005 and gain > 0 on every split.
A candidate overrides blend roles (whisper, w2v, omni_h, omni_lp, text, extra) with a member name or a list (= mean)."""
import os, numpy as np
from sklearn.metrics import f1_score
from hcv import y

S = (42, 7, 11)
L = lambda s, m: np.mean([np.load(f'oof/rcv_s{s}_{x}.npy') for x in m], 0) if isinstance(m, list) else np.load(f'oof/rcv_s{s}_{m}.npy')
F = lambda p: f1_score(y, p.argmax(1), average='macro')
BASE = dict(whisper='whisper_L24L32', w2v='w2vbert_L16', omni_h='omni_h', omni_lp='omni_lp', text='text_e5', extra=None, extra2=None)  # A′
A2 = dict(whisper='whisper_aug_L24L32', w2v='w2vbert_L10L16')  # A″
A3 = {**A2, 'extra': 'whisper_attn_text'}  # A‴ = A″ + P3 head (0.2) = current base; 10-Oct members enter as nested extra2


def full(s, **kw):
    r = {**BASE, **kw}
    audio = 1 / 3 * (L(s, r['whisper']) + L(s, r['w2v'])) + 1 / 6 * (L(s, r['omni_h']) + L(s, r['omni_lp']))
    for e in (r['extra'], r['extra2']):  # nested: each extra takes 0.2 of the audio blend built so far
        if e: audio = 0.8 * audio + 0.2 * L(s, e)
    return 0.75 * audio + 0.25 * L(s, r['text'])


CANDIDATES = [
    ('C1 w2vbert L10+L16', dict(w2v='w2vbert_L10L16')),
    ('C2 + whisper-v2 L24+L32 (0.2)', dict(extra='whisperv2_L24L32')),
    ('S1 group weighting', dict(whisper='whisper_L24L32_gw', w2v='w2vbert_L16_gw', omni_h='omni_h_gw', omni_lp='omni_lp_gw', text='text_e5_gw')),
    ('S2 multi-pad TTA', dict(whisper=['whisper_L24L32', 'whisper_pad20_L24L32', 'whisper_pad12_L24L32'])),
    ('S3 aug train + TTA', dict(whisper='whisper_aug_L24L32')),
    ('S3b loop-pad TTA', dict(whisper=['whisper_L24L32', 'whisper_loop_L24L32'])),
    ('S3+C1 aug whisper + w2vbert L10+L16', dict(whisper='whisper_aug_L24L32', w2v='w2vbert_L10L16')),
]
CANDIDATES = [(name, kw, {}) for name, kw in CANDIDATES]  # historical: gain vs A′
CANDIDATES += [(name, {**A2, **kw}, A2) for name, kw in [  # gain vs A″
    ('R1 + Dasheng-1.2B (0.2)', dict(extra='dasheng')),
    ('L1 w2vbert aug + TTA', dict(w2v='w2vbert_aug_L10L16')),
    ('L5 nested label cleaning', dict(whisper='whisper_aug_L24L32_cl', w2v='w2vbert_L10L16_cl', omni_h='omni_h_cl', omni_lp='omni_lp_cl', text='text_e5_cl')),
    ('L4 + segment pooling (0.2)', dict(extra='whisper_seg')),
    ('L3 + attentive pooling head (0.2)', dict(extra='whisper_attn')),
    ('L2 whisper 8 aug views', dict(whisper='whisper_aug8_L24L32')),
    ('L3a whisper = mean(aug probe, attn head)', dict(whisper=['whisper_aug_L24L32', 'whisper_attn'])),
    ('L3b whisper = mean(aug probe, aug attn head)', dict(whisper=['whisper_aug_L24L32', 'whisper_attn_aug'])),
    ('P1 whisper + audio-mix views', dict(whisper='whisper_mix_L24L32')),
    ('P2 whisper 0.7 probe + 0.3 kNN', dict(whisper='whisper_aug_knn')),
    ('P3 + audio-text fusion head (0.2)', dict(extra='whisper_attn_text')),
]]
CANDIDATES += [(name, {**A3, **kw}, A3) for name, kw in [  # 10-Oct candidates: gain vs A‴
    ('omni-zs2 omni members v2', dict(omni_h='omni2_h', omni_lp='omni2_lp')),
    ('S5 + Qwen2-Audio-7B LLM', dict(extra2=['qa7b_h', 'qa7b_lp'])),
    ('R3 = S5 with +A prompt', dict(extra2=['qa7b_h', 'qa7b_lp3'])),
    ('R2 + Step-Audio-2-mini', dict(extra2=['sa2_h', 'sa2_lp'])),
    ('K1 + Gemma-4 E4B LLM', dict(extra2=['gemma4_h', 'gemma4_lp'])),
    ('K2 + Gemma-4 audio encoder', dict(extra2='gemma4_enc')),
    ('K3 + Qwen3-ASR AuT encoder', dict(extra2='qasr_enc')),
    ('K4 + Omni with transcript', dict(extra2=['omni_th', 'omni_tlp'])),
    ('F1 calibrated blend (all members T-scaled)', dict(whisper='whisper_aug_L24L32_cal', w2v='w2vbert_L10L16_cal', omni_h='omni_h_cal',
                                                    omni_lp='omni_lp_cal', text='text_e5_cal', extra='whisper_attn_text_cal')),
    ('F3 text = mean(e5 whisper-ASR, e5 qwen3-ASR)', dict(text=['text_e5', 'text_e5q'])),
]]
if __name__ == '__main__':
    print('base per split: A′ ' + ' '.join(f'{F(full(s)):.4f}' for s in S) + ' | A″ ' + ' '.join(f'{F(full(s, **A2)):.4f}' for s in S)
          + ' | A‴ ' + ' '.join(f'{F(full(s, **A3)):.4f}' for s in S))
    for name, kw, ref in CANDIDATES:
        names = [x for v in kw.values() for x in (v if isinstance(v, list) else [v])]
        if not all(os.path.exists(f'oof/rcv_s{s}_{x}.npy') for s in S for x in names):
            print(f'{name:32s} (members missing)'); continue
        g = [F(full(s, **kw)) - F(full(s, **ref)) for s in S]
        folds = []
        for s in S:
            fo = np.load(f'oof/rcv_s{s}_fold.npy'); a, b = full(s, **kw), full(s, **ref)
            fk = lambda p, k: f1_score(y[fo == k], p[fo == k].argmax(1), average='macro')
            folds.append(sum(fk(a, k) > fk(b, k) for k in range(5)))
        ok = np.mean(g) >= 0.005 and min(g) > 0
        print(f'{name:32s} gains ' + ' '.join(f'{x:+.4f}' for x in g) + f' | mean {np.mean(g):+.4f} | folds up per split {folds} | ADOPT: {ok}')
