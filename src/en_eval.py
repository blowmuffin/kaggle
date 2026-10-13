"""Evaluate English-translation emotion features on honest v2 folds.
Pre-registered decision: full = 0.75·audio + 0.25·mean(text_e5, en_emo). Others = robustness checks."""
import numpy as np, pandas as pd
from hcv import *

R7 = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
RL, DB, SE, EE = (np.load(f'feats/{f}.npy') for f in ['en_emo_rl', 'en_emo_db', 'en_sent', 'en_e5'])
zs = RL[:n][:, [R7.index(c) for c in ['anger', 'joy', 'neutral', 'sadness']]]
print(f'ZERO-SHOT roberta-large (4 of 7 classes): F1 {f1(zs):.4f}')
lg = lambda P: np.log(np.clip(P, 1e-6, 1))
C1, o_emo, t_emo = probe(np.hstack([lg(RL), lg(DB), lg(SE)]), Cs=(0.01, 0.03, 0.1, 0.3, 1, 3))
C2, o_ee, t_ee = probe(EE, Cs=(0.01, 0.03, 0.1, 0.3, 1))
for nm, o, t in [('en_emo', o_emo, t_emo), ('en_e5', o_ee, t_ee)]:
    np.save(f'oof/v2_{nm}.npy', o); np.save(f'oof/v2_{nm}_test.npy', t)
print(f'en_emo probe (C={C1}): {f1(o_emo):.4f} | en_e5 probe (C={C2}): {f1(o_ee):.4f} | hindi text_e5 probe: {f1(np.load("oof/v2_text_e5.npy")):.4f}')
L = lambda m, t='': np.load(f'oof/v2_{m}{t}.npy')
audio = lambda t='': 1 / 3 * (L('whisper_L32', t) + L('w2vbert_L16', t)) + 1 / 6 * (L('omni_h', t) + L('omni_lp', t))
cur = 0.75 * audio() + 0.25 * L('text_e5')
print(f'text-only combos: e5+en_emo {f1((L("text_e5") + o_emo) / 2):.4f} | e5+en_emo+en_e5 {f1((L("text_e5") + o_emo + o_ee) / 3):.4f}')
print(f'\ncurrent full blend: {f1(cur):.4f}')
cands = {
    'PRE-REG 0.25·mean(e5,en_emo)': lambda t='': 0.75 * audio(t) + 0.25 * (L('text_e5', t) + L('en_emo', t)) / 2,
    '0.25·mean(e5,en_emo,en_e5)': lambda t='': 0.75 * audio(t) + 0.25 * (L('text_e5', t) + L('en_emo', t) + L('en_e5', t)) / 3,
    '0.30·mean(e5,en_emo)': lambda t='': 0.70 * audio(t) + 0.30 * (L('text_e5', t) + L('en_emo', t)) / 2,
    '0.20·mean(e5,en_emo)': lambda t='': 0.80 * audio(t) + 0.20 * (L('text_e5', t) + L('en_emo', t)) / 2,
    '0.25·e5 + 0.10·en_emo': lambda t='': 0.65 * audio(t) + 0.25 * L('text_e5', t) + 0.10 * L('en_emo', t),
}
for nm, fn in cands.items():
    p = fn(); d = per_fold(p, cur)
    print(f'{nm:30s} {f1(p):.4f} ({f1(p) - f1(cur):+.4f}) folds up {sum(x > 0 for x in d)}/5 ' + ' '.join(f'{x:+.3f}' for x in d))
pt = cands['PRE-REG 0.25·mean(e5,en_emo)']('_test')
ss = pd.read_csv('data/SRCASW-BhavVaani/sample_submission.csv')
ss[['Id']].merge(pd.DataFrame({'Id': te.Id, 'emotion': C4[pt.argmax(1)]}), on='Id').to_csv('subs/v3_en_prereg.csv', index=False)
