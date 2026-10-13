import hashlib, numpy as np, pandas as pd, soundfile as sf
from collections import Counter
R = 'data/SRCASW-BhavVaani/'
tr = pd.read_csv(R+'train.csv'); te = pd.read_csv(R+'test.csv'); ss = pd.read_csv(R+'sample_submission.csv')
print(tr.head(3), te.head(3), ss.head(3), sep='\n')
print('train ids', tr.Id.min(), tr.Id.max(), len(tr), '| test ids', te.Id.min(), te.Id.max(), len(te))
print(tr.emotion.value_counts().to_dict())
def info(split, df):
    rows = []
    for f in df.filename:
        x, sr = sf.read(R+f'{split}/{f}', dtype='float32', always_2d=True)
        info = sf.info(R+f'{split}/{f}')
        m = x.mean(1)
        rows.append(dict(sr=sr, ch=x.shape[1], dur=len(m)/sr, sub=info.subtype,
                         h=hashlib.sha256(np.round(m, 4).tobytes()).hexdigest(), rms=float(np.sqrt((m**2).mean()))))
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
tr = info('train', tr); te = info('test', te)
for n, d in [('train', tr), ('test', te)]:
    print(n, 'sr', Counter(d.sr), 'ch', Counter(d.ch), 'sub', Counter(d['sub']))
    print(n, 'dur', d.dur.describe().round(2).to_dict())
print('dur by emotion', tr.groupby('emotion').dur.median().round(2).to_dict())
print('sr by emotion', pd.crosstab(tr.emotion, tr.sr).to_dict())
print('rms by emotion', tr.groupby('emotion').rms.median().round(4).to_dict())
# duplicates
ht = set(tr.h); dup_te = te[te.h.isin(ht)]
print('train dup rows', len(tr) - tr.h.nunique(), '| test rows w/ exact train twin', len(dup_te), '| test-test dups', len(te) - te.h.nunique())
lab = tr.groupby('h').emotion.agg(lambda s: '/'.join(sorted(set(s))))
print('test twin labels', Counter(dup_te.h.map(lab)))
# label vs id order (blocks of 50)
tr['blk'] = tr.Id // 100
print(pd.crosstab(tr.blk, tr.emotion))
tr.to_csv('feats/train_meta.csv', index=False); te.to_csv('feats/test_meta.csv', index=False)
