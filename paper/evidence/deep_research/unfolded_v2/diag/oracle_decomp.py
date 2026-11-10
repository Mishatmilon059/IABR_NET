"""Where does the candidate-oracle headroom live? Pooled official subset + val (4000 scenes, no heat-map families).
For scenes where the oracle candidate (most correct components; ties -> lowest residual) beats NOMP os4rc3, split the gain by
whether the oracle candidate has a LOWER residual than NOMP (search loss: a better optimiser / residual-based selector can get it)
or a HIGHER residual (objective-limited: needs information beyond the likelihood, i.e. the truth); also by its marginal
likelihood (est. noise) relative to the best-mll candidate, and by family. stdout -> oracle_decomp.log, json -> oracle_decomp.json"""
import json
from analyze import load_pool, pd_snr
from dcommon import *
out = {}
rows = []
for b in ['official', 'val']:
    P = load_pool(b); fam = P['fam']
    legal = np.where(~np.isin(fam, ['oracle_ref', 'truth_ref', 'heat', 'heatcomb']))[0]
    nc = P['ncorrect'][:, legal]; res = P['res'][:, legal]; mll = P['mll_est'][:, legal]
    key = nc * 1e6 - res / res.max(1, keepdims=True)                       # most correct, then lowest residual
    o = key.argmax(1); B = len(o); ar = np.arange(B)
    rows.append(dict(snr=P['snr'], gain=nc[ar, o] - nc[:, 0], lower=res[ar, o] < res[:, 0] * (1 - 1e-9),
                     mllrank=(mll > mll[ar, o][:, None]).sum(1), fam=fam[legal][o],
                     sep_ok=P['sep'][:, legal][ar, o] >= SEP))
R = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
snr = R['snr'].astype(int)
print('per SNR: oracle gain (components / 6 per scene, = Pd gain) split by residual of the oracle candidate vs NOMP')
print(' snr | total gain | from lower-residual cands (search loss) | from higher-residual cands (objective-limited) | median mll rank of oracle cand among 59')
for s in sorted(set(snr.tolist())):
    m = snr == s; g = R['gain'][m]; lo = R['lower'][m]
    tot = g.sum() / m.sum() / 6; a = g[lo].sum() / m.sum() / 6; c = g[~lo].sum() / m.sum() / 6
    mr = np.median(R['mllrank'][m][g > 0]) if (g > 0).any() else np.nan
    out[s] = dict(total=tot, search_loss=a, objective_limited=c, median_mll_rank=float(mr))
    print(f'{s:4d} | {tot:.4f} | {a:.4f} | {c:.4f} | {mr:.0f}')
m = R['gain'] > 0
print('\nall SNR: total', round(float(R['gain'].sum() / len(snr) / 6), 4), '| search-loss part', round(float(R['gain'][R['lower']].sum() / len(snr) / 6), 4),
      '| objective-limited part', round(float(R['gain'][~R['lower']].sum() / len(snr) / 6), 4))
f, c = np.unique(R['fam'][m], return_counts=True)
print('family of the oracle candidate in improved scenes:', dict(zip(f.tolist(), c.tolist())))
print('oracle candidate obeys min-sep in improved scenes:', round(float(R['sep_ok'][m].mean()), 3))
out['family'] = dict(zip(f.tolist(), c.tolist()))
json.dump(out, open(f'{HERE}/oracle_decomp.json', 'w'), indent=1, default=float)
