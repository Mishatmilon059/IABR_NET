"""Task 3 decomposition (full official bank, 8000 scenes): which part of the Bayesian score lets it certify the near-truth mode
over NOMP's answer?  For each scene choose between NOMP (cached) and the oracle-seeded local optimum (verify/cache/oracle3) by a
score; the Pd of that choice is the ceiling of a perfect search under that score.  Scores: residual (ML), residual + min-sep rule,
marginal likelihood (gain prior CN(0,1/L)), + sep, + (u,v) Jacobian prior (MAP in u), + window mass (approx. P(within 1 deg)),
using the TRUE noise variance and the ESTIMATED one (NOMP residual / 250).  Also: which scenes the MAP choice fixes (endfire share).
stdout -> map_decomp.log, map_decomp.json"""
import json
from dcommon import *
from analyze import window_logmass, boot_diff
b = load_bank('official'); N = len(b['L']); snr = b['snr'].astype(int)
pt, ft = b['psi'].astype(np.float64), b['phi'].astype(np.float64)
H = np.load(f'{CACHE}/net__official.npz')['Ht'].astype(np.complex128)
z = np.load(f'{CACHE}/pred__NOMP_no_network__official.npz'); o = np.load('D:/ai_ml_project/paper/evidence/deep_research/verify/cache/oracle3__official.npz')
UV = {'nomp': (np.pi * np.cos(z['psi'].astype(np.float64)), np.pi * np.cos(z['phi'].astype(np.float64))),
      'oracle': (np.pi * np.cos(o['psi'].astype(np.float64)), np.pi * np.cos(o['phi'].astype(np.float64)))}
s2hat = residual_energy(H, *UV['nomp']) / 250
T = {}
for nm, (U, V) in UV.items():
    T[nm] = {}
    for tag, s2 in [('true', sigma2(snr)), ('est', s2hat)]:
        parts = [bayes_terms_s2(H[s:s + 400], U[s:s + 400], V[s:s + 400], s2[s:s + 400]) for s in range(0, N, 400)]
        d = {k: np.concatenate([p[k] for p in parts]) for k in ['res', 'mll', 'lprior', 'sep']}
        g2 = np.abs(np.concatenate([p['gains'] for p in parts])) ** 2
        d['lm'] = window_logmass(U[:, None], V[:, None], g2[:, None], s2)[:, 0]
        T[nm][tag] = d
    e, _ = vmatch(pt, ft, *uv_to_angles(U, V)); T[nm]['nc'] = (e[:, 0] <= 1).sum((1, 2))


def score(d, kind):
    pen = np.where(d['sep'] < SEP, -1e12, 0.0)
    return {'res': -d['res'], 'res_sep': -d['res'] + pen, 'mll': d['mll'], 'mll_sep': d['mll'] + pen, 'mapU': d['mll'] + d['lprior'],
            'mapU_sep': d['mll'] + d['lprior'] + pen, 'mass': d['mll'] + d['lm'], 'mass_sep': d['mll'] + d['lm'] + pen}[kind]


out = {}
ss = sorted(set(snr.tolist()))
print('Pd when choosing between NOMP and the oracle-seeded optimum by each score (ceiling of perfect search under that score)')
print('score            noise | mean   | ' + ' '.join(f'{s:>6d}' for s in ss) + ' | diff vs NOMP [95% CI]')
for tag in ['true', 'est']:
    for kind in ['res', 'res_sep', 'mll', 'mll_sep', 'mapU', 'mapU_sep', 'mass', 'mass_sep']:
        pick = score(T['oracle'][tag], kind) > score(T['nomp'][tag], kind)
        nc = np.where(pick, T['oracle']['nc'], T['nomp']['nc'])
        per = [nc[snr == s].mean() / 6 for s in ss]; d = boot_diff(nc, T['nomp']['nc'], snr)
        out[f'{kind}_{tag}'] = dict(pd=dict(zip(ss, per)), mean=float(np.mean(per)), diff=d, frac_pick_oracle=float(pick.mean()))
        print(f'{kind:12s} {tag:5s} | {np.mean(per):.4f} | ' + ' '.join(f'{p:.4f}' for p in per) + f' | {d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}]')
# what the mapU/mass choice fixes
pick = score(T['oracle']['est'], 'mapU_sep') > score(T['nomp']['est'], 'mapU_sep')
gain = pick & (T['oracle']['nc'] > T['nomp']['nc']); loss = pick & (T['oracle']['nc'] < T['nomp']['nc'])
endf = ((np.abs(np.cos(pt)) > 0.95) | (np.abs(np.cos(ft)) > 0.95)).any(1)
viol = T['nomp']['est']['sep'] < SEP
print(f'\nmapU_sep(est) choice: scenes improved {gain.sum()}, worsened {loss.sum()}; among improved: truth has an endfire path (|cos|>0.95) '
      f'{endf[gain].mean():.3f} (all scenes {endf.mean():.3f}); NOMP answer violated min-sep {viol[gain].mean():.3f} (all {viol.mean():.3f})')
for s in [-10, -5, 0]:
    m = snr == s
    print(f'  {s} dB: improved {int(gain[m].sum())}, worsened {int(loss[m].sum())}, improved-with-NOMP-sep-violation {int((gain & viol)[m].sum())}, improved-with-endfire-truth {int((gain & endf)[m].sum())}')
out['mapU_sep_est_fix'] = dict(improved=int(gain.sum()), worsened=int(loss.sum()), endfire_share_improved=float(endf[gain].mean()), sepviol_share_improved=float(viol[gain].mean()))
json.dump(out, open(f'{HERE}/map_decomp.json', 'w'), indent=1)
