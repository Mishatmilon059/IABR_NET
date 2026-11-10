"""Task 5 (adaptivity headroom) from configs_<bank>.npz (configs.py: NOMP os{2,4,8,16} x rc{1,3,6} x self-cal{off,on}).
Rows (Pd = mean over SNR groups of correct components / 6; official 2000-scene subset, 250/SNR):
  fixed_os4rc3     : NOMP os4 rc3 (= cached NOMP, the baseline)
  best_fixed_val   : single config chosen on the val bank (legal)
  best_fixed_off   : single config chosen on the official subset itself (optimistic)
  perSNR_val       : per true-SNR config chosen on val (legal if the SNR is known)
  perSNR_oracle    : per true-SNR config chosen on the official subset itself (upper bound)
  perSNRhat_val    : per ESTIMATED-SNR bucket (from the os4rc3 residual, noise var = res/250, rounded to the 5 dB grid) chosen on val (legal)
  min_residual     : per scene, the config with the smallest LS residual (legal)
  perScene_oracle  : per scene, the config with most correct components (truth; upper bound of a perfect per-scene selector)
Paired bootstrap vs NOMP os4rc3 (2000 resamples within SNR groups, seed 0). stdout -> adapt.log, json -> adapt.json"""
import json
from dcommon import *
from analyze import boot_diff, pd_snr

Z = {b: np.load(f'{BIG}/configs_{b}.npz') for b in ['official', 'val']}
cfgs = [tuple(c) for c in Z['official']['cfgs'].tolist()]
k0 = cfgs.index((4, 3, 0))


def pdm(NC, snr):
    ss = sorted(set(snr.tolist()))
    return np.array([[NC[snr == s, k].mean() / 6 for s in ss] for k in range(NC.shape[1])])   # (K, S)


def snr_hat(RES):
    sh = -10 * np.log10(RES[:, k0] / 250)
    return np.clip(np.round(sh / 5) * 5, -10, 25).astype(int)


O = Z['official']; V = Z['val']
NCo, so = O['NC'], O['snr'].astype(int); NCv, sv = V['NC'], V['snr'].astype(int)
ss = sorted(set(so.tolist()))
Po, Pv = pdm(NCo, so), pdm(NCv, sv)
print('per-config Pd (official subset | val), mean over SNR')
for k, c in enumerate(cfgs):
    print(f'os{c[0]:2d} rc{c[1]} sc{c[2]} | off {Po[k].mean():.4f} | val {Pv[k].mean():.4f} | off per SNR ' + ' '.join(f'{p:.4f}' for p in Po[k]))
rows = {}
rows['fixed_os4rc3'] = NCo[:, k0]
kv = Pv.mean(1).argmax(); rows['best_fixed_val'] = NCo[:, kv]
ko = Po.mean(1).argmax(); rows['best_fixed_off'] = NCo[:, ko]
pick_v = {s: Pv[:, i].argmax() for i, s in enumerate(ss)}; pick_o = {s: Po[:, i].argmax() for i, s in enumerate(ss)}
rows['perSNR_val'] = NCo[np.arange(len(so)), [pick_v[s] for s in so]]
rows['perSNR_oracle'] = NCo[np.arange(len(so)), [pick_o[s] for s in so]]
shv, sho = snr_hat(V['RES']), snr_hat(O['RES'])
pick_h = {}
for s in ss:
    m = shv == s
    pick_h[s] = NCv[m].mean(0).argmax() if m.sum() >= 20 else kv
rows['perSNRhat_val'] = NCo[np.arange(len(so)), [pick_h[s] for s in sho]]
rows['min_residual'] = NCo[np.arange(len(so)), O['RES'].argmin(1)]
rows['perScene_oracle'] = NCo.max(1)
print(f'\nchosen: best_fixed_val {cfgs[kv]} | best_fixed_off {cfgs[ko]}')
print('perSNR_val picks:', {s: cfgs[pick_v[s]] for s in ss})
print('perSNR_oracle picks:', {s: cfgs[pick_o[s]] for s in ss})
print('perSNRhat_val picks:', {s: cfgs[pick_h[s]] for s in ss}, '| SNR-hat accuracy (exact 5 dB bucket):', round(float((sho == so).mean()), 3))
out = {}
print('\nrule              | mean   | ' + ' '.join(f'{s:>6d}' for s in ss) + ' | diff vs NOMP os4rc3 [95% CI]')
for r, nc in rows.items():
    per = pd_snr(nc, so.astype(float)); d = boot_diff(nc, rows['fixed_os4rc3'], so.astype(float))
    out[r] = dict(pd=per, diff_vs_nomp=d)
    print(f'{r:17s} | {per["mean"]:.4f} | ' + ' '.join(f'{per[s]:.4f}' for s in ss) +
          f' | {d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}]{"*" if d[1] > 0 or d[2] < 0 else ""}')
out['cfgs'] = cfgs; out['picks'] = dict(best_fixed_val=cfgs[kv], perSNR_val={s: cfgs[pick_v[s]] for s in ss},
                                         perSNR_oracle={s: cfgs[pick_o[s]] for s in ss}, perSNRhat_val={s: cfgs[pick_h[s]] for s in ss})
json.dump(out, open(f'{HERE}/adapt.json', 'w'), indent=1, default=str)
