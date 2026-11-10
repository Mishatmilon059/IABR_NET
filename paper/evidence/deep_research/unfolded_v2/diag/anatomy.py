"""Task 1: failure anatomy of NOMP (cached NOMP_no_network = os4, rc3) on the FULL official bank (8000 scenes; cheap).
Also compares NOMP's objective value with the value at the true angles (LS gains) and at the oracle-seeded local optimum
(verify/cache/oracle3__official.npz = true (u,v) -> cyclic 3 rounds), for the residual (ML) objective and for the Bayesian
score (marginal likelihood with x~CN(0,I/L) + angle prior Jacobian, + min-separation rule). stdout -> anatomy.log, anatomy.json"""
import json
from dcommon import *
b = load_bank('official'); N = len(b['L'])
pt, ft = b['psi'].astype(np.float64), b['phi'].astype(np.float64); snr = b['snr'].astype(int)
H = np.load(f'{CACHE}/net__official.npz')['Ht'].astype(np.complex128)
z = np.load(f'{CACHE}/pred__NOMP_no_network__official.npz'); pn, fn_ = z['psi'].astype(np.float64), z['phi'].astype(np.float64)
o = np.load('D:/ai_ml_project/paper/evidence/deep_research/verify/cache/oracle3__official.npz'); po, fo = o['psi'].astype(np.float64), o['phi'].astype(np.float64)
Un, Vn = np.pi * np.cos(pn), np.pi * np.cos(fn_); Uo, Vo = np.pi * np.cos(po), np.pi * np.cos(fo); Ut, Vt = np.pi * np.cos(pt), np.pi * np.cos(ft)
T = {}
for nm, (U, V) in dict(nomp=(Un, Vn), truth=(Ut, Vt), oracle=(Uo, Vo)).items():
    parts = [bayes_terms(H[s:s + 400], U[s:s + 400], V[s:s + 400], snr[s:s + 400]) for s in range(0, N, 400)]
    T[nm] = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    T[nm]['map'] = T[nm]['mll'] + T[nm]['lprior'] + np.where(T[nm]['sep'] < SEP, -1e9, 0.0)
e, _ = vmatch(pt, ft, pn, fn_); e = e[:, 0]            # (N,2,3) matched |err| deg in truth order
ok_c = e <= 1.0; ok_p = ok_c.all(1)                  # path correct = both components
eo, _ = vmatch(pt, ft, po, fo); eo = eo[:, 0]
# proximity-based outlier classification (Euclidean in (psi,phi), deg)
D_T = 5.0
dist = np.rad2deg(np.hypot(pt[:, :, None] - pn[:, None, :], ft[:, :, None] - fn_[:, None, :]))   # (N, true, est)
near_t = (dist <= D_T).sum(2)          # estimates within D of each true path
near_e = (dist <= D_T).sum(1)          # true paths within D of each estimate
emax = e.max(1)                        # (N,3) worst component error per true path after matching
endf = (np.abs(np.cos(pt)) > 0.95) | (np.abs(np.cos(ft)) > 0.95)
# u-domain error of wrong components in DFT bins (2 pi/16)
pe_m = None
out = dict(per_snr={}, thresholds=dict(bias_max_deg=3.0, outlier_proximity_deg=D_T, endfire_abs_cos=0.95))
print('NOMP (cached, os4 rc3) failure anatomy, official bank, 8000 scenes, 1000 per SNR')
print('snr | Pd | scenes with 0/1/2/3 paths correct | wrong paths: bias(<=3deg) / outlier | outlier types: missed / spurious-est / doubled | endfire share of bias')
for s in sorted(set(snr.tolist())):
    m = snr == s; k = ok_p[m].sum(1)
    hist = [float((k == j).mean()) for j in range(4)]
    wrong = ~ok_p[m]; nw = wrong.sum()
    bias = wrong & (emax[m] <= 3.0); outl = wrong & (emax[m] > 3.0)
    missed = (near_t[m] == 0).sum(); doubled = (near_t[m] >= 2).sum(); spur = (near_e[m] == 0).sum()
    # distance from each missed true path to the nearest spurious estimate (same scene)
    dd = []
    for i in np.where(m)[0]:
        for t in np.where(near_t[i] == 0)[0]:
            sp = np.where(near_e[i] == 0)[0]
            if len(sp): dd.append(dist[i, t, sp].min())
    dd = np.array(dd) if dd else np.array([np.nan])
    r = dict(pd=float(ok_c[m].mean()), paths_correct_hist=hist, wrong_paths=int(nw), wrong_bias=int(bias.sum()), wrong_outlier=int(outl.sum()),
             missed_paths=int(missed), spurious_estimates=int(spur), doubled_paths=int(doubled), scenes_with_double=float((near_t[m] >= 2).any(1).mean()),
             missed_to_spurious_deg_median=float(np.nanmedian(dd)), missed_to_spurious_deg_q25_q75=[float(np.nanpercentile(dd, 25)), float(np.nanpercentile(dd, 75))],
             bias_endfire_share=float(endf[m][bias].mean()) if bias.sum() else None, all_paths_endfire_share=float(endf[m].mean()),
             bias_median_err_deg=float(np.median(emax[m][bias])) if bias.sum() else None)
    # objective comparison on scenes where NOMP is not fully correct
    bad = m & ~ok_p.all(1)
    for ref in ['truth', 'oracle']:
        r[f'res_{ref}_lower_frac_badscenes'] = float((T[ref]['res'][bad] < T['nomp']['res'][bad]).mean())
        r[f'map_{ref}_higher_frac_badscenes'] = float((T[ref]['map'][bad] > T['nomp']['map'][bad]).mean())
        r[f'mll_{ref}_higher_frac_badscenes'] = float((T[ref]['mll'][bad] > T['nomp']['mll'][bad]).mean())
    r['bad_scenes'] = int(bad.sum())
    r['nomp_violates_sep_frac'] = float((T['nomp']['sep'][m] < SEP).mean()); r['nomp_violates_sep_frac_badscenes'] = float((T['nomp']['sep'][bad] < SEP).mean())
    r['oracle_seeded_pd'] = float((eo[m] <= 1).mean())
    # u-domain view of the wrong paths: wrapped (u,v) distance (DFT bins, 2pi/16) between each true path and its matched estimate
    w = np.where(m)[0]; DU = []
    for i in w:
        _, gt, pr = comp_errors(pt[i], ft[i], pn[i], fn_[i])
        uu = np.angle(np.exp(1j * np.pi * (np.cos(gt[0]) - np.cos(pr[0])))); vv = np.angle(np.exp(1j * np.pi * (np.cos(gt[1]) - np.cos(pr[1]))))
        okp = (np.abs(get_ang_difference(gt, pr, return_flat=False)) <= 1).all(0)
        for j in np.where(~okp)[0]:
            ang = np.abs(get_ang_difference(gt, pr, return_flat=False))[:, j].max()
            DU.append((max(abs(uu[j]), abs(vv[j])) / (2 * np.pi / 16), ang))
    DU = np.array(DU)
    r['wrong_uv_close_lt025bin_frac'] = float((DU[:, 0] < 0.25).mean())
    r['wrong_endfire_mirror_frac'] = float(((DU[:, 0] < 0.25) & (DU[:, 1] > 90)).mean())
    r['wrong_uv_far_gt1bin_frac'] = float((DU[:, 0] > 1).mean())
    out['per_snr'][s] = r
    print(f"{s:4d} | {r['pd']:.4f} | {' '.join(f'{h:.3f}' for h in hist)} | {r['wrong_bias']}/{r['wrong_outlier']} | {missed}/{spur}/{doubled} "
          f"| {r['bias_endfire_share'] if r['bias_endfire_share'] is None else round(r['bias_endfire_share'],3)} (all paths {r['all_paths_endfire_share']:.3f}) | missed->spurious median {r['missed_to_spurious_deg_median']:.1f} deg")
print('\nObjective on scenes where NOMP has < 3 paths correct: frac where reference has LOWER residual (search loss) / HIGHER MAP score / higher marginal lik.')
for s, r in out['per_snr'].items():
    print(f"{s:4d} bad={r['bad_scenes']:4d} | truth: res-lower {r['res_truth_lower_frac_badscenes']:.3f} MAP-higher {r['map_truth_higher_frac_badscenes']:.3f} mll-higher {r['mll_truth_higher_frac_badscenes']:.3f} "
          f"| oracle-seeded: res-lower {r['res_oracle_lower_frac_badscenes']:.3f} MAP-higher {r['map_oracle_higher_frac_badscenes']:.3f} | NOMP violates sep: all {r['nomp_violates_sep_frac']:.3f} bad {r['nomp_violates_sep_frac_badscenes']:.3f}")
# pick-better-of {NOMP, oracle-seeded} by each objective: how much Pd each objective can certify
print('\nPd if choosing between NOMP and the oracle-seeded optimum by objective (upper bound for a perfect search under that objective):')
sel = {}
for crit in ['res', 'mll', 'map']:
    better = (T['oracle'][crit] < T['nomp'][crit]) if crit == 'res' else (T['oracle'][crit] > T['nomp'][crit])
    ec = np.where(better[:, None, None], eo, e); sel[crit] = ec
pdr = {}
for nm, ec in [('nomp', e), ('oracle_seeded', eo)] + [(f'pick_by_{k}', v) for k, v in sel.items()]:
    pdr[nm] = {int(s): float((ec[snr == s] <= 1).mean()) for s in sorted(set(snr.tolist()))}
    pdr[nm]['mean'] = float(np.mean(list(pdr[nm].values())))
    print(f"{nm:18s} mean {pdr[nm]['mean']:.4f} | " + ' '.join(f"{s}:{pdr[nm][s]:.4f}" for s in sorted(set(snr.tolist()))))
out['pick_between_nomp_and_oracle'] = pdr
json.dump(out, open(f'{HERE}/anatomy.json', 'w'), indent=1, default=float)
print('\nWrong paths in the (u,v) domain: frac within 0.25 DFT bin of the truth (precision / endfire-limited) | of which endfire mirror (angle err > 90 deg, u=+-pi alias) | frac > 1 bin away (true outliers)')
for s, r in out['per_snr'].items():
    print(f"{s:4d} | close {r['wrong_uv_close_lt025bin_frac']:.3f} | mirror {r['wrong_endfire_mirror_frac']:.3f} | far {r['wrong_uv_far_gt1bin_frac']:.3f}")
