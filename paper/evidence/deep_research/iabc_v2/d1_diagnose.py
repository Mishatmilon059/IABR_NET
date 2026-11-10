"""Diagnosis: why the current IABC (458 params) removes so little.
Old IABC = notebook IABC (lines 440-470) run through the CNN front (icommon.net). Its per-antenna inputs are
  mag = 0.5 log(row/col energy / mean)    -> equals the log-gain exactly only if every row of Ht has the same energy (L = 1)
  dph = detrended phase of sum_m Ht[n,m] exp(...) after de-rotating by the STRONGEST beam only -> equals the phase only for L = 1
plus a 4-number context; one shared per-antenna MLP (10 -> 32 -> 2).
Experiments (fresh seeds, recorded):
 A. % of identifiable MSE removed vs number of paths L in {1,2,3,5,9} and SNR in {0,10,25} dB, in-range cell (8 deg, 3 dB),
    300 scenes per (L,SNR), seed 8920 + 3*iL + iS (noise [seed,1], impairment [seed,3]). Estimators: old IABC; the raw input
    features used directly as the estimate ("feature = estimate"); the best linear read-out of those features (least-squares scale
    per L/SNR/component, fitted on the first half, scored on the second half); oracle ALS (true angles, 4 it, lam 1, tau 0).
    Also: regression slope of the IABC output on the target (1 = unbiased; < 1 = shrinkage).
 B. vs power ratio |alpha_2|^2/|alpha_1|^2, L=2, 25 dB, cell (8,3): 1500 scenes, seed 8940.
 C. vs severity (training range phase <= 8 deg / gain <= 3 dB): slope of IABC output on target and % removed, L=3, 25 dB,
    cells ph8, ph20, ph45, g3, g6; 600 scenes, seed 8941 (same angles/noise/uniforms in every cell).
Bootstrap not needed for the qualitative trends; 95% CIs given for A by resampling scenes (2000, seed 0)."""
import os
for _k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[_k] = '2'
from v2common import *
torch.set_num_threads(2)

LS = [1, 2, 3, 5, 9]; SN = [0, 10, 25]; N = 300


def feat_est(Ht, Y):
    with torch.no_grad():
        f = cnn.iabc.features(torch.from_numpy(Ht.astype(np.complex64)), torch.from_numpy(Y.astype(np.complex64))).double().numpy()
    return f[..., [1, 0]]          # (B,32,2) as [phase, log-gain]: dph, mag


def removed(tgt, est, j, idx=None):
    idx = np.arange(len(tgt)) if idx is None else idx
    e = ident_err(tgt[idx], est[idx])[..., j]; b = tgt[idx][..., j]
    return 100 * (1 - (e ** 2).sum() / (b ** 2).sum())


def ci_removed(tgt, est, j):
    e = (ident_err(tgt, est)[..., j] ** 2).sum(1); b = (tgt[..., j] ** 2).sum(1)
    R = np.random.default_rng(0).integers(0, len(e), (2000, len(e)))
    v = 100 * (1 - e[R].sum(1) / b[R].sum(1)); return np.percentile(v, 2.5), np.percentile(v, 97.5)


def slope(tgt, est, j):
    t = tgt[..., j].ravel(); e = est[..., j].ravel(); return float((t @ e) / (t @ t))


res = dict(A={}, B={}, C={})
print('## A. % identifiable MSE removed vs L and SNR, cell (8 deg, 3 dB) [in training range]; 300 scenes each')
print('| L | SNR | phase: IABC [95% CI] | feature=est | best linear | oracle ALS | IABC slope | gain: IABC [95% CI] | feature=est | best linear | oracle ALS | IABC slope |')
print('|---|---|---|---|---|---|---|---|---|---|---|---|')
for iL, L in enumerate(LS):
    for iS, snr in enumerate(SN):
        seed = 8920 + 3 * iL + iS
        d = make_scenes(seed, N, L, snr, 8, 3); Ht = to_ant(d['Y'])
        _, _, imp = net(d['Y']); fe = feat_est(Ht, d['Y']); tgt = d['tgt']
        U0, V0 = true_uv(d['psi'][:, :L], d['phi'][:, :L])
        est_or = selfcal_ret(Ht, U0, V0, 1.0, 0.0, iters=4, fix_angles=True)[2]
        h1, h2 = np.arange(N // 2), np.arange(N // 2, N)
        row = {}
        for j, comp in enumerate(['phase', 'gain']):
            k = float((tgt[h1][..., j].ravel() @ fe[h1][..., j].ravel()) / (fe[h1][..., j].ravel() @ fe[h1][..., j].ravel()))
            lin = fe.copy(); lin[..., j] *= k
            row[comp] = dict(iabc=removed(tgt, imp, j), iabc_ci=ci_removed(tgt, imp, j), feat=removed(tgt, fe, j),
                             lin=removed(tgt, lin, j, h2), lin_k=k, oracle=removed(tgt, est_or, j), slope=slope(tgt, imp, j))
        res['A'][f'L{L}_snr{snr}'] = row
        p, g = row['phase'], row['gain']
        print(f"| {L} | {snr} | {p['iabc']:.1f} [{p['iabc_ci'][0]:.1f},{p['iabc_ci'][1]:.1f}] | {p['feat']:.1f} | {p['lin']:.1f} (k={p['lin_k']:.2f}) | {p['oracle']:.1f} | {p['slope']:.2f} "
              f"| {g['iabc']:.1f} [{g['iabc_ci'][0]:.1f},{g['iabc_ci'][1]:.1f}] | {g['feat']:.1f} | {g['lin']:.1f} (k={g['lin_k']:.2f}) | {g['oracle']:.1f} | {g['slope']:.2f} |", flush=True)

print('\n## B. vs path-power ratio |a2|^2/|a1|^2 (L=2, 25 dB, cell 8 deg / 3 dB, 1500 scenes, seed 8940)')
d = make_scenes(8940, 1500, 2, 25, 8, 3); Ht = to_ant(d['Y']); _, _, imp = net(d['Y']); fe = feat_est(Ht, d['Y']); tgt = d['tgt']
ratio = np.abs(d['alpha'][:, 1]) ** 2 / np.abs(d['alpha'][:, 0]) ** 2
print('| ratio bin | n | phase: IABC | feature=est | gain: IABC | feature=est |'); print('|---|---|---|---|---|---|')
for lo, hi in [(0, 0.05), (0.05, 0.15), (0.15, 0.3), (0.3, 0.6), (0.6, 1.01)]:
    q = np.where((ratio >= lo) & (ratio < hi))[0]
    r = [removed(tgt, imp, 0, q), removed(tgt, fe, 0, q), removed(tgt, imp, 1, q), removed(tgt, fe, 1, q)]
    res['B'][f'{lo}-{hi}'] = dict(n=len(q), iabc_phase=r[0], feat_phase=r[1], iabc_gain=r[2], feat_gain=r[3])
    print(f'| {lo}-{hi} | {len(q)} | {r[0]:.1f} | {r[1]:.1f} | {r[2]:.1f} | {r[3]:.1f} |', flush=True)

print('\n## C. vs severity (L=3, 25 dB, 600 scenes, seed 8941): IABC output slope on target and % removed')
print('| cell | in range | phase slope | phase % removed | gain slope | gain % removed |'); print('|---|---|---|---|---|---|')
for name, pd, gd in [('ph8', 8, 0), ('ph20', 20, 0), ('ph45', 45, 0), ('g3', 0, 3), ('g6', 0, 6)]:
    d = make_scenes(8941, 600, 3, 25, pd, gd); _, _, imp = net(d['Y']); tgt = d['tgt']
    r = dict(phase_slope=slope(tgt, imp, 0) if pd else None, phase_rem=removed(tgt, imp, 0) if pd else None,
             gain_slope=slope(tgt, imp, 1) if gd else None, gain_rem=removed(tgt, imp, 1) if gd else None)
    res['C'][name] = r
    f = lambda x: '-' if x is None else f'{x:.2f}' if abs(x) < 5 else f'{x:.1f}'
    print(f"| {name} | {'yes' if pd <= 8 and gd <= 3 else 'no'} | {f(r['phase_slope'])} | {f(r['phase_rem'])} | {f(r['gain_slope'])} | {f(r['gain_rem'])} |", flush=True)
json.dump(res, open(f'{V2}/d1_diagnose.json', 'w'), indent=1, default=float)
print('wrote d1_diagnose.json')
