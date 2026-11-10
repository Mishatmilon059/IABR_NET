"""Sanity of the stress pipeline against the existing impairment-free banks phase_d0_snr0 / phase_d0_snr15 (v1 suite, n=500, L=3).
 A. pipeline equivalence: regenerate the two banks with stress_sim (seeds 1000 / 1015, reproduces the stored Y to float32 rounding, see
    check_sim.log), run scommon.run_methods, and compare paper-style Pd and per-scene agreement with the existing predictions:
    NOMP = pred__NOMP_no_network, NOMP+selfcal = new_results/preds/selfcal_pred, IABR-v2 dual = pred__IABR-v2,
    IABR-Net heat / network only = verify/cache/heat__, netonly__ (verify stage, notebook recipes).
 B. the impairment-free stress banks (seeds 5000 / 5015, parts/clean.json, cache/preds__clean_clean_snr*.npz) vs the existing banks:
    unpaired difference of paper-style Pd with a 95% CI from independent bootstraps (2000 resamples each, seeds 0 and 1); pass = CI contains 0.
usage: python sanity_clean.py  (stdout -> sanity_clean.log, writes sanity_clean.json)"""
from scommon import *

V = 'D:/ai_ml_project/paper/evidence/deep_research/verify/cache'
EX = {'NOMP': lambda n: f'{C.CACHE}/pred__NOMP_no_network__{n}.npz', 'NOMP+selfcal': lambda n: f'{C.OUT}/preds/selfcal_pred__{n}.npz',
      'IABR-v2 dual': lambda n: f'{C.CACHE}/pred__IABR-v2__{n}.npz', 'IABR-Net heat': lambda n: f'{V}/heat__{n}.npz',
      'network only': lambda n: f'{V}/netonly__{n}.npz'}


def boot1(S, seed, B=2000):
    rng = np.random.default_rng(seed); n = len(S); R = rng.integers(0, n, (B, n))
    return S[:, 0][R].sum(1) / (S[:, 0][R].sum(1) + S[:, 1][R].sum(1))


out = {}
for snr, seed, name in [(0, 1000, 'phase_d0_snr0'), (15, 1015, 'phase_d0_snr15')]:
    bank = C.load_bank(name); b = SS.base_scenes(500, 3, float(snr), seed)
    assert np.allclose(b['psi'], bank['psi'], atol=1e-6)
    pr, _, _ = run_methods(SS.generate(b, 'clean'), b['psi'], b['phi'], 3)
    mine_new = json.load(open(f'{HERE}/parts/clean.json'))
    newb = [r for r in mine_new['banks'] if r['snr'] == snr][0]
    zc = np.load(f'{MYCACHE}/preds__clean_clean_snr{snr}.npz'); bn = SS.base_scenes(500, 3, float(snr), 5000 + snr)
    out[name] = {}
    print(f'{name}:')
    for m, f in EX.items():
        ex = C.load_preds(f(name)); S_ex = C.scene_stats(ex, bank); S_me = stats(pr[m], b['psi'], b['phi'], 3)
        pd_ex = C.metric_from_stats(S_ex, np.arange(500))['pd_paper']; pd_me = C.metric_from_stats(S_me, np.arange(500))['pd_paper']
        mp = as_list(pr[m])
        agree = np.mean([len(e[0]) == len(q[0]) and np.allclose(np.sort(e[0]), np.sort(q[0]), atol=np.deg2rad(0.01)) and
                         np.allclose(np.sort(e[1]), np.sort(q[1]), atol=np.deg2rad(0.01)) for e, q in zip(ex, mp)])
        # B: new clean bank vs existing
        S_new = stats((zc[f'{m}__psi'], zc[f'{m}__phi']), bn['psi'], bn['phi'], 3)
        pd_new = C.metric_from_stats(S_new, np.arange(500))['pd_paper']
        assert abs(pd_new - newb['methods'][m]['pd_paper']) < 1e-12
        d = boot1(S_new, 0) - boot1(S_ex, 1); lo, hi = np.percentile(d, [2.5, 97.5])
        out[name][m] = dict(pd_existing=pd_ex, pd_rerun=pd_me, scene_agreement_0p01deg=float(agree), pd_new_clean_bank=pd_new,
                            diff_new_minus_existing=[pd_new - pd_ex, float(lo), float(hi)], within_ci=bool(lo <= 0 <= hi))
        print(f'  {m:14s} A: existing {pd_ex:.4f} | rerun {pd_me:.4f} | scenes agreeing < 0.01 deg {agree:.4f}   '
              f'B: new clean bank (seed {5000 + snr}) {pd_new:.4f}, new - existing {pd_new - pd_ex:+.4f} [{lo:+.4f},{hi:+.4f}] '
              f'{"PASS (CI contains 0)" if lo <= 0 <= hi else "FAIL"}', flush=True)
json.dump(out, open(f'{HERE}/sanity_clean.json', 'w'), indent=1)
