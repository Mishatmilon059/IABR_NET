"""Assemble verify.json (all JSON outputs) and REPORT.md (method notes + tables.md + latency). No new numbers are computed here.
usage: python make_report.py"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vcommon import HERE

J = {k: json.load(open(f'{HERE}/{k}.json')) for k in ['check_metric', 'check_ht', 'run_nomp', 'verify_tables', 'latency']}
json.dump(J, open(f'{HERE}/verify.json', 'w'), indent=1)
cm, ch, rn, vt, lat = J['check_metric'], J['check_ht'], J['run_nomp'], J['verify_tables'], J['latency']
hd = vt['headroom']

L = []
L.append('# Independent verification of the NOMP-only (no network) result\n')
L.append('Folder: `paper/evidence/deep_research/verify/`. Every number below is printed by a script in this folder; its stdout is in the `.log` next to it. '
         'Large arrays are in `cache/` (gitignored). No existing file was modified.\n')
L.append('## Verdict\n')
o = vt['official']['methods']
L.append(f'* An independently written NOMP (`nomp_mine.py`, separable-correlation Newton, written from the algorithm description; the notebook decoder was read only for '
         f'the sign/angle conventions) reproduces the cached `NOMP_no_network` predictions on all 10 banks, scene by scene: '
         + ', '.join(f'{k} {v["frac_lt_1e6rad"]:.4f}' for k, v in rn.items()) + ' of scenes agree to < 1e-6 rad '
         f'(max difference over all banks {max(v["max_diff_rad"] for v in rn.values()):.1e} rad).')
L.append(f'* An independently written metric (`metric_mine.py`) agrees with `common.scene_stats/metric_by_snr` and with the notebook `metrics()` on all '
         f'{len(cm["methods"])} cached prediction sets of the official bank: per-scene counts identical, worst |difference| of a mean metric {cm["worst"]:.1e}; '
         f'paired bootstrap identical to `common.boot_metrics` (max |diff| {cm["bootstrap_max_diff"]:.1e}). No mismatch.')
L.append(f'* Input check: cached `Ht` = `to_ant(Y)` recomputed from the frozen bank, 200 scenes, max relative error {ch["max_rel"]:.1e} (complex64 storage).')
L.append(f'* **The NOMP-only claim is confirmed**: official bank paper-style Pd NOMP {o["NOMP (mine)"]["mean"]["pd_paper"]:.4f} '
         f'[{o["NOMP (mine)"]["ci"]["pd_paper"][0]:.4f}, {o["NOMP (mine)"]["ci"]["pd_paper"][1]:.4f}] vs U-Net {o["U-Net"]["mean"]["pd_paper"]:.4f} '
         f'(strict {o["U-Net"]["mean"]["pd_strict"]:.4f}), IABR-v2 dual {o["IABR-v2 dual"]["mean"]["pd_paper"]:.4f}, IABR-Net heat-seeded '
         f'{o["IABR-Net heat"]["mean"]["pd_paper"]:.4f}, network only {o["network only"]["mean"]["pd_paper"]:.4f}. Neither network-based decoder beats NOMP on the '
         f'official bank (both paired differences are negative with CIs excluding 0). Of the 10 banks, the networked decoders are clearly ahead of NOMP only on gain_g4_snr15.')
L.append(f'* Headroom (official): seeding the same Newton+cyclic decoder at the true (u,v) gives Pd {hd["all"]["oracle3"]:.4f} (+{hd["all"]["gap"]:.4f}), but almost all of '
         f'that gap is at low SNR and comes from scenes where the true-seeded local optimum fits the data WORSE than NOMP\'s solution. Keeping the oracle start only '
         f'where the objective prefers it (lower residual) gives {hd["all"]["minres"]:.4f} (+{hd["all"]["minres_gap"]:.4f} '
         f'[{hd["all"]["minres_gap_ci"][0]:+.4f}, {hd["all"]["minres_gap_ci"][1]:+.4f}]): NOMP\'s search loses about half a point of Pd against what its objective can certify.\n')
L.append('## Method notes\n')
L.append('* **Channel**: Y (16x16 beamspace) -> Hant = W_IDEAL Y F_IDEAL^H; atom a(u,v)[n,m] = exp(-j n u + j m v), u = pi cos(psi) (AoA), v = pi cos(phi) (AoD); '
         'psi = arccos(wrap(u)/pi), phi = arccos(wrap(v)/pi). Input = `Ht` of `outputs/cache/net__<bank>.npz` (complex64, cast to complex128).')
L.append('* **NOMP (mine)**, known L, per step: 64x64 zero-padded FFT of the residual (bin (k1,k2) <-> u = -2 pi k1/64, v = 2 pi k2/64), argmax seed; 4 single-path Newton steps on '
         '|<a,r>|^2 (step only where the Hessian is negative definite; each component clipped to 2 pi/64); cyclic re-estimation of all detected paths, 3 rounds, '
         '2 Newton steps per path against y minus the LS fit of the others (LS gains recomputed before every path, ridge 1e-9); LS gains; residual update. '
         'Correlation derivatives use the separable form p(u)^T R q(v) (different arithmetic from the notebook\'s 256-term sums). Deterministic.')
L.append('* **Oracle-seeded**: same cyclic re-estimation started at the true (u,v) of all L paths (3 rounds; 10 rounds as sensitivity), known L.')
L.append('* **Metric (mine)**: scenes with fewer than L predictions are dropped (paper-style) or count 2L misses (strict); otherwise the first L predictions are '
         'matched to the truth by the minimum total Euclidean distance in raw (psi,phi) (exhaustive over L! permutations), component errors wrapped to (-180,180] deg, '
         'good if |e| <= 1 deg; RMSE over good components; per SNR group, then mean over groups. CIs: paired bootstrap, 2000 resamples within SNR groups, seed 0, '
         'percentile 95%; the same resample indices for all methods of a bank.')
L.append('* **Method sources**: U-Net = `pred__U-Net_paper` cache; IABR-v2 dual = `pred__IABR-v2`; network only = `pred__v2_network_only_no_physics`; NOMP (cached) = '
         '`pred__NOMP_no_network`; NOMP+selfcal = `new_results/preds/selfcal_pred__<bank>`; IABR-Net heat = `new_results/preds_heat/<bank>`. For sep_2deg_snr15, '
         'phase_d0_snr0 and phase_d0_snr15 there was no network-only cache and no heat-seeded prediction; `fill_missing.py` recomputed them with the notebook '
         '`parabolic_heat_decode` and the `heat_main.py` recipe after checking that both recipes reproduce existing caches exactly (`fill_missing.log`: max |diff| 4.4e-16 and 0 rad).')
L.append(f'* **Latency**: single thread (OMP/OPENBLAS/MKL = 1 before import, torch.set_num_threads(1)), {lat["platform"]}, numpy {lat["numpy"]}, torch {lat["torch"]}; '
         f'official scenes, L=3, 5 warm-ups, median of 60 runs. The CPU forward reproduces the cached heat/Hc up to max |diff| {lat["cpu_vs_cache"]["heat"]:.1e} / '
         f'{lat["cpu_vs_cache"]["Hc"]:.1e}.\n')
L.append('## Per-scene agreement, NOMP (mine) vs cached NOMP_no_network\n')
L.append('| bank | N | scenes max diff < 1e-6 rad | scenes max diff < 0.01 deg | max diff (rad) |\n|---|---|---|---|---|')
for k, v in rn.items():
    L.append(f'| {k} | {v["n"]} | {v["frac_lt_1e6rad"]:.4f} | {v["frac_lt_001deg"]:.4f} | {v["max_diff_rad"]:.1e} |')
L.append('\n## Metric cross-check (official bank, mean over SNR; mine / common.py / notebook metrics())\n')
L.append('| prediction set | pd_paper | pd_strict | pd_source | rmse | per-scene counts identical | max abs diff |\n|---|---|---|---|---|---|---|')
for k, v in cm['methods'].items():
    L.append(f'| {k} | {v["mine"]["pd_paper"]:.4f} / {v["common"]["pd_paper"]:.4f} / {v["notebook"]["pd_paper"]:.4f} | {v["mine"]["pd_strict"]:.4f} / {v["common"]["pd_strict"]:.4f} '
             f'| {v["mine"]["pd_source"]:.4f} / {v["common"]["pd_source"]:.4f} | {v["mine"]["rmse"]:.4f} / {v["common"]["rmse"]:.4f} | {v["per_scene_counts_identical"]} | {v["max_abs_mean_diff"]:.1e} |')
L.append('')
L.append(open(f'{HERE}/tables.md', encoding='utf-8').read())
L.append('\n## Latency (CPU, single thread)\n')
L.append('| what | batch | median ms / call | p10-p90 ms | median ms / scene |\n|---|---|---|---|---|')
for k, v in lat['timings'].items():
    L.append(f'| {k.split(" | ")[0]} | {v["batch"]} | {v["median_ms_per_call"]:.2f} | {v["p10_ms"]:.2f}-{v["p90_ms"]:.2f} | {v["median_ms_per_scene"]:.3f} |')
L.append('\n## Commands (run from this folder, in this order)\n')
L.append('```\npython check_metric.py > check_metric.log\npython check_ht.py    > check_ht.log\npython run_nomp.py    > run_nomp.log\n'
         'python fill_missing.py > fill_missing.log\npython tables.py      > tables.log\npython latency.py     > latency.log\npython make_report.py\n```')
L.append('Seeds: bootstrap `np.random.default_rng(0)`; latency scene draw `np.random.default_rng(0)`; decoders are deterministic.')
open(f'{HERE}/REPORT.md', 'w', encoding='utf-8').write('\n'.join(L) + '\n')
print('REPORT.md and verify.json written')
