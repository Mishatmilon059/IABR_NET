# Independent verification of the NOMP-only (no network) result

Folder: `paper/evidence/deep_research/verify/`. Every number below is printed by a script in this folder; its stdout is in the `.log` next to it. Large arrays are in `cache/` (gitignored). No existing file was modified.

## Verdict

* An independently written NOMP (`nomp_mine.py`, separable-correlation Newton, written from the algorithm description; the notebook decoder was read only for the sign/angle conventions) reproduces the cached `NOMP_no_network` predictions on all 10 banks, scene by scene: official 1.0000, T2_d5 1.0000, gain_g4_snr15 1.0000, oodphase_d15_snr15 1.0000, L8_snr0 1.0000, L8_snr15 1.0000, sep_5deg_snr15 1.0000, sep_2deg_snr15 1.0000, phase_d0_snr0 1.0000, phase_d0_snr15 1.0000 of scenes agree to < 1e-6 rad (max difference over all banks 5.6e-14 rad).
* An independently written metric (`metric_mine.py`) agrees with `common.scene_stats/metric_by_snr` and with the notebook `metrics()` on all 10 cached prediction sets of the official bank: per-scene counts identical, worst |difference| of a mean metric 1.1e-16; paired bootstrap identical to `common.boot_metrics` (max |diff| 2.2e-16). No mismatch.
* Input check: cached `Ht` = `to_ant(Y)` recomputed from the frozen bank, 200 scenes, max relative error 1.5e-07 (complex64 storage).
* **The NOMP-only claim is confirmed**: official bank paper-style Pd NOMP 0.7581 [0.7544, 0.7618] vs U-Net 0.7366 (strict 0.6940), IABR-v2 dual 0.7551, IABR-Net heat-seeded 0.7503, network only 0.4507. Neither network-based decoder beats NOMP on the official bank (both paired differences are negative with CIs excluding 0). Of the 10 banks, the networked decoders are clearly ahead of NOMP only on gain_g4_snr15.
* Headroom (official): seeding the same Newton+cyclic decoder at the true (u,v) gives Pd 0.8261 (+0.0680), but almost all of that gap is at low SNR and comes from scenes where the true-seeded local optimum fits the data WORSE than NOMP's solution. Keeping the oracle start only where the objective prefers it (lower residual) gives 0.7632 (+0.0051 [+0.0041, +0.0061]): NOMP's search loses about half a point of Pd against what its objective can certify.

## Method notes

* **Channel**: Y (16x16 beamspace) -> Hant = W_IDEAL Y F_IDEAL^H; atom a(u,v)[n,m] = exp(-j n u + j m v), u = pi cos(psi) (AoA), v = pi cos(phi) (AoD); psi = arccos(wrap(u)/pi), phi = arccos(wrap(v)/pi). Input = `Ht` of `outputs/cache/net__<bank>.npz` (complex64, cast to complex128).
* **NOMP (mine)**, known L, per step: 64x64 zero-padded FFT of the residual (bin (k1,k2) <-> u = -2 pi k1/64, v = 2 pi k2/64), argmax seed; 4 single-path Newton steps on |<a,r>|^2 (step only where the Hessian is negative definite; each component clipped to 2 pi/64); cyclic re-estimation of all detected paths, 3 rounds, 2 Newton steps per path against y minus the LS fit of the others (LS gains recomputed before every path, ridge 1e-9); LS gains; residual update. Correlation derivatives use the separable form p(u)^T R q(v) (different arithmetic from the notebook's 256-term sums). Deterministic.
* **Oracle-seeded**: same cyclic re-estimation started at the true (u,v) of all L paths (3 rounds; 10 rounds as sensitivity), known L.
* **Metric (mine)**: scenes with fewer than L predictions are dropped (paper-style) or count 2L misses (strict); otherwise the first L predictions are matched to the truth by the minimum total Euclidean distance in raw (psi,phi) (exhaustive over L! permutations), component errors wrapped to (-180,180] deg, good if |e| <= 1 deg; RMSE over good components; per SNR group, then mean over groups. CIs: paired bootstrap, 2000 resamples within SNR groups, seed 0, percentile 95%; the same resample indices for all methods of a bank.
* **Method sources**: U-Net = `pred__U-Net_paper` cache; IABR-v2 dual = `pred__IABR-v2`; network only = `pred__v2_network_only_no_physics`; NOMP (cached) = `pred__NOMP_no_network`; NOMP+selfcal = `new_results/preds/selfcal_pred__<bank>`; IABR-Net heat = `new_results/preds_heat/<bank>`. For sep_2deg_snr15, phase_d0_snr0 and phase_d0_snr15 there was no network-only cache and no heat-seeded prediction; `fill_missing.py` recomputed them with the notebook `parabolic_heat_decode` and the `heat_main.py` recipe after checking that both recipes reproduce existing caches exactly (`fill_missing.log`: max |diff| 4.4e-16 and 0 rad).
* **Latency**: single thread (OMP/OPENBLAS/MKL = 1 before import, torch.set_num_threads(1)), Windows-11-10.0.26200-SP0, numpy 2.3.5, torch 2.7.1+cu118; official scenes, L=3, 5 warm-ups, median of 60 runs. The CPU forward reproduces the cached heat/Hc up to max |diff| 2.8e-04 / 3.3e-06.

## Per-scene agreement, NOMP (mine) vs cached NOMP_no_network

| bank | N | scenes max diff < 1e-6 rad | scenes max diff < 0.01 deg | max diff (rad) |
|---|---|---|---|---|
| official | 8000 | 1.0000 | 1.0000 | 4.3e-14 |
| T2_d5 | 8000 | 1.0000 | 1.0000 | 5.6e-14 |
| gain_g4_snr15 | 500 | 1.0000 | 1.0000 | 1.7e-14 |
| oodphase_d15_snr15 | 500 | 1.0000 | 1.0000 | 8.4e-15 |
| L8_snr0 | 500 | 1.0000 | 1.0000 | 1.2e-14 |
| L8_snr15 | 500 | 1.0000 | 1.0000 | 2.2e-14 |
| sep_5deg_snr15 | 500 | 1.0000 | 1.0000 | 2.3e-15 |
| sep_2deg_snr15 | 500 | 1.0000 | 1.0000 | 4.2e-15 |
| phase_d0_snr0 | 500 | 1.0000 | 1.0000 | 6.2e-15 |
| phase_d0_snr15 | 500 | 1.0000 | 1.0000 | 1.7e-14 |

## Metric cross-check (official bank, mean over SNR; mine / common.py / notebook metrics())

| prediction set | pd_paper | pd_strict | pd_source | rmse | per-scene counts identical | max abs diff |
|---|---|---|---|---|---|---|
| pred__IABR-v2__official | 0.7551 / 0.7551 / 0.7551 | 0.7551 / 0.7551 | 0.6681 / 0.6681 | 0.3337 / 0.3337 | True | 5.6e-17 |
| pred__NOMP_no_network__official | 0.7581 / 0.7581 / 0.7581 | 0.7581 / 0.7581 | 0.6741 / 0.6741 | 0.3277 / 0.3277 | True | 5.6e-17 |
| pred__U-Net_paper__official | 0.7366 / 0.7366 / 0.7366 | 0.6940 / 0.6940 | 0.6035 / 0.6035 | 0.3611 / 0.3611 | True | 5.6e-17 |
| pred__v2_greedy_init_only__official | 0.7548 / 0.7548 / 0.7548 | 0.7548 / 0.7548 | 0.6680 / 0.6680 | 0.3334 / 0.3334 | True | 5.6e-17 |
| pred__v2_heatmap_init_only__official | 0.7503 / 0.7503 / 0.7503 | 0.7503 / 0.7503 | 0.6642 / 0.6642 | 0.3329 / 0.3329 | True | 1.1e-16 |
| pred__v2_network_only_no_physics__official | 0.4507 / 0.4507 / 0.4507 | 0.4507 / 0.4507 | 0.2401 / 0.2401 | 0.5660 / 0.5660 | True | 1.1e-16 |
| pred__v2_no_IABC__official | 0.7580 / 0.7580 / 0.7580 | 0.7580 / 0.7580 | 0.6721 / 0.6721 | 0.3291 / 0.3291 | True | 5.6e-17 |
| pred__v2_no_self-cal__official | 0.7503 / 0.7503 / 0.7503 | 0.7503 / 0.7503 | 0.6602 / 0.6602 | 0.3424 / 0.3424 | True | 0.0e+00 |
| selfcal_pred__official | 0.7576 / 0.7576 / 0.7576 | 0.7576 / 0.7576 | 0.6720 / 0.6720 | 0.3288 / 0.3288 | True | 1.1e-16 |
| official | 0.7503 / 0.7503 / 0.7503 | 0.7503 / 0.7503 | 0.6642 / 0.6642 | 0.3329 / 0.3329 | True | 1.1e-16 |

## Side-by-side, per bank (mean over SNR groups; 95% paired-bootstrap percentile CI, 2000 resamples within SNR groups, seed 0)

### official  (N=8000, L=[3], SNR groups=[-10, -5, 0, 5, 10, 15, 20, 25])

| method | Pd paper-style [95% CI] | Pd strict [95% CI] | RMSE deg [95% CI] | dropped scenes |
|---|---|---|---|---|
| U-Net | 0.7366 [0.7327, 0.7405] | 0.6940 [0.6890, 0.6991] | 0.3611 [0.3577, 0.3644] | 538 |
| IABR-v2 dual | 0.7551 [0.7513, 0.7587] | 0.7551 [0.7513, 0.7587] | 0.3337 [0.3304, 0.3368] | 0 |
| IABR-Net heat | 0.7503 [0.7463, 0.7540] | 0.7503 [0.7463, 0.7540] | 0.3329 [0.3297, 0.3361] | 0 |
| network only | 0.4507 [0.4460, 0.4551] | 0.4507 [0.4460, 0.4551] | 0.5660 [0.5625, 0.5696] | 0 |
| NOMP (cached) | 0.7581 [0.7544, 0.7618] | 0.7581 [0.7544, 0.7618] | 0.3277 [0.3243, 0.3307] | 0 |
| NOMP (mine) | 0.7581 [0.7544, 0.7618] | 0.7581 [0.7544, 0.7618] | 0.3277 [0.3243, 0.3307] | 0 |
| NOMP+selfcal | 0.7576 [0.7540, 0.7611] | 0.7576 [0.7540, 0.7611] | 0.3288 [0.3256, 0.3319] | 0 |

| paired difference | d Pd paper-style [95% CI] | d Pd strict [95% CI] | d RMSE deg [95% CI] |
|---|---|---|---|
| IABR-Net heat - NOMP | -0.0078 [-0.0095, -0.0061] | -0.0078 [-0.0095, -0.0061] | +0.0053 [+0.0032, +0.0075] |
| IABR-v2 dual - NOMP | -0.0030 [-0.0043, -0.0016] | -0.0030 [-0.0043, -0.0016] | +0.0060 [+0.0041, +0.0079] |
| NOMP - U-Net | +0.0215 [+0.0181, +0.0248] | +0.0641 [+0.0602, +0.0680] | -0.0334 [-0.0373, -0.0297] |
| network only - NOMP | -0.3074 [-0.3117, -0.3032] | -0.3074 [-0.3117, -0.3032] | +0.2384 [+0.2339, +0.2430] |
| NOMP (mine) - NOMP (cached) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [-0.0000, +0.0000] |

### T2_d5  (N=8000, L=[3], SNR groups=[-10, -5, 0, 5, 10, 15, 20, 25])

| method | Pd paper-style [95% CI] | Pd strict [95% CI] | RMSE deg [95% CI] | dropped scenes |
|---|---|---|---|---|
| U-Net | 0.7277 [0.7238, 0.7318] | 0.6946 [0.6897, 0.6995] | 0.3713 [0.3679, 0.3746] | 427 |
| IABR-v2 dual | 0.7452 [0.7414, 0.7489] | 0.7452 [0.7414, 0.7489] | 0.3442 [0.3407, 0.3473] | 0 |
| IABR-Net heat | 0.7412 [0.7372, 0.7452] | 0.7412 [0.7372, 0.7452] | 0.3433 [0.3399, 0.3465] | 0 |
| network only | 0.4478 [0.4430, 0.4526] | 0.4478 [0.4430, 0.4526] | 0.5660 [0.5626, 0.5696] | 0 |
| NOMP (cached) | 0.7468 [0.7429, 0.7506] | 0.7468 [0.7429, 0.7506] | 0.3415 [0.3383, 0.3446] | 0 |
| NOMP (mine) | 0.7468 [0.7429, 0.7506] | 0.7468 [0.7429, 0.7506] | 0.3415 [0.3383, 0.3446] | 0 |
| NOMP+selfcal | 0.7476 [0.7438, 0.7513] | 0.7476 [0.7438, 0.7513] | 0.3412 [0.3379, 0.3442] | 0 |

| paired difference | d Pd paper-style [95% CI] | d Pd strict [95% CI] | d RMSE deg [95% CI] |
|---|---|---|---|
| IABR-Net heat - NOMP | -0.0056 [-0.0073, -0.0038] | -0.0056 [-0.0073, -0.0038] | +0.0018 [-0.0002, +0.0038] |
| IABR-v2 dual - NOMP | -0.0017 [-0.0031, -0.0003] | -0.0017 [-0.0031, -0.0003] | +0.0027 [+0.0007, +0.0045] |
| NOMP - U-Net | +0.0191 [+0.0158, +0.0223] | +0.0522 [+0.0484, +0.0559] | -0.0299 [-0.0338, -0.0264] |
| network only - NOMP | -0.2990 [-0.3035, -0.2946] | -0.2990 [-0.3035, -0.2946] | +0.2245 [+0.2201, +0.2292] |
| NOMP (mine) - NOMP (cached) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | -0.0000 [-0.0000, +0.0000] |

### gain_g4_snr15  (N=500, L=[3], SNR groups=[15])

| method | Pd paper-style [95% CI] | Pd strict [95% CI] | RMSE deg [95% CI] | dropped scenes |
|---|---|---|---|---|
| U-Net | 0.8126 [0.7938, 0.8293] | 0.8093 [0.7900, 0.8263] | 0.3143 [0.3023, 0.3252] | 2 |
| IABR-v2 dual | 0.9123 [0.8990, 0.9247] | 0.9123 [0.8990, 0.9247] | 0.2473 [0.2344, 0.2593] | 0 |
| IABR-Net heat | 0.9110 [0.8967, 0.9237] | 0.9110 [0.8967, 0.9237] | 0.2495 [0.2374, 0.2618] | 0 |
| network only | 0.4780 [0.4583, 0.4973] | 0.4780 [0.4583, 0.4973] | 0.5742 [0.5602, 0.5876] | 0 |
| NOMP (cached) | 0.8697 [0.8547, 0.8840] | 0.8697 [0.8547, 0.8840] | 0.2727 [0.2616, 0.2835] | 0 |
| NOMP (mine) | 0.8697 [0.8547, 0.8840] | 0.8697 [0.8547, 0.8840] | 0.2727 [0.2616, 0.2835] | 0 |
| NOMP+selfcal | 0.8950 [0.8800, 0.9080] | 0.8950 [0.8800, 0.9080] | 0.2408 [0.2293, 0.2519] | 0 |

| paired difference | d Pd paper-style [95% CI] | d Pd strict [95% CI] | d RMSE deg [95% CI] |
|---|---|---|---|
| IABR-Net heat - NOMP | +0.0413 [+0.0307, +0.0520] | +0.0413 [+0.0307, +0.0520] | -0.0232 [-0.0366, -0.0095] |
| IABR-v2 dual - NOMP | +0.0427 [+0.0327, +0.0530] | +0.0427 [+0.0327, +0.0530] | -0.0254 [-0.0389, -0.0115] |
| NOMP - U-Net | +0.0571 [+0.0430, +0.0718] | +0.0603 [+0.0457, +0.0757] | -0.0416 [-0.0537, -0.0293] |
| network only - NOMP | -0.3917 [-0.4140, -0.3693] | -0.3917 [-0.4140, -0.3693] | +0.3015 [+0.2827, +0.3200] |
| NOMP (mine) - NOMP (cached) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [-0.0000, +0.0000] |

### oodphase_d15_snr15  (N=500, L=[3], SNR groups=[15])

| method | Pd paper-style [95% CI] | Pd strict [95% CI] | RMSE deg [95% CI] | dropped scenes |
|---|---|---|---|---|
| U-Net | 0.8454 [0.8300, 0.8593] | 0.8437 [0.8280, 0.8580] | 0.3414 [0.3301, 0.3526] | 1 |
| IABR-v2 dual | 0.8693 [0.8547, 0.8833] | 0.8693 [0.8547, 0.8833] | 0.3104 [0.2996, 0.3217] | 0 |
| IABR-Net heat | 0.8667 [0.8510, 0.8813] | 0.8667 [0.8510, 0.8813] | 0.3107 [0.2998, 0.3216] | 0 |
| network only | 0.4917 [0.4723, 0.5100] | 0.4917 [0.4723, 0.5100] | 0.5901 [0.5766, 0.6032] | 0 |
| NOMP (cached) | 0.8633 [0.8487, 0.8777] | 0.8633 [0.8487, 0.8777] | 0.3199 [0.3081, 0.3316] | 0 |
| NOMP (mine) | 0.8633 [0.8487, 0.8777] | 0.8633 [0.8487, 0.8777] | 0.3199 [0.3081, 0.3316] | 0 |
| NOMP+selfcal | 0.8730 [0.8583, 0.8873] | 0.8730 [0.8583, 0.8873] | 0.3126 [0.3019, 0.3240] | 0 |

| paired difference | d Pd paper-style [95% CI] | d Pd strict [95% CI] | d RMSE deg [95% CI] |
|---|---|---|---|
| IABR-Net heat - NOMP | +0.0033 [-0.0057, +0.0123] | +0.0033 [-0.0057, +0.0123] | -0.0092 [-0.0194, +0.0008] |
| IABR-v2 dual - NOMP | +0.0060 [-0.0017, +0.0140] | +0.0060 [-0.0017, +0.0140] | -0.0095 [-0.0196, +0.0004] |
| NOMP - U-Net | +0.0180 [+0.0070, +0.0284] | +0.0197 [+0.0083, +0.0297] | -0.0216 [-0.0333, -0.0095] |
| network only - NOMP | -0.3717 [-0.3917, -0.3510] | -0.3717 [-0.3917, -0.3510] | +0.2702 [+0.2523, +0.2873] |
| NOMP (mine) - NOMP (cached) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [-0.0000, +0.0000] |

### L8_snr0  (N=500, L=[8], SNR groups=[0])

| method | Pd paper-style [95% CI] | Pd strict [95% CI] | RMSE deg [95% CI] | dropped scenes |
|---|---|---|---|---|
| U-Net | 0.4746 [0.4593, 0.4901] | 0.2924 [0.2691, 0.3144] | 0.4974 [0.4867, 0.5087] | 192 |
| IABR-v2 dual | 0.5028 [0.4898, 0.5158] | 0.5028 [0.4898, 0.5158] | 0.4748 [0.4665, 0.4833] | 0 |
| IABR-Net heat | 0.4983 [0.4851, 0.5110] | 0.4983 [0.4851, 0.5110] | 0.4794 [0.4714, 0.4883] | 0 |
| network only | 0.4422 [0.4296, 0.4545] | 0.4422 [0.4296, 0.4545] | 0.5257 [0.5164, 0.5342] | 0 |
| NOMP (cached) | 0.5019 [0.4880, 0.5146] | 0.5019 [0.4880, 0.5146] | 0.4741 [0.4659, 0.4823] | 0 |
| NOMP (mine) | 0.5019 [0.4880, 0.5146] | 0.5019 [0.4880, 0.5146] | 0.4741 [0.4659, 0.4823] | 0 |
| NOMP+selfcal | 0.5016 [0.4879, 0.5141] | 0.5016 [0.4879, 0.5141] | 0.4744 [0.4662, 0.4828] | 0 |

| paired difference | d Pd paper-style [95% CI] | d Pd strict [95% CI] | d RMSE deg [95% CI] |
|---|---|---|---|
| IABR-Net heat - NOMP | -0.0036 [-0.0096, +0.0022] | -0.0036 [-0.0096, +0.0022] | +0.0053 [+0.0001, +0.0111] |
| IABR-v2 dual - NOMP | +0.0009 [-0.0023, +0.0041] | +0.0009 [-0.0023, +0.0041] | +0.0007 [-0.0029, +0.0046] |
| NOMP - U-Net | +0.0272 [+0.0133, +0.0413] | +0.2095 [+0.1894, +0.2296] | -0.0233 [-0.0364, -0.0106] |
| network only - NOMP | -0.0596 [-0.0695, -0.0499] | -0.0596 [-0.0695, -0.0499] | +0.0516 [+0.0396, +0.0625] |
| NOMP (mine) - NOMP (cached) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [-0.0000, +0.0000] |

### L8_snr15  (N=500, L=[8], SNR groups=[15])

| method | Pd paper-style [95% CI] | Pd strict [95% CI] | RMSE deg [95% CI] | dropped scenes |
|---|---|---|---|---|
| U-Net | 0.8230 [0.8111, 0.8349] | 0.6864 [0.6561, 0.7131] | 0.3226 [0.3141, 0.3306] | 83 |
| IABR-v2 dual | 0.8628 [0.8530, 0.8725] | 0.8628 [0.8530, 0.8725] | 0.2850 [0.2778, 0.2924] | 0 |
| IABR-Net heat | 0.8343 [0.8219, 0.8460] | 0.8343 [0.8219, 0.8460] | 0.2859 [0.2787, 0.2933] | 0 |
| network only | 0.4689 [0.4567, 0.4805] | 0.4689 [0.4567, 0.4805] | 0.5843 [0.5760, 0.5929] | 0 |
| NOMP (cached) | 0.8651 [0.8547, 0.8756] | 0.8651 [0.8547, 0.8756] | 0.2739 [0.2668, 0.2812] | 0 |
| NOMP (mine) | 0.8651 [0.8547, 0.8756] | 0.8651 [0.8547, 0.8756] | 0.2739 [0.2668, 0.2812] | 0 |
| NOMP+selfcal | 0.8648 [0.8552, 0.8745] | 0.8648 [0.8552, 0.8745] | 0.2825 [0.2755, 0.2897] | 0 |

| paired difference | d Pd paper-style [95% CI] | d Pd strict [95% CI] | d RMSE deg [95% CI] |
|---|---|---|---|
| IABR-Net heat - NOMP | -0.0309 [-0.0394, -0.0229] | -0.0309 [-0.0394, -0.0229] | +0.0121 [+0.0054, +0.0186] |
| IABR-v2 dual - NOMP | -0.0024 [-0.0082, +0.0035] | -0.0024 [-0.0082, +0.0035] | +0.0111 [+0.0045, +0.0175] |
| NOMP - U-Net | +0.0421 [+0.0315, +0.0528] | +0.1788 [+0.1550, +0.2074] | -0.0487 [-0.0582, -0.0393] |
| network only - NOMP | -0.3963 [-0.4096, -0.3833] | -0.3963 [-0.4096, -0.3833] | +0.3104 [+0.2993, +0.3213] |
| NOMP (mine) - NOMP (cached) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | -0.0000 [-0.0000, +0.0000] |

### sep_5deg_snr15  (N=500, L=[2], SNR groups=[15])

| method | Pd paper-style [95% CI] | Pd strict [95% CI] | RMSE deg [95% CI] | dropped scenes |
|---|---|---|---|---|
| U-Net | 0.8693 [0.8458, 0.8924] | 0.5720 [0.5315, 0.6095] | 0.3898 [0.3718, 0.4075] | 171 |
| IABR-v2 dual | 0.8595 [0.8380, 0.8805] | 0.8595 [0.8380, 0.8805] | 0.3798 [0.3631, 0.3949] | 0 |
| IABR-Net heat | 0.5280 [0.4940, 0.5610] | 0.5280 [0.4940, 0.5610] | 0.4206 [0.3988, 0.4417] | 0 |
| network only | 0.4435 [0.4150, 0.4705] | 0.4435 [0.4150, 0.4705] | 0.5490 [0.5321, 0.5662] | 0 |
| NOMP (cached) | 0.8970 [0.8750, 0.9190] | 0.8970 [0.8750, 0.9190] | 0.3427 [0.3221, 0.3619] | 0 |
| NOMP (mine) | 0.8970 [0.8750, 0.9190] | 0.8970 [0.8750, 0.9190] | 0.3427 [0.3221, 0.3619] | 0 |
| NOMP+selfcal | 0.9430 [0.9265, 0.9590] | 0.9430 [0.9265, 0.9590] | 0.2798 [0.2609, 0.2982] | 0 |

| paired difference | d Pd paper-style [95% CI] | d Pd strict [95% CI] | d RMSE deg [95% CI] |
|---|---|---|---|
| IABR-Net heat - NOMP | -0.3690 [-0.4005, -0.3360] | -0.3690 [-0.4005, -0.3360] | +0.0780 [+0.0509, +0.1044] |
| IABR-v2 dual - NOMP | -0.0375 [-0.0555, -0.0195] | -0.0375 [-0.0555, -0.0195] | +0.0371 [+0.0155, +0.0586] |
| NOMP - U-Net | +0.0277 [-0.0031, +0.0577] | +0.3250 [+0.2905, +0.3610] | -0.0471 [-0.0708, -0.0245] |
| network only - NOMP | -0.4535 [-0.4835, -0.4240] | -0.4535 [-0.4835, -0.4240] | +0.2063 [+0.1806, +0.2316] |
| NOMP (mine) - NOMP (cached) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [-0.0000, +0.0000] |

### sep_2deg_snr15  (N=500, L=[2], SNR groups=[15])

| method | Pd paper-style [95% CI] | Pd strict [95% CI] | RMSE deg [95% CI] | dropped scenes |
|---|---|---|---|---|
| U-Net | 0.2917 [0.2095, 0.3690] | 0.0245 [0.0150, 0.0345] | 0.6284 [0.5551, 0.7027] | 458 |
| IABR-v2 dual | 0.4550 [0.4355, 0.4725] | 0.4550 [0.4355, 0.4725] | 0.5324 [0.5101, 0.5543] | 0 |
| IABR-Net heat (recomputed (fill_missing.py)) | 0.3415 [0.3220, 0.3620] | 0.3415 [0.3220, 0.3620] | 0.5632 [0.5395, 0.5886] | 0 |
| network only (recomputed (fill_missing.py)) | 0.2885 [0.2720, 0.3045] | 0.2885 [0.2720, 0.3045] | 0.5437 [0.5225, 0.5639] | 0 |
| NOMP (cached) | 0.5880 [0.5625, 0.6120] | 0.5880 [0.5625, 0.6120] | 0.5518 [0.5356, 0.5675] | 0 |
| NOMP (mine) | 0.5880 [0.5625, 0.6120] | 0.5880 [0.5625, 0.6120] | 0.5518 [0.5356, 0.5675] | 0 |
| NOMP+selfcal | 0.7010 [0.6750, 0.7270] | 0.7010 [0.6750, 0.7270] | 0.5077 [0.4918, 0.5227] | 0 |

| paired difference | d Pd paper-style [95% CI] | d Pd strict [95% CI] | d RMSE deg [95% CI] |
|---|---|---|---|
| IABR-Net heat - NOMP | -0.2465 [-0.2740, -0.2180] | -0.2465 [-0.2740, -0.2180] | +0.0114 [-0.0086, +0.0338] |
| IABR-v2 dual - NOMP | -0.1330 [-0.1530, -0.1130] | -0.1330 [-0.1530, -0.1130] | -0.0194 [-0.0391, -0.0017] |
| NOMP - U-Net | +0.2963 [+0.2202, +0.3749] | +0.5635 [+0.5370, +0.5890] | -0.0767 [-0.1531, -0.0032] |
| network only - NOMP | -0.2995 [-0.3270, -0.2730] | -0.2995 [-0.3270, -0.2730] | -0.0081 [-0.0358, +0.0190] |
| NOMP (mine) - NOMP (cached) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | -0.0000 [-0.0000, +0.0000] |

### phase_d0_snr0  (N=500, L=[3], SNR groups=[0])

| method | Pd paper-style [95% CI] | Pd strict [95% CI] | RMSE deg [95% CI] | dropped scenes |
|---|---|---|---|---|
| U-Net | 0.6784 [0.6576, 0.6982] | 0.5997 [0.5723, 0.6257] | 0.4466 [0.4337, 0.4590] | 58 |
| IABR-v2 dual | 0.6837 [0.6637, 0.7020] | 0.6837 [0.6637, 0.7020] | 0.4237 [0.4114, 0.4355] | 0 |
| IABR-Net heat (recomputed (fill_missing.py)) | 0.6803 [0.6603, 0.6987] | 0.6803 [0.6603, 0.6987] | 0.4239 [0.4116, 0.4358] | 0 |
| network only (recomputed (fill_missing.py)) | 0.5487 [0.5273, 0.5690] | 0.5487 [0.5273, 0.5690] | 0.5420 [0.5299, 0.5541] | 0 |
| NOMP (cached) | 0.6817 [0.6610, 0.6997] | 0.6817 [0.6610, 0.6997] | 0.4243 [0.4124, 0.4367] | 0 |
| NOMP (mine) | 0.6817 [0.6610, 0.6997] | 0.6817 [0.6610, 0.6997] | 0.4243 [0.4124, 0.4367] | 0 |
| NOMP+selfcal | 0.6813 [0.6610, 0.6993] | 0.6813 [0.6610, 0.6993] | 0.4243 [0.4125, 0.4367] | 0 |

| paired difference | d Pd paper-style [95% CI] | d Pd strict [95% CI] | d RMSE deg [95% CI] |
|---|---|---|---|
| IABR-Net heat - NOMP | -0.0013 [-0.0087, +0.0050] | -0.0013 [-0.0087, +0.0050] | -0.0004 [-0.0066, +0.0053] |
| IABR-v2 dual - NOMP | +0.0020 [-0.0030, +0.0067] | +0.0020 [-0.0030, +0.0067] | -0.0006 [-0.0065, +0.0047] |
| NOMP - U-Net | +0.0033 [-0.0135, +0.0201] | +0.0820 [+0.0623, +0.1023] | -0.0223 [-0.0361, -0.0090] |
| network only - NOMP | -0.1330 [-0.1507, -0.1147] | -0.1330 [-0.1507, -0.1147] | +0.1177 [+0.1010, +0.1340] |
| NOMP (mine) - NOMP (cached) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [-0.0000, +0.0000] |

### phase_d0_snr15  (N=500, L=[3], SNR groups=[15])

| method | Pd paper-style [95% CI] | Pd strict [95% CI] | RMSE deg [95% CI] | dropped scenes |
|---|---|---|---|---|
| U-Net | 0.9344 [0.9231, 0.9452] | 0.9063 [0.8877, 0.9227] | 0.2542 [0.2435, 0.2652] | 15 |
| IABR-v2 dual | 0.9413 [0.9300, 0.9520] | 0.9413 [0.9300, 0.9520] | 0.2216 [0.2104, 0.2327] | 0 |
| IABR-Net heat (recomputed (fill_missing.py)) | 0.9370 [0.9250, 0.9487] | 0.9370 [0.9250, 0.9487] | 0.2190 [0.2081, 0.2300] | 0 |
| network only (recomputed (fill_missing.py)) | 0.4370 [0.4200, 0.4543] | 0.4370 [0.4200, 0.4543] | 0.5777 [0.5640, 0.5903] | 0 |
| NOMP (cached) | 0.9497 [0.9390, 0.9593] | 0.9497 [0.9390, 0.9593] | 0.2126 [0.2024, 0.2227] | 0 |
| NOMP (mine) | 0.9497 [0.9390, 0.9593] | 0.9497 [0.9390, 0.9593] | 0.2126 [0.2024, 0.2227] | 0 |
| NOMP+selfcal | 0.9470 [0.9363, 0.9573] | 0.9470 [0.9363, 0.9573] | 0.2164 [0.2058, 0.2271] | 0 |

| paired difference | d Pd paper-style [95% CI] | d Pd strict [95% CI] | d RMSE deg [95% CI] |
|---|---|---|---|
| IABR-Net heat - NOMP | -0.0127 [-0.0183, -0.0077] | -0.0127 [-0.0183, -0.0077] | +0.0063 [-0.0014, +0.0148] |
| IABR-v2 dual - NOMP | -0.0083 [-0.0130, -0.0040] | -0.0083 [-0.0130, -0.0040] | +0.0089 [+0.0011, +0.0175] |
| NOMP - U-Net | +0.0153 [+0.0049, +0.0256] | +0.0433 [+0.0313, +0.0573] | -0.0416 [-0.0539, -0.0304] |
| network only - NOMP | -0.5127 [-0.5317, -0.4940] | -0.5127 [-0.5317, -0.4940] | +0.3651 [+0.3474, +0.3819] |
| NOMP (mine) - NOMP (cached) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | -0.0000 [-0.0000, +0.0000] |

## Official bank, per SNR (paper-style Pd; strict Pd differs only for U-Net)

| SNR dB | U-Net | IABR-v2 dual | IABR-Net heat | network only | NOMP (cached) | NOMP (mine) | NOMP+selfcal | oracle-seeded (3 rounds) | oracle-seeded (10 rounds) |
|---|---|---|---|---|---|---|---|---|---|
| -10 | 0.2225 | 0.2630 | 0.2602 | 0.2517 | 0.2618 | 0.2618 | 0.2617 | 0.5655 | 0.5647 |
| -5 | 0.4708 | 0.5067 | 0.5042 | 0.4548 | 0.5063 | 0.5063 | 0.5063 | 0.6363 | 0.6333 |
| 0 | 0.6788 | 0.6915 | 0.6855 | 0.5537 | 0.6907 | 0.6907 | 0.6910 | 0.7400 | 0.7375 |
| 5 | 0.8129 | 0.8187 | 0.8142 | 0.5407 | 0.8198 | 0.8198 | 0.8207 | 0.8452 | 0.8418 |
| 10 | 0.8885 | 0.8955 | 0.8878 | 0.5027 | 0.8993 | 0.8993 | 0.9002 | 0.9143 | 0.9125 |
| 15 | 0.9244 | 0.9390 | 0.9355 | 0.4508 | 0.9448 | 0.9448 | 0.9425 | 0.9490 | 0.9487 |
| 20 | 0.9440 | 0.9582 | 0.9512 | 0.4278 | 0.9662 | 0.9662 | 0.9628 | 0.9743 | 0.9732 |
| 25 | 0.9508 | 0.9687 | 0.9640 | 0.4235 | 0.9758 | 0.9758 | 0.9758 | 0.9843 | 0.9842 |
| mean | 0.7366 | 0.7551 | 0.7503 | 0.4507 | 0.7581 | 0.7581 | 0.7576 | 0.8261 | 0.8245 |

U-Net strict Pd per SNR: -10: 0.2027, -5: 0.4143, 0: 0.5967, 5: 0.7478, 10: 0.8325, 15: 0.9032, 20: 0.9280, 25: 0.9270; mean 0.6940

RMSE (deg, detected components) per SNR:

| SNR dB | U-Net | IABR-v2 dual | IABR-Net heat | network only | NOMP (cached) | NOMP (mine) | NOMP+selfcal | oracle-seeded (3 rounds) | oracle-seeded (10 rounds) |
|---|---|---|---|---|---|---|---|---|---|
| -10 | 0.5497 | 0.5355 | 0.5360 | 0.5511 | 0.5350 | 0.5350 | 0.5346 | 0.4293 | 0.4285 |
| -5 | 0.5072 | 0.4953 | 0.4935 | 0.5272 | 0.4966 | 0.4966 | 0.4970 | 0.4692 | 0.4686 |
| 0 | 0.4548 | 0.4328 | 0.4312 | 0.5433 | 0.4331 | 0.4331 | 0.4328 | 0.4271 | 0.4277 |
| 5 | 0.3783 | 0.3514 | 0.3510 | 0.5636 | 0.3481 | 0.3481 | 0.3492 | 0.3483 | 0.3488 |
| 10 | 0.3049 | 0.2881 | 0.2864 | 0.5808 | 0.2803 | 0.2803 | 0.2844 | 0.2811 | 0.2816 |
| 15 | 0.2556 | 0.2269 | 0.2266 | 0.5872 | 0.2193 | 0.2193 | 0.2220 | 0.2177 | 0.2186 |
| 20 | 0.2300 | 0.1839 | 0.1833 | 0.5858 | 0.1759 | 0.1759 | 0.1757 | 0.1737 | 0.1745 |
| 25 | 0.2081 | 0.1557 | 0.1556 | 0.5891 | 0.1329 | 0.1329 | 0.1350 | 0.1263 | 0.1277 |
| mean | 0.3611 | 0.3337 | 0.3329 | 0.5660 | 0.3277 | 0.3277 | 0.3288 | 0.3091 | 0.3095 |

## Headroom: oracle-seeded decoder vs NOMP (official bank, paper-style Pd; 95% CI = paired bootstrap 2000 resamples within the SNR group, seed 0)

| SNR dB | NOMP | oracle-seeded (3 rounds) | gap (oracle - NOMP) [95% CI] | oracle-seeded (10 rounds) | min-residual(NOMP, oracle) | gap (min-res - NOMP) [95% CI] | scenes where oracle residual < NOMP residual | scenes NOMP misses >=1 comp. | of those, oracle all-correct | of those, oracle residual < NOMP residual |
|---|---|---|---|---|---|---|---|---|---|---|
| -10 | 0.2618 | 0.5655 | +0.3037 [+0.2882, +0.3182] | 0.5647 | 0.2630 | +0.0012 [+0.0002, +0.0023] | 7 | 998 | 55 | 0 |
| -5 | 0.5063 | 0.6363 | +0.1300 [+0.1185, +0.1422] | 0.6333 | 0.5083 | +0.0020 [+0.0003, +0.0038] | 52 | 945 | 58 | 0 |
| 0 | 0.6907 | 0.7400 | +0.0493 [+0.0418, +0.0573] | 0.7375 | 0.6942 | +0.0035 [+0.0010, +0.0062] | 83 | 840 | 31 | 1 |
| 5 | 0.8198 | 0.8452 | +0.0253 [+0.0192, +0.0317] | 0.8418 | 0.8265 | +0.0067 [+0.0033, +0.0103] | 121 | 636 | 28 | 6 |
| 10 | 0.8993 | 0.9143 | +0.0150 [+0.0102, +0.0200] | 0.9125 | 0.9083 | +0.0090 [+0.0050, +0.0133] | 211 | 417 | 25 | 15 |
| 15 | 0.9448 | 0.9490 | +0.0042 [+0.0018, +0.0067] | 0.9487 | 0.9478 | +0.0030 [+0.0010, +0.0053] | 263 | 265 | 7 | 4 |
| 20 | 0.9662 | 0.9743 | +0.0082 [+0.0047, +0.0122] | 0.9732 | 0.9737 | +0.0075 [+0.0042, +0.0115] | 309 | 154 | 14 | 12 |
| 25 | 0.9758 | 0.9843 | +0.0085 [+0.0050, +0.0122] | 0.9842 | 0.9840 | +0.0082 [+0.0047, +0.0118] | 437 | 111 | 20 | 20 |
| mean over SNR | 0.7581 | 0.8261 | +0.0680 [+0.0650, +0.0708] | 0.8245 | 0.7632 | +0.0051 [+0.0041, +0.0061] | 1483 | 4366 | 238 | 58 |

(residual = ||Ht - A x_LS||^2 at the decoded (u,v). "min-residual" keeps, per scene, whichever of the NOMP and oracle-seeded (3 rounds) solutions has the lower residual -- the oracle start is used only where the decoder objective itself prefers it, so its gap over NOMP is the search-failure headroom of this objective; the rest of the oracle-seeded gap comes from scenes where the oracle start sits in a local optimum near the truth that fits the data worse than the NOMP solution, i.e. the objective (noise/impairments) rather than the search is the limit.)


## Latency (CPU, single thread)

| what | batch | median ms / call | p10-p90 ms | median ms / scene |
|---|---|---|---|---|
| (a) NOMP greedy, mine | 1 | 8.53 | 5.01-12.68 | 8.530 |
| (a) NOMP greedy, mine | 256 | 379.33 | 308.73-457.23 | 1.482 |
| (a) NOMP greedy, notebook nomp_greedy | 1 | 11.69 | 7.08-19.57 | 11.694 |
| (a) NOMP greedy, notebook nomp_greedy | 256 | 814.92 | 671.34-1093.70 | 3.183 |
| (c) CNN forward only | 1 | 17.63 | 15.41-23.25 | 17.631 |
| (c) CNN forward only | 256 | 1111.35 | 996.29-1304.87 | 4.341 |
| (b) CNN forward + heat-seeded decode | 1 | 24.71 | 17.93-33.02 | 24.707 |
| (b) CNN forward + heat-seeded decode | 256 | 2334.13 | 1757.48-2803.03 | 9.118 |

## Commands (run from this folder, in this order)

```
python check_metric.py > check_metric.log
python check_ht.py    > check_ht.log
python run_nomp.py    > run_nomp.log
python fill_missing.py > fill_missing.log
python tables.py      > tables.log
python latency.py     > latency.log
python make_report.py
```
Seeds: bootstrap `np.random.default_rng(0)`; latency scene draw `np.random.default_rng(0)`; decoders are deterministic.
