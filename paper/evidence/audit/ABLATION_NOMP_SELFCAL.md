# Missing ablation: network-free NOMP + self-calibration (paper-style Pd, paper's own evaluator)
Script: ablate_nomp_selfcal.py ; raw numbers: ablate_nomp_selfcal_results.json ; log: ablate_nomp_selfcal.log
Sanity check: greedy-only recompute is bit-identical to the cached NOMP_no_network predictions on all 8 banks.
Self-calibration (lam, tau) = (1.0, 5.0), tuned for the hybrid on separate tuning banks, NOT re-tuned for the network-free decoder. Input = raw antenna-domain channel Ht (no IABC).
Paired bootstrap (2000 resamples over samples of one bank, one seed of data): difference = (NOMP+selfcal) - (IABR-v2).

| bank | n | L | IABR-v2 | NOMP | NOMP+selfcal | diff vs IABR-v2 [95% CI] |
|---|---|---|---|---|---|---|
| official | 8000 | 3 | 0.7551 | 0.7581 | 0.7576 | +0.0025 [+0.0014, +0.0037] |
| T2_d5 | 8000 | 3 | 0.7452 | 0.7468 | 0.7476 | +0.0025 [+0.0014, +0.0036] |
| gain 4 dB, SNR 15 | 500 | 3 | 0.9123 | 0.8697 | 0.8950 | -0.0173 [-0.0257, -0.0100] |
| gain 2 dB, SNR 0 | 500 | 3 | 0.7147 | 0.7143 | 0.7160 | +0.0013 [-0.0043, +0.0073] |
| phase 15 deg, SNR 15 | 500 | 3 | 0.8693 | 0.8633 | 0.8730 | +0.0037 [+0.0003, +0.0073] |
| L=8, SNR 0 | 500 | 8 | 0.5028 | 0.5019 | 0.5016 | -0.0011 [-0.0045, +0.0021] |
| L=8, SNR 15 | 500 | 8 | 0.8628 | 0.8651 | 0.8648 | +0.0020 [-0.0018, +0.0060] |
| two paths 5 deg apart, SNR 15 | 500 | 2 | 0.8595 | 0.8970 | 0.9430 | +0.0835 [+0.0695, +0.0985] |

Reading: the network-free decoder with self-calibration equals or exceeds IABR-v2 on 7 of 8 banks (differences inside the CI on 3 of them). IABR-v2 leads only at gain error 4 dB / SNR 15
(0.912 vs 0.895; self-calibration closes about 60% of the 0.870 -> 0.912 gap, the rest may come from IABC and the network, not separable without a retrained no-IABC network).
Limits: 500-sample banks, single data draw, lam/tau not re-tuned for the network-free path, L=3 for the main banks.
