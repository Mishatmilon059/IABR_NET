# Fine-tuning IABR-Net for severe array imperfections: results

All numbers come from logs in this folder: `m4_physics.log`, `eval_main.log`, `eval_fresh.log`, `final_tables.log`, `make_report.log`, `train*.log`, `unit_gen.log`.
Pd is the paper Pd (|err| <= 1 deg), averaged over the SNR groups. Every method is given the true L = 3.
Differences are paired bootstrap 95% CIs (2000 resamples within SNR groups, seed 0). `*` means the CI excludes 0.
"Best classical" is the best of 9 classical methods, picked on each test bank itself, which favours the classical side. The 9 are:
- NOMP
- NOMP+SC default
- NOMP+SC tuned
- classical dual, with and without SC
- NOMP os8 rc6
- FFT-parabolic os4 / os16
- EB-ML-g5 (the strongest classical self-calibration, from the IABC-v2 adversarial stage)

## Status / provenance
- **Training:** `train_imp.py`, 3300 steps, batch 128, 61.7 min.
  - Run detached. After a crash it was resumed from the step-3000 checkpoint; the batch seeds are per step, so the resumed steps are deterministic.
  - Output: `cache/imp_ft.pt`. `final.pt` and `lurd.pt` were not touched.
- **Main evaluation:** `evaluate.py stress clean official` -> `eval_main.log`.
- **Fresh-seed replicates:** `evaluate.py fresh:M2 fresh:M6 fresh:M3 fresh:clean fresh:M8` -> `eval_fresh.log`.
  - There is no fresh:M1 run because M1 had no candidate win on the stress seeds.
- **Physics control:** `m4_physics.py` -> `m4_physics.log` / `.json`, full run on the stress and fresh banks.
- **Tables and figures:**
  - `final_tables.py` -> `final_tables.log`.
  - `make_report.py` -> `make_report.log`, `results.json`, `fig_M1/M2/M3/M4/M6/M8/train.png`. Re-run at 20:12 to include the fresh banks; the earlier log is kept as `make_report_attempt2.log`.
- **Checks:**
  - Y is identical to the stress run: NOMP estimates equal `stress/cache/preds__*` with max |diff| 0 in all 40 stress banks and all 8 M4 banks.
  - `unit_gen.py`: the vectorised training generator equals `stress_sim.generate` (max diff 0).
  - Smoke test: the reused decoder pieces equal the notebook `decode()` (max diff 0).
- **Seeds** (all used nowhere else, checked with grep):
  - training: `default_rng([71000000, step])`
  - monitoring batch: `[71000001, 0]`
  - stress: 5000 / 5015 (0 / 15 dB, 500 scenes)
  - fresh: 8400 / 8415 (1000 scenes per cell)
  - official: the frozen bank (8000)
  - bootstrap: seed 0
- **Large caches:** per-scene stats are at `C:/Users/Milon/AppData/Local/Temp/claude/D--thesis/a5db9a29-d75e-4561-834b-cd0638a16d91/scratchpad/dr_cache/finetune_imp/` (17 MB, temporary, can be regenerated from the seeds).
- **Corrector used:** the IABC-v2 stage decided REPLACE and its adversarial check confirmed it, with the rewording "fast learned alternative to classical ML self-cal". So the fine-tuned trunk was evaluated with three front ends:
  - its own fine-tuned IABC (`FT ...`)
  - IABC-v2b (`FT v2b ...`, v2b read-only from `iabc_v2/cache`)
  - no correction (`FT nocorr ...`)

## Step 0: M4 physics control (known quantised codebooks, no learning)
The receiver knows its quantised codebooks Wq, Fq. Two decoders use them:
- **LS-NOMP:** H = pinv(Wq^H) Y pinv(Fq), then NOMP.
- **MD-NOMP:** NOMP in the measurement domain, with the exact dictionary.

Conditioning:
- b = 4/3/2: Wq is unitary (cond 1).
- b = 1: rank 12 of 16, cond = inf.

Pd, stress seeds (fresh seeds in brackets):

| bits / SNR | NOMP | NOMP+SC tuned | LS-NOMP | MD-NOMP | clean NOMP | gap closed by MD |
|---|---|---|---|---|---|---|
| b3 / 15 dB | 0.882 (0.888) | 0.884 | 0.928 (0.933) | 0.927 (0.933) | 0.929 (0.933) | ~96% (~100%) |
| b2 / 15 dB | 0.755 (0.747) | 0.745 | 0.925 (0.936) | 0.924 (0.936) | 0.929 (0.933) | ~97% (~100%) |
| b1 / 15 dB | 0.295 (0.282) | 0.296 | 0.021 (0.019) | 0.902 (0.909) | 0.929 (0.933) | ~96% (~96%) |
| b3 / 0 dB | 0.675 (0.688) | 0.671 | 0.699 (0.706) | 0.700 (0.706) | 0.690 (0.704) | >100% |
| b2 / 0 dB | 0.574 (0.587) | 0.575 | 0.693 (0.699) | 0.693 (0.698) | 0.690 (0.704) | >100% (95%) |
| b1 / 0 dB | 0.191 (0.206) | 0.191 | 0.018 (0.018) | 0.657 (0.669) | 0.690 (0.704) | 93% (93%) |

- MD-NOMP minus NOMP is `*` in every cell with b <= 3, on both seed sets.
- For b <= 2 the known-codebook decoder beats NOMP by +0.11 to +0.63.
- At b = 1, LS fails (pinv of a rank-12 matrix) but MD works. Clean minus MD is +0.024 to +0.035*.
- **Verdict M4: the gap is a model-mismatch artefact. No network is needed.** A receiver that uses its own quantised codebook closes 93-100% of it.

## Step 1: Fine-tuning
- **Data mix:** 40% original distribution and 60% extended impairments, one family per sample:
  - M3: phase U(0, 45 deg) and gain U(0, 6 dB)
  - M2: position sigma U(0, 0.2)
  - M6: phase noise U(0, 30 deg)
  - M1: coupling rho U(0, 0.3)
  - L 1..9, SNR -15..24 dB
- **Loss:** the original loss, focal heatmap + 50 x IABC MSE.
  - The IABC term is masked for M1, M2 and M6. There, no per-antenna diagonal truth exists (coupling mixes antennas, position error is path-dependent, phase noise is per measurement), and a zero target would teach the IABC to switch off.
- **Optimiser:** Adam, warm-up 200 steps, cosine 3e-4 -> 1e-5, clip 5.
- **Heat loss on the fixed monitoring batch, first -> last step:**

  | slice | heat loss | IABC MSE |
  |---|---|---|
  | orig | 0.460 -> 0.468 | |
  | M1 | 0.535 -> 0.539 | |
  | M2 | 0.589 -> 0.486 | |
  | M3 | 0.654 -> 0.544 | 0.048 -> 0.039 |
  | M6 | 0.637 -> 0.610 | |

  M1 did not learn. The original distribution got slightly worse.

## Step 2: Main results
**Regression on the official bank (8000 scenes)** vs NOMP 0.7581:

| method | Pd | diff vs NOMP |
|---|---|---|
| FT dual+SC | 0.7501 | -0.008 [-0.010, -0.006]* (orig dual+SC 0.7551, so FT is -0.005* vs orig) |
| FT heat+SC | 0.7458 | -0.012* |
| FT nocorr dual+SC | 0.7580 | -0.000 |
| FT v2b dual+SC | 0.7571 | -0.001 [-0.002, +0.000] |
| FT nocorr heat+SC (= FT seeds on Ht+SC) | 0.7520 | -0.006* |
| FT network only | 0.4716 | orig 0.4506 |
| best classical (NOMP os8 rc6) | 0.7598 | |

- **Clean banks:** FT dual+SC costs -0.012* / -0.016* at 15 dB (stress / fresh). The nocorr and v2b front ends are neutral there.
- **The fine-tuned IABC itself is harmful on ideal hardware**, worse than the original.

**Candidate wins on the stress seeds and their fresh-seed replication.** Wins are against the best classical method, with the same variant on the fresh seeds; the protocol is fixed in `final_tables.py`.

| cell | variant | stress diff vs best | fresh diff vs best | fresh diff vs v2b+NOMP+SC tuned |
|---|---|---|---|---|
| M2 0.2 / 15 dB | FT heat+SC | +0.033* | +0.036 [+0.028, +0.044]* | +0.038* |
| M2 0.2 / 15 dB | FT seeds on raw Ht+SC | +0.032* | +0.035 [+0.028, +0.043]* | +0.038* |
| M6 30 / 15 dB | FT seeds on raw Ht+SC | +0.039* | +0.046 [+0.035, +0.057]* | +0.052* |
| M6 30 / 15 dB | FT heat+SC | +0.033* | +0.030* | +0.037* |
| M6 20 / 15 dB | FT seeds on raw Ht+SC | +0.025* | +0.016 [+0.009, +0.025]* | +0.021* |
| M6 20 / 15 dB | FT heat+SC (own IABC) | +0.016* | +0.006 [-0.002, +0.015] (not replicated) | +0.011* |
| M6 30 / 0 dB | FT seeds on raw Ht+SC | +0.010* | +0.016 [+0.009, +0.022]* | +0.021* |
| M8 0.5 / 15 dB (OOD) | FT seeds on raw Ht+SC | +0.013* | +0.008 [+0.002, +0.013]* | +0.008* |
| M3 45 deg / 0 dB | FT v2b heat+SC | +0.063* | +0.066 [+0.054, +0.077]* | +0.002 [-0.001, +0.005] |
| M3 30 deg / 0 dB | FT v2b dual+SC | +0.018* | +0.025* | +0.001 [-0.001, +0.003] |
| M3 45 deg / 0 dB | FT dual+SC | +0.006* | +0.012* | -0.052* |
| M3 6 dB / 15 dB | FT v2b dual+SC | +0.009* | +0.001 (not replicated) | +0.000 |
| M3 30 deg / 4 dB / 0 dB | FT dual+SC | +0.012* | +0.006 (not replicated) | -0.007 |

**Absolute Pd in the replicated win cells (fresh seeds):**

| cell | NOMP | FT seeds on raw Ht | oracle-seeded |
|---|---|---|---|
| M2 0.2 / 15 dB | 0.627 | 0.662 | 0.690 |
| M6 30 / 15 dB | 0.761 | 0.807 | 0.839 |
| M6 20 / 15 dB | 0.861 | 0.877 | 0.902 |

**Where the fine-tuned net does not help:**
- **M1:** no win. The FT pipelines are -0.005 to -0.020* vs the best classical method at 15 dB.
- **Mild M2 (0.02-0.05) and M6 (5-10 deg):** losses of -0.01 to -0.02*.
- **M3 at 15 dB:** EB-ML-g5 is best (e.g. 45 deg: 0.696). FT dual+SC is -0.024 to -0.084* vs it; FT v2b heat+SC 0.704 ties it.

**The dual decoder throws the M2/M6 gain away.**
- With learned seeds on raw Ht, the dual pick returns to NOMP: FT nocorr dual+SC is 0.626 at M2 0.2 and 0.757 at M6 30 (fresh).
- The lower-residual rule prefers NOMP's solution under the mismatched model, as in the ADC stage.
- So the win needs the heat-only path, which costs -0.006* on the official bank and -0.012* on clean 15 dB (fresh).

## Step 3: Ablation (heat seeds + SC)
The three variants:
- (a) FT seeds on raw Ht: learned seeds only
- (b) greedy NOMP on the FT-corrected Hc: learned correction only
- (c) FT seeds on FT Hc: both

| bank | (a) - NOMP+SC | (b) - NOMP+SC | (c) - (a): value of the correction | (c) - (b): value of the seeds |
|---|---|---|---|---|
| fresh M2 0.2 / 15 dB | +0.037* | +0.004 | +0.001 | +0.033* |
| fresh M6 30 / 15 dB | +0.050* | +0.006 | -0.016* | +0.028* |
| fresh M6 20 / 15 dB | +0.022* | +0.001 | -0.010* | +0.011* |
| fresh M6 30 / 0 dB | +0.017* | -0.005 | -0.009* | +0.013* |
| fresh M3 45 deg / 15 dB | +0.037* | +0.011* | +0.008* | +0.033* |
| fresh M3 30 deg / 4 dB / 15 dB | +0.035* | +0.063* | +0.038* | +0.011* |
| official | -0.006* | -0.008* | -0.005* | -0.004* |

- **M2 and M6:** the whole gain comes from the learned seeds. The learned correction adds nothing in M2 and is harmful in M6. This matches the physics: no per-antenna correction exists for those families.
- **M3:** both parts help. The fine-tuned correction is still far below IABC-v2b / EB-ML: at 45 deg / 15 dB, FT NOMP on Hc+SC scores 0.612 vs v2b+NOMP 0.700 and EB-ML-g5 0.696.
- **Where the M3 wins come from:** every replicated M3 win comes from the v2b corrector, not from fine-tuning. FT v2b minus v2b+NOMP+SC tuned is +0.000 to +0.002, n.s.
- **Fine-tuning vs the original CNN:**
  - The fine-tuning is what makes the seeds better: FT nocorr heat+SC minus orig heat+SC is +0.033 to +0.059* on the fresh M2/M6 win cells.
  - The original CNN's heat seeds on raw Ht were not run as a separate arm.
  - FT network only beats orig network only in 67 of 72 cells but stays at 0.45-0.61 Pd, far below every decoder.

## Step 4: Verdict per family
- **M4 (phase-shifter bits): NETWORK NOT NEEDED.** Decoding with the known quantised codebook (MD-NOMP) closes 93-100% of the gap at b = 3/2/1, at 0 and 15 dB, replicated. The M4 "NN opportunity" was a model-mismatch artefact.
- **M1 (coupling): NO GAIN.** Fine-tuning did not learn M1 (monitoring loss flat). FT pipelines lose to classical by up to 0.02*.
- **M2 (position errors): SMALL REPLICATED WIN, sigma = 0.2 at 15 dB only.**
  - Learned heat seeds on raw Ht: +0.035* vs the best classical method (NOMP), closing 56% of the oracle gap. The gain is from the learned seeds, not the correction.
  - There is no win at sigma <= 0.1 or at 0 dB, and mild cells lose about 0.01-0.02*.
- **M6 (phase noise): REPLICATED WIN at 20-30 deg.**
  - Learned seeds on raw Ht:

    | cell | diff vs NOMP (= best classical) |
    |---|---|
    | 30 deg / 15 dB | +0.046* |
    | 20 deg / 15 dB | +0.016* |
    | 30 deg / 0 dB | +0.016* |

  - These close 59% / 39% / 18% of the oracle gap. The gain is from the seeds; the learned correction hurts here.
  - At 5-10 deg the learned seeds lose about 0.01*.
- **M3 (large phase/gain): NO WIN FROM FINE-TUNING.**
  - The replicated wins (+0.025* at 30 deg / 0 dB, +0.066* at 45 deg / 0 dB vs EB-ML-g5, which gates itself off at 0 dB) all come from the IABC-v2b corrector. They equal v2b+NOMP+SC tuned (n.s.).
  - At 15 dB, EB-ML-g5 / v2b are best and the fine-tuned IABC is far worse.
- **M8 (near field, out-of-distribution control):** one small replicated win, +0.008* at 0.5 m / 15 dB (learned seeds on raw Ht). Losses elsewhere at 15 dB.
- **Regression:**
  - FT dual+SC with its own IABC fails the no-regression check on the official bank (-0.008*) and on clean (-0.012 to -0.016*).
  - With the v2b or no-correction front end, FT dual+SC is neutral (-0.001 / -0.000) but then has no wins, because the dual pick reverts to NOMP.
  - The winning heat-only path costs -0.006* on the official bank.
- **Bottom line:**
  - Fine-tuning buys real but small, replicated gains (+0.016 to +0.046 Pd) only under severe position error (sigma 0.2) and phase noise (20-30 deg). Even there, 41-82% of the oracle gap remains.
  - It costs about 0.006-0.012 Pd on ideal hardware, and there is no impairment-aware switch to avoid that cost.
  - It does not support "the network is needed" for M1, M3 or M4.
  - For M3 use IABC-v2b / EB-ML. For M4 use the known-codebook decoder.

## Caveats
- Every method is given the true L = 3.
- The best classical method is picked on the test bank.
- Candidate wins were found post hoc among 11 learned variants. Only fresh-seed replications are claimed.
- No classical decoder designed for M2/M6 was implemented:
  - for M2, joint position-error calibration;
  - for M6, a phase-noise-robust estimator such as per-measurement phase marginalisation.
  Such a decoder is the strongest baseline still missing.
- Training ran 3300 steps (61.7 min), not the planned ~75 min.
- Latency was not re-measured: same architecture, 191,702 params.
