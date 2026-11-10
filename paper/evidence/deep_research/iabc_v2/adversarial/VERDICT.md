# Adversarial verdict: IABC-v2 claims and the keep / replace / drop decision

Folder: `D:/ai_ml_project/paper/evidence/deep_research/iabc_v2/adversarial/`. Every number below comes from a script in this folder, with its stdout in the `.log` next to it.

## Seeds

All seeds are fresh and disjoint from every earlier training, validation, tuning and test seed:

| Seeds | Script | Use |
|---|---|---|
| 9300-9307 (+[s,1], [s,3]) | `a3_test.py` | test cells |
| 9330-9332 | `a1_crb_check.py` | CRB check |
| 9340-9347 | `a0_proto.py` | prototype; timing and sanity only |
| 9350-9352 | `a2_unit.py` | unit checks |
| 9360-9370 | `a6_d1_check.py` | old-IABC diagnosis check |

The official bank is the fixed 8000-scene bank (`net__official.npz` Ht, true L). The plan file `a3_plan.json` was written by `a3_test.py` at 10:25:13, before its first test computation. It first landed in `../../iabc/` because `icommon.HERE` overwrote the script's own folder path; I moved it here and fixed the path in the script.

Large outputs are in `C:/Users/Milon/AppData/Local/Temp/claude/D--thesis/a5db9a29-d75e-4561-834b-cd0638a16d91/scratchpad/dr_cache/iabc_v2_adv/` (cell__*.npz, bank__official.npz, bank chunk .pkl files, 10 MB). They are temporary and can be regenerated with `a3_test.py`, which takes about 25 min.

Statistics: paired bootstrap, 2000 resamples within SNR groups, seed 0. `*` means the 95% CI excludes 0.

## 1. Code audit

**Seed disjointness: CONFIRMED** (`a2_unit.log`).
- Training and validation streams: v2a `[62000000,k]`, val `62999999`; v2b data `[63000000,k]`, val `[63999999,k]`; shuffling `[64000000,step]`. Test streams: `8900+k`, `[8900+k,1]`, `[8900+k,3]`.
- numpy aliases `default_rng(s)` with `default_rng([s,0])`. I verified this is true. After normalising the aliasing, the training/validation and test seed keys still have no element in common.
- No identical first-draw streams were found.
- Model selection used only the validation seeds: `train_v2a.py` selects on 62999999 and `train_v2b.py` on the 63999999 chunks.
- Timeline: v2b training ended 08:59, `e1_eval` started 09:01, `criteria.json` was written 06:30.
- One remark: `criteria.json`'s approximate bounds were computed on the test-seed angles (`d2_crb.py`). That is not model selection.

**Identifiable projection is the same for target and estimate: CONFIRMED.** `to_ident` (torch) and `ident_vec` / `S.project_impairment` differ by at most 2.6e-7 (float32), and the projection is idempotent.

**Correction sign and conjugation: CONFIRMED.**
- Noiseless 30 deg / 6 dB: apply_corr(true target) leaves 1.7e-6 deg of non-identifiable-free phase and 2.8e-7 dB of gain.
- Flipping the Tx or Rx phase sign instead leaves 77 / 84 deg.
- Trained models (20 deg / 4 dB, 25 dB): v2b leaves 0.99 deg as used, against 15.1 deg with either sign flipped.

**CRB derivation: CONFIRMED** (`a1_crb_check.log`).
- An independent model (angle parametrisation, simulator steering vectors, finite-difference Jacobian) gives CRB / `d2_crb` CRB ratios of 1.000000 (min and max) for phase and gain, with known and unknown angles. The impairment-block Jacobian differs by 6e-9.
- Monte-Carlo efficient-estimator check (300 scenes, 30 deg / 6 dB, Levenberg-Marquardt ML) gives MSE/CRB of:

  | SNR (dB) | Phase MSE / CRB | Gain MSE / CRB |
  |---|---|---|
  | 15 | 1.017 | 0.982 |
  | 25 | 0.992 | 0.991 |
  | 35 | 1.020 | 0.985 |

  All CIs contain 1, so the bound is attained.
- Noise variance of to_ant(noise) is 0.19956 against 10^-0.7 = 0.19953.
- "Knowing the angles barely changes it": at 0 dB the bound is 11.96 deg with unknown angles and 11.57 deg with known angles.

**Fair classical comparators: REFUTED (the comparison set was too weak).**
- The "tuned self-cal" comparator (lam 1, tau 2) gates itself off at large impairments, because model mismatch lowers its own SNR estimate. Its removal at SNR >= 5 dB is only 22% at 30/6 and 42% at 45 deg (my run), while the same ALS with the gate open removes 88% and 96%.
- The Toeplitz estimator is a strawman when there are several paths.
- The column labelled "Oracle (true angles)" is a 4-iteration, unshrunk ALS. Against the known-angle CRB its MSE is 2.2x at 25 dB and 13x at 35 dB (`a1`), so it is not an oracle bound.
- The strong classical estimator that was missing is joint ML plus shrinkage. It is added in section 3.

## 2. Re-measured headline numbers (fresh seeds 9300-9307, 200 scenes x 8 SNR, L=3)

**% identifiable MSE removed at SNR >= 5 dB** (`a5_summary.log` section 1; RMS % in parentheses):

| Cell | Claimed v2b | Replicated v2b | Tuned SC | Gate-open SC | EB-ML (classical) |
|---|---|---|---|---|---|
| 8 deg phase | 68 | 67.4 [66.3,68.5] (43) | 47.5 | 18.1 | 59.4 [49.5,65.9] |
| 3 dB gain | 90 | 90.3 [89.7,91.0] (69) | 85.0 | 86.4 | 88.8 [87.3,90.0] |
| 20 deg phase | 91 | 90.6 (69) | 84.0 | 86.6 | 89.2 |
| 6 dB gain | 96.5 | 96.5 (81) | 78.9 | 95.0 | 96.0 |
| 30 deg / 6 dB | 93 / 96 | 92.9 / 96.2 (73 / 81) | 21.9 / 22.2 | 87.7 / 93.8 | 92.5 / 95.6 |
| 45 deg phase | 97 | 97.2 (83) | 42.5 | 95.6 | 95.4 |
| 8 deg / 3 dB | - | 68.2 / 90.2 | 47.6 / 84.6 | - | 58.8 / 88.9 |

**Residual RMS per SNR** (`a5_summary.log` section 2) matches the claims:
- 8 deg at -5 dB: v2b 4.93 deg against 4.28 deg uncorrected (claim: 5.1 vs 4.3).
- 30/6 at 25 dB: v2b 1.60 deg / 0.24 dB.
- v2b on ideal hardware, official bank: 1.54 deg / 0.21 dB added (claim: 1.6 / 0.21).

**Downstream paper Pd, v2b + NOMP + tuned SC minus NOMP + tuned SC** (claimed value in parentheses):

| Cell | Difference |
|---|---|
| clean | -0.000 [-0.002, +0.002] (claimed -0.002) |
| 8 deg | +0.001 (ns) |
| 3 dB | +0.002 (ns) |
| 8 / 3 | +0.003 [-0.002, +0.007] (claimed +0.004*, **now ns**) |
| 20 deg | +0.006* (+0.004*) |
| 6 dB | +0.008* (+0.010*) |
| 30 / 6 | +0.049 [0.041, 0.058]* (+0.049*) |
| 45 deg | +0.057 [0.049, 0.064]* (+0.057*) |

v2b + NOMP minus NOMP: 45 deg +0.067*, 30/6 +0.057*, 6 dB +0.031*, 20 deg +0.014*, 3 dB +0.012*, 8 deg +0.003*. Strict Pd gives the same differences.

**Official bank (deterministic; reproduced exactly):**

| Pipeline | Pd | Difference vs NOMP |
|---|---|---|
| NOMP | 0.7581 | - |
| v2b + NOMP | 0.7576 | -0.001 [-0.002, +0.001] |
| v2b + NOMP + SC tuned | 0.7559 | -0.002 [-0.004, -0.001]* |
| NOMP + SC tuned | 0.7565 | -0.002 [-0.003, -0.000]* |

v2b + NOMP + SC tuned minus NOMP + SC tuned is -0.001 [-0.002, +0.000].

## 3. Does a simple classical change close the gap? (constants fixed before the test run)

**Comparators:**
- **More ALS iterations** (tuned SC, 10 iterations): no effect anywhere (|diff| <= 0.001). It does not close the gap.
- **Re-detection** (2 rounds of NOMP on the corrected channel, then tuned SC): +0.004 to +0.017* over tuned SC. It closes only a small part.
- **EB-ML** (`a_classical.py`), fully classical with no learning:
  1. NOMP, then gate-open ALS.
  2. Joint Levenberg-Marquardt ML over all parameters.
  3. Positive-part James-Stein shrinkage per block, using the estimate's own CRB.
  4. Re-detect; 2 rounds.
- **EB-ML-g5 / g10:** EB-ML applied only where its own blind SNR estimate is >= 5 / 10 dB.
  - The blind estimate tracks the nominal SNR: median 4.2 dB at 5 dB nominal, 24.5 dB at 25 dB nominal.
  - The prototype on seeds 9340-9347 showed ungated EB-ML is harmful on clean scenes at <= 0 dB. The 5 dB gate was fixed from that, before testing; 10 dB is the earlier snrgate threshold.

**Paper Pd differences (`a5_summary.log` section 4):**

| Cell | EB-ML-g5+NOMP − NOMP+SC tuned | v2b+NOMP+SC tuned − EB-ML-g5+NOMP+SC tuned | v2b+NOMP − EB-ML-g5+NOMP |
|---|---|---|---|
| clean | +0.001 | +0.000 | +0.000 |
| 8 deg | +0.003* | +0.000 | +0.000 |
| 3 dB | +0.009* | **-0.005*** | **-0.006*** |
| 8 / 3 | +0.009* | **-0.006*** | **-0.007*** |
| 20 deg | +0.003* | +0.001 | +0.002 |
| 6 dB | +0.024* | **-0.014*** | **-0.016*** |
| 30 / 6 | +0.059* | **-0.009*** | **-0.009*** |
| 45 deg | +0.036* | **+0.021*** | **+0.021*** |

- Official bank: EB-ML-g5 + NOMP minus NOMP is -0.000 [-0.001, +0.000]; EB-ML-g10 + NOMP is -0.000. **No loss on ideal hardware.**
- Ungated EB-ML loses: -0.015* on the official bank, -0.008* on the clean cell.

**Conclusion.** A classical joint-ML + shrinkage corrector with a blind 5 dB gate:
- beats NOMP + tuned SC in all 7 impaired cells;
- has no loss on the official bank;
- is significantly **better** than the full v2b pipeline in 4 of 7 impaired cells (3 dB, 8/3, 6 dB, 30/6);
- is not significantly different in 2 cells (8 deg, 20 deg);
- is worse only at 45 deg phase (-0.021).

At 45 deg the gate is the weak point: its blind SNR is under-estimated at 5 dB under large mismatch. Ungated EB-ML + NOMP scores 0.564 against v2b + NOMP's 0.572 there (point estimates only). On error removal, EB-ML is within 0.4-1.5 MSE-points of v2b at >= 20 deg / >= 3 dB, and 8 points behind at 8 deg.

Costs:
- EB-ML takes about 37 ms/scene, against about 6.5 ms for the v2b estimate (`a0_proto.log`). v2b is about 6x faster.
- EB-ML-g5's gate was chosen on prototype data, not on test data.

**Perfect correction is not an upper bound** (the claim says this for gain). Confirmed and stronger: EB-ML-g5 + NOMP beats CEILING in 3 dB (0.751 vs 0.741), 6 dB (0.744 vs 0.704) and 30/6 (0.640 vs 0.622).

## 4. Verdict per claim

1. **Old IABC diagnosis** (`d1`; fresh seeds 9360-9370, `a6_d1_check.log`): **CONFIRMED, two wording fixes needed.**
   - Replicated: phase removed 43.9 / 24.4 / 7.7% with slope 0.25 / 0.15 / 0.05 for L = 1 / 3 / 9 at 25 dB; the phase feature removes 83.7% for L=1 and adds 650-2900% error for L >= 3; outside the training range the slope is 0.03 at 20 deg and 0.01 at 45 deg; gain removal by the IABC output is 95.0 -> 62.6%.
   - Fix 1: "95% -> 63%" is the IABC output, not "the gain input" (the raw gain feature gives 98.5 -> 55.5%).
   - Fix 2: the phase feature also fails for L=1 at low SNR (original log: -4302% at 0 dB). Say "exact only for one path at high SNR".
2. **CRB values, CRB efficiency, "angles barely matter": CONFIRMED.** One wording fix: "for 8 deg / 3 dB at SNR <= 5 dB the bound exceeds the error" is true for phase (7.1 deg > 4.3 deg) but not for gain at 5 dB (1.13 dB < 1.68 dB). For gain it holds only at <= 0 dB.
3. **"Perfect correction" Pd table and "the unremovable ramp causes most of the loss": CONFIRMED.** CEILING on fresh seeds: 45 deg 0.596, 8 deg 0.739. "Not an upper bound for gain" is CONFIRMED and holds for 30/6 too.
4. **"v2b removes 90-97% of the impairment, close to the bound": NEEDS REWORDING.**
   - The numbers replicate: 90.3-97.2% MSE at SNR >= 5 dB for >= 20 deg / >= 3 dB, and 67-68% at 8 deg.
   - But they are **MSE** percentages. In RMS terms they are 69-83%, and 43-44% at 8 deg.
   - The comparison with "tuned self-cal" and the "oracle" overstates the network's edge. A classical joint-ML + shrinkage estimator removes 88.8-96.0% in the same cells, and 59% at 8 deg.
5. **"Gains up to +0.057 Pd over tuned self-cal at 45 deg and 30/6": CONFIRMED numerically (+0.057*, +0.049*). Misleading as evidence for a network:**
   - the reference is a self-gated comparator that switches itself off at large impairments;
   - against EB-ML-g5 the v2b pipeline is +0.021* at 45 deg and **-0.009*** at 30/6;
   - "7 of 10 cells significant" shrinks on fresh seeds: 8/3 is no longer significant (+0.003 [-0.002, 0.007]); 4 of 7 tested cells are significant (5 deg and 30 deg were not re-run).
6. **Official bank:**
   - "v2b+NOMP -0.001 neutral; full pipeline -0.002*, mostly from the SC stage": **CONFIRMED** (identical numbers; NOMP+SC tuned alone is -0.002*).
   - "Old IABC -0.008*": not re-run here; taken from the cached `e1_eval`.
7. **Closely spaced paths (-0.105* at 3 deg), 30-35 dB banks, IABR-Net swap (+0.074*, +0.041*), latency 5.8 vs 1.6 ms: NOT RE-RUN** (taken from `e1_eval` / `t3_latency` logs; unverified here). No contradicting evidence. My prototype timing of the v2b estimate (6.5 ms/scene at batch 40) is consistent.
8. **Decision rule** (i pass, ii pass, iii fail by 0.002): **CONFIRMED as applied.**
   - (iii) reproduces exactly.
   - (ii) on fresh seeds: 4 significant of the 7 tested cells, none negative. ph30 was significant in the original and is very likely significant again, so (ii) still passes by about one cell.
   - "v2b+NOMP without SC would pass (iii)": CONFIRMED (-0.001 [-0.002, +0.001]). The author rightly does not count it.
9. **"Drop the 'network is needed' claim": CONFIRMED, and strengthened.**
   - It is not needed on ideal hardware.
   - On impaired hardware a classical EB-ML corrector matches or beats it in 6 of 7 cells.
10. **"Replace the old IABC with v2b as an optional module for impaired hardware": NEEDS REWORDING.**
    - v2b is clearly better than the old IABC.
    - But it is presented without the strongest classical alternative: classical joint ML + shrinkage + gate is significantly better downstream at 3 dB, 8/3, 6 dB and 30/6, and equal at 8 and 20 deg.
    - v2b keeps a significant advantage only for very large phase errors (45 deg: +0.021*) and in speed (about 6x faster).
    - Defensible wording: "a learned corrector (v2b) is a fast alternative to a classical ML/shrinkage self-calibration; neither is needed on ideal hardware; the learned one is better only for >= 45 deg phase errors."

## Files
- `a0_proto.py/.log` (prototype)
- `a1_crb_check.py/.log` (CRB)
- `a2_unit.py/.log` (seeds, projection, sign)
- `a3_test.py/.log`, `a3_plan.json` (fresh-seed test and official bank)
- `a5_summary.py/.log`, `a5_results.json` (tables and CIs)
- `a6_d1_check.py/.log` (old-IABC diagnosis)
- `a_classical.py`, `a_lib.py` (classical comparators)
- Attempt-1 crash log: `a1_crb_check_attempt1.*` (an SVD failure in an unguarded Gauss-Newton step; replaced by guarded LM)
