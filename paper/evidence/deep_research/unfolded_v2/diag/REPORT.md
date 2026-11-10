# Headroom diagnostics for an unfolded decoder (unfolded_v2/diag)

Decoder family studied: decoders **unfolded from NOMP (Mamandipoor et al. 2016)**. Baselines: NOMP os4 rc3 (cached
NOMP_no_network) and the strongest classical setting NOMP os8 rc6. Pd = fraction of angle components with |err| <= 1 deg,
mean over SNR groups. Claims marked * have a 95% paired-bootstrap CI (2000 resamples within SNR groups, seed 0) excluding 0.
Ideal hardware, L = 3 unless stated. All numbers come from the scripts and logs in this folder.

Data used: official bank (8000 scenes) for task 1 and the task-3 ceiling; official 2000-scene subset (first 250 per SNR) plus a
fresh val bank (seed 71,000,000, 2000 scenes) and train bank (seed 70,000,000, 4000 scenes, used only to train the learned
selector); prior-violation banks nosep (no minimum separation, seed 9500) and sep10 (10 deg separation, seed 9501),
1000 scenes each at -10/-5/0/5 dB. Arrays in C:/Users/Milon/iabr_deep_research_cache/dr_cache/unfolded_v2/diag.

Min-separation rule (verified in dldoa_dataset_generation.generate_points, used by iabr2_sim.sample_paths): rejection sampling
of (AoD, AoA) uniform on [0,pi]^2 with Euclidean distance >= pi/6 (30 deg) between every pair. Official truth: min distance
30.01 deg, 0 violations. Gains CN(0,1/L) sorted strongest first (|x|^2 per slot 0.60/0.27/0.11 at high SNR).

## 1. NOMP failure anatomy (official bank, 8000 scenes; anatomy.py/.log/.json)

| SNR | Pd | scenes with 0/1/2/3 paths correct | wrong paths: small bias (<=3 deg) / outlier | missed path to spurious estimate, median [IQR] deg | wrong paths that are endfire u=+-pi mirrors | bad scenes where truth has LOWER residual (search loss) |
|---|---|---|---|---|---|---|
| -10 | 0.2618 | .618/.326/.054/.002 | 603 / 1957 | 60 [33, 92] | 0.018 | 0.000 |
| -5 | 0.5063 | .280/.433/.232/.055 | 778 / 1160 | 68 [34, 111] | 0.049 | 0.001 |
| 0 | 0.6907 | .103/.332/.405/.160 | 709 / 669 | 90 [35, 166] | 0.086 | 0.001 |
| 5 | 0.8198 | .014/.186/.436/.364 | 470 / 380 | 126 [49, 173] | 0.131 | 0.016 |
| 10 | 0.8993 | .007/.072/.338/.583 | 300 / 203 | 172 [70, 175] | 0.173 | 0.046 |
| 15 | 0.9448 | .000/.027/.238/.735 | 165 / 127 | 176 [171, 177] | 0.247 | 0.038 |
| 20 | 0.9662 | .001/.020/.133/.846 | 87 / 89 | 177 [174, 177] | 0.341 | 0.104 |
| 25 | 0.9758 | .000/.013/.098/.889 | 57 / 67 | 177 [171, 178] | 0.379 | 0.234 |

* Outliers are always "one path missed, one estimate placed elsewhere"; two estimates on one path occur 0-2 times per 1000
  scenes. At <= 0 dB the spurious estimate is a genuinely different location (median 60-90 deg away; 59% of wrong paths are
  > 1 DFT bin away in (u,v) at -10 dB). At >= 10 dB the missed-to-spurious distance is ~177 deg: the endfire alias u=+pi vs
  u=-pi (angle ~0 vs ~180 deg), where the two atoms are (almost) identical, an ambiguity no decoder can resolve from y.
* Small-bias errors (median 1.5-1.7 deg) concentrate near endfire at high SNR (endfire share of bias errors 0.82-0.97 at
  >= 10 dB vs 0.37 of all paths): a < 0.25-bin error in u becomes > 1 deg in angle there. A Bayesian endfire read-out (angle
  maximising the posterior mass of its +-1 deg window) changes nothing (nomp_ro -0.0001, n.s.).
* Objective-limited vs search loss: in scenes where NOMP gets < 3 paths, the true angles have a LOWER LS residual than the NOMP
  answer in 0-0.1% of scenes at <= 0 dB, 1.6-4.6% at 5-15 dB, 10-23% at 20-25 dB. The marginal likelihood (gains integrated
  under the true CN(0,1/L) prior) gives the same picture (0-0.3% at <= 0 dB). At low SNR NOMP does not lose the search: the
  objective prefers its wrong answer. Search loss exists only at high SNR, where little Pd is left.

## 2. Candidate-selection headroom (gen_cands.py, analyze.py, supp.py, oracle_decomp.py)

Pool per scene (all refined with the NOMP Newton + cyclic steps): NOMP os4rc3, NOMP os8rc6, 20 triples of the top-6 FFT peaks,
12 greedy branches, 9 support swaps of the NOMP answer, 16 prior-drawn random starts (59 legal candidates); on the official
subset also the CNN-heatmap decoder and 20 heat-peak triples (80 candidates). Pooled official subset + val, 4000 scenes, 59 candidates:

| selection rule | Pd | vs NOMP os4rc3 | vs NOMP os8rc6 |
|---|---|---|---|
| NOMP os4rc3 | 0.7585 | - | -0.0013* |
| NOMP os8rc6 (best classical) | 0.7597 | +0.0013* | - |
| ML over pool (min residual) | 0.7600 | +0.0015* | +0.0002 [-0.0004, +0.0008] |
| ML + min-sep rule | 0.7603 | +0.0018* | +0.0005 [-0.0008, +0.0018] |
| marginal lik. + posterior window mass, no sep | 0.7604 | +0.0020* | +0.0007 [-0.0001, +0.0015] |
| marginal lik. + mass + sep (mass_sep) | 0.7602 | +0.0017* | +0.0004 [-0.0009, +0.0017] |
| MBR / voting | 0.7597 | +0.0012 | -0.0001 |
| learned GBM selector (trained on 4000 train scenes; official subset / val) | 0.7558 / 0.7586 | -0.0007 / -0.0018 (n.s.) | - |
| **ORACLE pick (truth)** | 0.7872 | +0.0287* [+0.0265, +0.0311] | +0.0274* |
| oracle of {NOMP, swaps} / {NOMP, FFT triples} / {NOMP, branches} / {NOMP, random} | 0.7789 / 0.7791 / 0.7713 / 0.7648 | +0.020 / +0.021 / +0.013 / +0.006 | |
| oracle-seeded optimum (truth as start; reference) | 0.8261 | +0.0677* | +0.0664* |

With the 80-candidate pool including the CNN-heatmap families (official subset): oracle +0.0307*; the heat-map families add
only +0.002 oracle headroom; legal rules unchanged (+0.0025 to +0.0033* vs NOMP os4rc3, i.e. about os8rc6 level).

Where the oracle headroom lives (oracle_decomp.log): of the +0.0287, only **+0.0028** comes from candidates with a LOWER residual
than NOMP (search loss, reachable by a better optimiser or residual-based selection; located at 15-25 dB); **+0.0259 (90%)**
comes from candidates with a HIGHER residual (median marginal-likelihood rank of the winning candidate 17-23 of 59 at <= 0 dB):
selection luck under noise, identifiable only with the truth. Oracle gain per SNR: -10 dB +0.104, -5 +0.049, 0 +0.032,
5 +0.013, 10 +0.010, 15 +0.007, 20 +0.008, 25 +0.006.

## 3. Prior / Bayesian (MAP) headroom (map_decomp.py, analyze.py)

* NOMP answers violate the pi/6 rule in 19.2 / 10.8 / 5.6 / 2.0 % of scenes at -10/-5/0/5 dB (official subset); 67% of the
  scenes a MAP rule repairs had a violating NOMP answer.
* Ceiling (official 8000): choosing between NOMP and the truth-seeded optimum by MAP (marginal likelihood + uniform-angle
  Jacobian + separation rule) gives +0.0175 [+0.0156, +0.0194]* (-10 dB 0.2618 -> 0.3243); by residual only +0.0051*.
  The prior explains ~1/4 of the oracle-seeded gap (+0.068), but only when a candidate near the truth is already available.
* Legal (candidate pool): adding the separation rule to ML gives +0.0003 (0.7600 -> 0.7603, n.s.); the gain prior (marginal
  likelihood) adds nothing over the residual; the uniform-angle Jacobian (mapU) hurts slightly.
* Prior violated: on nosep / sep10 every sep-aware rule LOSES -0.031* / -0.029* vs NOMP (ml_sep 0.5375 vs 0.5687;
  0.5448 vs 0.5738); the learned selector (trained under the prior) loses -0.018* / -0.020*; prior-free rules (ML, mass_nosep)
  stay at +0.000 to +0.002 (n.s.). Any gain built on the min-separation rule is a benchmark-prior exploit and is fragile.

## 4. Low-SNR posterior-mean / voting (supp.py; pooled 500 scenes per SNR)

| SNR | NOMP | ML pool | mass_sep | mass_nosep | MBR vote | posterior mean | oracle |
|---|---|---|---|---|---|---|---|
| -10 | 0.2680 | +0.0014 | **+0.0070*** [+0.0010, +0.0130] | +0.0051* | +0.0056 | +0.0064* | +0.104 |
| -5 | 0.5130 | -0.0003 | +0.0009 | -0.0003 | +0.0002 | +0.0006 | +0.049 |
| 0 | 0.7040 | -0.0013 | -0.0023 | -0.0020 | -0.0017 | -0.0023 | +0.032 |

Approximate MMSE / posterior-mode decisions help only at -10 dB (+0.5 to +0.7 points, partly from the separation prior) and not
at -5 or 0 dB; they are ~15x smaller than the oracle gap.

## 5. Adaptivity headroom (configs.py, adapt.py; official subset; grid os{2,4,8,16} x rc{1,3,6} x self-cal{off,on})

| rule | Pd | vs NOMP os4rc3 |
|---|---|---|
| best single config chosen on val (os8 rc6, self-cal off) | 0.7582 | +0.0016* |
| per-true-SNR config chosen on val | 0.7559 | -0.0006 (n.s.) |
| per-estimated-SNR config chosen on val (residual-based SNR estimate hits the right 5 dB bucket in 100%) | 0.7559 | -0.0006 (n.s.) |
| per-SNR ORACLE config | 0.7588 | +0.0022* (+0.0006 over the best fixed config) |
| per-scene min-residual config | 0.7582 | +0.0017* |
| per-scene ORACLE config | 0.7631 | +0.0066* |

Self-calibration never helps on ideal hardware (on vs off: -0.001 to -0.002 for every os/rc). SNR adaptation of decoder settings
has essentially no headroom (+0.0006 oracle over the best fixed config); the per-scene oracle (+0.0066) is again mostly noise luck.

## Verdict: which mechanisms could beat NOMP by a significant margin on the official benchmark?

* **No tested legal mechanism beats the best classical setting (NOMP os8 rc6)**: every legal rule is within +0.0008 of it,
  all n.s. Against NOMP os4rc3 they give +0.0015 to +0.0020*, the same as simply running os8 rc6.
* Truth-based headrooms are large (candidate oracle +0.027, oracle-seeded +0.068), but ~90% of them sit in candidates that the
  likelihood ranks worse than the NOMP answer. A selector or unfolded step rule that only sees y and the benchmark prior cannot
  reach them: the Bayes-type decisions here (marginal likelihood, posterior window mass, MBR vote, posterior mean) use the exact
  generative model and still gain at most +0.002 overall, and a learned selector gains nothing.
* Small genuine headrooms: (a) high-SNR search loss (+0.003 oracle, residual-detectable, 15-25 dB; ML over the pool already
  captures it); (b) the -10 dB posterior-mass decision (+0.007*); (c) the min-separation prior (+0.0003 legal, +0.0175 only with a
  truth-near candidate), a benchmark-prior exploit that costs -0.03* when the rule is violated.
* Not worth pursuing on ideal hardware: SNR-adaptive settings (+0.0006 oracle), self-calibration, endfire read-out, learned
  candidate selection, low-SNR averaging above -10 dB.
* Where real headroom exists (earlier verified work, not re-run here): hardware-impaired data (1-2-bit ADC), where ML from the
  true angles is +0.05 to +0.09 above the best classical method and a 5 dB assumed-SNR mismatch costs 0.04-0.07. That is the
  place for an unfolded quantisation-aware ML refinement with learned noise-level adaptation.

## Files / provenance

probe.py/.log (noise scale, prior checks, timings); anatomy.py/.log/.json (task 1, 8000 scenes); map_decomp.py/.log/.json
(task-3 ceiling); make_banks.py/.log, make_stress.py/.log (fresh banks); gen_cands.py + gen_cands*.log (pools); analyze.py/.log/.json
(tasks 2-4 per bank, incl. learned selector and nosep/sep10); supp.py/.log/.json (pooled 4000 scenes, vs os8rc6, low-SNR CIs);
oracle_decomp.py/.log/.json; configs.py + configs_*.log, adapt.py/.log/.json (task 5); ../lit/oracle_select.py/.log/.json
(method-level oracle over cached decoders: {NOMP, heat, dual, U-Net, net-only} 0.7904 vs NOMP 0.7581); make_diag.py -> diag.json
(merged summary). check_vmatch.py/.log verifies that the vectorised metric matching equals the official matching (0 mismatches).
Runs used 2 threads, chunks <= 400 scenes, one process at a time. Free RAM stayed at ~0.8 GB throughout because of desktop
applications (no other heavy job was running); these jobs peak well below that, so they were run anyway. Fix applied on resume:
two print statements in analyze.py had a literal line break inside a string (syntax error); fixed before running.