export const meta = {
  name: 'deep-research-resume-sequential',
  description: 'Resume the interrupted IABC study, all-neural pipeline evaluation and CPU fine-tuning, strictly one heavy job at a time, long jobs detached so a usage limit cannot kill them; adversarial checks; final synthesis',
  phases: [
    { title: 'IABC', detail: 'how much impairment IABC removes; NOMP-only vs IABC+NOMP; claim sentences' },
    { title: 'IABC-v2', detail: 'why IABC fails, CRB / perfect-correction ceiling, design + train a much stronger corrector, keep / replace / drop decision' },
    { title: 'All-neural', detail: 'CNN seeds -> learned unfolded decoder, full scale vs U-Net / NOMP' },
    { title: 'FT-ADC', detail: 'CNN fine-tuned on low-resolution ADC data vs fair classical baselines; learned decoder adapted to ADC' },
    { title: 'FT-Imperfections', detail: 'phase-shifter physics control; CNN fine-tuned on severe array imperfections; ablation' },
    { title: 'Adversarial', detail: 'refute the fine-tuning claims' },
    { title: 'Synthesis', detail: 'final research report' },
  ],
}

const DR = 'D:/ai_ml_project/paper/evidence/deep_research'
const RES = `
RESOURCES AND INTERRUPTION RULES (important):
- This PC has 7.7 GB RAM in total and other apps use 2-3 GB. Run only ONE heavy python process at a time. Process scenes in chunks of at most 400 for NOMP / decoders (nomp_greedy allocates (B,64,64) float64 arrays) and CNN batches of at most 250; free arrays between banks. A previous attempt crashed with MemoryError because five jobs ran at once. Use torch.set_num_threads(6) and OMP/OPENBLAS/MKL threads 6 (set the env vars before importing numpy/torch); nothing else heavy is running now.
- A previous attempt of this task was interrupted by a usage limit at ~03:42 (no new data after that). FIRST inspect your folder: reuse finished scripts, logs and results; rerun only what is missing or crashed. Do not delete earlier logs (rename them *_attempt1.log if you must re-run).
- Any job expected to take more than ~10 minutes must be launched DETACHED so an interruption of you cannot kill it: from the PowerShell tool run Start-Process -FilePath python -ArgumentList '-u','<abs path to script>' -WorkingDirectory '<abs dir>' -WindowStyle Hidden -RedirectStandardOutput '<abs log>' -RedirectStandardError '<abs err>' -PassThru . The script must checkpoint / write partial results and resume from them when restarted. Wait for it by polling its log (Monitor tool with an until-loop, or one bash command with an internal until-loop and sleep, at most ~10 minutes per call). Check that the process is alive and not erroring.`

const CTX = `
PROJECT CONTEXT:
- Joint AoA/AoD for analog mmWave MIMO, 16x16 ULAs, DFT codebooks; Y (16x16 beamspace) = W^H diag(d_r)^H H diag(d_t) F + Z; antenna domain Ht = to_ant(Y) = conj(D_r) H D_t (+ noise): rows g_r e^{-j eps_r}, cols g_t e^{+j eps_t}. Mean phase/log-gain and the linear phase ramp are unidentifiable.
- Code: D:/ai_ml_project/IABR_v2_extracted/IABR_v2/notebook_src.py. IABC lines 440-470 (458 params; output imp (B,32,2) = identifiable [phase rad, natural-log gain], rows 0-15 Rx, 16-31 Tx); IABRv2.front 497-509 (Hc = Ht * c_r * c_t, c = exp(-imp_lg - j imp_ph)); CNN slice [419:522] (IABRv2(32,10,32), 191,702 params, outputs/checkpoints/IABRv2/final.pt - never overwrite); decoder slice [527:647] (nomp_greedy, heat_peaks, selfcal, decode(H, L, heat, 'greedy'|'heat'|'dual', sc)); parabolic_heat_decode 650-668; training focal_loss/lr_at/train 686-750, CFG ~line 82 (lr_max 1e-3, lr_min 1e-5, warmup 500, clip 5, lam_imp 50). iabr2_sim.py (set os.environ['IABR2_SUITE']='D:/ai_ml_project' first): sample_paths, simulate(...)->Y,D_r,D_t,H, to_ri, heat_targets, impairment_targets, make_batch(rng,B,cfg) with cfg train_L=(1,9), train_snr=(-15,24), clean_frac=0.2, train_phase_max=8, train_gain_max=3, grid=32, heat_sigma=1.0. CNN trained on phase <= 8 deg, gain <= 3 dB.
- Helpers: D:/ai_ml_project/paper/evidence/new_results/common.py (load_bank, scene_stats, metric_by_snr, boot_metrics (paired, within SNR groups), ci, SC=(1.0,5.0)); learned decoder code D:/ai_ml_project/paper/evidence/learned_decoder/ (exp.py loads the CNN at import and sets torch threads 6; exp3.py RefinerV3/Staged; exp5.py StagedSC; lurd.pt 4,803 params). Stress simulator ${DR}/stress/stress_sim.py (base_scenes(n,L,snr,seed), generate(base, kind, sev): clean, M1 coupling rho, M2 position sigma, M3 (phase_deg, gain_db), M4 phase-shifter bits, M5 ADC bits (peak AGC), M6 phase-noise deg, M8 near-field m; stress test seeds 5000 (0 dB) / 5015 (15 dB)), ${DR}/stress/scommon.py. Caches: D:/ai_ml_project/IABR_v2_extracted/IABR_v2/outputs/cache/net__<bank>.npz (heat, Hc, Ht; 56 banks), pred__{U-Net_paper,NOMP_no_network,IABR-v2}__<bank>.npz (52 banks); new_results/preds_heat/, new_results/preds/selfcal_pred__*.npz.
- Verified so far (read the reports): ${DR}/verify/REPORT.md - official bank (8000, ideal hardware) NOMP-only 0.7581, IABR-v2 dual 0.7551, IABR-Net heat (classical decoder) 0.7503, U-Net 0.7366, network only 0.4507; oracle headroom search loss only +0.005; latency NOMP 8.5 ms vs CNN+decoder 24.7 ms. ${DR}/stress/RESULTS.md + stress/adversarial/VERDICT.md - NOMP breaks under position errors, large phase/gain, low-res phase shifters, low-res ADC, phase noise; FAIR classical baseline = NOMP + TUNED self-calibration (lam 1, tau 2, iters 4; stress/adversarial/replicate.py; tuned on seeds 9900/9915); original network beats it only at gain 6 dB / 15 dB (+0.028 heat, +0.019 dual) and tiny margins at 2-bit phase shifters; IABR-v2 dual == NOMP on Hc within 0.001 (gain comes from IABC, not heat seeds); oracle-seeded search-loss gaps: 1-bit ADC +0.40/+0.34 (15/0 dB), 1-bit phase shifters +0.26/+0.34, phase 45 deg +0.06/+0.11, position 0.2 +0.06/+0.10, phase noise 30 deg +0.07/+0.10. Literature: ${DR}/lit/baselines_survey.md, ${DR}/lit/when_dl_helps.md (verified references).
RULES: never modify or delete existing files outside your own folder; never overwrite final.pt or lurd.pt; large files (*.npz, *.pt) in a cache/ subfolder (gitignored); no git. Every number from a script you ran, stdout saved as .log next to it. Fixed seeds, disjoint from all earlier test/tuning seeds (record them). Paired bootstrap (2000 resamples within SNR groups, seed 0); a claim needs the 95% CI to exclude 0. Report null/negative results plainly.
${RES}`

const IABC_M = `${CTX}

YOUR TASK (folder ${DR}/iabc; resume - s1_signcheck done; s2_sweep re-run in 400-scene chunks and FINISHED ('done 515s' in s2_sweep.log); s4_banks.py was relaunched detached by a previous agent and may still be running - check running python processes (PowerShell Get-CimInstance Win32_Process) before launching anything and wait for it rather than starting a duplicate; s3_analyze and partial results exist): produce the evidence for two statements the project owner wants to make, and find out exactly how far each is true:
 (S1) "The learned IABC module removes most of the per-antenna gain/phase impairment."
 (S2) "NOMP alone achieves X; IABR-Net (network + NOMP-type decoder) achieves Y; so the network is needed."
Partial results from the crashed run (averaged over SNR -10..25; treat as preliminary): identifiable RMS phase error before -> after IABC: phase 8 deg cell 4.31 -> 4.08 deg; gain 3 dB cell 1.66 -> 1.10 dB; gain 6 dB 3.33 -> 2.59 dB; self-cal with the gate forced open had ~13-14 deg phase error even on clean data (probably dominated by low SNR - check per SNR); Pd at gain 6 dB: NOMP 0.695, IABC+NOMP 0.717, IABR-Net heat 0.732, NOMP+SC tuned 0.717, IABC+NOMP+SC tuned 0.732.
Steps (keep what is already done and verified):
1. Sign/convention check (done in s1 - confirm from its log).
2. Impairment-estimation accuracy sweeps (L=3, 200 scenes per cell, 8 SNRs -10..25): phase-only max {0,2,5,8,12,20,30,45} deg; gain-only max {0,1,2,3,4,6} dB; combined (5,2),(8,3),(20,4),(30,6). Per scene RMS over 32 antennas of identifiable phase error (deg) and log-gain error (dB): (a) before correction, (b) after IABC, (c) after classical self-calibration from NOMP's angles (own copy of selfcal returning its cumulative correction; gate forced open, lam 1, iters 2 and 4), (d) oracle ALS with true angles. Report PER SNR (this matters: low SNR dominates averages) and averaged; median and mean; "% of identifiable impairment removed" (1 - after/before on MSE and on RMS); mark the IABC training range (<= 8 deg, <= 3 dB); also the unidentifiable part for context.
3. Downstream Pd on the same scenes: NOMP on Ht (NOMP-only), NOMP on Hc (IABC + NOMP), IABR-Net heat (decode(Hc,L,heat,'heat',(1,5))), IABR-v2 dual, NOMP+SC default and tuned, IABC+NOMP+SC tuned. Paired differences with CIs: (IABC+NOMP) - NOMP-only; IABR-Net heat - NOMP-only; (IABC+NOMP) - (NOMP+SC tuned); (IABC+NOMP+SC tuned) - (NOMP+SC tuned). Per severity (averaged over SNR) and per SNR for the largest severities.
4. The 56 cached banks: NOMP on cached Hc vs cached NOMP-on-Ht, paired CI per bank; counts of CI above 0 / below 0 / containing 0.
5. Figures: fig_iabc_reduction.png, fig_iabc_vs_snr.png, fig_pd_vs_severity.png (regenerate from the final results).
6. iabc/REPORT.md with all tables + "CLAIMS": (i) sentences the owner CAN write, each with numbers, CI, file; (ii) sentences the owner CANNOT write and why (say explicitly if NN+NOMP does not beat NOMP on the ideal-hardware official bank).
Return the CLAIMS section plus the two or three key tables (<= 80 lines).`

const IABC_A = (m) => `${CTX}

CLAIMS AND TABLES TO ATTACK:
${m}

YOUR TASK (folder ${DR}/iabc/adversarial): try to REFUTE every claim sentence; default to REFUTED if you cannot reproduce. 1) Read ${DR}/iabc/ scripts: correction sign/conjugation, identifiable projection applied consistently to target, IABC estimate and self-cal estimate, dB conversion, "before" computed on the identifiable part only, fair self-calibration, correct oracle ALS, same scenes for all methods. 2) With your own short script and FRESH seeds (8800+), re-measure 4-6 headline numbers (impairment removed at phase 8 deg / gain 3 dB / gain 6 dB at 15 and 0 dB; IABC vs self-cal error; IABC+NOMP - NOMP-only at the largest severities; the ideal-hardware difference). 3) Flag sentences that are true but misleading (only high SNR, only inside the training range, only for gain errors) and give an exact corrected sentence. Write iabc/adversarial/VERDICT.md + logs. Return a verdict per claim (CONFIRMED / REFUTED / NEEDS REWORDING + corrected sentence) with replicated numbers.`

const IV2 = (m, a) => `${CTX}

IABC STUDY RESULTS (current IABC, 458 params):
${m}
--- adversarial verdict ---
${a}

THE OWNER'S QUESTION: "If IABC does nothing, what is the need of it? Should I keep it? If yes, make it better so it removes impairments significantly - any change it needs, no size limit."
YOUR TASK (folder ${DR}/iabc_v2): answer with experiments, then build the strongest corrector you can and decide keep / replace / drop.
1. Diagnose why the current IABC removes so little (read notebook_src.py lines 420-470: its features are per-row/column energy and the detrended phase after de-rotating by the STRONGEST beam only, a shared 458-parameter per-antenna MLP, trained only on phase <= 8 deg / gain <= 3 dB, jointly with the heatmap loss). Show evidence (e.g. error vs number of paths L, vs SNR, vs path-power ratio).
2. What is achievable at all? (a) Cramer-Rao bound for the identifiable per-antenna phase (deg) and log-gain (dB) per SNR (L=3; angles and complex path gains unknown nuisance parameters; also the known-angle case), computed numerically from the Fisher information of the model Y = W^H conj(D_r)... (state the exact model you use, verify the FIM by finite differences); (b) the PERFECT-CORRECTION ceiling: NOMP on Ht corrected with the TRUE identifiable impairment, per severity (phase 0..45 deg, gain 0..6 dB, combined) and SNR - the most any corrector can buy downstream; compare with NOMP alone and NOMP + tuned self-calibration.
3. Build a much stronger corrector ("IABC-v2"), no size limit, any architecture. Ideas to try (pick by evidence, not all are required): inputs = complex Ht plus covariance features R_r = Ht Ht^H and R_t = Ht^H Ht (for an ideal ULA these are close to Toeplitz, and the per-antenna distortion breaks the Toeplitz / shift-invariance structure - the classical structure-based calibration cue), plus the old IABC features; antenna-token transformer or 2-D CNN over the 16x16 matrices with per-antenna output heads; iterative / unrolled refinement (estimate -> correct -> re-estimate, 2-4 stages); optional decoder-in-the-loop features (residual after a quick decode). Output the identifiable phase/log-gain (same projection as IABC) so it is a drop-in replacement. Train it ALONE (the CNN trunk frozen) with MSE on impairment_targets over wide ranges: phase U(0,45 deg), gain U(0,6 dB), 20% clean, SNR -15..25, L 1..9; fresh synthetic batches, seeds disjoint from all test seeds (use 62,000,000+). Launch training DETACHED and resumable; budget up to ~3 hours of CPU in total; log loss curves; keep a held-out validation set (separate seed) for model selection - never the test banks.
4. Classical comparators on the same scenes: no correction, old IABC, ALS self-calibration from NOMP angles (tuned recipe and gate-open variants), a Toeplitz/structure-based classical estimator you implement, oracle ALS with true angles, and the CRB.
5. Evaluation (fresh test seeds 8900+, 200-500 scenes per cell, 8 SNRs): per-SNR residual identifiable error and % removed for every method vs CRB; downstream Pd (paper-style and strict, paired CIs): NOMP alone, NOMP on old-IABC Hc, NOMP on IABC-v2-corrected channel, NOMP + tuned self-cal, IABC-v2 + NOMP + tuned self-cal, perfect-correction ceiling, and IABR-Net (heat-seeded) with IABC-v2 swapped in (trunk frozen; plus, if time, a short trunk fine-tune). Severities: phase {0,5,8,20,30,45} deg, gain {0,3,6} dB, combined (8,3), (20,4), (30,6). Also the official bank (8000; ideal hardware) as a no-regression check.
6. DECISION with explicit criteria, stated before looking at the final test numbers: KEEP/REPLACE with IABC-v2 if (i) it removes >= 50% of the identifiable RMS impairment at SNR >= 5 dB for errors of >= 8 deg / >= 3 dB, (ii) downstream Pd beats NOMP + tuned self-calibration with the 95% CI excluding 0 on impaired banks, and (iii) no significant loss on the official bank. Otherwise recommend dropping IABC from the claims (or keeping it only as an optional module) and say exactly why (e.g. CRB says low-SNR correction is impossible; NOMP is insensitive to small errors).
Write iabc_v2/REPORT.md (diagnosis, CRB, ceiling, architecture with parameter count, training curve, all tables, figures fig_residual_vs_snr.png and fig_pd_vs_severity.png, decision), results.json, scripts, logs, checkpoints in iabc_v2/cache/ (gitignored).
Return a concise markdown summary (<= 80 lines): diagnosis, CRB/ceiling numbers, the new corrector, key tables, and the decision.`

const IV2A = (s) => `${CTX}
${NOTE}

CLAIMS TO ATTACK:
${s}

YOUR TASK (folder ${DR}/iabc_v2/adversarial): try to REFUTE every claim about IABC-v2 and the keep/replace/drop decision. Default REFUTED if not reproducible. 1) Read ${DR}/iabc_v2/ scripts: training seeds disjoint from validation and test seeds (prove from code), no test bank used for model selection, identifiable projection identical for target and estimate, correction applied with the right sign/conjugation, the CRB derivation (re-check the FIM numerically by finite differences and with a Monte-Carlo efficient-estimator check at high SNR), fair classical comparators (tuned ALS, structure-based estimator). 2) Re-measure the 4-6 headline numbers (impairment removed per SNR at 8 deg / 3 dB / 30 deg+6 dB; downstream Pd vs NOMP + tuned self-cal; official-bank regression) with your own short script on FRESH seeds (9300+). 3) Check whether a simple classical change closes any claimed gap (more ALS iterations, tuned shrinkage, structure-based estimator). Write iabc_v2/adversarial/VERDICT.md + logs. Return a verdict per claim (CONFIRMED / REFUTED / NEEDS REWORDING) with replicated numbers and whether the keep/replace/drop decision stands.`

const NOTE = `
DISK AND REPORT NOTES (added after the last interruption):
- Drive D: is almost full (about 100 MB free; it holds the owner's other data - never delete anything outside your own folder). Keep what you write on D: small (scripts, logs, json, figures, checkpoints of a few MB). Put large arrays (prediction caches, generated training data) under C:/Users/Milon/AppData/Local/Temp/claude/D--thesis/a5db9a29-d75e-4561-834b-cd0638a16d91/scratchpad/dr_cache/<your stage>/ (C: has about 4 GB free; temporary and regenerable from fixed seeds) and record those paths in your report. Check free space (df -h /d /c) before long runs; if a write fails for lack of space, stop that job and report it instead of deleting anything.
- A previous agent found that creating report files with the Write tool was blocked. Write your report with the Bash tool (python or a heredoc) as REPORT.md; if that is also blocked, write REPORT.txt with the same content. Never skip the written report.
- The IABC-v2 stage finished (${DR}/iabc_v2/: s_summary.log, results.json, criteria.json, crb.json, figures, cache/v2a_best.pt, cache/v2b_best.pt; no REPORT.md). Its v2b corrector (v2a transformer + NOMP + classical self-calibration + learned refiner, 854k params) removes 90-97% of the correctable identifiable impairment at SNR >= 5 dB and is neutral on the official bank (v2b+NOMP -0.001 [-0.002, +0.001]).`

const NN_E = `${CTX}
${NOTE}

YOUR TASK (folder ${DR}/nn_decoder; resume - s0_inventory and s1_reproduce are done: s1 reproduced results5.json exactly (learned_no_sc official 0.7437 etc.)): evaluate the ALL-NEURAL pipeline at full scale so the project can compare it with published work.
Pipeline "IABR-Net (all-neural)": CNN on Y -> heat = sigmoid(heat logits), Hc -> top-L heatmap local maxima (heat_peaks) as seeds -> learned decoder StagedSC(RefinerV3(), pre=6, rounds=3, inner=2) from lurd.pt, use_sc=True (and use_sc=False ablation). No greedy FFT search, no classical Newton step.
1. Run on the full official bank (8000), T2_d1, T2_d2, T2_d5 (8000 each), and every other bank with a net__ cache and truth (the learned decoder was trained with L=3; report L != 3 banks separately).
2. Per bank: paper-style Pd, strict Pd, RMSE with CI for: all-neural (SC on/off), IABR-Net with the classical heat-seeded decoder (preds_heat or heat_main recipe), IABR-v2 dual, NOMP-only, U-Net, network only (where cached). Paired differences: all-neural - U-Net; all-neural - NOMP-only; all-neural - classical-decoder IABR-Net. Per-SNR table (official). Bank counts (CI above / below / containing 0).
3. Latency, CPU single thread: CNN + learned decoder at batch 1 and 256 (median of >= 30 runs), next to NOMP and the classical-decoder pipeline measured the same way.
4. Model facts: parameter counts; precise description of what the learned decoder computes (which analytic quantities it takes as features - score, analytic gradient, probes, LS gains, residual subtraction - and what is learned).
5. Optional only if time remains and it can be validated: port Lloria et al.'s ResNet (D:/ai_ml_project/DL_DOA/models/inf_model_007_256_resnet.h5; h5py works, TensorFlow does not; UNetPaper port in notebook_src.py ~line 839 as template) and validate against numbers printed in the paper PDFs under D:/ai_ml_project/papers; otherwise skip and say why.
Write nn_decoder/REPORT.md (tables; a "comparison with published work" table: all-neural IABR-Net vs U-Net (and ResNet if validated) on the same benchmark, with NOMP-only listed as the strong model-based reference), results.json, fig_pd_vs_snr.png, scripts, logs.
Return a concise markdown summary (<= 70 lines) with the official-bank table, per-SNR table, bank counts and latency.`

const NN_A = (s) => `${CTX}
${NOTE}

CLAIMS TO ATTACK:
${s}

YOUR TASK (folder ${DR}/nn_decoder/adversarial): try to REFUTE every claim; default REFUTED if not reproducible. 1) Read ${DR}/nn_decoder/ scripts and exp.py/exp3.py/exp5.py: no ground truth at inference (truth only in the loss), seeds only from the CNN heatmap, lurd.pt and use_sc used as stated, same scenes and L for all methods, caches aligned scene-by-scene with truth, common.py evaluator; argue whether official-bank scenes could appear in training (gen_batch draws fresh synthetic scenes). 2) Own short script: official bank (all 8000) all-neural vs U-Net vs NOMP-only, plus 3 other banks; agreement within CI. 3) Check the description of what the decoder computes against the code; correct anything overstated (e.g. "no classical component" when LS gains and analytic gradients are used). Write nn_decoder/adversarial/VERDICT.md + logs. Return a verdict per claim (CONFIRMED / REFUTED / NEEDS REWORDING + corrected sentence) with replicated numbers.`

const FTA = `${CTX}
${NOTE}

YOUR TASK (folder ${DR}/finetune/adc; resume - acommon.py, diag.py/diag2.py (+logs/json), evaluate.py, latency.py, make_report.py, train_ft.py exist; training attempt 2 (THREADS 4, BS 64, 900 steps, 2.0 s/step under load) stopped at step 50 without a checkpoint; train_ft.py writes cache/adc_ft_ckpt.pt every CK steps and final cache/adc_ft.pt and resumes): can a CNN trained on low-resolution-ADC measurements give the decoder better starting points than NOMP's greedy FFT search? (With a 1-bit ADC the oracle-seeded decoder beats NOMP by +0.40 / +0.34 Pd: search loss a learned front-end could recover.)
1. Classical baselines on the stress M5 banks (b = 1,2,3,4; SNR 0 and 15; stress seeds; confirm Y identical to the stress run): why NOMP fails; (i) FFT-peak seeds (top-L local maxima of the 4x-oversampled |FFT2(Hant)| as the heatmap for decode 'heat'); (ii) classical dual (greedy vs FFT-peak seeds by residual); (iii) NOMP oversampling 8, rc 6; (iv) one more cheap robustness trick if a good one exists. Reuse diag results if complete.
2. Fine-tune the CNN from final.pt (50/50: original training distribution and the same passed through stress_sim.adc with b ~ U{1,2,3,4}, peak AGC, before to_ri; original loss; Adam, short warm-up, cosine ~3e-4 -> 1e-5, clip 5). Launch train_ft.py DETACHED (see the rules) with 6 threads, size the run to ~60 minutes, save cache/adc_ft.pt (state_dict) - this exact path is used by the next stage. Log the loss curve.
3. Evaluate (fine-tuned CNN heat-seeded + SC, dual + SC, NOMP greedy on its Hc, network only) against step-1 baselines, NOMP+SC default and tuned, the ORIGINAL CNN (heat, dual), oracle-seeded: stress M5 banks (500 per cell), a fresh-seed replicate (1000 per cell, new seeds) for every claimed win, the official bank (all 8000; regression vs the original CNN), clean SNR 0/15 banks. Paired bootstrap vs the best classical method per cell.
4. Verdict: where the fine-tuned network beats every classical baseline, by how much, what it costs on official/clean banks, latency.
Write finetune/adc/REPORT.md, results.json, fig_adc.png, scripts, logs. Return a concise markdown summary (<= 60 lines) with the key table and verdict.`

const LDA = (a) => `${CTX}
${NOTE}

ADC FINE-TUNING RESULT:
${a}

YOUR TASK (folder ${DR}/nn_decoder_adapt): adapt the learned (unfolded) decoder to low-resolution-ADC data. The script ${DR}/nn_decoder_adapt/ld_adapt_adc.py is already written and smoke-tested (ld_adapt_adc_smoke.log/json; the smoke run used final.pt as a stand-in): it waits until ${DR}/finetune/adc/cache/adc_ft.pt exists (unchanged for 10 min, no train_ft process), fine-tunes StagedSC from lurd.pt for 1500 steps (batch 64, 4 threads; training seeds 61,000,000+; resumable checkpoint cache/ld_adc_run.pt), then evaluates on the stress M5 banks (seeds 5000/5015) and a 2000-scene official subset: NOMP, FT-CNN + classical heat decoder, FT-CNN + original learned decoder, FT-CNN + adapted learned decoder, original CNN + original learned decoder, with paired bootstrap.
1. Read the script; check it is correct (seed disjointness, same Y for all methods, evaluator, checkpoint loading of adc_ft.pt). Fix bugs if any (keep a copy of the original as ld_adapt_adc_v1.py) and re-smoke-test.
2. If adc_ft.pt does not exist (ADC stage failed), report that and stop. Otherwise launch it DETACHED (rules above; you may raise threads to 6 by editing NTH) and wait for 'done' in ld_adapt_adc_run.log.
3. Write nn_decoder_adapt/REPORT.md: the tables from ld_adapt_adc_run.json, and a verdict: does the adapted all-neural pipeline beat NOMP and the best classical method (take the best classical numbers on the same banks from ${DR}/finetune/adc/results.json) under low-resolution ADCs, and at what cost on the official subset.
Return a concise markdown summary (<= 40 lines).`

const FTB = (a, l, v) => `${CTX}
${NOTE}

EARLIER STAGES:
--- ADC fine-tuning ---
${a}
--- learned decoder adapted to ADC ---
${l}

--- IABC-v2 (stronger impairment corrector) decision ---
${v}

YOUR TASK (folder ${DR}/finetune/imp; resume - a previous agent wrote icom.py, unit_gen.py (+log), m4_physics.py (smoke run only), train_imp.py, evaluate.py, bench.log; its training (3300 steps) reached step ~3100 when the laptop crashed and restarted at ~17:22; the orchestrator relaunched it detached from the step-3000 checkpoint at ~17:27 (output in train_resume.log / train_resume.err). Training then FINISHED (cache/imp_ft.pt, 3300 steps, 61.7 min). A second agent then ran the main evaluation (evaluate.py -> eval_main.log, results.json, make_report.py/make_report.log, fig_M1/M2/M4/train.png) and launched a detached queue run_queue.py (started 18:19; eval_fresh.log; fresh-seed replicates fresh:M2, fresh:M6, fresh:M3, fresh:clean, fresh:M8) before it was cut off by a network outage at ~18:40. UPDATE 20:10: the fresh-seed queue FINISHED at 19:42 (eval_fresh.log, run_queue2.log; nothing is running now); a later agent also wrote final_tables.py (18:46) before being cut off by a usage limit. Check for running python processes anyway (Get-CimInstance Win32_Process) and never start duplicates; read the existing logs; then finish whatever is still missing (M4 physics control full run if only the smoke run exists, ablation, official-bank regression if not yet in results.json, report): severe array imperfections. If the IABC-v2 stage decided KEEP/REPLACE and its adversarial check confirmed it, use the IABC-v2 corrector (${DR}/iabc_v2/cache/) in place of the old IABC in the fine-tuned pipeline (and include a no-correction and an old-IABC variant for comparison); otherwise keep the old IABC.
0. Physics control for low-resolution phase shifters (M4, no training): decode the M4 banks with an antenna-domain transform built from the KNOWN quantised codebooks (least squares / pinv; check conditioning). Does it close the M4 gap (b = 3,2,1; SNR 0/15)? If yes, the network is not needed there - say so.
1. Fine-tune the CNN from final.pt on 40% original training distribution + 60% extended impairments (one family per sample): M3 phase U(0,45 deg) with gain U(0,6 dB); M2 position sigma U(0,0.2); M6 phase noise U(0,30 deg); M1 coupling rho U(0,0.3) (own vectorised generator following stress_sim.generate; L 1..9, SNR -15..24). IABC target = identifiable diagonal part where it exists (M3 and the original distribution); mask the IABC loss otherwise (justify). Same loss; Adam, warm-up + cosine ~3e-4 -> 1e-5, clip 5. Launch DETACHED, resumable; ~75 minutes; save cache/imp_ft.pt.
2. Evaluate on the stress banks M1, M2, M3 (all five settings), M6 (all severities, SNR 0/15, same scenes as the stress run - confirm Y), M8 as out-of-distribution control, a fresh-seed replicate (1000 per cell) for every claimed win, the official bank (all 8000; regression), clean SNR 0/15. Methods: NOMP; NOMP+SC default and tuned; classical dual (greedy vs FFT-peak seeds; reuse finetune/adc code); original CNN (heat, dual); fine-tuned CNN (heat + SC, dual + SC, network only); oracle-seeded.
3. Ablation: (a) fine-tuned heat seeds on raw Ht; (b) NOMP greedy on fine-tuned Hc; (c) full fine-tuned heat-seeded on Hc; paired bootstrap between them - is any gain from the learned correction or the learned seeds?
4. Verdict per family.
Write finetune/imp/REPORT.md, results.json, fig_<family>.png, scripts, logs. Return a concise markdown summary (<= 70 lines) with key tables and verdict.`

const FADV = (a, l, b) => `${CTX}
${NOTE}

FINE-TUNING RESULTS TO ATTACK:
--- ADC ---
${a}
--- learned decoder adapted to ADC ---
${l}
--- imperfections ---
${b}

YOUR TASK (folder ${DR}/finetune/adversarial; RESUME NOTE 22:45: a previous agent wrote and ran run_all.py, which FINISHED all of s3_adc_test (quantiser-ML baseline on fresh ADC banks, 1000 scenes per cell, b=1/2 at 0/15 dB), s4_imp, s5_official and s6_latency ('ALL DONE' in run_all.log; logs + json + .done markers here); it also wrote s7_restarts.py but did not run it; it was then cut off by a network outage before writing VERDICT.md. Read those logs/json and the scripts, check them, run s7_restarts.py only if it is needed for a claim, then write the verdict): try to REFUTE every claim that a fine-tuned network (CNN and/or adapted learned decoder) beats the best classical method, and every claim of no regression. Default REFUTED if not reproducible. 1) Read all scripts under ${DR}/finetune/ and ${DR}/nn_decoder_adapt/: training seeds disjoint from test/tuning seeds (prove from code), no test bank used for checkpoint selection or thresholds, identical Y for all methods, consistent L, generators matching stress_sim.generate (numerical cross-check), right checkpoints, fair classical baselines (tuned self-cal active; FFT-peak and dual correct). 2) Re-evaluate the 3-6 most important claimed wins and the official-bank regression with your own short script on FRESH seeds (1000 scenes per cell, disjoint from everything so far). 3) Look for a simple classical fix closing any claimed gap (tuned self-cal with more iterations, more restarts, larger oversampling) and report it with numbers. 4) PRIORITY - the headline win so far is the ADC-fine-tuned CNN read out directly ('network only') beating every classical baseline by +0.13 to +0.18 Pd at 1-2 ADC bits / 15 dB (finetune/adc/REPORT.md; best classical there = quantisation-consistent NOMP). The ADC stage itself said a stronger quantisation-aware classical estimator is still missing. Implement the strongest feasible one: maximum-likelihood refinement of the angles and complex gains under the exact quantiser likelihood (per-component probability that the noisy real/imag value falls in the observed ADC cell, Gaussian noise of known variance, peak AGC known), initialised from consistent NOMP and from FFT-peak seeds with several restarts (and, if feasible, a GAMP-style or Bussgang-linearised variant); tune its few constants on separate seeds only. Test the network-only win against it on fresh seeds (1000 scenes per cell, b = 1, 2 at 0 and 15 dB), report latency, and say plainly whether the win survives. Write finetune/adversarial/VERDICT.md + logs. Return a verdict per claim (CONFIRMED / REFUTED / UNCERTAIN) with replicated numbers and any bug (file:line).`

const SYN = (parts) => `${CTX}
${NOTE}

STAGE RESULTS:
${parts}

YOUR TASK: write the final research report ${DR}/REPORT.md for the project owner (undergraduate team; plain, direct English; no hype). Read the underlying reports (verify/, stress/, stress/adversarial/, iabc/, iabc/adversarial/, iabc_v2/, iabc_v2/adversarial/, nn_decoder/, nn_decoder/adversarial/, finetune/*/, nn_decoder_adapt/, lit/) - use only numbers that appear in those logged reports and only references marked verified there. Structure:
1. The question and the short answer (5-8 bullets).
2. Is NOMP alone enough? Official-bank table, paired differences, oracle headroom.
3. How much impairment does IABC remove? (iabc/ results, per SNR, inside/outside training range, vs classical self-calibration) and what that buys downstream; then IABC-v2 (iabc_v2/: diagnosis, CRB, perfect-correction ceiling, new corrector, adversarial status) and the keep / replace / drop decision.
4. The all-neural pipeline (CNN + learned unfolded decoder) vs U-Net vs NOMP on the benchmark; latency; what the decoder computes.
5. Where NOMP breaks (stress table + adversarial status).
6. Can the network do what NOMP cannot? Fine-tuning results (ADC, adapted learned decoder, imperfections, phase-shifter physics control), ablation, adversarial verdicts, regressions. Be exact about what survived adversarial verification.
7. Why recent DL AoA/AoD papers don't compare with NOMP (stated reasons vs inference).
8. Honest journal framing: 2-3 candidate paper stories ranked by the evidence actually obtained, each with title idea, central claim, supporting tables/figures (file paths), the reviewer attack and the answer, any experiment still needed. NOMP (Mamandipoor et al. 2016) and gain/phase self-calibration (Weiss & Friedlander 1990) must be named and credited in every framing; an unfolded decoder must be described as unfolded from NOMP. Say plainly why hiding them would fail review.
9. File index.
Return the full text of sections 1, 3, 4 and 8 (<= 160 lines).`

const n = (x) => x || '(stage returned nothing - read its folder)'
phase('IABC')
const im = await agent(IABC_M, { label: 'IABC effectiveness (resume)', phase: 'IABC', effort: 'high' })
const ia = await agent(IABC_A(n(im)), { label: 'adversarial: IABC claims', phase: 'IABC', effort: 'high' })
phase('IABC-v2')
const iv = await agent(IV2(n(im), n(ia)), { label: 'IABC-v2: diagnose, bound, redesign, decide', phase: 'IABC-v2', effort: 'high' })
const iva = await agent(IV2A(n(iv)), { label: 'adversarial: IABC-v2 claims', phase: 'IABC-v2', effort: 'high' })
phase('All-neural')
const ne = await agent(NN_E, { label: 'all-neural pipeline eval (resume)', phase: 'All-neural', effort: 'high' })
const na = await agent(NN_A(n(ne)), { label: 'adversarial: all-neural claims', phase: 'All-neural', effort: 'high' })
phase('FT-ADC')
const fa = await agent(FTA, { label: 'fine-tune: low-resolution ADC (resume)', phase: 'FT-ADC', effort: 'high' })
const la = await agent(LDA(n(fa)), { label: 'learned decoder adapted to ADC', phase: 'FT-ADC', effort: 'high' })
phase('FT-Imperfections')
const fb = await agent(FTB(n(fa), n(la), `${n(iv)}\n--- adversarial ---\n${n(iva)}`), { label: 'fine-tune: array imperfections', phase: 'FT-Imperfections', effort: 'high' })
phase('Adversarial')
const fv = await agent(FADV(n(fa), n(la), n(fb)), { label: 'adversarial: fine-tuning claims', phase: 'Adversarial', effort: 'high' })
phase('Synthesis')
const parts = `--- IABC ---\n${n(im)}\n--- IABC adversarial ---\n${n(ia)}\n--- IABC-v2 ---\n${n(iv)}\n--- IABC-v2 adversarial ---\n${n(iva)}\n--- all-neural ---\n${n(ne)}\n--- all-neural adversarial ---\n${n(na)}\n--- ADC ---\n${n(fa)}\n--- adapted learned decoder ---\n${n(la)}\n--- imperfections ---\n${n(fb)}\n--- fine-tuning adversarial ---\n${n(fv)}`
const sy = await agent(SYN(parts), { label: 'synthesis report', phase: 'Synthesis', effort: 'high' })
return { iabc: im, iabc_adv: ia, iabc_v2: iv, iabc_v2_adv: iva, nn: ne, nn_adv: na, adc: fa, ld_adapt: la, imp: fb, ft_adv: fv, synthesis: sy }
