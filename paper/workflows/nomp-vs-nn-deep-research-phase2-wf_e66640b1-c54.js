export const meta = {
  name: 'nomp-vs-nn-deep-research-phase2',
  description: 'Fine-tune the IABR-Net CNN on CPU for the regimes where NOMP breaks (low-res ADC, severe array imperfections), test against fair classical baselines, verify, synthesize',
  phases: [
    { title: 'FT-ADC', detail: 'classical baselines + CNN fine-tuned on low-resolution ADC data' },
    { title: 'FT-Imperfections', detail: 'phase-shifter physics control + CNN fine-tuned on severe array imperfections, ablation of where the gain comes from' },
    { title: 'Adversarial', detail: 'refute the fine-tuning claims, fresh seeds, leakage and fairness checks' },
    { title: 'Synthesis', detail: 'final research report with the honest journal framing' },
  ],
}

const DR = 'D:/ai_ml_project/paper/evidence/deep_research'
const CTX = `
PROJECT CONTEXT (everything is on disk; read what you need):
- Task: joint AoA/AoD estimation for analog mmWave MIMO, 16x16 ULAs, DFT beam codebooks. Y (16x16 complex beamspace) = W^H diag(d_r)^H H diag(d_t) F + Z; Hant = to_ant(Y); u = pi cos(psi), v = pi cos(phi).
- Code (source of truth): D:/ai_ml_project/IABR_v2_extracted/IABR_v2/notebook_src.py — decoder slice [527:647] (newton, cyclic, ls_gains, residual_energy, nomp_greedy, heat_peaks, selfcal, uv_to_angles, decode(Hant, L, heat, mode='greedy'|'heat'|'dual', sc)); parabolic_heat_decode lines 650-668; CNN IABRv2 slice [419:522] (191,702 params; IABC per-antenna corrector -> trunk -> 32x32 heatmap); training: focal_loss/lr_at/train lines 686-750, CFG line ~82 (steps 48000, batch 256, lr_max 1e-3, lr_min 1e-5, warmup 500, clip 5, lam_imp 50); checkpoint outputs/checkpoints/IABRv2/final.pt. Simulator iabr2_sim.py (set os.environ['IABR2_SUITE']='D:/ai_ml_project' before import): sample_paths, simulate, to_ri, heat_targets, impairment_targets, make_batch(rng, B, cfg) with cfg keys train_L=(1,9), train_snr=(-15,24), clean_frac=0.2, train_phase_max=8, train_gain_max=3, grid=32, heat_sigma=1.0.
- How to build/run the CNN on CPU: D:/ai_ml_project/paper/evidence/learned_decoder/exp.py (exec src[419:522] with np, torch, nn, TF, W_IDEAL, F_IDEAL, P=16, Q=16; IABRv2(32,10,32); o = cnn(y) with y = to_ri(Y).transpose(0,3,1,2) float32; heat = sigmoid(o['heat']); o['Hc'] = corrected antenna-domain channel; o['imp'] = IABC impairment estimate).
- Metrics/helpers: D:/ai_ml_project/paper/evidence/new_results/common.py (scene_stats, metric_by_snr, boot_metrics (paired, within SNR groups), ci, SC=(1.0,5.0)). Paper-style Pd = fraction of angle components with |err| <= 1 deg (scenes with < L detections dropped), strict Pd counts them as misses.
- PHASE-1 RESULTS (all verified; read the reports for detail):
  * ${DR}/verify/REPORT.md: an independent NOMP reproduces the cached NOMP-only predictions scene by scene. Official bank (8000 scenes, L=3, SNR -10..25): NOMP 0.7581, NOMP+selfcal 0.7576, IABR-v2 dual 0.7551, IABR-Net heat-seeded 0.7503, U-Net 0.7366, network only 0.4507. Oracle-seeded decoder 0.8261, but only +0.0051 of that is search loss (rest is objective-limited at low SNR). Latency CPU 1 thread: NOMP 8.5 ms/scene (batch 1), CNN+heat decode 24.7 ms.
  * ${DR}/stress/RESULTS.md + stress_sim.py (base_scenes(n, L, snr, seed), generate(base, kind, sev) for kinds clean, M1 coupling rho, M2 position sigma, M3 (phase_deg, gain_db), M4 phase-shifter bits, M5 ADC bits (peak AGC), M5c, M6 per-measurement phase noise deg, M8 near-field range; adc(); SEVERITIES; the stress seeds are in run_stress.py) and scommon.py. NOMP degrades > 0.05 Pd under: M2 sigma>=0.05 (15 dB), M3 phase>=20 deg, gain 6 dB, M4 b<=2, M5 b<=4 (15 dB), M6 >=20 deg, M1 rho=0.3 (15 dB, borderline). Oracle-seeded gaps over NOMP (= search loss NOMP leaves): M5 b=1 +0.40/+0.34 (15/0 dB), M4 b=1 +0.26/+0.34, M3 45 deg +0.06/+0.11, M2 sigma 0.2 +0.06/+0.10, M6 30 deg +0.07/+0.10.
  * ${DR}/stress/adversarial/VERDICT.md + replicate.py + adv_common.py: the default self-calibration gate (tau=5) is shut on NOMP's raw channel under large impairments, so the FAIR classical baseline is NOMP + TUNED self-calibration (lam 1, tau 2, iters 4, tuned on separate tuning banks seeds 9900/9915 — reuse that recipe/constants). Against it the original network wins only at gain 6 dB / 15 dB (+0.028 heat, +0.019 dual) and tiny margins at 2-bit phase shifters (+0.006..0.012 dual). IABR-v2 dual equals plain NOMP run on the CNN-corrected channel Hc (within 0.001): its gains come from the learned IABC correction, not from the heatmap seeds. Heat-seeded decoding collapses for closely spaced paths and at 2-bit phase shifters / near field.
  * Literature: ${DR}/lit/baselines_survey.md and ${DR}/lit/when_dl_helps.md (verified references; DL gains appear mainly under unmodelled hardware effects, nonlinear/quantised measurements, unknown model order, and as learned initialisation + model-based refinement; no paper benchmarks a pure network against NOMP for analog-mmWave joint AoA/AoD).
- Hardware: CPU only, 8 logical CPUs, no CUDA; torch 2.7.1 CPU. Use torch.set_num_threads(6) for training. One heavy process at a time.
RULES:
- Do NOT modify or delete existing files (in particular never overwrite final.pt). Write only inside your own folder under ${DR}/finetune/ (or the folder named in your task). Large files (*.pt, *.npz) go in a cache/ subfolder (gitignored). No git commit/push.
- Never fabricate a number: every number you report comes from a script you ran with its stdout saved to a .log next to it. Report null/negative results plainly — a clean "the network does not help here" is a valid, valuable result.
- Test banks must never be used for training, threshold tuning or model selection. Training data comes from fresh random draws with seeds disjoint from all test/tuning seeds (record them). Fixed seeds everywhere. Paired bootstrap (2000 resamples within SNR groups, seed 0) for every claimed difference; a claim needs the 95% CI to exclude 0.
- The network must be compared against the BEST classical method in each cell (choose per cell, which is conservative): NOMP, NOMP+selfcal default (1,5), NOMP+selfcal tuned (1,2,4 iters), and the classical alternatives you add. Also report an oracle-seeded reference.
- Measure seconds per training step first and size the run to the time budget stated in your task; log the loss curve; save checkpoints so an interrupted run can resume.`

const FTA = `${CTX}

YOUR TASK (folder ${DR}/finetune/adc): can a CNN trained on low-resolution-ADC measurements give the decoder better starting points than NOMP's greedy FFT search? Motivation: with a b=1 ADC the oracle-seeded decoder beats NOMP by +0.40 (15 dB) / +0.34 (0 dB) Pd, so the refinement objective still peaks near the truth but NOMP's detection fails — that is search loss a learned front-end could recover.
1. Diagnose and build classical baselines first (cheap): on the stress M5 banks (b = 1, 2, 3, 4; SNR 0 and 15; regenerate with the stress seeds and confirm Y identical to the stress run), find why NOMP fails (wrong first detection? spurious quantisation harmonics? which path is missed?). Implement the classical alternatives a reviewer would ask for: (i) FFT-peak seeds = top-L local maxima of the 4x-oversampled |FFT2(Hant)| used as the "heatmap" for decode(..., 'heat'); (ii) classical dual = greedy NOMP vs FFT-peak seeds, picked by residual energy; (iii) NOMP with oversampling 8 and rc 6; (iv) one other cheap robustness trick if you think of a good one (e.g. dithering-aware / Bussgang reasoning, multi-start from top-K peaks). Report all on the M5 banks.
2. Fine-tune the CNN from final.pt on a 50/50 mixture: the original training distribution (make_batch with the original cfg) and the same distribution passed through stress_sim.adc with b ~ U{1,2,3,4} (peak AGC) applied to Y before to_ri. Keep the original loss (focal heatmap + lam_imp * IABC MSE on impairment_targets). Adam, short warm-up, cosine from about 3e-4 to 1e-5, clip 5, batch 128 (or what fits). Time budget for training: about 75 minutes of CPU. Save finetune/adc/cache/adc_ft.pt; log the loss curve.
3. Evaluate the fine-tuned CNN (heat-seeded + selfcal SC, dual + SC, NOMP greedy on its Hc, network only) against everything in step 1, NOMP+selfcal default and tuned, the ORIGINAL CNN (heat, dual), and the oracle-seeded decoder, on: the stress M5 banks (500 scenes per cell), a fresh-seed replicate (1000 scenes per cell, new seeds) for the cells where you claim a win, the official bank (all 8000; check for regression vs the original CNN), and clean SNR 0/15 banks. Paired bootstrap vs the best classical method per cell.
4. Verdict: where (bits, SNR) does the fine-tuned network beat every classical baseline, by how much, what does it cost on the official/clean banks, and what is the latency.
Write finetune/adc/REPORT.md, results.json, fig_adc.png (Pd vs ADC bits, one line per method, panels SNR 15 / 0), scripts and logs.
Return a concise markdown summary (<= 60 lines) with the key table and the verdict.`

const FTB = (adcSummary) => `${CTX}

ADC FINE-TUNING STAGE RESULT (for context):
${adcSummary}

YOUR TASK (folder ${DR}/finetune/imp): severe array imperfections.
0. Physics control for low-resolution phase shifters (M4, no training): the receiver knows its own quantised phase-shifter states, so decode the M4 banks with an antenna-domain transform built from the KNOWN quantised codebooks (least squares / pinv; check conditioning) instead of the ideal to_ant. Report whether this closes the M4 gap (b = 3, 2, 1; SNR 0/15). If it does, the network is not needed there — say so.
1. Fine-tune the CNN from final.pt on a mixture: 40% the original training distribution (make_batch, original cfg); 60% extended impairments, each sample drawn from one family: M3 phase U(0,45 deg) with gain U(0,6 dB); M2 position sigma U(0,0.2); M6 phase noise U(0,30 deg); M1 coupling rho U(0,0.3) (generate with your own vectorised code following stress_sim.generate; L 1..9, SNR -15..24 as in training). IABC target: the identifiable diagonal part where one exists (M3 and the original distribution); mask the IABC loss for families without a diagonal ground truth (justify). Same loss, Adam, warm-up + cosine ~3e-4 -> 1e-5, clip 5, batch 128 (or what fits). Time budget for training: about 90 minutes of CPU. Save finetune/imp/cache/imp_ft.pt; log the loss curve.
2. Evaluate on the stress banks M1, M2, M3 (all five settings), M6 (all severities, SNR 0/15, same scenes as the stress run — confirm identical Y), M8 as an out-of-distribution control, a fresh-seed replicate (1000 scenes per cell) for every cell where you claim a win, the official bank (all 8000; regression check), and clean SNR 0/15 banks. Methods: NOMP; NOMP+selfcal default and tuned; classical dual (greedy vs FFT-peak seeds by residual — reuse ${DR}/finetune/adc code if present); original CNN (heat, dual); fine-tuned CNN (heat-seeded + SC, dual + SC, network only); oracle-seeded.
3. Ablation to locate any gain (this matters for the paper's story): (a) fine-tuned heat seeds on the raw channel Ht (learned seeds, no learned correction); (b) NOMP greedy on the fine-tuned Hc (learned correction, classical seeds); (c) full fine-tuned heat-seeded on Hc. Paired bootstrap between them.
4. Verdict per impairment family: does the fine-tuned network beat the best classical method, by how much, is the gain from the learned correction or the learned seeds, and what does it cost on the official/clean banks.
Write finetune/imp/REPORT.md, results.json, figures fig_<family>.png (Pd vs severity per method, panels SNR 15 / 0), scripts and logs.
Return a concise markdown summary (<= 70 lines) with the key tables and the verdict.`

const ADV = (a, b) => `${CTX}

FINE-TUNING RESULTS TO ATTACK:
--- ADC stage ---
${a}
--- Imperfections stage ---
${b}

YOUR TASK (folder ${DR}/finetune/adversarial): try to REFUTE every claim that a fine-tuned network beats the best classical method, and every claim that it does not regress elsewhere. Default to REFUTED if you cannot reproduce.
1. Read all scripts under ${DR}/finetune/. Check: training seeds disjoint from test and tuning seeds (prove it from the code), test banks never used for checkpoint selection or thresholds, identical Y for all methods, L passed consistently, the impairment generators match stress_sim.generate (cross-check numerically on the same draws), the right checkpoint loaded, the classical baselines given the same information and a fair configuration (tuned self-calibration actually active; FFT-peak and dual baselines correct).
2. Independently re-evaluate the 3-6 most important claimed wins AND the official-bank regression numbers with your own short script on FRESH seeds (new banks, 1000 scenes per cell, seeds disjoint from everything used so far); report agreement within CI.
3. Look for a simple classical fix that closes any claimed gap (e.g. tuned selfcal with more iterations, more NOMP restarts, larger oversampling); if one exists, the network claim is weakened — say so with numbers.
Write finetune/adversarial/VERDICT.md and logs. Return a markdown verdict per claim (CONFIRMED / REFUTED / UNCERTAIN) with replicated numbers and any bug (file:line) found.`

const SYN = (a, b, adv) => `${CTX}

STAGE RESULTS:
--- ADC fine-tuning ---
${a}
--- Imperfection fine-tuning ---
${b}
--- Adversarial verdict ---
${adv}

YOUR TASK: write the final research report ${DR}/REPORT.md for the project owner (an undergraduate team; plain, direct English; no hype). Read the underlying reports (verify/, stress/, stress/adversarial/, finetune/*/, lit/) — use only numbers that appear in those logged reports and only references marked verified there. Structure:
1. The question and the short answer (5-8 bullets).
2. Is NOMP alone enough? The verified official-bank table and paired differences; the oracle headroom analysis; what this means.
3. Where NOMP breaks: the stress-test table (severity at which NOMP loses > 0.05 Pd; oracle search-loss gaps), with the adversarially confirmed status of each claim.
4. Can the network do what NOMP cannot? The fine-tuning results (ADC; imperfections; phase-shifter physics control), the ablation (learned correction vs learned seeds), the adversarial verdicts, regressions, latency. Be exact about what survived adversarial verification.
5. Why recent DL AoA/AoD papers don't compare with NOMP: stated reasons vs inference, from lit/baselines_survey.md.
6. Honest journal framing: 2-3 candidate paper stories ranked by strength of the evidence actually obtained, each with: title idea, central claim, which tables/figures support it (file paths), what a reviewer will attack and the answer, and what extra experiment (if any) is still needed. NOMP (Mamandipoor et al. 2016) and gain/phase self-calibration (Weiss & Friedlander 1990) must be named and credited as baselines/components in every framing; do not propose hiding them, and say plainly why hiding them would fail review.
7. File index of all evidence.
Return the full text of the report's sections 1 and 6 (<= 120 lines).`

phase('FT-ADC')
const a = await agent(FTA, { label: 'fine-tune: low-resolution ADC', phase: 'FT-ADC', effort: 'high' })

phase('FT-Imperfections')
const b = await agent(FTB(a || '(ADC stage returned nothing)'), { label: 'fine-tune: array imperfections', phase: 'FT-Imperfections', effort: 'high' })

phase('Adversarial')
const adv = await agent(ADV(a || '(none)', b || '(none)'), { label: 'adversarial: fine-tuning claims', phase: 'Adversarial', effort: 'high' })

phase('Synthesis')
const syn = await agent(SYN(a || '(none)', b || '(none)', adv || '(none)'), { label: 'synthesis report', phase: 'Synthesis', effort: 'high' })

return { adc: a, imperfections: b, adversarial: adv, synthesis: syn }
