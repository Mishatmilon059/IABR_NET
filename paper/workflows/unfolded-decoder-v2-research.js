export const meta = {
  name: 'unfolded-decoder-v2-research',
  description: 'Research and build an adaptive unfolded (NOMP-derived) neural decoder that can genuinely beat classical NOMP: literature + headroom diagnosis now (light CPU), heavy training after the running workflow finishes, adversarial check, report',
  phases: [
    { title: 'Research', detail: 'verified literature on unfolded/learned estimators that beat ML-type methods; headroom diagnosis of where NOMP fails' },
    { title: 'Design', detail: 'rank architectures by measured headroom, pre-register success criteria' },
    { title: 'Build', detail: 'wait for the CPU, implement + train the top designs, evaluate vs every classical baseline' },
    { title: 'Adversarial', detail: 'refute the claims on fresh seeds' },
    { title: 'Report', detail: 'final unfolded-v2 report' },
  ],
}

const DR = 'D:/ai_ml_project/paper/evidence/deep_research'
const OUT = DR + '/unfolded_v2'
const CACHE = 'C:/Users/Milon/iabr_deep_research_cache/dr_cache/unfolded_v2'
const CTX = `
PROJECT CONTEXT (read the files you need; everything is on disk):
- Joint AoA/AoD estimation, analog mmWave MIMO, 16x16 ULAs, DFT codebooks. Y (16x16 beamspace) -> antenna domain Ht = to_ant(Y); atoms a[n,m] = exp(-j n u + j m v), u = pi cos(AoA), v = pi cos(AoD). Benchmark of Lloria et al. (IEEE TVT 2026): L=3 paths, SNR -10..25 dB; the scene generator (iabr2_sim.sample_paths -> DG.generate_points(L, sep=pi/6)) enforces a minimum angular separation, path gains CN(0,1/L) sorted strongest first.
- Code: D:/ai_ml_project/IABR_v2_extracted/IABR_v2/notebook_src.py (decoder slice [527:647]: atoms, ls_gains, newton, cyclic, residual_energy, nomp_greedy, heat_peaks, selfcal, decode(H, L, heat, mode, sc); CNN slice [419:522]; final.pt never overwrite). iabr2_sim.py (set os.environ['IABR2_SUITE']='D:/ai_ml_project'). Helpers D:/ai_ml_project/paper/evidence/new_results/common.py (load_bank, scene_stats, metric_by_snr, boot_metrics, ci). Cached CNN outputs per bank: D:/ai_ml_project/IABR_v2_extracted/IABR_v2/outputs/cache/net__<bank>.npz (heat, Hc, Ht), cached predictions pred__{NOMP_no_network,U-Net_paper,IABR-v2}__<bank>.npz. Stress simulator ${DR}/stress/stress_sim.py (generate(base, kind, sev): M1 coupling, M2 position, M3 phase/gain, M4 phase-shifter bits, M5 ADC bits, M6 phase noise, M8 near field).
- THE EXISTING UNFOLDED DECODER (the thing to beat and improve): D:/ai_ml_project/paper/evidence/learned_decoder/ exp.py, exp3.py (RefinerV3: MLP 7->64->64->2 fed with score-normalised gradient, 4 probe second differences, coherence; Staged = 6 pre-steps + 3 cyclic rounds x 2 inner steps with LS gains and residual subtraction), exp5.py (StagedSC adds a gated self-calibration with learnable strength), lurd.pt (4,803 params). Verified results (${DR}/nn_decoder/REPORT.md + adversarial/VERDICT.md): official bank (8000 scenes) all-neural (CNN + lurd) 0.7508 vs NOMP 0.7581 (-0.0073*), vs classical heat decoder +0.0005 (same), vs U-Net +0.0142*, vs ResNet +0.0404*; its MLP step correlates 0.91-0.92 with the clipped Newton step (it just re-learned Newton); latency ~98 ms/scene (unrolled torch). Adapted to 1-2-bit ADC data it beats NOMP (+0.07-0.08) but loses to the best classical method (quantisation-consistent NOMP) and to the plain ADC-fine-tuned CNN read-out (${DR}/nn_decoder_adapt/REPORT.md, ${DR}/finetune/adc/REPORT.md).
- WHY IT CANNOT BEAT NOMP (verified, ${DR}/verify/REPORT.md): NOMP's Newton search is converged; the oracle-seeded decoder (true angles as starts) reaches 0.8261 vs NOMP 0.7581, but only +0.0051 is search loss - at low SNR the true angles are NOT the maximum of the residual / likelihood objective (ML threshold effect: -10 dB NOMP 0.2618 vs oracle-seeded 0.5655; -5 dB 0.5063 vs 0.6363; 0 dB 0.6907 vs 0.7400). Any decoder that maximises the same objective, with whatever learned step rule, inherits this. Best classical on the official bank: NOMP os8 rc6 0.7598. NEW (final adversarial check, ${DR}/finetune/adversarial/s3_adc_test.log, s6_latency.json): for 1-2-bit ADCs the strongest classical method is maximum-likelihood refinement under the exact quantiser likelihood (cell probabilities, known noise variance) with 13 starts + EM: Pd 1-bit/15 dB 0.540, 2-bit/15 dB 0.682, 1-bit/0 dB 0.275, 2-bit/0 dB 0.476, at 482 ms per scene; the ADC-fine-tuned CNN read-out alone reaches 0.419/0.593/0.239/0.438 at 6 ms; ML started from the network's seeds (hybrid, ~100 ms) gives 0.572/0.687/0.274/0.471 (beats multi-start ML only at 1-bit/15 dB, +0.032*); ML from the TRUE angles gives 0.626/0.719/0.439/0.540, so a better objective/decision rule still has headroom; ML loses 0.04-0.07 when its assumed SNR is off by 5 dB. An unfolded version of this quantisation-aware ML refinement (learned steps, learned noise-level/SNR adaptation, learned candidate selection) is a natural design to consider.
- Other verified context: ${DR}/RESULTS_SUMMARY.md (read it first), literature ${DR}/lit/when_dl_helps.md and baselines_survey.md.
RESOURCES / RULES:
- CPU only (8 logical CPUs), 7.7 GB RAM; another research workflow may still be running heavy jobs. Drive D: is almost full (~100 MB): on D: write only small files (scripts, logs, json, figures, small checkpoints < 20 MB) inside ${OUT}/<your subfolder>; put large arrays/training data in ${CACHE}/<your subfolder>/ (C:, permanent, ~20 GB free). Never delete anything outside your own folders. No git.
- Long jobs (> 10 min) must run DETACHED (PowerShell: Start-Process -FilePath python -ArgumentList '-u','<abs script>',... -WorkingDirectory '<abs dir>' -WindowStyle Hidden -RedirectStandardOutput '<abs log>' -RedirectStandardError '<abs err>' -PassThru), checkpoint and resume; wait by polling logs (one bash call with an internal until-loop, <= 10 min per call). Process scenes in chunks <= 400; one heavy process at a time.
- If the Write tool refuses a report file, write it with Bash (python or heredoc); if .md is refused use .txt. Never skip the written report.
- Every number from a script you ran, stdout saved as .log next to it. Fixed seeds, disjoint from all earlier ones (training 70,000,000+; validation 71,000,000+; fresh test banks 9500-9599). Paired bootstrap (2000 resamples within SNR groups, seed 0); a claim needs the 95% CI to exclude 0. Report null results plainly.
- Honesty: the decoder is derived from NOMP and must be described as "unfolded from NOMP (Mamandipoor et al. 2016)"; NOMP and the strongest classical variants stay as named baselines. If a gain comes from exploiting the benchmark's data prior (e.g. the minimum-separation rule or the gain distribution), say so explicitly and test what happens when the prior is violated (sep_*deg banks, L != 3 banks).`

const LIT = `${CTX}

YOUR TASK (folder ${OUT}/lit; RESUME NOTE: a previous attempt was cut off by a usage limit - it left oracle_select.py/.log/.json in this folder (probably misplaced diagnosis work: leave it, do not rely on it); light work only, no heavy CPU): verified literature (load WebSearch/WebFetch via ToolSearch "select:WebSearch,WebFetch"; open every source) on HOW an unfolded / learned estimator can beat a converged ML-type greedy estimator like NOMP, and on adaptive unfolding. Cover: (1) the ML threshold effect / outliers at low SNR in line-spectral and DOA estimation and Bayesian / MMSE / MAP estimators that beat ML there (e.g. Barthelme & Utschick 2021 NN + likelihood refinement; Deep-MLE; Bayesian line spectral estimation such as VALSE; hybrid NN-initialised ML); (2) unfolded networks that learn the OBJECTIVE or score (learned priors, learned proximal/denoiser = plug-and-play, learned likelihood) rather than only step sizes (LISTA/LAMP/ALISTA, deep unfolding with learned priors, NeuPLSE, learned AMP for channel estimation, mpNet learning the dictionary); (3) multi-hypothesis / learned model selection (choosing among candidate solutions, learned stopping / model-order); (4) adaptive unfolding: hypernetworks conditioned on SNR / impairment level / quantiser bits, variable-depth unfolding (e.g. mpNet variable depth), learned early stopping; (5) quantisation-aware unfolding (1-bit / few-bit likelihoods: GAMP with probit, unrolled MM for one-bit DOA). For each: what is learned, what classical method it beat, by how much, under what conditions, and whether the baseline was a converged ML/greedy-Newton method. End with 4-6 concrete design recommendations for this decoder, each with the mechanism by which it could beat NOMP. Write ${OUT}/lit/unfolding_survey.md (via Bash if needed). Return a concise summary (<= 60 lines) with the recommendations and the verified reference list (title, authors, venue, year, DOI/URL, verified yes/no).`

const DIAG = `${CTX}

YOUR TASK (folder ${OUT}/diag; RESUME NOTE: two previous attempts were cut off by usage limits (~19:35 and ~00:12; the second one updated gen_cands.py, make_stress.py (+log), analyze.py and wrote run_stage2.cmd but did not finish running them) - it wrote probe.py, anatomy.py (+log/json), make_banks.py, gen_cands.py (+log; a detached candidate-generation run), dcommon.py, configs.py, analyze.py, map_decomp.py (+log/json) here and oracle_select.py (+log/json) in ../lit/; read them, check what finished, reuse it and only redo what is missing; LIGHT CPU ONLY while the other workflow may be running: torch/numpy threads 2, chunks <= 400 scenes, check free RAM (PowerShell Get-CimInstance Win32_OperatingSystem FreePhysicalMemory) is > 1.5 GB before each run; use the cached net__ files and cached NOMP predictions, subsets of at most 2000 official scenes (250 per SNR) unless a full run is cheap): measure the HEADROOM for each way an unfolded decoder could beat NOMP, so the design is driven by evidence.
1. Failure anatomy of NOMP on the official subset per SNR: fraction of scenes with 0/1/2/3 paths correct; for wrong paths, is the error an outlier (a path placed at a wrong location / missed path / two estimates on one path) or a small bias; how far apart are the missed true path and NOMP's estimate; residual of NOMP's solution vs residual at the true angles (how often the truth has the LOWER residual = search loss vs HIGHER residual = objective-limited).
2. Candidate-selection headroom: generate K candidate solutions per scene cheaply (NOMP greedy, heatmap-seeded refinement, top-K FFT-peak / heatmap-peak combinations each refined with Newton+cyclic, multi-start) and report: Pd if an ORACLE picks the best candidate per scene (by true error) vs picking by residual vs picking by a simple prior-aware score. The oracle-selection gap is the headroom for a learned selector.
3. Prior / Bayesian headroom: does the benchmark prior (minimum separation, gain distribution, L) explain the objective-limited cases? E.g. a MAP score = residual + penalty for violating the minimum separation / implausible gain ordering; report Pd vs NOMP per SNR. Check the actual minimum-separation rule in DG.generate_points (D:/ai_ml_project/... find the module via iabr2_sim's import) and how often NOMP's solutions violate it.
4. Low-SNR posterior-mean idea: on a small subset at -10/-5/0 dB, does averaging / voting over candidates (an approximate MMSE / mode-of-posterior decision) beat the single ML answer?
5. Adaptivity headroom: how much would per-SNR (or per-scene estimated SNR) choice of decoder settings (oversampling, rounds, self-cal on/off, number of candidates) gain vs one fixed setting (oracle choice per SNR bucket and per scene)?
Write ${OUT}/diag/REPORT.md (via Bash if needed), diag.json, scripts, logs. Return a concise markdown summary (<= 60 lines): per-SNR failure anatomy, the headroom numbers for (2)-(5), and which mechanisms have enough headroom to possibly beat NOMP by a significant margin (and which do not).`

const DESIGN = (lit, diag) => `${CTX}

LITERATURE RESULT:
${lit}

HEADROOM DIAGNOSIS:
${diag}

YOUR TASK (folder ${OUT}/design; no heavy CPU): choose and specify the decoder(s) to build. Requirements from the owner: an unfolded (NOMP-derived) neural decoder that genuinely BEATS the classical decoder and is MORE ADAPTIVE (handles different SNR, impairments, ADC bits, ideally unknown number of paths). The owner said explicitly: there is NO limit on model size or number of parameters for now - do not shrink a design to save parameters; the only real constraint is CPU training/inference time on this machine (no GPU). Use the measured headroom: do not pick mechanisms the diagnosis shows cannot gain. Produce:
1. Ranked list of 2-3 designs (at most 2 will be trained), each with: architecture (exact layers, what is learned vs fixed classical structure, parameter count), inputs, what objective/score it effectively optimises and why that can beat NOMP's ML objective, adaptivity mechanism (conditioning on estimated SNR / impairment / bits, variable depth / learned stopping), training data mix (ideal benchmark distribution + impairments + 1-4-bit ADC + variable L; seeds 70,000,000+), loss (matched angle loss / detection-aware loss / auxiliary), CPU training-time estimate (budget up to ~6 hours of CPU in total for both designs; no parameter limit), inference latency estimate.
2. Pre-registered success criteria written to ${OUT}/design/criteria.json BEFORE any test run, e.g.: PRIMARY = beats NOMP AND NOMP os8 rc6 on the full official bank (8000) with the CI excluding 0 and no significant loss at any SNR >= 10 dB; SECONDARY = beats the best classical method (per cell, including quantisation-consistent NOMP for ADC and tuned/EB-ML self-calibration for impairments) in >= half of the regime cells with no regression on the official bank; plus the honesty test on prior-violating banks (sep_2/3/5deg, L != 3).
3. Evaluation plan (banks, baselines, per-SNR tables, fresh seeds, latency).
Write ${OUT}/design/DESIGN.md (via Bash if needed) and criteria.json. Return the design summary (<= 60 lines).`

const BUILD = (design) => `${CTX}

DESIGN (from the previous stage; criteria pre-registered in ${OUT}/design/criteria.json):
${design}

YOUR TASK (folder ${OUT}/build; HEAVY CPU): implement, train and evaluate the top design (and the second one if time allows; total training budget up to ~6 hours CPU; the owner set NO limit on model size or parameter count).
0. WAIT FOR THE CPU FIRST: another workflow (run wf_57f76d2f-fc7) may still be evaluating/fine-tuning. Before starting any training, poll (one bash call with an internal until-loop, up to ~10 min per call, repeat as needed) until BOTH: (a) the file ${DR}/REPORT.md exists (that workflow's final synthesis) OR no python process whose command line or working directory contains 'deep_research' (other than keepawake.py and your own) has been running for 15 minutes (check with PowerShell Get-CimInstance Win32_Process); and (b) free RAM > 2.5 GB. While waiting you may write and smoke-test code with 1 thread and tiny batches.
1. Implement the decoder(s) as specified (reuse exp.py/exp3.py/exp5.py helpers where sensible; new files only in your folder). Unit-test: gradients flow, outputs sane, identical inputs give identical outputs, a classical-equivalent configuration reproduces the classical decoder's numbers on 200 official scenes.
2. Train DETACHED and resumable (seeds 70,000,000+; validation set seeds 71,000,000+ for checkpoint selection only). Log curves.
3. Evaluate per the plan: full official bank (8000) per SNR; T2_d5; ADC b=1..4 at 0/15 dB (stress seeds 5000/5015 via stress_sim, plus fresh 9500+); impairment cells (M3 phase 20/45 deg, gain 6 dB, 30 deg + 6 dB; M6 20/30 deg; M2 0.1/0.2); variable-L (L1..L10 cached banks if the design supports unknown L, else known L) and separation banks sep_2/3/5deg_snr15. Baselines on the same scenes: NOMP, NOMP os8 rc6, classical heat decoder, NOMP + tuned self-cal (lam 1, tau 2, 4 iters), the existing unfolded decoder (lurd.pt), quantisation-consistent NOMP for ADC cells (code in ${DR}/finetune/adc/), the ADC-fine-tuned CNN read-out (finetune/adc/cache/adc_ft.pt) where relevant, U-Net (cached, official/T2 only). Paired bootstrap vs NOMP and vs the best classical per cell. Latency (1 thread, batch 1 and 256).
4. Apply the pre-registered criteria literally and state PASS/FAIL for each.
Write ${OUT}/build/REPORT.md (via Bash if needed), results.json, figures (fig_pd_vs_snr.png official; fig_regimes.png), scripts, logs; checkpoints small on D: (< 20 MB) else on C:. Return a concise summary (<= 80 lines): architecture, training, official per-SNR table vs NOMP, regime table vs best classical, criteria verdicts, latency.`

const ADV = (b) => `${CTX}

CLAIMS TO ATTACK:
${b}

YOUR TASK (folder ${OUT}/adversarial): try to REFUTE every claim that the new unfolded decoder beats classical methods. Default REFUTED if not reproducible. 1) Read ${OUT}/build scripts: no ground truth at inference; training/validation/test seeds disjoint (prove from code); no test bank used for model selection or thresholds; same Y and L for all methods; fair baselines (NOMP os8 rc6, tuned self-cal active, consistent NOMP correct); criteria.json written before the test runs (check timestamps/logs). 2) Re-evaluate the headline claims with your own short script on FRESH seeds (9600-9699; 1000-2000 scenes per cell) including the full official bank. 3) Check whether a simple classical change gives the same gain (e.g. NOMP multi-start with residual selection, MAP penalty with the same prior, more oversampling/rounds) - if a classical variant matches it, say so with numbers. 4) Check the prior-violation banks and say whether the gain depends on the benchmark's data prior. Write ${OUT}/adversarial/VERDICT.md (via Bash if needed) + logs. Return a verdict per claim (CONFIRMED / REFUTED / NEEDS REWORDING) with replicated numbers.`

const REP = (lit, diag, design, build, adv) => `${CTX}

STAGE RESULTS:
--- literature ---
${lit}
--- headroom diagnosis ---
${diag}
--- design ---
${design}
--- build / evaluation ---
${build}
--- adversarial ---
${adv}

YOUR TASK: write ${OUT}/REPORT.md (via Bash if needed) for the project owner in plain direct English: (1) the question and short answer; (2) why the first unfolded decoder only re-learned Newton; (3) headroom diagnosis; (4) the new design(s) and what makes them adaptive; (5) results vs NOMP and the best classical method (official per SNR, regimes), only numbers that survived the adversarial check marked as such; (6) pre-registered criteria verdicts; (7) what can and cannot be claimed in the paper, with exact wording (credit NOMP; say "unfolded from NOMP"); (8) next steps; (9) file index. Also append a short section "Unfolded decoder v2" to ${DR}/RESULTS_SUMMARY.md (add it at the end with Bash; do not rewrite the rest of that file). Return sections 1, 5, 6 and 7 (<= 100 lines).`

const n = (x) => x || '(stage returned nothing - read its folder)'
phase('Research')
const [lit, diag] = await parallel([
  () => agent(LIT, { label: 'literature: beating ML with unfolding', phase: 'Research' }),
  () => agent(DIAG, { label: 'headroom diagnosis (light CPU)', phase: 'Research', effort: 'high' }),
])
phase('Design')
const design = await agent(DESIGN(n(lit), n(diag)), { label: 'design + pre-registered criteria', phase: 'Design', effort: 'high' })
phase('Build')
const build = await agent(BUILD(n(design)), { label: 'build, train, evaluate', phase: 'Build', effort: 'high' })
phase('Adversarial')
const adv = await agent(ADV(n(build)), { label: 'adversarial: unfolded v2', phase: 'Adversarial', effort: 'high' })
phase('Report')
const rep = await agent(REP(n(lit), n(diag), n(design), n(build), n(adv)), { label: 'unfolded v2 report', phase: 'Report', effort: 'high' })
return { lit, diag, design, build, adv, report: rep }
