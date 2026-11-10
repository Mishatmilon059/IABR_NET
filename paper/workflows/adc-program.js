export const meta = {
  name: 'adc-program',
  description: 'Low-resolution-ADC program: read and explain the nearest papers, re-implement the published baselines faithfully, design and train a network that tries to beat all of them, evaluate comprehensively, verify adversarially, write a presentation pack',
  phases: [
    { title: 'Papers', detail: 'one reader per key paper + a sweep for newer work; plain-language takeaways with verified quotes' },
    { title: 'Synthesis', detail: 'takeaways document + positioning matrix + list of baselines that must be beaten' },
    { title: 'Baselines', detail: 'faithful re-implementations (GNOMP, Mo-Schniter-Heath GAMP, Bussgang+NOMP, SignalNet-style) with reproduction checks' },
    { title: 'Design', detail: 'three independent designs, judge, pre-registered criteria' },
    { title: 'Build', detail: 'implement + train (detached, resumable, memory-light)' },
    { title: 'Evaluate', detail: 'full grid vs every baseline, paired CIs, latency, criteria verdicts' },
    { title: 'Adversarial', detail: 'baseline fidelity/fairness, generalisation/prior-learning, independent reproduction' },
    { title: 'Pack', detail: 'presentation pack: paper takeaways, positioning, verified results, slide outline' },
  ],
}

const DR = 'D:/ai_ml_project/paper/evidence/deep_research'
const OUT = 'D:/ai_ml_project/paper/evidence/adc_program'
const BIG = 'C:/Users/Milon/iabr_deep_research_cache/adc_program'
const CTX = `
PROJECT (read ${DR}/RESULTS_SUMMARY.md first; it holds every verified number):
- Joint AoA/AoD estimation, analog mmWave MIMO, 16x16 ULAs, 16x16 DFT beam codebooks; beamspace measurement Y = W^H H F + Z (16x16 complex); antenna domain H~ = W Y F^H; L = 3 paths in the benchmark (Lloria et al., IEEE TVT 2026); SNR -10..25 dB. Metric: paper-style Pd (angle components within 1 deg after Hungarian matching; mean over SNR groups), strict Pd, RMSE. Classical reference: NOMP (Mamandipoor et al., IEEE TSP 2016).
- LOW-RESOLUTION ADC MODEL (stress_sim.adc, ${DR}/stress/stress_sim.py): per-scene peak AGC A = max(|Re|,|Im|) over the 256 beam samples, b-bit uniform mid-rise quantiser on Re and Im separately (step 2A/2^b). Code that already exists and must be reused where sensible: ${DR}/finetune/adc/ (acommon.py: adc_cells etc.; ADC-fine-tuned CNN cache/adc_ft.pt; consistent NOMP), ${DR}/finetune/adversarial/ (mlq.py: quantiser maximum likelihood with exact cell probabilities, Levenberg-Marquardt, 13 classical starts + EM; s3_adc_test.py/log; s6_latency.json; s9_hybrid_rep.log), ${DR}/unfolded_v2/build/ (ucore.py unfolded core, mlt.py torch quantised ML, g0_gate.log, g0_pool12.log, train logs, deviations.txt), ${DR}/unfolded_v2/design/ (DESIGN.md, criteria.json), ${DR}/unfolded_v2/lit/unfolding_survey.md (41 verified references), ${DR}/unfolded_v2/diag/REPORT.md.
- VERIFIED ADC RESULTS SO FAR (fresh seeds, 1000 scenes/cell, Pd): 1-bit/15 dB: NOMP 0.191, consistent NOMP 0.237, ADC-CNN read-out alone 0.419, quantiser-ML (13 starts + EM) 0.540, hybrid (quantiser-ML from CNN seeds) 0.572 (+0.032* vs ML, replicated +0.028/+0.036), ML from TRUE angles 0.626. 2-bit/15 dB: 0.289 / 0.451 / 0.593 / 0.682 / 0.687 (tie) / 0.719. 1-bit/0 dB: 0.163 / 0.226 / 0.239 / 0.275 / 0.274 / 0.439. 2-bit/0 dB: 0.373 / 0.422 / 0.438 / 0.476 / 0.471 / 0.540. Latency ms/scene (1 thread): NOMP 5.8, CNN 6.2, consistent NOMP 76.9, ML 1 start 93.8, ML all starts 482.4. ML loses 0.04-0.07 when its assumed SNR is wrong by 5 dB; at 1 bit the noise level is not identifiable from the data.
- UNFOLDED-V2 FINDINGS (${DR}/unfolded_v2/build/): a candidate pool (greedy, FFT triples, CNN seeds, branches; 12 LM polish layers) has an ORACLE pick +0.036 to +0.060* above 13-start ML in the four ADC cells, but lowest-NLL selection only reaches about ML's level; the learned parts (hypernet, fusion, step MLP, selector, stop head) did NOT improve validation over their initialisation (validation 1200 scenes; front-end drift; a zero-gradient saddle found and fixed; S3 never run); test evaluation never run. Learn from deviations.txt and the train logs.
RULES (strict):
- NOTHING WITHOUT A VERIFIED SOURCE: every statement about a paper must be checked in the paper itself (full text: arXiv abs/html/pdf via WebFetch - load with ToolSearch "select:WebSearch,WebFetch"; if a page is blocked try the built-in browser tools via ToolSearch "+Claude_Browser navigate get_page_text"); quote <= 15 words with section/page; mark UNVERIFIED anything you could not open. Every number about our system must come from a script you ran (stdout saved as .log) or from the verified reports. Never invent.
- Credit NOMP, GNOMP, quantiser ML, self-calibration etc. by name; a learned decoder derived from an algorithm is "unfolded from <algorithm>". Report null and negative results plainly.
- RESOURCES: CPU only (8 logical CPUs), 7.7 GB RAM of which often only ~0.8-1.2 GB is free (desktop apps). Before any heavy run check free RAM (PowerShell (Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory); if < 1.5 GB run memory-light (torch/numpy threads 2, chunks <= 100 scenes); never two heavy python processes at once (check running python processes first). Long jobs (> 10 min) run DETACHED via PowerShell Start-Process -FilePath python -ArgumentList '-u','<abs script>',... -WorkingDirectory '<abs dir>' -WindowStyle Hidden -RedirectStandardOutput '<abs log>' -RedirectStandardError '<abs err>' -PassThru, with checkpoints/partial results and resume; wait by polling (one bash call with an internal until-loop, <= 10 min per call).
- DISK: D: has < 100 MB free: on D: write only small text/json/png (< 300 KB each) inside ${OUT}/<your folder>; everything else (caches, checkpoints, data, PDFs) goes to ${BIG}/<your folder>/ (C: has ~5 GB free - keep your total < 1 GB). Never delete anything outside your own folders. No git.
- If the Write tool refuses a .md file, write it with Bash (python or heredoc). Never skip the written report.
- SEEDS (disjoint from everything used before): training 80,000,000+; validation 81,000,000+; test banks 12000-12999; adversarial fresh banks 13000-13999. Paired bootstrap (2000 resamples within SNR groups, seed 0); a claim needs the 95% CI to exclude 0.`

const PAPER_TASK = (key, ref, focus) => `${CTX}

YOUR TASK (folder ${OUT}/papers; write ${OUT}/papers/${key}.md with Bash; PDFs/HTML copies, if you save any, go to ${BIG}/papers/): read this paper in FULL and explain it so that an undergraduate who finds it hard can present it confidently:
${ref}
Focus: ${focus}
Write these sections:
1. One-paragraph plain-English summary (no jargon).
2. The problem and the system model; a table mapping their symbols to our project's (Y, W, F, H, L, b, A, SNR, u, v).
3. The method step by step in plain words, with the few key equations each explained in one sentence; pseudo-code detailed enough to re-implement (if it is an algorithm).
4. Assumptions (known noise variance? known number of paths? dithering? AGC? snapshots? grid?).
5. Reported results with exact numbers, the figure/table they come from, and the conditions.
6. 4-8 citable claims, each quoted (<= 15 words) with section/page, and what we may cite it for.
7. Limitations / what the paper does not show.
8. Relevance to us: what to reuse, whether it must be a baseline (and how to make the comparison fair: same metric, same setting), and what it implies for our network design.
9. "If a reviewer asks..." - 3 likely questions about this paper vs our work, with honest answers.
10. Glossary of the paper's terms in plain English.
Mark anything you could not verify as UNVERIFIED. Return a concise summary (<= 40 lines) with the verified bibliographic entry (authors, title, venue, vol, pages, year, DOI, arXiv).`

const PAPERS = [
  { key: 'gnomp', label: 'paper: GNOMP (few-bit NOMP)', ref: 'J. Zhu, H. Zhang, N. Zhang, J. Fang, F. Qu, "Line Spectrum Estimation and Detection With Few-Bit ADCs: Theoretical Analysis and Generalized NOMP Algorithm," IEEE Trans. Aerospace and Electronic Systems, vol. 62, pp. 3933-3951, 2026, DOI 10.1109/TAES.2025.3649136, arXiv 2307.00491.', focus: 'This is the strongest classical competitor for our ADC result. Extract the full algorithm (probit/cell-probability likelihood, Rao-test detection on an FFT grid, Newton refinement on the quantised likelihood, cyclic refinement, CFAR stopping, the unknown-noise-variance variant) precisely enough to re-implement, the theoretical analysis (CRB / detection SNR loss per bit: 4.8 / 2.3 / 0.9 dB at 1/2/3 bits as recorded in our survey - verify), whether it handles 2-D (multi-dimensional) frequencies, dithering, AGC, and every experimental setting so we can reproduce one of their results.' },
  { key: 'mo2018', label: 'paper: Mo-Schniter-Heath few-bit mmWave', ref: 'J. Mo, P. Schniter, R. W. Heath Jr., "Channel Estimation in Broadband Millimeter Wave MIMO Systems With Few-Bit ADCs," IEEE Trans. Signal Processing, vol. 66, no. 5, pp. 1141-1154, 2018, DOI 10.1109/TSP.2017.2781644, arXiv 1610.02735 (verify the exact author list).', focus: 'mmWave few-bit channel estimation with EM-GM-GAMP / EM-VAMP on a sparse angle-delay grid with an exact quantised likelihood; the training/pilot design; what they report vs bits/SNR; how to adapt it to our narrowband 16x16 beamspace model as a baseline (grid size, prior, peak extraction to angles).' },
  { key: 'lordnet', label: 'paper: LoRD-Net (unfolded low-res receiver)', ref: 'S. Khobahi, N. Shlezinger, M. Soltanalian, Y. C. Eldar, "LoRD-Net: Unfolded Deep Detection Network With Low-Resolution Receivers," IEEE Trans. Signal Processing, vol. 69, pp. 5651-5664, 2021, DOI 10.1109/TSP.2021.3117503, arXiv 2102.02993.', focus: 'How unfolding gradient steps of the relaxed one-bit ML with a learned preconditioner and a learned surrogate model beats the non-learned ML; the training procedure (two-stage), parameter count, what made it work; exactly which ideas transfer to angle estimation from quantised beamspace data.' },
  { key: 'signalnet', label: 'paper: SignalNet (low-res sinusoid network)', ref: 'R. M. Dreifuerst, R. W. Heath Jr., "SignalNet: A Low Resolution Sinusoid Decomposition and Estimation Network," IEEE Trans. Signal Processing, vol. 70, pp. 4454-4467, 2022, DOI 10.1109/TSP.2022.3201336, arXiv 2106.05490.', focus: 'Architecture, training data, baselines, results at 1/2/3 bits, and especially the "worst-case learning threshold" (what it is, how to compute it, why the network often fails to exceed it at 1 bit) so we can apply the same test to our ADC network; how to build a SignalNet-style learned baseline for our 2-D problem.' },
  { key: 'onebit_unrolled', label: 'paper: one-bit DOA unrolled (Hu-Sun-Zhang)', ref: 'Y. Hu, S. Sun, Y. D. Zhang, "Model-Based Learning for DOA Estimation with One-Bit Single-Snapshot Sparse Arrays," arXiv 2502.17473 (accepted to IEEE JSTSP per arXiv; verify).', focus: 'The MAP model with logistic one-bit likelihood, the unrolled MM network (SBRI-X-Net), what it learns, results vs its iterative parent across SNR, generalisation outside the training SNR range; relevance to single-snapshot one-bit angle estimation like ours.' },
  { key: 'valse_ep', label: 'paper: VALSE-EP (gridless quantised LSE)', ref: 'J. Zhu, Q. Zhang, X. Meng, "Grid-less Variational Bayesian Inference of Line Spectral Estimation from Quantized Samples," arXiv 1811.05680 (China Communications 2021 per our survey; verify venue).', focus: 'How expectation propagation turns quantised data into pseudo-unquantised data with heteroscedastic noise; model-order estimation; accuracy vs CRB; computational cost; whether it is a practical baseline for our 16x16 2-D case.' },
  { key: 'bussgang_power', label: 'papers: Bussgang one-bit MIMO + ADC power motivation', ref: '(a) Y. Li, C. Tao, G. Seco-Granados, A. Mezghani, A. L. Swindlehurst, L. Liu, "Channel Estimation and Performance Analysis of One-Bit Massive MIMO Systems," IEEE Trans. Signal Processing, 2017, arXiv 1609.07427. (b) R. H. Walden, "Analog-to-digital converter survey and analysis," IEEE JSAC, vol. 17, no. 4, pp. 539-550, 1999 (check what can be verified: abstract, figure of merit; mark the rest UNVERIFIED). (c) J. Mo, R. W. Heath Jr., "Capacity Analysis of One-Bit Quantized MIMO Systems with Transmitter Channel State Information," arXiv 1410.7353 (verify venue).', focus: 'The Bussgang decomposition explained simply (why a quantiser = scaled signal + uncorrelated distortion, and why that distortion breaks least-squares methods such as NOMP); the BLMMSE estimator (a classical baseline to implement: Bussgang de-quantisation then NOMP); and the verified motivation for low-resolution ADCs (power vs bits) - only what the sources actually state.' },
  { key: 'foundations', label: 'papers: NOMP, CFAR-NOMP, GAMP, unrolling', ref: '(a) B. Mamandipoor, D. Ramasamy, U. Madhow, "Newtonized Orthogonal Matching Pursuit: Frequency Estimation Over the Continuum," IEEE TSP 64(19):5066-5081, 2016, arXiv 1509.01942. (b) CFAR based NOMP, arXiv 2210.10302 (verify authors/venue). (c) S. Rangan, "Generalized Approximate Message Passing for Estimation with Random Linear Mixing," ISIT 2011, arXiv 1010.5141. (d) V. Monga, Y. Li, Y. C. Eldar, "Algorithm Unrolling," IEEE SPM 38(2):18-44, 2021, arXiv 1912.10557.', focus: 'Compact takeaways of each (one section per paper, same structure but shorter): what each contributes to our pipeline (NOMP = our classical decoder; CFAR stopping = unknown L; GAMP = the engine of the Mo et al. baseline; unrolling = the principle of our learned decoder), citable claims with quotes.' },
  { key: 'sweep', label: 'sweep: newest learned few-bit estimators', ref: 'No fixed paper: search (WebSearch/WebFetch, Google Scholar-like queries on arXiv, IEEE Xplore abstracts) for 2021-2026 work on deep learning / unfolding for angle (DOA/AoA/AoD) or channel-parameter estimation with 1-4-bit ADCs, especially mmWave and single-snapshot, and any work that compares against GNOMP or quantised ML.', focus: 'Find the 6-12 nearest learned works; for each: verified citation, what it estimates, data model (snapshots, array, bits, known noise?), method, baselines it beat, its numbers, and whether it would be a STRONGER baseline than those we have. Conclude: what is the real state of the art we must beat, and is any of it reproducible on our problem within CPU limits. Write this as ${OUT}/papers/sweep.md (sections per work + a ranked summary table).' },
]

const SYN = (paperSummaries) => `${CTX}

PAPER READER SUMMARIES:
${paperSummaries}

YOUR TASK (folder ${OUT}/synthesis; read every ${OUT}/papers/*.md in full): write two documents with Bash:
(1) ${OUT}/synthesis/TAKEAWAYS.md - for the project owner to present: per paper, a half-page plain-English takeaway (what problem, what idea, what result, why it matters to us, one verified quote), then "the big picture of low-resolution-ADC estimation in 10 bullet points", then a glossary. Only verified content.
(2) ${OUT}/synthesis/POSITIONING.md - a comparison matrix (rows = papers + our previous hybrid + our planned network; columns = task, data model, bits, snapshots, known noise/L, method family, learned?, reported accuracy, cost, code availability), the exact gap our work can fill (stated cautiously and only where the evidence supports it), the list of baselines we MUST implement to claim we beat the state of the art (with priority and feasibility on our CPU), and the fair-comparison rules (same data, same metric, matched information).
Return the list of mandatory baselines and the gap statement (<= 50 lines).`

const BASE = (b, syn) => `${CTX}

SYNTHESIS (mandatory baselines and gap):
${syn}

YOUR TASK (folder ${OUT}/baselines/${b.key}; caches in ${BIG}/baselines/${b.key}/): ${b.task}
General requirements: (i) implement faithfully from the paper notes ${OUT}/papers/*.md and the paper itself - list every deviation and why; (ii) REPRODUCTION CHECK: reproduce at least one quantitative result or qualitative trend from the original paper in its own setting (small scale is fine) and report agreement; (iii) adapt to our 16x16 beamspace model; tune any free constants ONLY on validation seeds 81,000,000+; (iv) evaluate on test banks seeds 12000+ for bits {1,2,3,4} x SNR {-10,0,10,15,25} dB, L=3, 300 scenes per cell (more if cheap), save per-scene predictions to the cache in the common format {psi, phi, n} and per-scene stats via new_results/common.py scene_stats; (v) latency per scene (1 thread, batch 1). Write ${OUT}/baselines/${b.key}/REPORT.md (Bash) with tables + deviations + reproduction check. Return a concise summary (<= 40 lines).`

const BASELINES = [
  { key: 'gnomp', label: 'baseline: GNOMP (faithful)', task: 'Implement GNOMP (few-bit NOMP with the cell-probability likelihood, Rao-test detection on an oversampled FFT grid, Newton refinement on the quantised likelihood, cyclic refinement, both known-L and CFAR-stopping modes, and the unknown-noise-variance variant if specified). Extend to our 2-D (u, v) frequency model if the paper is 1-D (state exactly how). Compare against our existing quantiser-ML (mlq.py) and consistent NOMP on the same scenes.' },
  { key: 'gamp', label: 'baseline: Mo-Schniter-Heath GAMP', task: 'Implement the few-bit sparse channel estimator of Mo, Schniter & Heath (EM-GM-GAMP or EM-VAMP with the exact quantised output channel) on an oversampled 2-D angle grid matched to our beamspace model, then extract L angle pairs (peak picking + optional off-grid refinement, documented). If the full EM version is infeasible on this CPU, implement GAMP with a Bernoulli-Gaussian prior and EM-tuned parameters and say so.' },
  { key: 'bussgang', label: 'baseline: Bussgang LMMSE + NOMP', task: 'Implement Bussgang-based de-quantisation (BLMMSE-style linear estimate of the unquantised beamspace samples given the known quantiser) followed by standard NOMP (os4 rc3) and by NOMP os8 rc6; this is the simplest classical fix a reviewer will ask about.' },
  { key: 'signalnet', label: 'baseline: SignalNet-style learned', task: 'Build and train a SignalNet-style learned estimator adapted to our 2-D quantised beamspace input (follow the paper architecture as closely as feasible; same training data budget as our ADC CNN: bits 1-4, SNR -15..25, L 1..9, seeds 80,000,000+), evaluate it, and apply SignalNet\'s worst-case learning threshold test to BOTH this model and our existing ADC CNN (finetune/adc/cache/adc_ft.pt) to check whether either merely learns the prior at 1 bit.' },
]

const DESIGNER = (angle, syn, bsum) => `${CTX}

SYNTHESIS:
${syn}

BASELINE RESULTS:
${bsum}

YOUR TASK (folder ${OUT}/design; write ${OUT}/design/proposal_${angle.key}.md with Bash; NO heavy CPU): propose a network design for 1-4-bit ADC joint AoA/AoD estimation that can beat EVERY baseline above (GNOMP, quantiser ML multi-start, Mo-GAMP, Bussgang+NOMP, SignalNet-style, our previous hybrid) comprehensively, from this angle: ${angle.desc}
Every design choice must cite its source (a verified paper in ${OUT}/papers/ or our own verified evidence) or be marked as our own new idea. Include: architecture (exact layers, learned vs fixed, parameter count), inputs (incl. the known quantiser cells and bits), how it handles unknown noise level and unknown L, training data mix and losses, the training schedule sized for this CPU (RAM often ~1 GB free; see RULES), why it should beat each baseline (mechanism + evidence + expected size of the gain), risks, and how the unfolded-v2 training failures (no learning beyond initialisation; zero-gradient saddle; front-end drift; small validation) are avoided. Return the proposal summary (<= 50 lines).`

const ANGLES = [
  { key: 'select', desc: 'SELECTION-CENTRIC: CNN seeds + a rich candidate pool (classical + learned) + a learned selector trained on oracle labels to capture the measured +0.036..+0.060 pool-oracle headroom, with GNOMP / quantised-LM polishing.' },
  { key: 'unfold', desc: 'UNFOLDED QUANTISED LIKELIHOOD: unfold GNOMP / quantised LM from CNN seeds with learned preconditioners (LoRD-Net idea), learned damping/annealing and a learned noise-level estimate (SNR-blind), trained end-to-end with a detection-oriented loss.' },
  { key: 'robust', desc: 'NETWORK-FIRST ROBUSTNESS: a stronger front-end network (trained on bits 1-4, SNR range, unknown L, quantiser variants such as clipping AGC and dither) that outputs seeds plus uncertainty and the path count, followed by a short deterministic quantised-likelihood polish; emphasis on generalisation and on passing the SignalNet learning-threshold test.' },
]

const JUDGE = (props, syn, bsum) => `${CTX}

SYNTHESIS:
${syn}

BASELINES:
${bsum}

PROPOSALS:
${props}

YOUR TASK (folder ${OUT}/design; Bash for files): act as the design judge. Score each proposal (expected gain over the strongest baseline per cell, evidence quality, CPU feasibility, risk, novelty vs the papers) and write ${OUT}/design/FINAL_DESIGN.md: the chosen design (graft the best ideas from the others), exact architecture, training plan, ablations that isolate each component's contribution, and the evaluation plan. Then write ${OUT}/design/criteria.json BEFORE any training or test run (timestamp inside), with: the test grid (bits {1,2,3,4} x SNR {-10,-5,0,5,10,15,20,25} dB at L=3; unknown-L banks L in {1..6}; SNR-mismatch +-5 dB for methods that need sigma; quantiser mismatch: dithered and +-3RMS-clipping AGC; prior violation: no minimum separation; ADC + per-antenna errors (8 deg / 3 dB); the official ideal bank as a no-regression check), the baselines, the success labels (FULL = significantly better than the best baseline in >= 75% of the 1-2-bit cells and never significantly worse in any 1-4-bit cell; PARTIAL; NULL), the latency budget, and the honesty checks (learning-threshold test, OOD tests). Return the design summary + criteria (<= 60 lines).`

const BUILD = (design) => `${CTX}

FINAL DESIGN AND CRITERIA (${OUT}/design/FINAL_DESIGN.md, criteria.json):
${design}

YOUR TASK (folder ${OUT}/build for small files, ${BIG}/build for checkpoints/data): implement and train the final design. Unit-test first (gradients reach every learned module; a classical configuration reproduces the corresponding classical baseline on 100 validation scenes; determinism). Train DETACHED and resumable (seeds 80,000,000+; validation set >= 3000 scenes from seeds 81,000,000+, used only for checkpoint selection). Monitor learning: if validation does not improve over initialisation within the first 20% of the budget, DIAGNOSE (gradient norms per module, ablate modules, check for saddles/drift as in unfolded-v2 deviations.txt) and apply at most two documented fixes before continuing. Total training budget up to ~8 hours of CPU; keep memory-light when free RAM is low. Write ${OUT}/build/REPORT.md (Bash): what was built (parameter counts), training curves (figure < 300 KB), validation results vs the classical configuration, every deviation from FINAL_DESIGN.md. Do NOT touch test banks. Return a concise summary (<= 40 lines).`

const EVAL = (build) => `${CTX}

BUILD RESULT:
${build}

YOUR TASK (folder ${OUT}/eval; caches in ${BIG}/eval): run the full pre-registered evaluation in ${OUT}/design/criteria.json (do not edit it; amendments go to criteria_amendments.json with reasons, never after seeing test results). Every baseline from ${OUT}/baselines/* and the previous hybrid on the SAME test scenes (seeds 12000+), the new network and its ablations; paired bootstrap vs each baseline and vs the best baseline per cell; per-cell tables, a summary heatmap of differences (figure < 300 KB), latency (1 thread, batch 1 and batch 64), and the literal criteria verdicts (FULL / PARTIAL / NULL). Run detached in parts with resume; order: 1-2-bit cells first, then 3-4 bits, then mismatch/OOD, then official bank. Write ${OUT}/eval/REPORT.md (Bash) and results.json. Return the verdicts and the key tables (<= 60 lines).`

const ADV = (lens, ev) => `${CTX}

EVALUATION RESULT TO ATTACK:
${ev}

YOUR TASK (folder ${OUT}/adversarial/${lens.key}; Bash for files): try to REFUTE the evaluation from this angle: ${lens.desc} Default to REFUTED when you cannot reproduce. Write ${OUT}/adversarial/${lens.key}/VERDICT.md with a verdict per claim (CONFIRMED / REFUTED / NEEDS REWORDING + corrected sentence), replicated numbers and any bug (file:line). Return the verdict list (<= 40 lines).`

const LENSES = [
  { key: 'fidelity', desc: 'BASELINE FIDELITY AND FAIRNESS: are GNOMP, Mo-GAMP, Bussgang+NOMP, SignalNet-style and quantiser ML implemented faithfully (compare against the papers and the reproduction checks) and tuned as well as the new network (same information: noise level, L, quantiser; tuning only on validation seeds)? Is any baseline handicapped? Re-run the weakest-looking baseline with a better configuration and report whether the network still wins.' },
  { key: 'generalise', desc: 'GENERALISATION AND PRIOR-LEARNING: does the network merely learn the training prior (SignalNet learning-threshold test), does it hold under the minimum-separation rule being violated, unknown L, SNR outside the training range, different quantiser/AGC, a different angle distribution, and ADC + hardware errors? Run the tests yourself on fresh seeds 13000+.' },
  { key: 'reproduce', desc: 'INDEPENDENT REPRODUCTION: re-run the 6-10 headline numbers with your own short scripts on fresh seeds 13000+ (1000 scenes per cell where feasible), check leakage (training/validation/test seeds disjoint - prove from code), check that criteria.json predates the test runs, and recompute the verdict labels.' },
]

const PACK = (all) => `${CTX}

ALL STAGE RESULTS:
${all}

YOUR TASK (folder ${OUT}; Bash for files): write ${OUT}/PRESENTATION_PACK.md for the project owner, plain English, only verified content and only adversarially CONFIRMED results:
1. ADC in one page (what, why low-bit, what breaks) with citations.
2. The papers: for each, the 5-line takeaway from ${OUT}/synthesis/TAKEAWAYS.md, plus how our work relates.
3. Positioning table (from POSITIONING.md) and the gap.
4. Our method: what it is, what is learned vs classical (credit every algorithm it builds on), why it should work (with sources).
5. Results: the comparison with every baseline, per bits/SNR, with the verdict label, the ablations, latency; what was refuted.
6. Limitations and next steps.
7. A 12-slide outline with talking points and the figure/table to show on each slide.
8. A "how to answer reviewer questions" section.
Also append a short section "Low-resolution ADC program" to ${DR}/RESULTS_SUMMARY.md (append with Bash; do not rewrite the rest). Return sections 1, 5 and 7 (<= 120 lines).`

const n = (x) => x || '(stage returned nothing - read its folder)'
phase('Papers')
const DONE_KEYS = ['valse_ep', 'mo2018']
const RESUME = (k) => `
RESUME NOTE: a previous attempt of this task was cut off by a usage limit (~07:40). Partial work may exist: ${OUT}/papers/${k}.md, scripts/logs in ${OUT}/papers/ whose names start with ${k} or relate to this paper (e.g. repro_*, check_*, signalnet_*, lordnet_check/), and files in ${BIG}/papers/. Read what exists, verify it against the paper, and complete or correct it instead of starting over.`
const papers = await parallel(PAPERS.map(p => () => agent(PAPER_TASK(p.key, p.ref, p.focus) + (DONE_KEYS.includes(p.key) ? '' : RESUME(p.key)), { label: p.label, phase: 'Papers' })))
const paperText = PAPERS.map((p, i) => `--- ${p.key} ---\n${n(papers[i])}`).join('\n')
phase('Synthesis')
const syn = await agent(SYN(paperText), { label: 'takeaways + positioning', phase: 'Synthesis', effort: 'high' })
phase('Baselines')
const bres = []
for (const b of BASELINES) {
  bres.push(await agent(BASE(b, n(syn)), { label: b.label, phase: 'Baselines', effort: 'high' }))
}
const bsum = BASELINES.map((b, i) => `--- ${b.key} ---\n${n(bres[i])}`).join('\n')
phase('Design')
const props = await parallel(ANGLES.map(a => () => agent(DESIGNER(a, n(syn), bsum), { label: `design: ${a.key}`, phase: 'Design', effort: 'high' })))
const propText = ANGLES.map((a, i) => `--- ${a.key} ---\n${n(props[i])}`).join('\n')
const design = await agent(JUDGE(propText, n(syn), bsum), { label: 'design judge + criteria', phase: 'Design', effort: 'high' })
phase('Build')
const build = await agent(BUILD(n(design)), { label: 'build + train', phase: 'Build', effort: 'high' })
phase('Evaluate')
const ev = await agent(EVAL(n(build)), { label: 'full evaluation', phase: 'Evaluate', effort: 'high' })
phase('Adversarial')
const advs = []
for (const l of LENSES) {
  advs.push(await agent(ADV(l, n(ev)), { label: `adversarial: ${l.key}`, phase: 'Adversarial', effort: 'high' }))
}
const advText = LENSES.map((l, i) => `--- ${l.key} ---\n${n(advs[i])}`).join('\n')
phase('Pack')
const all = `--- synthesis ---\n${n(syn)}\n--- baselines ---\n${bsum}\n--- design ---\n${n(design)}\n--- build ---\n${n(build)}\n--- evaluation ---\n${n(ev)}\n--- adversarial ---\n${advText}`
const pack = await agent(PACK(all), { label: 'presentation pack', phase: 'Pack', effort: 'high' })
return { synthesis: syn, baselines: bsum, design, build, evaluation: ev, adversarial: advText, pack }
