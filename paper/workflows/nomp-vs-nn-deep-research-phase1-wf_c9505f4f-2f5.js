export const meta = {
  name: 'nomp-vs-nn-deep-research-phase1',
  description: 'Verify NOMP-only results independently, stress-test NOMP under unmodelled hardware effects, survey literature on DL vs model-based AoA/AoD',
  phases: [
    { title: 'Verify', detail: 'from-scratch NOMP + metric, re-run official and key banks, side-by-side table' },
    { title: 'Literature', detail: 'baselines used by DL AoA/AoD papers; regimes where DL beats model-based' },
    { title: 'Stress', detail: 'mismatch simulator (coupling, position error, low-res PS, ADC, phase noise, unknown L, large impairments) + all methods' },
    { title: 'Adversarial', detail: 'try to refute the stress-test headline claims, fresh-seed replication' },
  ],
}

const DR = 'D:/ai_ml_project/paper/evidence/deep_research'
const CTX = `
PROJECT CONTEXT (read carefully; everything you need is on disk):
- Task: joint AoA/AoD estimation for analog mmWave MIMO, 16x16 ULAs, DFT beam codebooks. Measurement Y (16x16 complex, beamspace) = W^H diag(d_r)^H H diag(d_t) F + Z. Antenna-domain channel Hant = to_ant(Y) (notebook_src.py line 274), u = pi cos(psi) (AoA), v = pi cos(phi) (AoD). Per-antenna gain/phase impairments d_r, d_t; the mean and the linear phase ramp are unidentifiable.
- Source of truth code: D:/ai_ml_project/IABR_v2_extracted/IABR_v2/notebook_src.py. Decoder = lines 528-647 (python slice [527:647]): atoms, ls_gains, newton, cyclic, residual_energy, nomp_greedy, heat_peaks, selfcal, uv_to_angles, decode(Hant, L, heat=None, mode='greedy'|'heat'|'dual', sc=None|(lam,tau), rc=None); parabolic_heat_decode at lines 650-668 (network-only read-out). Model (IABRv2 CNN, 191,702 params) = lines 420-522 (slice [419:522]). metrics() lines 315-343. to_ant line 274. make_bank line 356.
- Simulator: D:/ai_ml_project/IABR_v2_extracted/IABR_v2/iabr2_sim.py (needs os.environ['IABR2_SUITE']='D:/ai_ml_project' before import): sample_paths, simulate(psi,phi,alpha,snr_db,rng,phase_deg_max,gain_db_max), to_ri, W_IDEAL, F_IDEAL.
- Helpers: D:/ai_ml_project/paper/evidence/new_results/common.py (load_bank -> psi,phi,L,snr; scene_stats; metric_by_snr; boot_metrics (paired bootstrap within SNR groups); ci; decoder compiled from the notebook; SC=(1.0,5.0) self-calibration setting). heat_main.py in the same folder = the IABR-Net heat-seeded method (decode(Hc, L, heat, 'heat', SC) on the IABC-corrected channel Hc). D:/ai_ml_project/paper/evidence/learned_decoder/exp.py shows how to build/load the CNN on CPU: exec src[419:522] with namespace (np, torch, nn, TF, W_IDEAL, F_IDEAL, P=16, Q=16), cnn = IABRv2(32,10,32), load outputs/checkpoints/IABRv2/final.pt; input y = to_ri(Y).transpose(0,3,1,2) float32; o = cnn(y); heat = sigmoid(o['heat']) (32x32); o['Hc'] = IABC-corrected antenna-domain channel.
- Caches: D:/ai_ml_project/IABR_v2_extracted/IABR_v2/outputs/cache/net__<bank>.npz (heat, Hc, Ht; Ht = raw antenna-domain channel) and pred__<method>__<bank>.npz (psi, phi, n) for methods 'U-Net_paper', 'IABR-v2' (dual-init hybrid), 'NOMP_no_network' (greedy NOMP on Ht, known L), 'v2_network_only_no_physics', etc. New-results preds: paper/evidence/new_results/preds/selfcal_pred__<bank>.npz (NOMP+selfcal), preds_heat/<bank>.npz (IABR-Net heat-seeded). JSON summaries: new_results/bank_json/*.json (51 banks), bank_json_heat/*.json (21 banks). Narrative: new_results/RESULTS_NEW.md.
- Known official-bank numbers (8,000 scenes, L=3, SNR -10..25 dB step 5, mean over SNR, paper-style Pd): U-Net 0.7366 (strict 0.6940); IABR-v2 dual 0.7551; NOMP no network 0.7581; NOMP+selfcal 0.7576; IABR-Net heat-seeded 0.7503; network only 0.451. The CNN was trained on L in 1..9, SNR -15..24 dB, phase error up to 8 deg, gain error up to 3 dB, 20% clean.
- Metrics: evaluator D:/ai_ml_project/IABR_Net_TestSuite/DL_DOA/src/TVT_Blob_Inference.py (prepare_for_metric, get_ang_difference). Paper-style Pd = fraction of angle components with |error| <= 1 deg, scenes with fewer than L detections dropped from the denominator; strict Pd = those scenes count as misses; RMSE over detected components only.
- Hardware: CPU only (8 logical CPUs, no CUDA), Python 3.12 ('python'), numpy 2.3, scipy 1.17, torch 2.7.1 CPU. Measured NOMP greedy latency ~10 ms/scene single-thread. The U-Net weights exist only as a 7z archive and 7-Zip is not installed, so the U-Net can only be used through its cached predictions on existing banks.
RULES:
- Do NOT modify or delete any existing file. Write only inside your own folder under ${DR}/ . Put large arrays (*.npz, *.pt) in a cache/ subfolder (gitignored). Do not git commit or push.
- Never fabricate or estimate a number: every number you report must come from a script you wrote and ran, with its stdout saved to a .log file next to it. Negative and null results are reported plainly.
- Seeds fixed and recorded. Use paired bootstrap (2000 resamples within SNR groups, seed 0) for differences between methods on the same scenes.
- Be efficient with CPU: vectorise over scenes, batch, avoid needless repeats.`

const VERIFY = `${CTX}

YOUR TASK (folder ${DR}/verify): independently verify the "NOMP-only (no network)" result, and put every method side by side.
1. Write your OWN NOMP implementation from the algorithm description; do NOT import/exec the notebook decoder. Algorithm (Mamandipoor, Ramasamy, Madhow 2016, 2-D): known L; for l=1..L: 4x oversampled 2-D FFT of the residual, take the max cell as (u,v) seed; single-path Newton on |<a(u,v), r>|^2 with atom a[n,m] = exp(-j n u + j m v) (4 steps, Hessian negative-definite guard, step clip 2pi/64); then cyclic re-estimation of all detected paths (3 rounds; each path re-refined against y minus the others with 2 Newton steps, LS gains); LS gains; residual update. Read the notebook decoder only to confirm conventions (sign of u/v, uv->angle map), then write your own code.
2. Write your OWN metric from the TVT evaluator definition (read prepare_for_metric / get_ang_difference, then implement independently) and cross-check it against common.scene_stats/metric_by_snr on the cached predictions of all methods on the official bank (must agree to printed digits; report any mismatch).
3. Input: the raw antenna-domain channel Ht from net__<bank>.npz. Also check on 200 official scenes that Ht equals to_ant(Y) recomputed from the frozen bank (D:/ai_ml_project/IABR_Net_TestSuite/data/frozen_banks/eval_bank.npz, see notebook load_bank line 383: Y16 = downsample16(data)).
4. Run your NOMP on the full official bank (8,000 scenes) and on: T2_d5, gain_g4_snr15, oodphase_d15_snr15, L8_snr0, L8_snr15, sep_5deg_snr15, sep_2deg_snr15, phase_d0_snr0, phase_d0_snr15. Report per-scene agreement with the cached NOMP_no_network predictions (fraction of scenes with max angle difference < 1e-6 rad, < 0.01 deg) and the metrics.
5. Side-by-side table per bank (paper-style Pd, strict Pd, RMSE, with 95% bootstrap CI): U-Net (cached), IABR-v2 dual (cached), IABR-Net heat-seeded (new_results/preds_heat if present), network only (cached), NOMP (cached), NOMP (yours), NOMP+selfcal (new_results/preds). Plus paired differences: IABR-Net heat - NOMP, IABR-v2 dual - NOMP, NOMP - U-Net, network only - NOMP. Per-SNR table for the official bank.
6. Headroom: on the official bank, oracle-seeded decoder (true (u,v) as seeds -> Newton + cyclic, known L) to show how far NOMP is from what this objective can reach. Report the Pd gap NOMP vs oracle-seeded per SNR.
7. Latency on CPU, single thread (torch.set_num_threads(1), OMP/OPENBLAS threads 1 via env before import): (a) NOMP greedy per scene, (b) CNN forward + heat-seeded decode per scene, (c) CNN forward alone; batch 1 and batch 256; median of >= 50 runs.
Write: verify/REPORT.md (all tables, method notes, exact commands), verify/verify.json, scripts, logs.
Return a concise markdown summary (<= 60 lines) with the key tables and whether the NOMP-only claim is confirmed.`

const LIT1 = `${CTX}

YOUR TASK (folder ${DR}/lit, file lit/baselines_survey.md): literature research with VERIFIED citations (open each source with WebFetch/WebSearch; load those tools via ToolSearch with query "select:WebSearch,WebFetch"). No CPU-heavy work.
Question: Why do recent deep-learning AoA/AoD papers not compare against (or not acknowledge) NOMP / gridless super-resolution, and do any papers compare DL against NOMP?
1. Identify the exact citations of the benchmark papers this project uses: Lloria et al. 2026 (U-Net/blob-detection joint AoA/AoD in mmWave with phase/gain impairments; the TVT evaluator 'TVT_Blob_Inference' suggests IEEE Trans. Veh. Technol.) and Meneses-Albala et al. 2026 (fine-tuned U-Net, energy). Check local file D:/ai_ml_project/paper/latex/references.bib first.
2. Survey >= 15 DL papers (2018-2026) on AoA/AoD / DOA / channel-parameter estimation for mmWave or massive MIMO (beamspace, hybrid/analog beamforming, super-resolution). For each: task, DL model, classical baselines compared (on-grid OMP, MUSIC, ESPRIT, SBL, ANM/atomic norm, NOMP/Newtonized OMP, CRB), whether a gridless super-resolution method is included, whether baselines got matched side information (known L, noise variance), claimed gain. Make a table.
3. Find papers that DO compare a learned method against NOMP (mmWave channel estimation, frequency estimation e.g. DeepFreq (Izacard, Mohan, Fernandez-Granda 2019), cResFreq, learned NOMP / unfolded NOMP, etc.) and what they found.
4. Evidence-based reasons for the omission: e.g., comparing only to on-grid OMP; assuming NOMP/ANM too slow; different measurement models; critiques of weak baselines in DL-for-signal-processing (find any published critique or position paper). Distinguish clearly between what a source states and your own inference.
Return: a markdown summary (<= 80 lines) with the table and the list of references (title, authors, venue, year, URL/DOI, verified yes/no, the specific claim each supports).`

const LIT2 = `${CTX}

YOUR TASK (folder ${DR}/lit, file lit/when_dl_helps.md): literature research with VERIFIED citations (load WebSearch/WebFetch via ToolSearch "select:WebSearch,WebFetch" and open each source). No CPU-heavy work.
Question: Under which conditions do data-driven or hybrid (model-based deep learning) estimators genuinely beat model-based gridless estimators such as NOMP / MUSIC / ESPRIT / atomic-norm for angle / line-spectral estimation?
Cover with specific papers and numbers: (a) array imperfections, mutual coupling, antenna position errors (e.g. Liu, Zhang, Yu 2018 IEEE TAP DNN DOA robust to array imperfections; others); (b) unknown number of paths / model-order estimation (DeepFreq counting module; DL source enumeration); (c) coherent / closely spaced sources (SubspaceNet, Shmuel et al.); (d) 1-bit / low-resolution ADCs and low-resolution phase shifters in mmWave; (e) phase noise; (f) near-field; (g) deep unfolding / algorithm unrolling of greedy or Newton-type estimators (LISTA, unfolded OMP/NOMP, learned AMP for channel estimation); (h) latency / complexity arguments; (i) learned initialisation + model-based refinement hybrids. For each: what the NN does better, by how much, under what conditions, and whether the baseline was a gridless super-resolution method with matched information.
Also find the framing references: Shlezinger, Whang, Eldar, Dimakis "Model-based deep learning" (Proc. IEEE 2023); Monga, Li, Eldar "Algorithm unrolling" (IEEE SPM 2021); any survey on DL for DOA estimation.
End with 3-5 concrete, defensible journal framings for a hybrid "CNN heatmap + model-based refinement + self-calibration" AoA/AoD estimator, given the honest fact that a network-free NOMP matches it under the nominal per-antenna gain/phase impairment model. Each framing must name the experiment that would support it.
Return: a markdown summary (<= 80 lines) plus the reference list (title, authors, venue, year, URL/DOI, verified yes/no, claim supported).`

const STRESS = (verSummary) => `${CTX}

VERIFICATION RESULT FROM THE PREVIOUS STAGE (for context):
${verSummary}

YOUR TASK (folder ${DR}/stress): find where NOMP breaks. Build a mismatch simulator and run every method on it. The decoder's model assumes: ideal DFT codebooks, per-antenna diagonal gain/phase distortion, plane waves on an ideal half-wavelength ULA, unquantised measurements, a common phase across the 256 measurements, and known L. Break each assumption:
 M1 mutual coupling: banded symmetric Toeplitz C with C[i,i]=1, C[i,i+-1]=c1, C[i,i+-2]=c2, c1 = rho*exp(j*theta1), c2 = (rho/2)*exp(j*theta2), theta ~ U(0,2pi) per scene, independent for Rx and Tx; effective channel C_r H C_t^T. rho in {0.05, 0.1, 0.2, 0.3}.
 M2 antenna position errors: element n at (n + delta_n) half-wavelengths, delta_n ~ N(0, sigma^2), sigma in {0.02, 0.05, 0.1, 0.2}, both arrays, per scene (angle-dependent; not a diagonal distortion).
 M3 large per-antenna impairments in the existing model (outside the CNN training range): phase 20, 30, 45 deg; gain 6 dB; phase 30 deg + gain 4 dB.
 M4 low-resolution phase shifters: W and F phases quantised to b bits (b = 3, 2, 1) when generating Y; the decoder/to_ant still assume the ideal DFT codebook. (The ideal 16-point DFT codebook needs 4 bits; check that b=4 reproduces the ideal case exactly.)
 M5 ADC quantisation of Y: b-bit uniform mid-rise quantiser on Re and Im after per-scene AGC (scale so the quantiser range covers +-3 RMS; choose and document), b in {1 (sign), 2, 3, 4}.
 M6 per-measurement phase noise: each of the 256 beam measurements multiplied by exp(j*phi_qp), phi ~ N(0, sigma^2), sigma in {5, 10, 20, 30} deg.
 M7 unknown number of paths: L ~ U{1..8} per scene; the methods must estimate L. NOMP stopping rules: (a) oracle-noise-variance CFAR threshold (Mamandipoor et al. 2016 style; false-alarm rate 1e-2; derive and document), (b) blind (noise variance estimated from the data, e.g. MDL/AIC on the residual energy sequence up to Lmax=10, or the median residual after Lmax steps). IABR-Net: count local maxima of the heatmap above a threshold t; then heat-seeded decode with that count. Choose t and any NOMP constants on a SEPARATE tuning bank (different seed), never on the test bank. Metrics: model-order accuracy P(L_hat = L), path recall and precision (a path matched when both angle errors <= 1 deg, Hungarian assignment), plus RMSE of matched paths.
 M8 (optional, only if time remains) near-field: spherical wavefront at range r in {0.5, 1, 2} m, 28 GHz, half-wavelength spacing.
Sanity: an impairment-free bank from your simulator at SNR 0 and 15 dB must reproduce NOMP/IABR-Net Pd of the existing phase_d0_snr0 / phase_d0_snr15 banks within CI. Extend iabr2_sim.simulate by writing your own module (do not edit iabr2_sim.py).
Banks: L=3 (except M7), SNR in {0, 15} dB, 500 scenes per bank (M7: 1000 scenes per SNR), fixed seeds.
Methods per bank (all on the same Y): (1) NOMP greedy, known L, on Hant = to_ant(Y); (2) NOMP + selfcal SC=(1,5); (3) IABR-Net heat-seeded: CNN forward on Y -> heat, Hc -> decode(Hc, L, heat, 'heat', SC); (4) IABR-v2 dual: decode(Hc, L, heat, 'dual', SC); (5) network only: parabolic_heat_decode(heat, L); (6) oracle-seeded decoder (true (u,v) seeds -> Newton+cyclic) as an upper reference. Use the notebook decoder (exec slice [527:647] as common.py does).
Report paper-style Pd, strict Pd, RMSE with 95% CI; paired bootstrap differences: heat - NOMP, dual - NOMP, heat - (NOMP+selfcal), network only - NOMP.
Write stress/RESULTS.md (one table per impairment; rows severity x SNR; columns methods; a short "NOMP degradation" column = NOMP Pd minus NOMP Pd on the matching clean bank), stress/results.json, figures stress/fig_<impairment>.png (Pd vs severity per method, SNR 15 and 0 panels), all scripts and logs. Run CPU work sequentially (one heavy process at a time).
Return a concise markdown summary (<= 80 lines): for each impairment the severity at which NOMP loses > 0.05 Pd vs clean, and every cell where an NN-containing method is significantly better than BOTH NOMP and NOMP+selfcal (CI excluding 0), with numbers; and the regimes you judge most promising for a trained NN (to be fine-tuned next).`

const ADV = (stressSummary) => `${CTX}

STRESS-TEST SUMMARY FROM THE PREVIOUS STAGE:
${stressSummary}

YOUR TASK (folder ${DR}/stress/adversarial): try to REFUTE the stress-test headline claims. Default to "refuted" if you cannot reproduce a claim.
1. Read every script in ${DR}/stress/. Check the simulator for bugs: is each impairment implemented as specified (coupling matrix applied on the correct side and transpose, quantiser AGC, phase-shifter quantisation on W/F, phase noise per measurement, position errors angle-dependent), is the same Y used for all methods, is L passed consistently, NaN/endfire handling, angle wrapping, evaluator use.
2. Check fairness: thresholds/constants tuned on the test bank? NOMP given less information than the NN? Self-calibration settings identical? Any method seeing ground truth?
3. Replicate the 3-5 most important headline cells (largest NOMP degradations and any cell where an NN method beats NOMP) with a FRESH seed (new bank, same spec), using your own short script; report whether the fresh-seed result agrees within CI.
4. For unknown-L results, recompute model-order accuracy and recall/precision independently.
Write stress/adversarial/VERDICT.md and logs. Return a markdown verdict per headline claim: CONFIRMED / REFUTED / UNCERTAIN, with the replicated numbers, plus any bug found (file:line) and its impact.`

phase('Literature')
const litPromise = parallel([
  () => agent(LIT1, { label: 'lit: baselines in DL AoA/AoD papers', phase: 'Literature' }),
  () => agent(LIT2, { label: 'lit: when DL beats model-based', phase: 'Literature' }),
])

phase('Verify')
const ver = await agent(VERIFY, { label: 'verify NOMP-only independently', phase: 'Verify', effort: 'high' })
log('verification done; starting stress tests')

phase('Stress')
const stress = await agent(STRESS(ver || '(verification agent returned nothing)'), { label: 'stress-test NOMP under mismatch', phase: 'Stress', effort: 'high' })

phase('Adversarial')
const adv = await agent(ADV(stress || '(stress agent returned nothing)'), { label: 'adversarial check of stress claims', phase: 'Adversarial', effort: 'high' })

const lit = await litPromise
return { verify: ver, stress: stress, adversarial: adv, lit_baselines: lit[0], lit_when_dl_helps: lit[1] }
