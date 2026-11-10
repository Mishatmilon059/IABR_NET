export const meta = {
  name: 'iabc-effectiveness-and-nn-essential-proof',
  description: 'Measure how much per-antenna impairment the learned IABC removes, and where NN+NOMP beats NOMP-only, then adversarially check every claim sentence',
  phases: [
    { title: 'Measure', detail: 'impairment before/after IABC vs classical self-calibration; downstream Pd NOMP-only vs IABC+NOMP across severity and SNR' },
    { title: 'Adversarial', detail: 'refute each claim sentence; fresh seeds; sign/projection/dB checks' },
  ],
}

const DR = 'D:/ai_ml_project/paper/evidence/deep_research'
const CTX = `
PROJECT CONTEXT:
- Joint AoA/AoD for analog mmWave MIMO, 16x16 ULAs, DFT codebooks. Y = W^H diag(d_r)^H H diag(d_t) F + Z; antenna domain Ht = Wi Y Fi^H (to_ant). With per-antenna distortion d = g e^{j eps}, the antenna-domain channel is conj(D_r) H D_t: rows g_r e^{-j eps_r}, cols g_t e^{+j eps_t}. Only part of the distortion is identifiable: the mean (phase and log-gain) and the linear phase ramp cannot be estimated (a ramp is an angle shift).
- Code: D:/ai_ml_project/IABR_v2_extracted/IABR_v2/notebook_src.py. IABC class lines 440-470 (per-antenna gain/phase estimator, 458 params; output imp (B,32,2) = [identifiable phase (rad), identifiable natural-log gain], rows 0-15 Rx, 16-31 Tx, projected onto the identifiable subspace); IABRv2.front lines 497-509 builds Hc = Ht * c_r * c_t with c = exp(-imp_loggain - j imp_phase). Decoder slice [527:647] (nomp_greedy, selfcal (alternating-LS self-calibration with gate snr_est > tau), decode(H, L, heat, mode, sc)). iabr2_sim.py (set os.environ['IABR2_SUITE']='D:/ai_ml_project' first): sample_paths, simulate(psi,phi,alpha,snr,rng,phase_deg_max,gain_db_max) returns Y, D_r, D_t, H; impairment_targets(D_r, D_t) = the IABC training target (identifiable phase/log-gain, (B,32,2)); project_impairment. CNN load on CPU: see D:/ai_ml_project/paper/evidence/learned_decoder/exp.py (exec src[419:522] with np, torch, nn, TF, W_IDEAL, F_IDEAL, P=16, Q=16; IABRv2(32,10,32); final.pt at outputs/checkpoints/IABRv2/final.pt). The CNN/IABC was trained with phase error up to 8 deg and gain error up to 3 dB (uniform per antenna in +-max), L 1..9, SNR -15..24, 20% clean.
- Helpers: D:/ai_ml_project/paper/evidence/new_results/common.py (scene_stats, metric_by_snr, boot_metrics, ci). Fair classical baseline = NOMP + TUNED self-calibration (lam 1, tau 2, iters 4; recipe in ${DR}/stress/adversarial/replicate.py) and also default (1, 5). Caches: D:/ai_ml_project/IABR_v2_extracted/IABR_v2/outputs/cache/net__<bank>.npz has Hc (IABC-corrected) and Ht (raw) for 56 banks; pred__NOMP_no_network__<bank>.npz = NOMP on Ht.
- Already verified (${DR}/verify/REPORT.md, ${DR}/stress/RESULTS.md, ${DR}/stress/adversarial/VERDICT.md): official bank (ideal hardware) NOMP-only 0.7581 vs IABR-v2 dual 0.7551 vs IABR-Net heat 0.7503; IABR-v2 dual equals plain NOMP run on Hc within 0.001 (its gain comes from IABC); only verified network win vs tuned self-cal: gain 6 dB @15 dB (+0.028 heat, +0.019 dual).
- Another workflow is fine-tuning a CNN on this CPU right now: use torch.set_num_threads(2) and OMP/OPENBLAS/MKL threads 2 (env before import), batch your work, keep total compute under ~40 minutes.
RULES: write only inside ${DR}/iabc/ (large arrays in iabc/cache/, gitignored); never modify existing files; no git. Every number comes from a script you ran with its stdout saved as a .log next to it. Fixed seeds, disjoint from all previous seeds (use base seed 7700+). Paired bootstrap (2000 resamples within SNR groups, seed 0) for every difference; a claim needs the 95% CI to exclude 0. Report null results plainly.`

const MEASURE = `${CTX}

YOUR TASK (folder ${DR}/iabc): produce the evidence for two statements the project owner wants to make, and find out exactly how far each is true:
 (S1) "The learned IABC module removes most of the per-antenna gain/phase impairment."
 (S2) "NOMP alone achieves X; IABR-Net (network + NOMP-type decoder) achieves Y; so the network is needed."
Steps:
1. Sign/convention check first: on noiseless single scenes, confirm that applying the TRUE identifiable correction (from impairment_targets) to Ht the same way IABRv2.front does removes the identifiable distortion (residual ~0). Confirm the dB conversion (dB = 20/ln(10) * natural-log gain).
2. Impairment-estimation accuracy. Sweeps (L=3, 200 scenes per cell, 8 SNRs -10..25 step 5): phase-only max in {0, 2, 5, 8, 12, 20, 30, 45} deg; gain-only max in {0, 1, 2, 3, 4, 6} dB; combined (phase, gain) in {(5,2), (8,3), (20,4), (30,6)}. For each scene compute the RMS over the 32 antennas of the identifiable phase error (deg) and log-gain error (dB):
   (a) before correction (= RMS of the identifiable target itself: what NOMP faces),
   (b) after IABC (target minus IABC estimate),
   (c) after classical self-calibration: write your own copy of the notebook selfcal that also RETURNS its cumulative per-antenna correction, run it from NOMP's angles with the gate forced open (tau 0) and with lam 1, iters 2 and 4; convert its correction to the same identifiable representation (verify the sign convention on noiseless data),
   (d) oracle ALS: the same alternating LS but with the TRUE angles (best case for classical calibration).
   Report per cell: median and mean RMS error, and "% of identifiable impairment removed" = 1 - after/before (on mean squared error, and on RMS). Present tables per SNR and averaged; mark which cells are inside the IABC training range (<= 8 deg, <= 3 dB). Also report the unidentifiable part (mean + linear ramp) that no method can remove, for context.
3. Downstream Pd on the same scenes (paper-style and strict Pd, RMSE): NOMP on Ht (NOMP-only), NOMP on Hc (IABC + NOMP, no heatmap), IABR-Net heat-seeded (decode(Hc, L, heat, 'heat', (1,5))), IABR-v2 dual, NOMP + self-cal default (1,5) and tuned (1,2,4 iters), IABC + NOMP + tuned self-cal. Paired differences with CIs: (IABC+NOMP) - NOMP-only; IABR-Net heat - NOMP-only; (IABC+NOMP) - (NOMP+tuned self-cal); (IABC+NOMP+tuned self-cal) - (NOMP+tuned self-cal). Give these per severity (averaged over SNR) and per SNR for the largest severities.
4. Existing 56 cached banks: NOMP on cached Hc vs cached NOMP-on-Ht predictions, paired CI per bank; count banks where the CI is above 0 / below 0 / contains 0.
5. Figures (matplotlib, PNG, clear labels): fig_iabc_reduction.png (identifiable phase error before/after IABC/self-cal/oracle vs severity, SNR 15 and 0 panels; same for gain); fig_iabc_vs_snr.png (residual error vs SNR at phase 8 deg and gain 3 dB); fig_pd_vs_severity.png (Pd vs severity: NOMP-only, IABC+NOMP, IABR-Net heat, NOMP+tuned self-cal).
6. Write iabc/REPORT.md with all tables, method notes, commands; iabc/results.json; and a section "CLAIMS" listing: (i) sentences the owner CAN write in the paper, each with the exact supporting numbers, CI and file; (ii) sentences the owner CANNOT write and why (e.g. if NN+NOMP does not beat NOMP on the official/ideal-hardware bank, say so explicitly with the numbers).
Return the CLAIMS section plus the two or three key tables (<= 80 lines).`

const ADV = (m) => `${CTX}

CLAIMS AND TABLES TO ATTACK:
${m}

YOUR TASK (folder ${DR}/iabc/adversarial): try to REFUTE every claim sentence. Default to REFUTED if you cannot reproduce it.
1. Read ${DR}/iabc/ scripts. Check: correction sign and conjugation (rows vs cols), identifiable projection (mean and linear ramp removed consistently for target, IABC estimate and self-cal estimate), dB conversion, that "before" is computed on the identifiable part (not inflating the reduction with the unidentifiable part), that self-calibration was run fairly (gate, iterations, same inputs), that the oracle ALS is correct, that the same scenes are used for all methods.
2. With your own short script and FRESH seeds (disjoint; 8800+), re-measure the 4-6 headline numbers (e.g. % impairment removed at phase 8 deg / gain 3 dB / gain 6 dB at 15 dB and 0 dB; IABC vs self-cal error; IABC+NOMP - NOMP-only at the largest severities; official/ideal-hardware difference). Report agreement within CI.
3. Check wording: flag any claim sentence that is technically true but misleading (e.g. "removes 90% of the impairment" if that is only at high SNR or only inside the training range; "network is essential" if it holds only for large gain errors). Propose an exact corrected sentence for each.
Use torch.set_num_threads(2) and thread env vars = 2 (another job is training on this CPU). Write iabc/adversarial/VERDICT.md and logs.
Return a markdown verdict per claim (CONFIRMED / REFUTED / NEEDS REWORDING + corrected sentence) with the replicated numbers.`

phase('Measure')
const m = await agent(MEASURE, { label: 'IABC effectiveness + NN-vs-NOMP evidence', phase: 'Measure', effort: 'high' })
phase('Adversarial')
const v = await agent(ADV(m || '(measure stage returned nothing)'), { label: 'adversarial: IABC claims', phase: 'Adversarial', effort: 'high' })
return { measure: m, adversarial: v }
