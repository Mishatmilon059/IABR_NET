# Check of GEMINI_SIMULATION_BRIEF.md against code and verified results (2026-10-16)

The check went line by line through the brief, comparing it with the code and the evidence reports. A numeric check (`dldoa_dataset_generation.py` codebooks and channel) confirmed the corrected conventions. W_ideal[n,q] = exp(+j2πnq/16)/4 and F_ideal[m,p] = exp(-j2πmp/16)/4 (error 3e-15). Both codebooks are unitary. H̃ = H, and H̃ = conj(D_r)·H·D_t (error 7e-15). The row phase step is -π cos ψ and the column phase step is +π cos φ. In 200 of 200 single-path scenes, the brightest cell of |Y| lies within half a cell of q = 16·wrap(-π cos ψ)/2π, p = 16·wrap(π cos φ)/2π.

## Corrections (section: old -> new; source)

1. **§2 Codebooks (error):** "W_ideal is built the same way" -> W_ideal uses cos ψ_q = angle(exp(**-**j2πq/16))/π, giving explicit DFT forms for W and F and the beam-cell mapping. Source: `dldoa_dataset_generation.beamforming_vector_generation_Q`.
2. **§2 Decoder sanity check:** "phase step down a row is -π cos ψ" (ambiguous) -> row n to n+1: -π cos ψ; column m to m+1: +π cos φ. Stage-1 test (ii) is reworded the same way and adds a beam-cell check. Source: `notebook_src.atoms`, `iabr2_sim.simulate`.
3. **§2 ADC:** "The receiver knows b and A" -> A is recovered from the quantised output (top level = A(1-2^-b)). The cells are finite because peak AGC never clips. Source: `finetune/adc/acommon.adc_cells`.
4. **§2 Other impairments:** added θ1, θ2 ~ U(0,2π) per scene, separate draws for Rx and Tx (coupling). Position errors apply to both arrays and the codebooks stay ideal. Phase noise is applied after the noise. Source: `stress_sim.generate`.
5. **§0 rule 4 credits:** "U-Net and ResNet of Lloria et al., IEEE TVT 2026" -> U-Net from TVT 2026, ResNet from Lloria et al., IEEE PIMRC 2024. The venue of Weiss & Friedlander was added. Source: `lit/baselines_survey.md` R1, R3; `lit/when_dl_helps.md` #32.
6. **§4.3 IABC:** the features are now named; they also use the strongest beam of Y. Source: `notebook_src.IABC.features`.
7. **§4.4 head:** "A 1x1 head" -> two 1x1 convs (32->32 ReLU, 32->4), then pixel shuffle with factor 2. Source: `IABRv2.__init__`. The 191,702 parameter count was re-derived and is correct.
8. **§4.4 training:** added the focal-loss form, the 500-step warm-up, gradient clipping 5, and per-sample δ ~ U(0,8°) and γ ~ U(0,3 dB). Source: `notebook_src` CFG, `focal_loss`, `lr_at`, `iabr2_sim.make_batch`.
9. **§4.5 decoder:** now states that it runs on H_c and that seed Newton steps use the full channel. Self-calibration is now described exactly: 2 iterations, 3 ALS sweeps, identifiable cleaning, λ=1, then 2 cyclic rounds. The gate is the linear ratio of model energy to residual energy > τ=5, not an SNR in dB. Source: `notebook_src.selfcal`, `decode`, `learned_decoder/exp.py run_classical`.
10. **§4.8 all-neural decoder (made exact):** 4,803 = 4,802 MLP weights + 1 learned λ. The 7 inputs are listed: gradient/f (2), second differences at 0.02 and 0.08 rad (4), and normalised |S| (1). Step = (π/32)·tanh. Seeds come from heatmap peaks with no FFT search. The schedule is 6 pre-steps, 3 cyclic rounds × 2 steps, and 2 self-cal iterations (τ=5) each followed by 2 cyclic rounds. The Newton correlation is 0.91-0.92. Source: `learned_decoder/exp3.py` (RefinerV3, Staged), `exp5.py` (StagedSC), `RESULTS_SUMMARY.md` §5.
11. **§4.9 hybrid:** network seeds = top-3 peaks + parabolic interpolation. μ is defined in beamspace, and s = sqrt(10^(-SNR/10)/2). The solver is LM; the classical version uses 13 starts + EM, and the hybrid starts from the network seeds. Source: `finetune/adversarial/mlq.py`, `s3_adc_test.py`.
12. **§5 metric:** the description now matches the code. The first L estimates are kept, then Hungarian matching on Euclidean (ψ,φ) distance, then the wrapped error. Counts are pooled per SNR, and the headline is the mean over the 8 SNRs. Strict Pd adds 2L misses per dropped scene. The worked example is reworded to say the scene adds to the pooled counts. Source: `TVT_Blob_Inference.prepare_for_metric/get_ang_difference`, `tvt_data_generation_v3.permute_pairs`, `new_results/common.py scene_stats/metric_from_stats`.
13. **Scene 04:** the true-path beam-cell formula was added.
14. **Scene 07 citation:** "Walden, IEEE JSAC 1999; Mo & Heath; Li et al., IEEE TSP 2017" -> "Mo, Schniter & Heath, IEEE TSP 2018". Walden and Li et al. appear only in a reading list in RESULTS_SUMMARY and are not in any verified lit file. Mo/Schniter/Heath is ref. [32] in `unfolded_v2/lit/unfolding_survey.md`. The power claim is now phrased qualitatively.
15. **Scene 09:** "IABC-v2 removes 90-97% (MSE)" -> the same figure, plus its conditions (20-45° phase, 3-6 dB gain) and 67-68% at 8°. Source: `RESULTS_SUMMARY.md` §4 table.
16. **Scene 10 shapes:** "2x16x16 -> 32x16x16 ..." -> the 3x16x16 stem input (Re, Im, log-power) was inserted. Source: `IABRv2.forward`.
17. **Scene 12a:** the text said the true angles themselves were compared. It now says the solution refined from the true angles gets every component right in 55 scenes, and none of those 55 has a lower residual than NOMP's answer. Source: `verify/REPORT.md` headroom table (-10 dB: 55 scenes, 0 with lower residual).
18. **Scene 12c:** "perfect 64-grid seeds" -> true angles rounded to the 64-point grid; Pd 0.658 -> 0.430 after Newton/cyclic. Source: `finetune/adc/diag2.log`, `REPORT.md`.
19. **Scene 15:** "Every claim" -> "Every headline claim".
20. **§6.4 Option B:** names the namespace that lines 420-522 need (np, torch, nn, TF, W_IDEAL, F_IDEAL, P, Q). Source: `learned_decoder/exp.py` line 16.
21. **§8 labels:** the network-only row is now "heatmap peaks + parabolic interpolation". The ADC header "13 starts" -> "13 classical starts + EM", and the hybrid is "quantiser-ML started from the network seeds". Latency is marked 1 thread, batch 1. The M2 row is "fine-tuned network seeds on raw H̃ + self-cal". "+0.044 / +0.055" is now attributed to two further fresh banks (seed 8775 and seed 9415) rather than one replication. Source: `s3_adc_test.log`, `s6_latency.json`, `finetune/imp/REPORT.md`, `finetune/adversarial/VERDICT.md`.
22. **Stage 2 test:** "recovers 3 paths within 1° in > 90% of scenes", a scene-level criterion not backed by any report -> paper-style Pd ≥ 0.9 at 25 dB. The verified NOMP Pd at 25 dB is 0.9758.

## Verified unchanged

The following were verified and left unchanged:
- **Model:** steering vector, H scaling, D_r/D_t model, W^H usage, conj(D_r) on rows and D_t on columns, atom exp(-jnu+jmv), Newton objective and sign, clip 2π/64, FFT seed mapping u=-2πi/64, v=2πk/64, heat-target mapping, rounding and wrap.
- **Targets and sampling:** identifiable projection, impairment-target signs, quantiser formula and b=2 levels, noise std, gain distribution and sorting, π/6 Euclidean separation.
- **Model sizes:** IABC 458, IABR-Net v2 191,702, IABC-v2 597,250 / 256,898 / 854,148.
- **Tables:** every number in §8: official table, per-SNR table, NOMP-loss table, ADC table, latencies and M2 values.

## Not fully verifiable

- Weiss & Friedlander (1990) is marked "partial (Crossref metadata only)" in `lit/when_dl_helps.md`. REPORT.md cites it knowingly.
- "≥ 45°" in Scene 09 follows REPORT.md. The adversarial verdict tested only up to 45°.
- The "refuted" item "IABC is essential" is a "cannot claim" item in RESULTS_SUMMARY §9 and the IABC adversarial verdict. It is not a single refuted test.
- The unfolded-decoder v2 has no `unfolded_v2/REPORT.md` yet, so the placeholder stays.
- The `notebook_src.py` line numbers in the brief's own source list (e.g. CFG ~82) were not part of the brief text.
