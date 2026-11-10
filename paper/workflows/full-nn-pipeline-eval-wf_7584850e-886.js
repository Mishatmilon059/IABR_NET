export const meta = {
  name: 'full-nn-pipeline-eval',
  description: 'Evaluate the all-neural IABR-Net pipeline (CNN heatmap seeds -> learned unfolded decoder with learned self-calibration) at full scale against U-Net and NOMP, then adversarially verify',
  phases: [
    { title: 'Evaluate', detail: 'full official bank + all cached banks, paired CIs vs U-Net / NOMP / classical-decoder IABR-Net, latency' },
    { title: 'Adversarial', detail: 'refute the claims, fresh-seed replication, no-ground-truth check' },
  ],
}

const DR = 'D:/ai_ml_project/paper/evidence/deep_research'
const CTX = `
PROJECT CONTEXT:
- Joint AoA/AoD for analog mmWave MIMO, 16x16 ULAs, DFT codebooks; measurement Y (16x16 beamspace). IABR-Net = CNN (IABRv2: IABC per-antenna corrector -> SE-ResNet trunk -> 32x32 heatmap; 191,702 params; D:/ai_ml_project/IABR_v2_extracted/IABR_v2/outputs/checkpoints/IABRv2/final.pt) followed by a decoder.
- A LEARNED decoder already exists: D:/ai_ml_project/paper/evidence/learned_decoder/ — exp.py (CNN loading, gen_batch, LS-gain helpers, quantities, matched_loss, load_test, run_learned/score helpers), exp3.py (RefinerV3 = learned optimiser that sees the score, its gradient and score probes; no Hessian, no Newton formula, output initialised to zero; Staged = unrolled structure: 6 per-path pre-steps, then 3 cyclic rounds x 2 inner steps with LS gains and residual subtraction), exp5.py (StagedSC = Staged + differentiable self-calibration stage with learnable strength lam; trained 1500 steps batch 64 on CNN heat-peak seeds; saved as lurd.pt, 4,803 params; results5.json). Earlier 2000-scene subset results (250 per SNR): official learned+SC 0.7465 vs classical heat-seeded decoder 0.7417 (no SC) / 0.7448 (SC); T2_d5 0.7393 vs 0.7386; gain_g4_snr15 0.9097 vs 0.9097; oodphase_d15_snr15 0.8677 vs 0.8683. NOTE exp.py loads the CNN and calls torch.set_num_threads(6) at import: call torch.set_num_threads(2) right after importing.
- Caches: D:/ai_ml_project/IABR_v2_extracted/IABR_v2/outputs/cache/net__<bank>.npz (heat, Hc, Ht) for 56 banks; pred__U-Net_paper__<bank>.npz (U-Net of Lloria et al., IEEE TVT 2026 — the benchmark paper), pred__NOMP_no_network__<bank>.npz, pred__IABR-v2__<bank>.npz (dual-init classical decoder) for 52 banks; D:/ai_ml_project/paper/evidence/new_results/preds_heat/<bank>.npz (IABR-Net with the classical heat-seeded decoder) and bank_json_heat/*.json; truth via new_results/common.py load_bank(name). Metrics/bootstrap: common.py scene_stats, metric_by_snr, boot_metrics, ci (paper-style Pd = components with |err| <= 1 deg, scenes with < L detections dropped; strict Pd counts them as misses; RMSE over detected components).
- Verified reference numbers (official bank, 8000 scenes): NOMP-only 0.7581, IABR-v2 dual 0.7551, IABR-Net heat-seeded classical decoder 0.7503, U-Net 0.7366 (strict 0.6940), network only 0.4507 (${DR}/verify/REPORT.md).
- Other jobs are training on this CPU: use torch.set_num_threads(2) and OMP/OPENBLAS/MKL threads 2 (env before import). Batch work.
RULES: write only inside ${DR}/nn_decoder/ (large arrays in nn_decoder/cache/, gitignored); never modify existing files (do not overwrite lurd.pt); no git. Every number from a script you ran with stdout saved as .log. Paired bootstrap (2000 resamples within SNR groups, seed 0); a claim needs the 95% CI to exclude 0. Report null results plainly.`

const EVAL = `${CTX}

YOUR TASK (folder ${DR}/nn_decoder): evaluate the ALL-NEURAL pipeline at full scale, so the project can compare it with published work.
Pipeline "IABR-Net (all-neural)": CNN on Y -> heat = sigmoid(heat logits), Hc -> top-L local maxima of the heatmap (heat_peaks) as seeds -> learned decoder StagedSC(RefinerV3(), pre=6, rounds=3, inner=2) loaded from lurd.pt, use_sc=True (and also use_sc=False as an ablation) -> angles. No greedy FFT search, no classical Newton step.
1. Reproduce first: rerun the exp5 evaluation on its 2000-scene official subset and confirm 0.7465 (learned+SC) before scaling up.
2. Run on: the full official bank (8000), T2_d1, T2_d2, T2_d5 (8000 each), and every other bank with a net__ cache and truth available (use the cached heat/Hc; the learned decoder was trained with L=3 only: evaluate L != 3 banks too and report them separately).
3. For each bank: paper-style Pd, strict Pd, RMSE with CI for: all-neural IABR-Net (SC on/off), IABR-Net with the classical heat-seeded decoder (preds_heat or recompute with the heat_main recipe), IABR-v2 dual, NOMP-only, U-Net, network only (where cached). Paired differences: all-neural - U-Net; all-neural - NOMP-only; all-neural - classical-decoder IABR-Net. Per-SNR table for the official bank. Summary counts over banks (CI above 0 / below 0 / contains 0) for each difference.
4. Latency on CPU single thread (torch threads 1): CNN + learned decoder per scene at batch 1 and 256 (median of >= 30 runs), next to NOMP and the classical-decoder pipeline (re-measure them the same way).
5. Model facts: parameter counts (CNN, IABC, learned decoder); describe precisely what the learned decoder computes (which analytic quantities it uses as input features — e.g. score, analytic gradient, probes, LS gains, residual subtraction — and what is learned), so the paper can describe it accurately.
6. Optional only if time remains and it can be done reliably: Lloria et al.'s ResNet checkpoint D:/ai_ml_project/DL_DOA/models/inf_model_007_256_resnet.h5 is available unzipped; TensorFlow is broken but h5py works and notebook_src.py has a PyTorch port for the U-Net (UNetPaper ~line 839, unet_predict ~909) you can use as a template. Only include ResNet numbers if your port is validated (e.g. against numbers printed in the paper's local PDF under D:/ai_ml_project/papers or the PIMRC 2024 paper); otherwise skip and say why.
Write nn_decoder/REPORT.md (tables, a "comparison with published work" table: all-neural IABR-Net vs U-Net (and ResNet if validated) on the same benchmark, with NOMP-only listed as the strong model-based reference), results.json, fig_pd_vs_snr.png (official bank Pd vs SNR: all-neural IABR-Net, classical-decoder IABR-Net, NOMP-only, U-Net), scripts, logs.
Return a concise markdown summary (<= 70 lines) with the official-bank table, the per-SNR table, bank counts and latency.`

const ADV = (s) => `${CTX}

CLAIMS TO ATTACK:
${s}

YOUR TASK (folder ${DR}/nn_decoder/adversarial): try to REFUTE every claim. Default to REFUTED if you cannot reproduce it.
1. Read the scripts in ${DR}/nn_decoder/ and the learned-decoder code (exp.py, exp3.py, exp5.py). Check that at inference the learned decoder never sees ground truth (truth only in the loss); seeds come only from the CNN heatmap; the correct checkpoint (lurd.pt) and use_sc flag are used; the same scenes and L for all methods; the U-Net/NOMP caches are aligned scene-by-scene with the truth; the evaluator is the common.py one. Also check whether any official-bank scenes could have been seen in training (gen_batch draws fresh synthetic scenes — confirm the RNG seed/stream cannot reproduce official-bank scenes, or argue why overlap is impossible).
2. With your own short script re-evaluate the official bank (all 8000) for all-neural vs U-Net vs NOMP-only, and 3 other banks; agreement within CI.
3. Check the description of what the decoder computes (analytic features vs learned parts) against the code; correct anything overstated (e.g. "no classical component" if LS gains and analytic gradients are used).
Use torch.set_num_threads(2). Write nn_decoder/adversarial/VERDICT.md and logs. Return a markdown verdict per claim (CONFIRMED / REFUTED / NEEDS REWORDING + corrected sentence) with replicated numbers.`

phase('Evaluate')
const s = await agent(EVAL, { label: 'evaluate all-neural IABR-Net', phase: 'Evaluate', effort: 'high' })
phase('Adversarial')
const v = await agent(ADV(s || '(evaluation returned nothing)'), { label: 'adversarial: all-neural claims', phase: 'Adversarial', effort: 'high' })
return { evaluate: s, adversarial: v }
