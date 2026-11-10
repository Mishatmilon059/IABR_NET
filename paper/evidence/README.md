# Evidence index (do not overstate)

Main IABR-v2 results (from the earlier GPU run on a different computer, RTX 3050 laptop, i5-11400H):
  D:/ai_ml_project/IABR_v2_extracted/IABR_v2/outputs/RESULTS.md  (+ results/*.json, figures/*.png, logs/run_log.txt)
Source of the model, decoder and training: D:/ai_ml_project/IABR_v2_extracted/IABR_v2/notebook_src.py, iabr2_sim.py

learned_decoder/ = experiments run in this session on a CPU-only laptop (ASUS VivoBook, i5-1135G7), one seed each:
  exp.py / results.json / run.log   : V1 (design pasted by the user) and V2 (normalised, Newton step + learned correction)
  exp3.py / results3.json / run3.log: V3, learned optimiser from scratch (no Hessian, no Newton formula)
  exp4.py / results4.json / run4.log: V4, 7x larger refiner (no gain over V3)
  exp5.py / results5.json / run5.log: V3 + differentiable self-calibration; weights lurd.pt
  diag*.py + diag*_output.txt        : root-cause diagnostics (convergence, objective-vs-truth, Cramer-Rao comparison)
  worked_example_trace.py            : step-by-step numeric trace on one real sample

METRIC WARNING: the learned-decoder experiments score the fraction of the 6 angle components (3 psi, 3 phi) within 1 degree
after permutation matching, on 250 samples per SNR (official bank) or <=500 samples (generated banks), L=3 only.
This differs slightly from the paper-style Pd in RESULTS.md (e.g. 0.745 here vs 0.755 there for the full decoder). Do not mix the two.
The training data for the learned decoder are fresh simulated scenes (never the test banks); the CNN weights are the trained IABRv2 checkpoint.
Known open items: unknown-L never tested; end-fire wrap-around (arccos fold) unresolved in the decoder; GPU not used in the learned-decoder runs.
