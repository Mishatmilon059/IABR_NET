"""Audit: MSE of a zero predictor on the IABC impairment targets of the training recipe (5120 scenes, seed 123).
Expected: 5.600e-03 (phase channel 1.535e-03, gain channel 9.665e-03); the logged final IABC training MSE is 4.18e-03 (run_log.txt)."""
import os, sys, numpy as np
os.environ.setdefault('IABR2_SUITE', 'D:/ai_ml_project')
sys.path.insert(0, 'D:/ai_ml_project/IABR_v2_extracted/IABR_v2')
import iabr2_sim as sim
cfg = dict(train_L=(1, 9), train_snr=(-15, 24), clean_frac=0.2, train_phase_max=8.0, train_gain_max=3.0, grid=32, heat_sigma=1.0)
rng = np.random.default_rng(123)
imp = np.concatenate([sim.make_batch(rng, 256, cfg)['imp'] for _ in range(20)])
print('zero-predictor MSE %.3e | phase %.3e | gain %.3e' % (np.mean(imp ** 2), np.mean(imp[..., 0] ** 2), np.mean(imp[..., 1] ** 2)))
