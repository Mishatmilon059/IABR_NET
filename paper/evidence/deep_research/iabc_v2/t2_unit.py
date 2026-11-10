"""Unit checks of the new helper functions (seeds 61020-61023, small batches, 2 threads)."""
import os
os.environ['OMP_NUM_THREADS'] = '2'
from v2common import *
torch.set_num_threads(2)
from iabc2_model import Refiner, predict, IABCv2
# 1. Toeplitz estimator: L=1 noiseless-ish -> should remove ~all identifiable error; L=3 at 25 dB -> partial (cross-path bias)
for L, snr in [(1, 60.0), (1, 25.0), (3, 60.0), (3, 25.0)]:
    d = make_scenes(61020, 100, L, snr, 20, 4); Ht = to_ant(d['Y']); e = toeplitz_est(Ht)
    ph, g, mp, mg = err_rms(d['tgt'], e); ph0, g0, mp0, mg0 = err_rms(d['tgt'], np.zeros_like(e))
    print(f'Toeplitz L={L} SNR={snr}: phase MSE removed {100*(1-mp.sum()/mp0.sum()):.1f}%, gain {100*(1-mg.sum()/mg0.sum()):.1f}%', flush=True)
# 2. apply_corr with the true target removes the identifiable part: residual distortion = only mean/ramp
d = make_scenes(61021, 50, 3, 300.0, 30, 6); Ht = to_ant(d['Y']); Hc = apply_corr(Ht, d['tgt'])
r = Hc / d['H']; rr = r[:, :, :1] / r[:, :1, :1]; lg = np.log(np.abs(rr[..., 0])); ph = np.unwrap(np.angle(rr[..., 0]), axis=1)
print('apply_corr(true target): remaining row log-gain spread', np.abs(lg - lg.mean(1, keepdims=True)).max(), ' remaining row phase non-linear part', np.abs(proj_np(ph)).max())
# 3. hybrid_inputs with eA = 0: e_ALS must equal SC open estimate from NOMP angles
d = make_scenes(61022, 60, 3, 15.0, 20, 4); Ht = to_ant(d['Y']); z = np.zeros((60, 32, 2))
tok, glob, (U, V) = hybrid_inputs(Ht, 3, z)
Ug, Vg = C.nomp_greedy(Ht, 3, 4, 3); e_open = selfcal_ret(Ht, Ug, Vg, 1.0, 0.0, iters=4)[2]
print('hybrid e_ALS (eA=0) vs SC open:', np.abs(tok[..., 2:4] - e_open).max(), ' glob[0] (log10 snr_est) median', np.median(glob[:, 0]), '(true per-entry SNR 15 dB, signal/noise energy ratio ~ 10^1.5)')
# 4. Refiner forward shape / zero-init -> returns eA
rf = Refiner(); o = rf(torch.from_numpy(tok), torch.from_numpy(glob), torch.zeros(60, 32, 2)); print('refiner zero-init output max', float(o.abs().max()))
# 5. trunk_heat on the CNN's own Hc equals CNN heat
d = make_scenes(61023, 40, 3, 10.0, 8, 3); heat, Hc, imp = net(d['Y']); print('trunk_heat vs CNN heat max|d|', np.abs(trunk_heat(to_beam(Hc)) - heat).max())
