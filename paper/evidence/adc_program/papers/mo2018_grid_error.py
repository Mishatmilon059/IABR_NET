"""On-grid angle error of a DFT-type grid in u = pi*cos(psi) (benchmark convention, iabr2_sim.angles_to_cells; psi ~ U[0, pi]).
For an oversampling factor G the grid has 16*G points spaced 2*pi/(16G) in u (wrap-around). Each angle is snapped to the nearest
grid point and mapped back with arccos (u clipped to [-pi, pi]). Reports the fraction of single angle components whose snapped value
is within 1 deg of the truth (noise-free; an upper limit for 'grid peak only' decoding with no off-grid refinement), and error quantiles.
Pure arithmetic, no data, seed 0."""
import numpy as np
rng = np.random.default_rng(0)
psi = rng.uniform(0, np.pi, 2_000_000)
u = np.pi * np.cos(psi)
print('G  grid_pts  spacing_u(rad)  frac_within_1deg  median_err_deg  p95_err_deg  max_err_deg')
for G in [1, 2, 4, 8, 16, 32]:
    n = 16 * G; d = 2 * np.pi / n
    uh = np.round(u / d) * d
    uh = np.clip(uh, -np.pi, np.pi)
    err = np.abs(np.degrees(np.arccos(uh / np.pi) - psi))
    print(f'{G:<3d}{n:>6d}  {d:12.4f}  {np.mean(err <= 1.0):16.4f}  {np.median(err):14.3f}  {np.quantile(err, .95):11.3f}  {err.max():11.3f}')
