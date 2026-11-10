"""Steps 2+3: impairment-estimation accuracy and downstream Pd on the SAME scenes.
Scenes: L=3, 8 SNRs (-10..25 step 5), 200 scenes per SNR -> 1600 per cell. Seeds: SNR index k -> seed 7700+k;
angles/alpha from default_rng(seed) (iabr2_sim.sample_paths), noise from default_rng([seed, 1]), impairment uniforms from
default_rng([seed, 3]) passed to iabr2_sim.simulate -> every cell uses the SAME angles, gains, noise and SAME uniform draws (only scaled
by the severity), so all comparisons across cells and methods are paired.
Cells: clean (0,0); phase-only {2,5,8,12,20,30,45} deg; gain-only {1,2,3,4,6} dB; combined {(5,2),(8,3),(20,4),(30,6)}.
Per scene, identifiable per-antenna error (RMS over 32 antennas; phase deg, log-gain dB):
  before   = target itself (impairment_targets)                     what NOMP-only faces
  iabc     = target - IABC estimate (CNN imp output)
  sc_t0_i2 / sc_t0_i4 = target - self-cal estimate, NOMP start, lam 1, gate open (tau 0), 2 / 4 iterations
  sc_def / sc_tun     = same with the gated settings (1,5,2) / (1,2,4) (estimate 0 where the gate is shut)
  oracle   = target - ALS estimate from the TRUE angles (lam 1, tau 0, 4 iters, angles fixed)
  iabc_sct = target - (IABC + tuned self-cal run on Hc) combined estimate
  unident  = mean + linear-ramp part (phase) / mean part (gain): not removable by any method
Pd methods (true L=3): NOMP (on Ht) | IABC+NOMP (nomp_greedy on Hc, no self-cal) | IABR-Net heat = decode(Hc,3,heat,'heat',(1,5)) |
  IABR-v2 dual = decode(Hc,3,heat,'dual',(1,5)) | NOMP+SC default (1,5,2) | NOMP+SC tuned (1,2,4) | NOMP+SC tau0 i2/i4 |
  IABC+NOMP+SC tuned (selfcal(Hc, NOMP(Hc), 1,2,4)).
The heat/dual decodes are assembled from the same notebook functions as decode() (heat peaks->Newton->cyclic; dual pick by residual
energy; then selfcal) -- checked equal to decode() on the first cell (printed). Per-scene results -> cache/sweep__<cell>.npz.
usage: python s2_sweep.py > s2_sweep.log   (resumes: cells with an existing cache/sweep__<cell>.npz are skipped)"""
from icommon import *

SNRS = [-10, -5, 0, 5, 10, 15, 20, 25]; NPER = 200; L = 3
CELLS = [('clean', 0, 0), ('c8_3', 8, 3), ('ph8', 8, 0), ('g3', 0, 3), ('c30_6', 30, 6), ('ph20', 20, 0), ('g6', 0, 6), ('ph45', 45, 0),
         ('c20_4', 20, 4), ('c5_2', 5, 2), ('ph2', 2, 0), ('ph5', 5, 0), ('ph12', 12, 0), ('ph30', 30, 0), ('g1', 0, 1), ('g2', 0, 2), ('g4', 0, 4)]
if os.environ.get('CELLS'):
    CELLS = [c for c in CELLS if c[0] in os.environ['CELLS'].split(',')]

# fixed scenes (shared by all cells)
banks = []
for k, snr in enumerate(SNRS):
    seed = 7700 + k; rng = np.random.default_rng(seed)
    psi, phi, alpha = S.sample_paths(np.full(NPER, L), rng)
    g = np.random.default_rng([seed, 1]); s = np.sqrt(10 ** (-snr / 10) / 2)
    noise = s * (g.standard_normal((NPER, 16, 16)) + 1j * g.standard_normal((NPER, 16, 16)))
    banks.append(dict(seed=seed, snr=snr, psi=psi, phi=phi, alpha=alpha, noise=noise))
PSI = np.concatenate([b['psi'] for b in banks]); PHI = np.concatenate([b['phi'] for b in banks]); SNR = np.repeat(SNRS, NPER).astype(float)
LL = np.full(len(PSI), L)
U0, V0 = true_uv(PSI, PHI)


def run_chunk(Y, tgt, D_r, D_t, U0, V0, PSI, PHI, LL, name, c0):
    t0 = time.time(); Ht = to_ant(Y); heat, Hc, imp = net(Y); tn = time.time() - t0
    Ug, Vg = C.nomp_greedy(Ht, L, 4, 3)                                    # NOMP-only
    Uc, Vc = C.nomp_greedy(Hc, L, 4, 3)                                    # IABC + NOMP
    Uh, Vh = heat_seed(Hc, heat, L)                                        # heat-seeded (before self-cal)
    pick_h = C.residual_energy(Hc, Uh, Vh) < C.residual_energy(Hc, Uc, Vc)
    Uh_s, Vh_s = C.selfcal(Hc, Uh, Vh, 1.0, 5.0)                           # IABR-Net heat
    Ud_s, Vd_s = Uh_s.copy(), Vh_s.copy(); ng = ~pick_h
    if ng.any():
        a_, b_ = C.selfcal(Hc[ng], Uc[ng], Vc[ng], 1.0, 5.0); Ud_s[ng] = a_; Vd_s[ng] = b_   # IABR-v2 dual
    td = time.time() - t0 - tn
    Usd, Vsd, est_def, gate_def, _ = selfcal_ret(Ht, Ug, Vg, *DEFAULT[:2], iters=DEFAULT[2])
    Ust, Vst, est_tun, gate_tun, _ = selfcal_ret(Ht, Ug, Vg, *TUNED[:2], iters=TUNED[2])
    _, _, _, _, sn0 = selfcal_ret(Ht, Ug, Vg, 1.0, 0.0, iters=4, record=(2, 4))
    Uct, Vct, est_ct, gate_ct, _ = selfcal_ret(Hc, Uc, Vc, *TUNED[:2], iters=TUNED[2])
    _, _, est_or, _, _ = selfcal_ret(Ht, U0, V0, 1.0, 0.0, iters=4, fix_angles=True)
    if c0 == 0 and (name == 'clean' or os.environ.get('CHECK')):
        jj = np.arange(0, len(Y), 8)
        a1 = C.decode(Hc[jj], L, heat[jj], 'heat', (1.0, 5.0)); a2 = C.decode(Hc[jj], L, heat[jj], 'dual', (1.0, 5.0))
        b1 = C.uv_to_angles(Uh_s[jj], Vh_s[jj]); b2 = C.uv_to_angles(Ud_s[jj], Vd_s[jj])
        print(f'  check vs notebook decode() on {len(jj)} scenes: heat max|d| {max(np.abs(a1[0] - b1[0]).max(), np.abs(a1[1] - b1[1]).max()):.1e} rad, '
              f'dual max|d| {max(np.abs(a2[0] - b2[0]).max(), np.abs(a2[1] - b2[1]).max()):.1e} rad', flush=True)
    ANG = {'NOMP': (Ug, Vg), 'IABC+NOMP': (Uc, Vc), 'IABR-Net heat': (Uh_s, Vh_s), 'IABR-v2 dual': (Ud_s, Vd_s),
           'NOMP+SC default': (Usd, Vsd), 'NOMP+SC tuned': (Ust, Vst), 'NOMP+SC tau0 i2': sn0[2][:2], 'NOMP+SC tau0 i4': sn0[4][:2],
           'IABC+NOMP+SC tuned': (Uct, Vct)}
    EST = {'before': np.zeros_like(tgt), 'iabc': imp, 'sc_t0_i2': sn0[2][2], 'sc_t0_i4': sn0[4][2], 'sc_def': est_def, 'sc_tun': est_tun,
           'oracle': est_or, 'iabc_sct': imp + est_ct}
    rec = dict(gate_def=gate_def, gate_tun=gate_tun, gate_ct=gate_ct, pick_h=pick_h, _tn=tn, _td=td)
    for k, e in EST.items():
        ph, gn = rms_deg_db(ident_err(tgt, e)); rec[f'ph_{k}'] = ph; rec[f'g_{k}'] = gn
    up, ug = unident_part(D_r, D_t); rec['ph_unident'] = up; rec['g_unident'] = ug
    for m, uv in ANG.items():
        rec[f'S_{m}'] = stats(C.uv_to_angles(*uv), PSI, PHI, LL)
    return rec


t00 = time.time()
for name, pdeg, gdb in CELLS:
    fn = f'{CACHE}/sweep__{name}.npz'
    if os.path.exists(fn):
        print(f'{name}: cached, skip', flush=True); continue
    t0 = time.time(); Ys, Drs, Dts = [], [], []
    for b in banks:
        o = S.simulate(b['psi'], b['phi'], b['alpha'], float(b['snr']), np.random.default_rng([b['seed'], 3]), phase_deg_max=pdeg, gain_db_max=gdb, noise=b['noise'])
        Ys.append(o['Y']); Drs.append(o['D_r']); Dts.append(o['D_t'])
    Y = np.concatenate(Ys); D_r = np.concatenate(Drs); D_t = np.concatenate(Dts)
    tgt = S.impairment_targets(D_r, D_t).astype(np.float64)
    CH = 400; parts = []; tn = td = 0.0          # resume run: process in chunks of 400 scenes (memory); every step is per-scene
    for c0 in range(0, len(Y), CH):
        sl = slice(c0, c0 + CH); parts.append(run_chunk(Y[sl], tgt[sl], D_r[sl], D_t[sl], U0[sl], V0[sl], PSI[sl], PHI[sl], LL[sl], name, c0))
        tn += parts[-1].pop('_tn'); td += parts[-1].pop('_td')
    rec = {k: np.concatenate([p_[k] for p_ in parts]) for k in parts[0]}
    gate_def, gate_tun, gate_ct, pick_h = rec.pop('gate_def'), rec.pop('gate_tun'), rec.pop('gate_ct'), rec.pop('pick_h')
    ST = {k[2:]: v for k, v in rec.items() if k.startswith('S_')}
    np.savez_compressed(fn, snr=SNR, gate_def=gate_def, gate_tun=gate_tun, gate_ct=gate_ct, pick_h=pick_h, phase_deg=pdeg, gain_db=gdb, **rec)
    pdv = {m: C.metric_from_stats(s_, np.arange(len(s_)))['pd_paper'] for m, s_ in ST.items()}
    print(f'{name:6s} ph {pdeg:2d} g {gdb} | RMS phase deg before {rec["ph_before"].mean():.2f} iabc {rec["ph_iabc"].mean():.2f} sc_t0_i4 {rec["ph_sc_t0_i4"].mean():.2f} '
          f'oracle {rec["ph_oracle"].mean():.2f} | gain dB before {rec["g_before"].mean():.2f} iabc {rec["g_iabc"].mean():.2f} sc_t0_i4 {rec["g_sc_t0_i4"].mean():.2f} '
          f'oracle {rec["g_oracle"].mean():.2f}', flush=True)
    print('        Pd ' + ' | '.join(f'{m} {v:.3f}' for m, v in pdv.items()) +
          f' | gate open def {gate_def.mean():.2f} tun {gate_tun.mean():.2f} | net {tn:.0f}s decode {td:.0f}s total {time.time() - t0:.0f}s [{time.time() - t00:.0f}s]', flush=True)
# chunking check: recompute the first 400 scenes of a cell written by attempt 1 (whole-cell, 1600 at once) and compare
for name, pdeg, gdb in [c for c in CELLS if c[0] == 'g6']:
    z = np.load(f'{CACHE}/sweep__{name}.npz'); Ys, Drs, Dts = [], [], []
    for b in banks[:2]:
        o = S.simulate(b['psi'], b['phi'], b['alpha'], float(b['snr']), np.random.default_rng([b['seed'], 3]), phase_deg_max=pdeg, gain_db_max=gdb, noise=b['noise'])
        Ys.append(o['Y']); Drs.append(o['D_r']); Dts.append(o['D_t'])
    Y = np.concatenate(Ys); D_r = np.concatenate(Drs); D_t = np.concatenate(Dts); tgt = S.impairment_targets(D_r, D_t).astype(np.float64)
    r = run_chunk(Y, tgt, D_r, D_t, U0[:400], V0[:400], PSI[:400], PHI[:400], LL[:400], name, 1)
    dm = max(float(np.abs(np.asarray(r[k], float) - np.asarray(z[k][:400], float)).max()) for k in r if not k.startswith('_'))
    print(f'chunk check {name}: first 400 scenes recomputed in a 400-chunk vs attempt-1 1600-batch cache: max |diff| over all per-scene arrays = {dm:.2e}', flush=True)
print(f'done {time.time() - t00:.0f}s')
