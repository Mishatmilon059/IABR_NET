"""GNOMP on the 16x16 beamspace test banks (seeds 12000-12004, L = 3, SNR -10/0/10/15/25 dB, bits 1-4, 300 scenes/cell).
Modes: 'L' = GNOMP with exactly L = 3 detections (the threshold is not used; same knowledge as every other decoder here);
       'cfar' = paper's stopping rule, P_FA = 0.01 over N = 256 bins (tau = 20.29), at most 8 paths.
Also NOMP (common.py decoder, gamma_os 4, R_c 3) on the same scenes, for reference.
Predictions (psi, phi, n) and per-scene stats (common.scene_stats) are cached on C:; the summary goes to results_gnomp.json.

Usage:  python run_gnomp.py            full run (20 cells)
        python run_gnomp.py smoke      20 scenes, two cells (checks only)
        python run_gnomp.py latency    batch-1 latency per scene, 1 thread, 30 scenes at 1 bit / 15 dB

DEVIATIONS from the paper (each one is intentional and listed here, not hidden in the code):
 D1  1-D -> 2-D: the atom is the beamspace atom rho(u) (x) tau(v) of finetune/adversarial/mlq.py (two angles per path,
     theta = (u, v), P = 2). The Rao statistic is the 2x2 real-information form of the paper's eq. (19)/(32) (it reduces
     to the paper's form for unit-modulus atoms). This 2-D extension is OURS; the paper has no 2-D algorithm.
 D2  Newton steps are Fisher-scoring steps (expected curvature, always positive) with a backtracking line search that
     accepts only decreases of the exact NLL; the paper uses the exact second derivative and checks local concavity.
     Frequency steps are clipped to 0.1 rad per iteration.
 D3  The noise standard deviation is the true value (known, as in the paper); no kappa and no tuning. All free constants
     are the paper's defaults (P_FA = 0.01, gamma_os = 4, R_s = 1, R_c = 3, Newton iterations 3 (amplitude) and 2
     (frequency)). Nothing was tuned, so no validation seeds were used.
 D4  Quantiser: peak AGC, so the full scale A is estimated from each scene's own data (paper: gamma is a known design
     value). The observed cells are the closed outer cells +-A (finetune/adc acommon.adc_cells convention). The Fisher
     information h_B(zeta) is computed from the open-outer-cell thresholds (paper's model), not from the closed cells.
 D5  Stopping: the paper's CFAR stop applies only in mode 'cfar'; mode 'L' uses exactly L detections (as the paper does
     not define a fixed-L rule). The spurious-component suppression (paper lines 21-29) is applied in mode 'cfar' only.
 D6  Output: the L strongest detections by |amplitude| are kept in mode 'cfar' when more than L are found (paper's metric
     compares L truths); scenes with fewer than L detections are NaN-padded and dropped by the paper-style metric and
     counted as misses by the strict metric (common.scene_stats).
 D7  The unknown-noise-variance variant (paper Sec. V-I/K) is NOT implemented: the paper reports its results but does not
     describe the algorithm (v4 footnote 2), so there is nothing faithful to implement.
 D8  Identification grid: N-bin grid = the 16x16 DFT bins (every 4th point of the 64x64 oversampled grid); the start is
     the argmax over the 64x64 grid, as in the paper (line 6). The grid map is the one of acommon.fft_heat.
"""
import os, sys, time, json
MODE_ARG = sys.argv[1] if len(sys.argv) > 1 else 'full'
NT = '1' if MODE_ARG == 'latency' else '2'
for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[k] = NT
os.environ.setdefault('IABR2_SUITE', 'D:/ai_ml_project')
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, 'D:/ai_ml_project/paper/evidence/new_results')
sys.path.insert(0, 'D:/ai_ml_project/paper/evidence/deep_research/stress')
sys.path.insert(0, 'D:/ai_ml_project/IABR_v2_extracted/IABR_v2')
import numpy as np
import common as C
import stress_sim as SS
import gnomp_core as G

CACHE = 'C:/Users/Milon/iabr_deep_research_cache/adc_program/baselines/gnomp/cache'
os.makedirs(CACHE, exist_ok=True)
L = 3; NS = 300; CH = 100
SNRS = [-10, 0, 10, 15, 25]; BITS = [1, 2, 3, 4]
SEEDS = {s: 12000 + i for i, s in enumerate(SNRS)}           # test banks 12000-12004 (paper-independent)
M2 = G.Atoms2D(SS.W_IDEAL, SS.F_IDEAL)
GRID = M2.grid(4)
KMAX = 8; PFA = 0.01; NCFG = 256
np.seterr(all='ignore')   # overshooting Newton trial steps can produce inf/NaN in rejected candidates (line search rejects them)


def sigma_of(snr):
    return np.sqrt(10 ** (-snr / 10) / 2)


def cell_chunk(Y, bits, snr):
    rl, rh, il, ih, A = G.adc_cells(Y, bits)
    lo, hi = G.cells_ri(rl, rh, il, ih)
    thr = G.thresholds_peak(A, bits)
    s = np.full((len(Y), 1), sigma_of(snr))
    return lo, hi, thr, s


def as_list(p):
    return [(p[0][i], p[1][i]) for i in range(len(p[0]))]


def stats_with_drops(psi, phi, bank):
    """common.scene_stats for scenes with L finite detections. Scenes with fewer than L detections (NaN-padded) are not
    passed to scene_stats (its Hungarian step cannot take NaN); they get its own 'dropped' row [0,0,1,0,L,0,L]."""
    full = ~np.isnan(psi).any(1) & ~np.isnan(phi).any(1)
    S = np.zeros((len(psi), 7)); S[~full] = [0, 0, 1, 0, L, 0, L]
    if full.any():
        sub = dict(psi=bank['psi'][full], phi=bank['phi'][full], L=bank['L'][full])
        S[full] = C.scene_stats(as_list((psi[full], phi[full])), sub)
    return S


def run_cell(bits, snr, n, methods):
    base = SS.base_scenes(n, L, float(snr), SEEDS[snr])
    Y = SS.generate(base, 'M5', bits)
    out = {m: dict(psi=[], phi=[], n=[]) for m in methods}; extra = dict(spur=[], steps=[])
    t0 = time.time()
    for c0 in range(0, n, CH):
        sl = slice(c0, c0 + CH); Yc = Y[sl]
        lo, hi, thr, s = cell_chunk(Yc, bits, snr)
        for m in methods:
            if m == 'NOMP':
                H = SS.to_ant(Yc); U, V = C.nomp_greedy(H, L, 4, 3)
                psi, phi = C.uv_to_angles(U, V)
                psi = np.asarray(psi)[:, :L]; phi = np.asarray(phi)[:, :L]
                nd = np.full(len(Yc), L)
            else:
                mode = 'L' if m == 'GNOMP-L' else 'cfar'
                Z, valid, info = G.gnomp(M2, GRID, lo, hi, s, thr, NCFG, mode=mode, L=L, Kmax=KMAX, pfa=PFA)
                psi, phi = G.to_pred(M2, Z, valid, L)
                if m == 'GNOMP-CFAR':
                    extra['spur'].append(info['spur_removed']); extra['steps'].append(np.full(len(Yc), info['steps']))
            for i in range(len(psi)):
                out[m]['psi'].append(psi[i]); out[m]['phi'].append(phi[i])
    bank = dict(psi=base['psi'], phi=base['phi'], L=base['L'])
    res = {}
    for m in methods:
        psi = np.array(out[m]['psi']); phi = np.array(out[m]['phi'])
        nfound = np.sum(~np.isnan(psi), 1)
        np.savez(f'{CACHE}/{m}_b{bits}_s{snr}.npz', psi=psi, phi=phi, n=nfound)
        S = stats_with_drops(psi, phi, bank)
        np.save(f'{CACHE}/stats_{m}_b{bits}_s{snr}.npy', S)
        mt = C.metric_from_stats(S, np.arange(n))
        res[m] = dict(pd_paper=mt['pd_paper'], pd_strict=mt['pd_strict'], rmse=mt['rmse'], dropped=mt['dropped'],
                      n_eq_L_frac=float(np.mean(nfound == L)), mean_found=float(nfound.mean()))
    if extra['spur']:
        res['GNOMP-CFAR']['spurious_removed_mean'] = float(np.concatenate(extra['spur']).mean())
        res['GNOMP-CFAR']['greedy_steps_mean'] = float(np.concatenate(extra['steps']).mean())
    np.save(f'{CACHE}/snr_b{bits}_s{snr}.npy', base['snr'])
    res['_wall_s'] = time.time() - t0
    return res


def latency():
    n = 30; bits, snr = 1, 15
    base = SS.base_scenes(n, L, float(snr), SEEDS[snr]); Y = SS.generate(base, 'M5', bits)
    rows = {}
    for name in ['NOMP', 'GNOMP-L', 'GNOMP-CFAR']:
        t = []
        for i in range(n):
            t0 = time.perf_counter()
            if name == 'NOMP':
                C.nomp_greedy(SS.to_ant(Y[i:i + 1]), L, 4, 3)
            else:
                lo, hi, thr, s = cell_chunk(Y[i:i + 1], bits, snr)
                mode = 'L' if name == 'GNOMP-L' else 'cfar'
                G.gnomp(M2, GRID, lo, hi, s, thr, NCFG, mode=mode, L=L, Kmax=KMAX, pfa=PFA)
            t.append(time.perf_counter() - t0)
        rows[name] = dict(ms_per_scene_median=1e3 * float(np.median(t)), ms_per_scene_mean=1e3 * float(np.mean(t)))
        print(name, rows[name], flush=True)
    json.dump(rows, open(f'{HERE}/latency_gnomp.json', 'w'), indent=1)


if __name__ == '__main__':
    if MODE_ARG == 'latency':
        latency(); sys.exit(0)
    methods = ['NOMP', 'GNOMP-L', 'GNOMP-CFAR']
    if MODE_ARG == 'smoke':
        for bits, snr in [(1, 15), (4, 0)]:
            print(bits, snr, run_cell(bits, snr, 20, methods), flush=True)
        sys.exit(0)
    summary = {}
    for bits in BITS:
        for snr in SNRS:
            r = run_cell(bits, snr, NS, methods)
            summary[f'b{bits}_s{snr}'] = r
            print(f'b{bits} snr{snr:>3}: ' + '  '.join(f"{m} Pd {r[m]['pd_paper']:.4f} strict {r[m]['pd_strict']:.4f} n=L {r[m]['n_eq_L_frac']:.3f}" for m in methods)
                  + f"  wall {r['_wall_s']:.0f}s", flush=True)
            json.dump(summary, open(f'{HERE}/results_gnomp.json', 'w'), indent=1, default=float)
    print('done', flush=True)
