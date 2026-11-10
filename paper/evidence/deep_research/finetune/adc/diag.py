"""Step 1a: regenerate the stress M5 banks (b = 4,3,2,1; SNR 0/15; seeds 5000/5015; 500 scenes, L = 3) and diagnose why NOMP fails.
Checks:
  (0) Y identity: NOMP on the regenerated Y must reproduce stress/cache/preds__M5_<b>_snr<s>.npz NOMP predictions exactly (the stress run
      did not store Y), and stress_sim.adc must equal the independent adversarial adc_peak on the same pre-ADC Y.
  (1) per-path detection rate by strength rank (path 1 = strongest; sample_paths sorts by |alpha|), both angles within 1 deg.
  (2) first greedy detection (FFT argmax + Newton on the full channel): within 1 deg of the strongest path / of any path / distance in
      beam cells to the nearest true path.
  (3) the NOMP estimates that miss: distance (16-grid beam cells, max over u and v) to the nearest true path.
  (4) search loss vs objective: fraction of scenes where the oracle-seeded solution has a LOWER residual than NOMP's (a better search
      in the same objective would be preferred by it), and the Pd of 'oracle start only where the residual prefers it'.
  (5) peak structure: |FFT2(Hant)| value at the true (u,v) of each path relative to the global max, and the rank of the strongest
      local maximum lying within 0.5 cell of each true path among all local maxima (4x grid).
usage: python diag.py > diag.log"""
from acommon import *
sys.path.insert(0, 'D:/ai_ml_project/paper/evidence/deep_research/stress/adversarial')
from scipy.optimize import linear_sum_assignment

t00 = time.time()


def ang_err(pu, pv, tu, tv):
    """estimates (B,K) psi/phi (rad), truths (B,L) -> (B,K,L) max(|dpsi|,|dphi|) in deg"""
    return np.maximum(np.abs(np.rad2deg(pu[:, :, None] - tu[:, None, :])), np.abs(np.rad2deg(pv[:, :, None] - tv[:, None, :])))


def match(pu, pv, tu, tv):
    """Hungarian on max-angle error; returns det (B,L) bool (true path detected within 1 deg), est_ok (B,K)"""
    E = ang_err(pu, pv, tu, tv); B, K, L = E.shape
    det = np.zeros((B, L), bool); ok = np.zeros((B, K), bool)
    for i in range(B):
        r, c = linear_sum_assignment(E[i])
        g = E[i][r, c] <= 1.0; det[i, c[g]] = True; ok[i, r[g]] = True
    return det, ok


def cell_dist(U, V, tu, tv):
    """beam-cell (16 grid) distance max(|du|,|dv|)*16/(2pi), wrapped, from each estimate (B,K) to nearest truth -> (B,K)"""
    du = np.angle(np.exp(1j * (U[:, :, None] - tu[:, None, :]))); dv = np.angle(np.exp(1j * (V[:, :, None] - tv[:, None, :])))
    return (np.maximum(np.abs(du), np.abs(dv)) * 16 / (2 * np.pi)).min(2)


import adv_common as ADV
res = {}
for snr in [15, 0]:
    for bits in [None] + BITS:
        t0 = time.time()
        b, Y = m5_bank(500, snr, STRESS_SEEDS[snr], bits)
        lab = 'clean' if bits is None else f'b{bits}'
        H = to_ant(Y); L = 3
        U, V = C.nomp_greedy(H, L, 4, 3)
        ps, ph = C.uv_to_angles(U, V)
        rec = dict(snr=snr, bits=bits)
        # (0) identity
        if bits is not None:
            z = np.load(f'D:/ai_ml_project/paper/evidence/deep_research/stress/cache/preds__M5_{bits}_snr{snr}.npz')
            rec['nomp_vs_stress_cache_maxdiff_rad'] = float(max(np.nanmax(np.abs(z['NOMP__psi'] - ps)), np.nanmax(np.abs(z['NOMP__phi'] - ph))))
            Ypre = SS.generate(b, 'clean')
            rec['adc_vs_independent_adc_maxdiff'] = float(np.abs(SS.adc(Ypre, bits, 'peak') - ADV.adc_peak(Ypre, bits)).max())
            rec['Y_equals_adc_of_clean'] = bool(np.array_equal(Y, SS.adc(Ypre, bits, 'peak')))
        Sx = stats((ps, ph), b); rec['pd_nomp'] = pd(Sx)
        tu = np.pi * np.cos(b['psi'][:, :L]); tv = np.pi * np.cos(b['phi'][:, :L])
        # (1) per-rank detection
        det, ok = match(ps, ph, b['psi'][:, :L], b['phi'][:, :L])
        rec['det_rate_by_rank'] = det.mean(0).tolist()
        rec['n_detected_hist'] = np.bincount(det.sum(1), minlength=4).tolist()
        # (2) first greedy detection
        NF = 64; X = np.fft.fft2(H, s=(NF, NF)); idx = np.abs(X).reshape(len(H), -1).argmax(1)
        u1, v1 = C.newton(H, -2 * np.pi * (idx // NF) / NF, 2 * np.pi * (idx % NF) / NF)
        p1, f1 = C.uv_to_angles(u1[:, None], v1[:, None])
        E1 = ang_err(p1, f1, b['psi'][:, :L], b['phi'][:, :L])[:, 0, :]
        rec['first_within1deg_strongest'] = float(np.mean(E1[:, 0] <= 1)); rec['first_within1deg_any'] = float(np.mean(E1.min(1) <= 1))
        d1 = cell_dist(u1[:, None], v1[:, None], tu, tv)[:, 0]
        rec['first_cell_dist_quantiles'] = np.percentile(d1, [25, 50, 75, 90]).tolist()
        # (3) misses: distance of the unmatched estimates to the nearest truth
        dm = cell_dist(U, V, tu, tv)[~ok]
        rec['n_bad_estimates'] = int(dm.size)
        rec['bad_est_dist_frac'] = dict(lt0p25=float(np.mean(dm < 0.25)), c0p25_1=float(np.mean((dm >= 0.25) & (dm < 1))),
                                        c1_2=float(np.mean((dm >= 1) & (dm < 2))), ge2=float(np.mean(dm >= 2))) if dm.size else None
        # duplicates: two estimates within 0.5 cell of each other
        du = np.angle(np.exp(1j * (U[:, :, None] - U[:, None, :]))); dv = np.angle(np.exp(1j * (V[:, :, None] - V[:, None, :])))
        dd = np.maximum(np.abs(du), np.abs(dv)) * 16 / (2 * np.pi) + np.eye(L)[None] * 99
        rec['frac_scenes_with_duplicate_estimates'] = float(np.mean(dd.min((1, 2)) < 0.5))
        # (4) search loss vs objective
        Uo, Vo = C.cyclic(H, tu, tv, 3)
        So = stats(C.uv_to_angles(Uo, Vo), b); rec['pd_oracle'] = pd(So)
        ro, rn = C.residual_energy(H, Uo, Vo), C.residual_energy(H, U, V)
        lower = ro < rn; rec['frac_oracle_lower_residual'] = float(lower.mean())
        Uc = np.where(lower[:, None], Uo, U); Vc = np.where(lower[:, None], Vo, V)
        rec['pd_oracle_if_lower_residual'] = pd(stats(C.uv_to_angles(Uc, Vc), b))
        rec['rel_residual_median_nomp_oracle'] = [float(np.median(rn / (np.abs(H) ** 2).sum((1, 2)))), float(np.median(ro / (np.abs(H) ** 2).sum((1, 2))))]
        # (5) peak structure on the 4x FFT grid
        A = np.abs(X); A = A / A.max((1, 2), keepdims=True)
        qi = np.round(-tu * NF / (2 * np.pi)).astype(int) % NF; pi_ = np.round(tv * NF / (2 * np.pi)).astype(int) % NF
        val = np.take_along_axis(A.reshape(len(A), -1), qi * NF + pi_, 1)
        rec['fft_rel_value_at_truth_by_rank_median'] = np.median(val, 0).tolist()
        mx = scipy.ndimage.maximum_filter(A, size=(1, 3, 3), mode='wrap'); lm = A >= mx
        ranks = np.zeros((len(A), L))
        for i in range(len(A)):
            qq, pp = np.nonzero(lm[i]); vals = A[i, qq, pp]; order = np.argsort(-vals); qq, pp = qq[order], pp[order]
            for l in range(L):
                dq = np.abs((qq - qi[i, l] + NF // 2) % NF - NF // 2); dp = np.abs((pp - pi_[i, l] + NF // 2) % NF - NF // 2)
                near = np.nonzero(np.maximum(dq, dp) <= 2)[0]          # within 0.5 beam cell (2 bins of the 4x grid)
                ranks[i, l] = near[0] + 1 if near.size else 99
        rec['localmax_rank_of_truth_by_path_median'] = np.median(ranks, 0).tolist()
        rec['frac_truth_paths_in_top3_localmax'] = float(np.mean(ranks <= 3)); rec['frac_truth_paths_in_top6_localmax'] = float(np.mean(ranks <= 6))
        rec['frac_truth_paths_no_localmax_within_half_cell'] = float(np.mean(ranks == 99))
        rec['n_localmax_median'] = float(np.median(lm.sum((1, 2))))
        res[f'{lab}_snr{snr}'] = rec
        print(f'--- {lab} SNR {snr}  ({time.time() - t0:.0f}s)', flush=True)
        for k, v in rec.items():
            print(f'    {k}: {np.round(v, 4).tolist() if isinstance(v, list) else (round(v, 4) if isinstance(v, float) else v)}', flush=True)
save_json(res, f'{HERE}/diag.json')
print(f'total {time.time() - t00:.0f}s')
