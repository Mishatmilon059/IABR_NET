"""Adversarial replicate of the unfolded-decoder claims on FRESH seeds (not used by any earlier script).
Seeds: 9700 + 0..(n-1) per call; one bank per call: 8 SNR groups (-10..25 dB, step 5) x NPER scenes, L = 3 given, ideal hardware.
Decoders A / B and named classical baselines are produced by build/ev.py run_method (same code path as the build's own ev.py).
One extra classical variant is defined here: NOMP candidate pool (os 4/rc 3, os 8/rc 6, os 16/rc 12) with per-scene selection
of the candidate with the lowest least-squares residual energy (ML-style selection, no truth used).
usage: python adv_fresh.py NPER SEED [METHOD ...]   (stdout -> logs/fresh_<SEED>.log, results json + npz in this folder)"""
import os, sys, time, json
os.environ['IABR2_SUITE'] = 'D:/ai_ml_project'
os.environ.setdefault('THREADS', '3')
BUILD = 'D:/ai_ml_project/paper/evidence/deep_research/unfolded_v2/build'
ADV = 'D:/ai_ml_project/paper/evidence/deep_research/unfolded_v2/adversarial'
CACHE = 'C:/Users/Milon/iabr_deep_research_cache/dr_cache/unfolded_v2/adversarial'
os.makedirs(CACHE, exist_ok=True)
sys.path.insert(0, BUILD)
os.chdir(BUILD)
import numpy as np
import ev
import bcommon as BC

C = ev.C
NPER = int(sys.argv[1]); SEED = int(sys.argv[2])
SNRS = list(range(-10, 26, 5))
DEFAULT = ['A', 'B', 'NOMP', 'NOMP os8 rc6', 'NOMP os8 rc6+SC tuned', 'NOMP+SC tuned', 'consistent NOMP', 'classical skeleton']
METHODS = sys.argv[3:] if len(sys.argv) > 3 else DEFAULT + ['NOMP os16 rc12', 'NOMP pool+resid-select']
PAIRS = [('A', 'NOMP'), ('A', 'NOMP os8 rc6'), ('A', 'NOMP os8 rc6+SC tuned'), ('A', 'consistent NOMP'), ('A', 'classical skeleton'),
         ('A', 'NOMP os16 rc12'), ('A', 'NOMP pool+resid-select'), ('A', 'B'), ('B', 'NOMP os8 rc6')]


def steer(U, V):
    """atoms a_l[n,m] = exp(-j n u_l + j m v_l), n,m = 0..15 -> (B, 256, L)"""
    n = np.arange(16.)
    E = np.exp(-1j * n[None, :, None, None] * U[:, None, None, :] + 1j * n[None, None, :, None] * V[:, None, None, :])
    return E.reshape(len(U), 256, U.shape[1])


def ls_residual(H, U, V):
    """per-scene LS residual energy ||y - A x||^2 with complex gains x (no truth used)"""
    A = steer(U, V); y = H.reshape(len(H), 256); out = np.zeros(len(H))
    for i in range(len(H)):
        x, *_ = np.linalg.lstsq(A[i], y[i], rcond=None)
        out[i] = np.sum(np.abs(y[i] - A[i] @ x) ** 2)
    return out


def pool_select(H, L):
    cands = [C.nomp_greedy(H, L, 4, 3), C.nomp_greedy(H, L, 8, 6), C.nomp_greedy(H, L, 16, 12)]
    res = np.stack([ls_residual(H, U, V) for U, V in cands])          # (K, B)
    k = res.argmin(0); idx = np.arange(len(H))
    U = np.stack([c[0] for c in cands])[k, idx]; V = np.stack([c[1] for c in cands])[k, idx]
    return U, V


def run(meth, d):
    H = ev.SS.to_ant(d['Y']); Lv = np.asarray(d['L']) if not np.isscalar(d['L']) else np.full(len(d['Y']), d['L'])
    if meth == 'NOMP os16 rc12':
        return ev.by_L(Lv, lambda ii, L: C.nomp_greedy(H[ii], L, 16, 12))
    if meth == 'NOMP pool+resid-select':
        return ev.by_L(Lv, lambda ii, L: pool_select(H[ii], L))
    return ev.run_method(meth, d, slice(None))


def main():
    t00 = time.time()
    print(f'fresh bank seed {SEED}, {NPER} scenes per SNR group x {len(SNRS)} groups; methods {METHODS}', flush=True)
    d = ev._fresh(NPER, 3, SNRS, SEED)
    snr = d['snr']; N = len(snr)
    print(f'scenes {N}, L values {sorted(set(np.asarray(d["L"]).tolist()))}, bits {sorted(set(d["bits"].tolist()))}', flush=True)
    # sanity: LS residual at the TRUE (u,v) should be about the noise energy, and NOMP's own estimate should be no worse than the truth
    Htrue = ev.SS.to_ant(d['Y']); ut = np.pi * np.cos(d['psi']); vt = np.pi * np.cos(d['phi'])
    rt = ls_residual(Htrue, ut, vt)
    # if the atom convention is right, residual at the true angles ~ (256 - L) * noise variance (ratio ~ constant across SNR)
    ratio = rt / ((256 - 3) * 10 ** (-snr / 10))
    print('sanity: residual at TRUE (u,v) / ((256-L) * 10^(-SNR/10)) by SNR group: ' +
          ' '.join(f'{s}:{np.median(ratio[snr == s]):.2f}' for s in SNRS), flush=True)
    S = {}; preds_all = {}
    for m in METHODS:
        t0 = time.time()
        preds = run(m, d)
        preds_all[m] = preds
        S[m] = BC.stats(preds, d)
        print(f'{m:28s} Pd(mean over SNR) {BC.pd(S[m], snr):.4f}  ({time.time() - t0:.0f}s)', flush=True)
    # per-SNR table
    print('\nper-SNR pd_paper:', flush=True)
    print('method'.ljust(28) + ''.join(f'{s:>7d}' for s in SNRS), flush=True)
    for m in METHODS:
        per = [C.metric_by_snr(S[m], snr)[s]['pd_paper'] for s in SNRS]
        print(m.ljust(28) + ''.join(f'{p:7.3f}' for p in per), flush=True)
    # paired bootstrap (2000 resamples within SNR groups, seed 0)
    pairs = [p for p in PAIRS if p[0] in S and p[1] in S]
    pdm, cis, diffs = BC.boot_pairs(S, snr, pairs, B=2000)
    print('\npaired bootstrap, difference of mean-over-SNR pd_paper [95% CI]  (* = CI excludes 0):', flush=True)
    res = {}
    for k, v in diffs.items():
        flag = '*' if (v[1] > 0 or v[2] < 0) else ''
        print(f'  {k:45s} {v[0]:+.4f} [{v[1]:+.4f}, {v[2]:+.4f}] {flag}', flush=True)
        res[k] = dict(diff=v[0], lo=v[1], hi=v[2], sig=bool(flag))
    out = dict(seed=SEED, nper=NPER, n=N, pd={m: BC.pd(S[m], snr) for m in METHODS}, ci={m: cis[m] for m in cis},
               pd_by_snr={m: {str(s): C.metric_by_snr(S[m], snr)[s]['pd_paper'] for s in SNRS} for m in METHODS}, pairs=res,
               seconds=time.time() - t00)
    tag = f'{SEED}_n{NPER}'
    with open(f'{ADV}/fresh_{tag}.json', 'w') as f:
        json.dump(out, f, indent=1)
    np.savez(f'{CACHE}/stats_{tag}.npz', **{m.replace(' ', '_').replace('+', 'p').replace('/', '-'): S[m] for m in METHODS}, snr=snr)
    print(f'done in {time.time() - t00:.0f}s', flush=True)


if __name__ == '__main__':
    main()
