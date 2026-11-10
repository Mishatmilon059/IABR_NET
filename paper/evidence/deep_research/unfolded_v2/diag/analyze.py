"""Tasks 2-5 analysis on the candidate pools produced by gen_cands.py (and configs.py for task 5).
Selection rules (all legal: use only y, the candidate set and, where stated, the noise variance estimated from the NOMP residual):
  nomp            : NOMP os4 rc3 (baseline)
  ml              : min LS residual over the pool (ML over the candidate set)
  ml_sep          : min residual among candidates obeying the benchmark's min-separation rule (pi/6 in (psi,phi)); fallback ml
  mll_sep         : max marginal likelihood (gains ~ CN(0,I/L), estimated noise var) + separation rule
  mapU_sep        : mll_sep + log uniform-angle prior Jacobian in (u,v) (MAP in the (u,v) domain)
  mass_sep        : mll_sep + log posterior mass of a +-1 deg window around every angle component (Laplace/Gaussian in u,
                    Fisher 10880|x|^2/sigma^2, uniform prior in angle) = approx. log P(all components within 1 deg | y)
  mbr             : minimum-Bayes-risk / voting: maximise sum_c' w_c' * (#components of c within 1 deg of c'), w = softmax(mass_sep)
                    over de-duplicated candidates (approximate posterior-expected Pd)
  *_ro            : the same followed by the Bayesian endfire read-out (each component moved to the angle maximising the
                    posterior mass of its +-1 deg window; changes only components near endfire)
  postmean        : mass_sep choice, each path replaced by the posterior-weighted mean of matching paths (within 2 deg) over candidates
  oracle          : best candidate per scene by the true error (upper bound = headroom of a perfect learned selector)
  learned         : HistGradientBoosting regressor of #correct components on candidate features, trained on the 'train' bank pool
                    (seed 70,000,000), evaluated on the official subset (pool without heat-map families)
usage: python analyze.py      stdout -> analyze.log, results -> diag.json (merged)"""
import glob, json, time, sys
from dcommon import *
from sklearn.ensemble import HistGradientBoostingRegressor
NEG = -1e12
D1 = np.deg2rad(1.0)


def load_pool(bank):
    fs = sorted(glob.glob(f'{BIG}/cands_{bank}/chunk_*.npz'))
    zs = [dict(np.load(f)) for f in fs]
    P = {k: np.concatenate([z[k] for z in zs]) for k in zs[0] if k != 'fam'}
    P['fam'] = zs[0]['fam']
    return P


def window_logmass(U, V, gain2, s2, grid=21):
    """log of prod over components of (1/pi) int_{angle+-1deg} exp(-wrap(pi cos a - u_hat)^2 / (2 std_u^2)) da"""
    std = np.sqrt(s2[:, None, None] / (10880 * np.maximum(gain2, 1e-12)))      # (B,K,3)
    tot = 0
    for W in (U, V):
        w = np.angle(np.exp(1j * W)); a0 = np.arccos(np.clip(w / np.pi, -1, 1))
        off = np.linspace(-D1, D1, grid)
        a = np.clip(a0[..., None] + off, 0, np.pi)
        d = np.angle(np.exp(1j * (np.pi * np.cos(a) - w[..., None])))
        f = np.exp(-d ** 2 / (2 * std[..., None] ** 2))
        m = np.trapezoid(f, a, axis=-1) / np.pi
        tot = tot + np.log(np.maximum(m, 1e-300)).sum(-1)
    return tot


def endfire_readout(U, V, gain2, s2, thr_deg=25.0, step_deg=0.02):
    """Bayesian read-out per component: angle a* maximising the posterior mass of [a*-1deg, a*+1deg] under the Gaussian-in-u
    likelihood (centre u_hat, Fisher std) and the uniform angle prior; applied to components within thr_deg of endfire.
    Returns psi, phi arrays (B,3)."""
    out = []
    std = np.sqrt(s2[:, None] / (10880 * np.maximum(gain2, 1e-12)))         # (B,3)
    g = np.deg2rad(np.arange(0, 180 + step_deg / 2, step_deg)); hw = int(round(1.0 / step_deg))
    cg = np.pi * np.cos(g)
    for W in (U, V):
        w = np.angle(np.exp(1j * W)); a0 = np.arccos(np.clip(w / np.pi, -1, 1)); a = a0.copy()
        sel = np.abs(np.cos(a0)) > np.cos(np.deg2rad(thr_deg))
        ii = np.argwhere(sel)
        for s in range(0, len(ii), 400):
            jj = ii[s:s + 400]; wu = w[jj[:, 0], jj[:, 1]]; sd = std[jj[:, 0], jj[:, 1]]
            d = np.angle(np.exp(1j * (cg[None] - wu[:, None])))
            f = np.exp(-d ** 2 / (2 * sd[:, None] ** 2))
            cs = np.concatenate([np.zeros((len(jj), 1)), np.cumsum(f, 1)], 1)
            lo = np.clip(np.arange(len(g)) - hw, 0, len(g)); hi = np.clip(np.arange(len(g)) + hw + 1, 0, len(g))
            wm = cs[:, hi] - cs[:, lo]
            a[jj[:, 0], jj[:, 1]] = g[wm.argmax(1)]
        out.append(a)
    return out


def pd_snr(nc, snr):
    ss = sorted(set(snr.astype(int).tolist()))
    per = {int(s): float(nc[snr.astype(int) == s].mean() / 6) for s in ss}
    per['mean'] = float(np.mean([per[s] for s in ss]))
    return per


def boot_diff(a, b, snr, B=2000, seed=0):
    """paired bootstrap of mean-over-SNR Pd difference a-b (per-scene correct counts), resampling within SNR groups"""
    rng = np.random.default_rng(seed); snr = snr.astype(int); ss = sorted(set(snr.tolist())); acc = np.zeros(B)
    for s in ss:
        ix = np.where(snr == s)[0]; R = rng.integers(0, len(ix), (B, len(ix)))
        acc += ((a[ix] - b[ix])[R].mean(1)) / 6
    acc /= len(ss)
    return [float(np.mean(acc)), float(np.percentile(acc, 2.5)), float(np.percentile(acc, 97.5))]


def pairwise(P, legal):
    """pairwise #correct of candidate c judged against candidate c' as truth: (B,K,K), and duplicate mask"""
    ps, fs = uv_to_angles(P['U'][:, legal], P['V'][:, legal]); B, K, _ = ps.shape
    M = np.zeros((B, K, K), np.int8); dup = np.zeros((B, K), bool)
    for j in range(K):
        e, _ = vmatch(ps[:, j], fs[:, j], ps, fs)
        M[:, :, j] = (e <= 1.0).sum((2, 3))
        if j > 0:
            same = (e[:, :j].max((2, 3)) < 0.01).any(1)
            dup[:, j] = same
    return M, dup


def rules(P, legal, train_model=None, feats_fn=None):
    """returns dict rule -> chosen candidate index (into the legal list) or a (psi,phi) override"""
    B = len(P['snr']); s2 = P['s2hat']
    res = P['res'][:, legal]; mll = P['mll_est'][:, legal]; lpr = P['lprior'][:, legal]; sepok = P['sep'][:, legal] >= SEP
    lm = window_logmass(P['U'][:, legal], P['V'][:, legal], P['gain2'][:, legal], s2)
    pen = np.where(sepok, 0.0, NEG)
    S = dict(nomp=np.zeros(B, int), ml=res.argmin(1))
    S['ml_sep'] = np.where(sepok.any(1), np.where(sepok, -res, -np.inf).argmax(1), res.argmin(1))
    S['mll_sep'] = (mll + pen).argmax(1)
    S['mapU_sep'] = (mll + lpr + pen).argmax(1)
    S['mass_nosep'] = (mll + lm).argmax(1)
    S['mass_sep'] = (mll + lm + pen).argmax(1)
    M, dup = pairwise(P, legal)
    sc = mll + lm + pen; sc = np.where(dup, -np.inf, sc); w = np.exp(sc - sc.max(1, keepdims=True)); w /= w.sum(1, keepdims=True)
    exp_corr = (M * w[:, None, :]).sum(2)                                     # expected #correct of each candidate
    S['mbr'] = exp_corr.argmax(1)
    extra = dict(w=w, M=M, exp_corr=exp_corr, lm=lm)
    if train_model is not None:
        X = feats_fn(P, legal, extra)
        pred = train_model.predict(X.reshape(-1, X.shape[-1])).reshape(B, -1)
        S['learned'] = pred.argmax(1)
    return S, extra


def feats(P, legal, extra):
    s2 = P['s2hat'][:, None]; res = P['res'][:, legal] / s2; mll = P['mll_est'][:, legal]
    B, K = res.shape
    fam = P['fam'][legal]; fams = ['nomp', 'nomp_os8rc6', 'fftcomb', 'branch', 'swap', 'random']
    g2 = np.sort(P['gain2'][:, legal] / s2[..., None], -1)
    ps, fs = uv_to_angles(P['U'][:, legal], P['V'][:, legal])
    endf = ((np.abs(np.cos(ps)) > 0.95) | (np.abs(np.cos(fs)) > 0.95)).sum(-1)
    F = [res - res.min(1, keepdims=True), mll - mll.max(1, keepdims=True), P['lprior'][:, legal], np.rad2deg(P['sep'][:, legal]),
         (P['sep'][:, legal] >= SEP).astype(float), np.log10(g2[..., 0]), np.log10(g2[..., 1]), np.log10(g2[..., 2]),
         extra['lm'] - extra['lm'].max(1, keepdims=True), extra['exp_corr'], extra['w'], endf.astype(float),
         np.repeat(-10 * np.log10(P['s2hat'])[:, None], K, 1), np.argsort(np.argsort(res, 1), 1).astype(float)]
    F += [np.repeat((fam == f).astype(float)[None], B, 0) for f in fams]
    return np.stack(F, -1)


def chosen_counts(P, legal, idx):
    return np.take_along_axis(P['ncorrect'][:, legal], idx[:, None], 1)[:, 0]


def readout_counts(P, legal, idx):
    U = np.take_along_axis(P['U'][:, legal], idx[:, None, None], 1)[:, 0]; V = np.take_along_axis(P['V'][:, legal], idx[:, None, None], 1)[:, 0]
    g2 = np.take_along_axis(P['gain2'][:, legal], idx[:, None, None], 1)[:, 0]
    ps, fs = endfire_readout(U, V, g2, P['s2hat'])
    e, _ = vmatch(P['psi_t'], P['phi_t'], ps, fs)
    return (e[:, 0] <= 1).sum((1, 2))


def postmean_counts(P, legal, idx, extra):
    ps, fs = uv_to_angles(P['U'][:, legal], P['V'][:, legal]); B, K, _ = ps.shape
    p0 = np.take_along_axis(ps, idx[:, None, None], 1)[:, 0]; f0 = np.take_along_axis(fs, idx[:, None, None], 1)[:, 0]
    e, k = vmatch(p0, f0, ps, fs)          # candidates matched to the chosen one
    pp = np.take_along_axis(ps[..., PERMS], k[..., None, None], 2)[:, :, 0]; fp = np.take_along_axis(fs[..., PERMS], k[..., None, None], 2)[:, :, 0]
    near = (e.max(2) <= 2.0)                                                 # (B,K,3) path within 2 deg in both comps
    w = extra['w'][..., None] * near
    w = w / np.maximum(w.sum(1, keepdims=True), 1e-300)
    pm = (w * pp).sum(1); fm = (w * fp).sum(1)
    e2, _ = vmatch(P['psi_t'], P['phi_t'], pm, fm)
    return (e2[:, 0] <= 1).sum((1, 2))


def sl(P, a, b):
    return {k: (v if k == 'fam' else v[a:b]) for k, v in P.items()}


def evaluate(P, legal, model):
    """memory-light: rules and counts per chunk of 400 scenes (every rule is per-scene)"""
    out = {}; N = len(P['snr'])
    for a in range(0, N, 400):
        Q = sl(P, a, a + 400); S, E = rules(Q, legal, model, feats)
        NCs = {r: chosen_counts(Q, legal, i) for r, i in S.items()}
        NCs['oracle'] = Q['ncorrect'][:, legal].max(1)
        for r in ['nomp', 'ml_sep', 'mass_sep', 'mbr']:
            NCs[r + '_ro'] = readout_counts(Q, legal, S[r])
        NCs['postmean'] = postmean_counts(Q, legal, S['mass_sep'], E)
        NCs['oracle_seeded_ref'] = Q['ncorrect'][:, -2]
        nomp = Q['ncorrect'][:, 0]; fam = Q['fam'][legal]
        for f in ['nomp_os8rc6', 'heat', 'fftcomb', 'heatcomb', 'branch', 'swap', 'random']:
            if (fam == f).any():
                NCs[f'oracle_nomp+{f}'] = np.maximum(nomp, Q['ncorrect'][:, legal][:, fam == f].max(1))
        for r, i in S.items():
            NCs['changed__' + r] = (i != 0).astype(int)
        for r, v in NCs.items():
            out.setdefault(r, []).append(v)
    return {r: np.concatenate(v) for r, v in out.items()}


if __name__ == '__main__':
    t0 = time.time(); OUTJ = {}
    extra_banks = sys.argv[1:]                       # e.g. nosep sep10 (prior-violation banks)
    pools = {b: load_pool(b) for b in ['official', 'val', 'train'] + extra_banks}
    for b, P in pools.items():
        print(b, 'scenes', len(P['snr']), 'candidates', len(P['fam']), dict(zip(*np.unique(P['fam'], return_counts=True))))
    fam_off = pools['official']['fam']
    legal_all = np.where(~np.isin(fam_off, ['oracle_ref', 'truth_ref']))[0]
    legal_nh = np.where(~np.isin(fam_off, ['oracle_ref', 'truth_ref', 'heat', 'heatcomb']))[0]
    legal_v = np.where(~np.isin(pools['val']['fam'], ['oracle_ref', 'truth_ref']))[0]
    assert list(pools['val']['fam'][legal_v]) == list(fam_off[legal_nh])
    # ---- learned selector trained on the 'train' pool (features built per chunk)
    Pt = pools['train']; Xs = []
    for a in range(0, len(Pt['snr']), 400):
        Q = sl(Pt, a, a + 400); _, Et = rules(Q, legal_v); Xs.append(feats(Q, legal_v, Et).astype(np.float32))
    Xt = np.concatenate(Xs); yt = Pt['ncorrect'][:, legal_v].astype(float)
    model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, random_state=0)
    model.fit(Xt.reshape(-1, Xt.shape[-1]), yt.ravel()); del Xs, Xt
    print(f'learned selector trained on {len(yt)} train scenes x {len(legal_v)} candidates ({time.time()-t0:.0f}s)', flush=True)
    res_tab = {}
    runs = [('official_allpool', pools['official'], legal_all), ('official_noheat', pools['official'], legal_nh),
            ('val_noheat', pools['val'], legal_v)] + [(f'{b}_noheat', pools[b], legal_v) for b in extra_banks]
    for name, P, legal in runs:
        NCs = evaluate(P, legal, model)
        snr = P['snr']; nomp = P['ncorrect'][:, 0]
        tab = {}
        print()
        print(f'=== {name}: {len(snr)} scenes, {len(legal)} legal candidates; Pd mean over SNR | per SNR | diff vs NOMP [95% CI]')
        for r, nc in NCs.items():
            if r.startswith('changed__'):
                continue
            per = pd_snr(nc, snr); d = boot_diff(nc, nomp, snr)
            tab[r] = dict(pd=per, diff_vs_nomp=d)
            print(f'{r:22s} {per["mean"]:.4f} | ' + ' '.join(f'{per[s]:.4f}' for s in sorted(k for k in per if k != 'mean')) +
                  f' | {d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}]{"*" if d[1] > 0 or d[2] < 0 else ""}')
        ch = {r[9:]: round(float(v.mean()), 3) for r, v in NCs.items() if r.startswith('changed__')}
        print("fraction of scenes where the rule changes NOMP's answer:", ch)
        tab['changed_frac'] = ch
        res_tab[name] = tab
        if name == 'official_allpool':
            OUTJ['selection_counts'] = {r: v.tolist() for r, v in NCs.items() if not r.startswith('changed__')}
            OUTJ['selection_snr'] = P['snr'].tolist()
        elif name.split('_')[0] in extra_banks:
            OUTJ[f'counts_{name}'] = {r: v.tolist() for r, v in NCs.items() if not r.startswith('changed__')}
        print(f'({time.time()-t0:.0f}s)', flush=True)
    OUTJ['selection'] = res_tab
    print()
    print('Task 4 (official subset, all pool): Pd at -10/-5/0 dB')
    t = res_tab['official_allpool']
    for s in [-10, -5, 0]:
        print(f'{s:4d} dB | ' + ' | '.join(f'{r} {t[r]["pd"][s]:.4f}' for r in ['nomp', 'ml', 'mass_sep', 'mbr', 'postmean', 'oracle']))
    json.dump(OUTJ, open(f'{HERE}/analyze.json', 'w'), indent=1)
    print(f'done {time.time()-t0:.0f}s')
