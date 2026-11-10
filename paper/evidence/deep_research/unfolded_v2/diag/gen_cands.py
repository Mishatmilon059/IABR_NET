"""Tasks 2-4: generate a pool of candidate L=3 solutions per scene (all refined with the NOMP Newton + cyclic machinery,
Mamandipoor et al. 2016) and store their objective terms and their true error.
usage: python gen_cands.py <bank>   bank in {official (2000-scene subset, 250/SNR, with CNN-heatmap candidates), official_full,
val, train}. Chunks of 400 scenes, resumable (skips finished chunks). Arrays -> C: cache (BIG/cands_<bank>/chunk_*.npz).
Families (names saved per candidate in `fam`):
  nomp        : NOMP os4 rc3 (= cached NOMP_no_network)
  nomp_os8rc6 : NOMP os8 rc6 (best classical setting on the official bank)
  heat        : CNN-heatmap-seeded decoder (top-3 heat peaks -> Newton(4) -> cyclic 3)               [official only]
  fftcomb     : all 20 triples of the top-6 local maxima of |FFT64(y)| -> Newton(4) -> cyclic 3
  heatcomb    : all 20 triples of the top-6 heat-map local maxima -> Newton(4) -> cyclic 3             [official only]
  branch      : 12 greedy branches: 1st atom from top-4 FFT peaks, 2nd from top-3 residual peaks, 3rd greedy (cyclic 3 after each)
  swap        : 9 support swaps of the NOMP solution: drop path j, re-detect from the top-3 residual peaks, cyclic 3
  random      : 16 starts drawn from the benchmark prior (uniform psi,phi, min sep pi/6), cyclic 6 (rng seed 72,000,000 + offset)
  refs (NOT legal candidates, last two): oracle_ref = true (u,v) -> cyclic 3; truth_ref = the true (u,v)
Stored terms per candidate: res (LS residual energy), mll_true / mll_est (log marginal likelihood with gains ~ CN(0, I/L), noise
variance true 10^(-SNR/10) or estimated from the NOMP residual / 250), lprior (log Jacobian of the uniform-angle prior in (u,v)),
sep (min pairwise (psi,phi) distance, rad), gain2 (|LS gains|^2), ncorrect (components with |err| <= 1 deg), errsum."""
import sys, time
from dcommon import *
bank = sys.argv[1]
TEST = os.environ.get('DIAG_TEST') == '1'
if bank.startswith('official'):
    b = load_bank('official'); idx = subset(b, 250) if bank == 'official' else np.arange(len(b['L']))
    z = np.load(f'{CACHE}/net__official.npz'); H_all = z['Ht'][idx].astype(np.complex128); heat_all = z['heat'][idx].astype(np.float64)
    pt_all, ft_all, snr_all = b['psi'][idx].astype(np.float64), b['phi'][idx].astype(np.float64), b['snr'][idx].astype(np.float64)
else:
    z = np.load(f'{BIG}/bank_{bank}.npz'); H_all = z['Ht']; heat_all = None
    pt_all, ft_all, snr_all = z['psi'], z['phi'], z['snr']
N = len(H_all); OUT = f'{BIG}/cands_{bank}'; os.makedirs(OUT, exist_ok=True)
G64 = 64


def fft_peaks(r, k):
    X = np.abs(np.fft.fft2(r, s=(G64, G64)))
    mx = scipy.ndimage.maximum_filter(X, size=(1, 3, 3), mode='wrap')
    sc = np.where(X >= mx, X + X.max() + 1, X).reshape(len(r), -1)       # local maxima ranked first
    ix = np.argsort(-sc, 1)[:, :k]
    return -2 * np.pi * (ix // G64) / G64, 2 * np.pi * (ix % G64) / G64


def refine_single(r, u, v, it=4):
    B, k = u.shape
    uu, vv = newton(np.repeat(r, k, 0), u.ravel(), v.ravel(), iters=it)
    return uu.reshape(B, k), vv.reshape(B, k)


def fit_res(y, U, V):
    x, A = ls_gains(y, U, V); return y - np.einsum('bik,bk->bi', A, x).reshape(y.shape)


def prior_seeds(rng, B, n):
    out_p = np.zeros((B, n, 3)); out_f = np.zeros((B, n, 3))
    for i in range(B):
        for j in range(n):
            pts = []
            while len(pts) < 3:
                p = rng.uniform(0, np.pi, 2)
                if all(np.hypot(*(p - q)) >= SEP for q in pts): pts.append(p)
            pts = np.array(pts); out_p[i, j], out_f[i, j] = pts[:, 0], pts[:, 1]
    return np.pi * np.cos(out_p), np.pi * np.cos(out_f)


def candidates(y, heat, rng):
    C = []; fam = []
    def add(UV, f): C.append(UV); fam.append(f)
    Un, Vn = nomp_greedy(y, 3, 4, 3); add((Un, Vn), 'nomp')
    add(nomp_greedy(y, 3, 8, 6), 'nomp_os8rc6')
    if heat is not None:
        Uh, Vh = heat_peaks(heat, 3); Uh, Vh = refine_single(y, Uh, Vh); add(cyclic(y, Uh, Vh, 3), 'heat')
    pu, pv = fft_peaks(y, 6); pu, pv = refine_single(y, pu, pv)
    for c in combinations(range(6), 3):
        c = list(c); add(cyclic(y, pu[:, c], pv[:, c], 3), 'fftcomb')
    if heat is not None:
        hu, hv = heat_peaks(heat, 6); hu, hv = refine_single(y, hu, hv)
        for c in combinations(range(6), 3):
            c = list(c); add(cyclic(y, hu[:, c], hv[:, c], 3), 'heatcomb')
    for a in range(4):
        U1, V1 = cyclic(y, pu[:, [a]], pv[:, [a]], 3); r1 = fit_res(y, U1, V1)
        qu, qv = fft_peaks(r1, 3); qu, qv = refine_single(r1, qu, qv)
        for c in range(3):
            U2, V2 = cyclic(y, np.c_[U1, qu[:, c]], np.c_[V1, qv[:, c]], 3); r2 = fit_res(y, U2, V2)
            su, sv = fft_peaks(r2, 1); su, sv = refine_single(r2, su, sv)
            add(cyclic(y, np.c_[U2, su], np.c_[V2, sv], 3), 'branch')
    for j in range(3):
        keep = [k for k in range(3) if k != j]; r = fit_res(y, Un[:, keep], Vn[:, keep])
        qu, qv = fft_peaks(r, 3); qu, qv = refine_single(r, qu, qv)
        for c in range(3):
            add(cyclic(y, np.c_[Un[:, keep], qu[:, c]], np.c_[Vn[:, keep], qv[:, c]], 3), 'swap')
    ru, rv = prior_seeds(rng, len(y), 16)
    for j in range(16):
        add(cyclic(y, ru[:, j], rv[:, j], 6), 'random')
    U = np.stack([c[0] for c in C], 1); V = np.stack([c[1] for c in C], 1)
    return U, V, fam


t00 = time.time()
starts = [0] if TEST else range(0, N, 400)
for s in starts:
    fn = f'{BIG}/test_chunk.npz' if TEST else f'{OUT}/chunk_{s:05d}.npz'
    if os.path.exists(fn) and not TEST: continue
    t0 = time.time(); sl = slice(s, min(s + (40 if TEST else 400), N)); y = H_all[sl]; snr = snr_all[sl]; pt, ft = pt_all[sl], ft_all[sl]
    off = {'official': 0, 'official_full': 0, 'val': 100_000, 'train': 200_000, 'nosep': 300_000, 'sep10': 400_000}[bank]
    rng = np.random.default_rng(72_000_000 + off + s)
    U, V, fam = candidates(y, None if heat_all is None else heat_all[sl], rng)
    Ut, Vt = np.pi * np.cos(pt), np.pi * np.cos(ft); Uo, Vo = cyclic(y, Ut, Vt, 3)
    U = np.concatenate([U, Uo[:, None], Ut[:, None]], 1); V = np.concatenate([V, Vo[:, None], Vt[:, None]], 1); fam = fam + ['oracle_ref', 'truth_ref']
    B, K, _ = U.shape
    s2hat = residual_energy(y, U[:, 0], V[:, 0]) / 250
    terms = {}
    for tag, s2 in [('true', sigma2(snr)), ('est', s2hat)]:
        parts = [bayes_terms_s2(y, U[:, k], V[:, k], s2) for k in range(K)]
        terms[f'mll_{tag}'] = np.stack([p['mll'] for p in parts], 1)
    for key in ['res', 'lprior', 'sep']:
        terms[key] = np.stack([p[key] for p in parts], 1)
    terms['gain2'] = np.stack([np.abs(p['gains']) ** 2 for p in parts], 1)
    nc, es = score_uv(pt, ft, U, V)
    np.savez(fn, U=U, V=V, fam=np.array(fam), snr=snr, s2hat=s2hat, ncorrect=nc, errsum=es, psi_t=pt, phi_t=ft, **terms)
    print(f'{bank} chunk {s:5d}: {B} scenes x {K} candidates, {time.time()-t0:.1f}s (total {time.time()-t00:.0f}s); '
          f'Pd nomp {nc[:, 0].mean()/6:.4f} oracle-of-legal {nc[:, :-2].max(1).mean()/6:.4f}', flush=True)
print('done', bank)
