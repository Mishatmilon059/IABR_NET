"""Step 0: physics control for low-resolution phase shifters (stress family M4), no training, no learning.
The receiver KNOWS its own quantised codebooks Wq, Fq (stress_sim.quantise_codebook), so the measurement model is exactly
    Y = Wq^H H Fq + Z,   Z white CN(0, sigma^2)   (M4 banks: no other impairment).
The stress decoders instead use the IDEAL inverse Hant = W Y F^H, i.e. a mismatched model. Two known-codebook decoders:
  'LS-NOMP'   antenna-domain transform from the known codebooks, H_ls = pinv(Wq^H) Y pinv(Fq) (least squares; = exact inverse when
              Wq is full rank, minimum-norm projection when it is not), then the unchanged NOMP (os 4, rc 3) on H_ls. Conditioning of
              Wq, Fq is printed (noise in H_ls is coloured / amplified by the inverse).
  'MD-NOMP'   NOMP carried out in the MEASUREMENT domain with the exact dictionary G(u,v) = (Wq^H e_r(u)) (Fq^T e_t(v))^T
              (e_r[n] = exp(-j n u), e_t[m] = exp(j m v): the decoder's atom convention). Greedy: normalised correlation on a 64x64 grid
              -> 2-D zoom refinement (5x5 local grid, step halved 8 times) -> cyclic re-refinement of every path against the residual
              of the others (3 rounds, LS gains in the measurement domain). With white Z this is the least-squares = ML objective.
              Sanity: with Wq = W (b = 4) it must reproduce NOMP's Pd.
References: NOMP / NOMP+SC tuned on the ideal transform (the stress decoders), clean NOMP on the same scenes (b = 4 bank is identical to
clean), oracle-seeded (true (u,v) -> 3 cyclic rounds on the ideal Hant, as in the stress run) and oracle-seeded MD (true (u,v) ->
3 MD cyclic rounds; the known-codebook analogue). Gap closed = (method - NOMP) / (clean NOMP - NOMP).
Banks: stress M4 banks (seeds 5000 / 5015, 500 scenes, L = 3, identical Y to stress/run_stress.py: NOMP predictions compared with
stress/cache/preds__M4_*.npz) and fresh replicate banks (seeds 8400 / 8415, 1000 scenes) for every cell.
Paired bootstrap 2000 within the (single) SNR group, seed 0.  usage: python m4_physics.py [SMOKE]  (stdout -> m4_physics.log)"""
from icom import *

SMOKE = int(sys.argv[1]) if len(sys.argv) > 1 else 0
L = 3; NU = 64
nvec = np.arange(16.)


def gvec(U, Wc, side):
    """U (...,) -> (...,16) measurement-domain factor: rows g_r = Wq^H e_r(u); cols g_t = Fq^T e_t(v)."""
    if side == 'r':
        return np.exp(-1j * U[..., None] * nvec) @ np.conj(Wc)
    return np.exp(1j * U[..., None] * nvec) @ Wc


def md_atoms(U, V, Wq, Fq):
    gr = gvec(U, Wq, 'r'); gt = gvec(V, Fq, 't')                   # (B,K,16)
    return (gr[:, :, :, None] * gt[:, :, None, :]).reshape(U.shape[0], U.shape[1], 256).transpose(0, 2, 1)   # (B,256,K)


def md_ls(Y, U, V, Wq, Fq):
    A_ = md_atoms(U, V, Wq, Fq); AH = np.conj(np.transpose(A_, (0, 2, 1)))
    x = np.linalg.solve(AH @ A_ + 1e-9 * np.eye(U.shape[1])[None], AH @ Y.reshape(len(Y), -1, 1))[..., 0]
    return x, A_


def zoom(R, u, v, Wq, Fq, delta, levels=8):
    offs = np.array([-1, -0.5, 0, 0.5, 1.0])
    for _ in range(levels):
        Uc = u[:, None] + delta * offs; Vc = v[:, None] + delta * offs
        gr = gvec(Uc, Wq, 'r'); gt = gvec(Vc, Fq, 't')
        c = np.einsum('bkq,bqp,blp->bkl', np.conj(gr), R, np.conj(gt))
        obj = np.abs(c) ** 2 / ((np.abs(gr) ** 2).sum(-1)[:, :, None] * (np.abs(gt) ** 2).sum(-1)[:, None, :] + 1e-12)
        k = obj.reshape(len(R), -1).argmax(1)
        u = Uc[np.arange(len(R)), k // 5]; v = Vc[np.arange(len(R)), k % 5]
        delta = delta / 2
    return u, v


def md_cyclic(Y, U, V, Wq, Fq, rounds=3):
    U, V = U.copy(), V.copy()
    for _ in range(rounds):
        for j in range(U.shape[1]):
            x, A_ = md_ls(Y, U, V, Wq, Fq)
            others = np.einsum('bik,bk->bi', np.delete(A_, j, 2), np.delete(x, j, 1)).reshape(Y.shape)
            U[:, j], V[:, j] = zoom(Y - others, U[:, j], V[:, j], Wq, Fq, 2 * np.pi / 256, levels=6)
    return U, V


def md_nomp(Y, L, Wq, Fq, rc=3):
    B = len(Y); grid = 2 * np.pi * np.arange(NU) / NU
    Gr = gvec(grid, Wq, 'r'); Gt = gvec(grid, Fq, 't')            # (64,16)
    nr = (np.abs(Gr) ** 2).sum(1); nt = (np.abs(Gt) ** 2).sum(1)
    U = np.zeros((B, 0)); V = np.zeros((B, 0)); R = Y.copy()
    for _ in range(L):
        c = np.einsum('kq,bqp,lp->bkl', np.conj(Gr), R, np.conj(Gt))
        obj = np.abs(c) ** 2 / (nr[:, None] * nt[None, :] + 1e-12)
        k = obj.reshape(B, -1).argmax(1)
        u, v = zoom(R, grid[k // NU], grid[k % NU], Wq, Fq, 2 * np.pi / NU, levels=8)
        U = np.c_[U, u]; V = np.c_[V, v]
        U, V = md_cyclic(Y, U, V, Wq, Fq, rc)
        x, A_ = md_ls(Y, U, V, Wq, Fq); R = Y - np.einsum('bik,bk->bi', A_, x).reshape(Y.shape)
    return U, V


def run_bank(b, Y, Yclean, bits):
    Wq, Fq = SS.quantise_codebook(SS.W_IDEAL, bits), SS.quantise_codebook(SS.F_IDEAL, bits)
    Wp = np.linalg.pinv(np.conj(Wq).T); Fp = np.linalg.pinv(Fq)
    P = {}
    for s0 in range(0, len(Y), 400):
        sl = slice(s0, s0 + 400); Yc = Y[sl]; H = SS.to_ant(Yc)
        U, V = C.nomp_greedy(H, L, 4, 3); out = {'NOMP': C.uv_to_angles(U, V)}
        out['NOMP+SC tuned'] = C.uv_to_angles(*A.selfcal_tuned(H, U, V))
        out['clean NOMP'] = C.uv_to_angles(*C.nomp_greedy(SS.to_ant(Yclean[sl]), L, 4, 3))
        Hls = Wp[None] @ Yc @ Fp[None]
        Ul, Vl = C.nomp_greedy(Hls, L, 4, 3); out['LS-NOMP'] = C.uv_to_angles(Ul, Vl)
        out['LS-NOMP+SC tuned'] = C.uv_to_angles(*A.selfcal_tuned(Hls, Ul, Vl))
        out['MD-NOMP'] = C.uv_to_angles(*md_nomp(Yc, L, Wq, Fq))
        U0 = np.pi * np.cos(b['psi'][sl, :L]); V0 = np.pi * np.cos(b['phi'][sl, :L])
        out['oracle-seeded'] = C.uv_to_angles(*C.cyclic(H, U0, V0, 3))
        out['oracle-seeded MD'] = C.uv_to_angles(*md_cyclic(Yc, U0, V0, Wq, Fq, 3))
        for m, v in out.items():
            P.setdefault(m, []).extend(as_list(v))
    return P, Wq, Fq


meths = ['NOMP', 'NOMP+SC tuned', 'LS-NOMP', 'LS-NOMP+SC tuned', 'MD-NOMP', 'oracle-seeded', 'oracle-seeded MD', 'clean NOMP']
res = dict(banks={}, conditioning={})
_JP = f'{HERE}/m4_physics{"_smoke" if SMOKE else ""}.json'
if os.path.exists(_JP):                                  # resumable: banks already in the json are skipped
    res['banks'] = json.load(open(_JP))['banks']
for bits in [4, 3, 2, 1]:
    Wq = SS.quantise_codebook(SS.W_IDEAL, bits); s = np.linalg.svd(Wq, compute_uv=False)
    res['conditioning'][f'b{bits}'] = dict(sv_max=float(s[0]), sv_min=float(s[-1]), rank=int((s > 1e-10 * s[0]).sum()),
                                           cond=float(s[0] / s[-1]) if s[-1] > 1e-10 * s[0] else float('inf'),
                                           noise_gain_ls_db=float(10 * np.log10(np.mean(1 / s[s > 1e-10 * s[0]] ** 2) / np.mean(1 / np.linalg.svd(SS.W_IDEAL, compute_uv=False) ** 2))))
    print(f'b={bits}: Wq singular values max {s[0]:.3f} min {s[-1]:.2e}, rank {res["conditioning"][f"b{bits}"]["rank"]}, cond '
          f'{res["conditioning"][f"b{bits}"]["cond"]:.3g}, LS noise gain per side (mean 1/s^2 vs ideal) {res["conditioning"][f"b{bits}"]["noise_gain_ls_db"]:+.2f} dB', flush=True)
for set_, seeds, n in [('stress', STRESS_SEEDS, 500), ('fresh', FRESH_SEEDS, 1000)]:
    for snr in [15, 0]:
        nn_ = SMOKE or n
        b = SS.base_scenes(n, L, float(snr), seeds[snr])
        b = {k: (v[:nn_] if isinstance(v, np.ndarray) and v.ndim and len(v) == n else v) for k, v in b.items()}
        Yclean = SS.generate(b, 'clean')
        for bits in [4, 3, 2, 1]:
            t0 = time.time(); name = f'{set_}_M4_b{bits}_snr{snr}'
            if name in res['banks']:
                print(f'skip {name} (done)', flush=True); continue
            Y = SS.generate(b, 'M4', bits)
            P, Wq, Fq = run_bank(b, Y, Yclean, bits)
            S = {m: stats(P[m], b) for m in meths}
            rec = dict(n=nn_, seed=seeds[snr], bits=bits, snr=snr)
            if set_ == 'stress':
                z = np.load(f'D:/ai_ml_project/paper/evidence/deep_research/stress/cache/preds__M4_{bits}_snr{snr}.npz')
                pk = pack(P['NOMP'], 3)
                rec['nomp_vs_stress_cache_maxdiff'] = float(np.nanmax(np.abs(pk['psi'] - z['NOMP__psi'][:nn_])) + np.nanmax(np.abs(pk['phi'] - z['NOMP__phi'][:nn_])))
            pairs = [(m, 'NOMP') for m in meths] + [(m, 'NOMP+SC tuned') for m in meths] + [('MD-NOMP', 'LS-NOMP'), ('clean NOMP', 'MD-NOMP'),
                                                                                         ('oracle-seeded MD', 'MD-NOMP')]
            pdm, pci, diffs = boot_table(S, b['snr'], pairs)
            gap = pdm['clean NOMP'] - pdm['NOMP']
            rec.update(pd=pdm, pd_ci=pci, diffs=diffs, gap_closed={m: (pdm[m] - pdm['NOMP']) / gap if abs(gap) > 1e-9 else None for m in meths})
            res['banks'][name] = rec
            np.savez_compressed(f'{PCACHE}/m4stats_{name}{"_smoke" if SMOKE else ""}.npz', snr=b['snr'], **S)
            print(f'=== {name} (n={nn_}) [{time.time() - t0:.0f}s]' + (f'  NOMP vs stress cache max|diff| {rec["nomp_vs_stress_cache_maxdiff"]:.1e}' if 'nomp_vs_stress_cache_maxdiff' in rec else ''), flush=True)
            for m in meths:
                d1 = diffs.get(f'{m} - NOMP'); d2 = diffs.get(f'{m} - NOMP+SC tuned')
                g = rec['gap_closed'][m]
                print(f'   {m:18s} Pd {pdm[m]:.4f} [{pci[m][0]:.4f},{pci[m][1]:.4f}]' + (f'  vs NOMP {d1[0]:+.4f} [{d1[1]:+.4f},{d1[2]:+.4f}]' if d1 else '') +
                      (f'  vs NOMP+SC tuned {d2[0]:+.4f} [{d2[1]:+.4f},{d2[2]:+.4f}]' if d2 else '') + (f'  gap closed {100 * g:5.1f}%' if g is not None and bits < 4 else ''), flush=True)
            for k in ['MD-NOMP - LS-NOMP', 'clean NOMP - MD-NOMP', 'oracle-seeded MD - MD-NOMP']:
                d = diffs[k]; print(f'   {k:30s} {d[0]:+.4f} [{d[1]:+.4f},{d[2]:+.4f}]', flush=True)
            save_json(res, _JP)
print('done', flush=True)
