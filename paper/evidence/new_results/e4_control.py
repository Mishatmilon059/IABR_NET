"""E4: is the self-calibration gain on unimpaired close-separation banks calibration or just extra refinement?
Network-free decoder, input Ht. Variants (all paper-style/strict Pd via metrics() logic; paired bootstrap 2000 resamples):
 i    greedy (os=4, rc=3)                        [= cached NOMP_no_network, verified]
 ii   greedy + selfcal(lam=1.0, tau=5.0)          [from e13_decode.py output]
 iii  greedy + selfcal(lam=0.0, tau=5.0) called DIRECTLY (decode() would skip selfcal for lam=0): applies no correction, but performs
      selfcal's 2 outer iterations x cyclic(rc=2) = 4 extra cyclic rounds on the final path set
 iv   greedy with rc = 5, 8 (rc applies inside every greedy stage)
 v    greedy with os = 8 (rc=3)
usage: python e4_control.py bank1,bank2"""
import sys; sys.path.insert(0, 'D:/ai_ml_project/paper/evidence/new_results')
from common import *
os.makedirs(f'{OUT}/e4', exist_ok=True)
DIFFS = [('ii-i', 'sc(1,5)', 'greedy'), ('iii-i', 'sc(0,5)', 'greedy'), ('ii-iii', 'sc(1,5)', 'sc(0,5)'), ('rc5-i', 'rc5', 'greedy'), ('rc8-i', 'rc8', 'greedy'), ('os8-i', 'os8', 'greedy')]
for name in sys.argv[1].split(','):
    t0 = time.time(); b = load_bank(name); Ls = b['L']; L = int(Ls[0]); assert (Ls == L).all()
    Ht = np.load(f'{CACHE}/net__{name}.npz')['Ht'].astype(np.complex128)
    def run(f, chunk=1000):
        P, F = [], []
        for s in range(0, len(Ht), chunk):
            U, V = f(Ht[s:s + chunk]); a = uv_to_angles(U, V); P.append(a[0]); F.append(a[1])
        return np.concatenate(P), np.concatenate(F)
    preds = {}
    cached = np.load(f'{CACHE}/pred__NOMP_no_network__{name}.npz'); preds['greedy'] = (cached['psi'], cached['phi'])
    z = np.load(f'{OUT}/preds/selfcal_pred__{name}.npz'); preds['sc(1,5)'] = (z['psi'], z['phi'])
    # (iii) greedy then selfcal with lam=0
    def f3(H):
        U, V = nomp_greedy(H, L, 4, 3); return selfcal(H, U, V, 0.0, 5.0)
    preds['sc(0,5)'] = run(f3)
    preds['rc5'] = run(lambda H: nomp_greedy(H, L, 4, 5)); preds['rc8'] = run(lambda H: nomp_greedy(H, L, 4, 8))
    preds['os8'] = run(lambda H: nomp_greedy(H, L, 8, 3))
    # sanity: recomputed greedy rc=3 os=4 equals cache
    gre = run(lambda H: nomp_greedy(H, L, 4, 3)); same = bool(np.allclose(gre[0], cached['psi'], equal_nan=True) and np.allclose(gre[1], cached['phi'], equal_nan=True))
    S = {k: scene_stats([(p[0][i, :L], p[1][i, :L]) for i in range(len(Ls))], b) for k, p in preds.items()}
    np.savez_compressed(f'{OUT}/e4/preds__{name}.npz', **{f'{k}__psi': p[0] for k, p in preds.items()}, **{f'{k}__phi': p[1] for k, p in preds.items()})
    B = boot_metrics(S, b['snr']); res = dict(bank=name, n=len(Ls), L=L, recompute_greedy_equals_cache=same, variants={}, diffs={})
    for k in S:
        pt = metric_by_snr(S[k], b['snr'])['mean']; res['variants'][k] = dict(point=pt, ci_pd_paper=ci(B[k][:, 0])[1:], ci_pd_strict=ci(B[k][:, 1])[1:])
    for nm, a_, b_ in DIFFS:
        d = B[a_] - B[b_]; pa = metric_by_snr(S[a_], b['snr'])['mean']; pb = metric_by_snr(S[b_], b['snr'])['mean']
        res['diffs'][nm] = dict(pd_paper=[pa['pd_paper'] - pb['pd_paper'], *ci(d[:, 0])[1:]], pd_strict=[pa['pd_strict'] - pb['pd_strict'], *ci(d[:, 1])[1:]])
    json.dump(res, open(f'{OUT}/e4/e4__{name}.json', 'w'), indent=1)
    print(f'{name:18s} {time.time()-t0:.0f}s greedy==cache {same} | ' + ' | '.join(f'{k} {res["variants"][k]["point"]["pd_paper"]:.4f}' for k in S) + ' | ' +
          ' | '.join(f'{nm} {res["diffs"][nm]["pd_paper"][0]:+.4f} [{res["diffs"][nm]["pd_paper"][1]:+.4f},{res["diffs"][nm]["pd_paper"][2]:+.4f}]' for nm, _, _ in DIFFS), flush=True)
