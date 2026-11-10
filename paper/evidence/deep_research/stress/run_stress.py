"""Stress banks M1-M6, M8 (+ M5c) and their clean reference: L=3, SNR 0 and 15 dB, 500 scenes per bank.
Base scenes (angles, gains, M3 draws, noise) come from seeds 5000 (SNR 0) and 5015 (SNR 15) and are SHARED by the clean bank and every
impaired bank of the same SNR (common random numbers): the NOMP degradation vs clean is a paired difference on identical scenes.
Impairment draws: default_rng([seed, code]) (stress_sim.IMP_CODE), the same standard draws scaled across severities.
Six methods on the same Y (scommon.run_methods). Paired bootstrap (2000 resamples, seed 0) within the (single) SNR group.
Writes parts/<kind>.json, cache/preds__<kind>_<sev>_snr<s>.npz.   usage: python run_stress.py clean,M1,M2,...   (stdout -> run_stress.log)"""
from scommon import *

SEEDS = {0: 5000, 15: 5015}
N = 500
os.makedirs(f'{HERE}/parts', exist_ok=True)
kinds = sys.argv[1].split(',')
DIFFS = [(f'{m} - NOMP', m, 'NOMP') for m in NN_METHODS] + [(f'{m} - NOMP+selfcal', m, 'NOMP+selfcal') for m in NN_METHODS] + \
        [('oracle-seeded - NOMP', 'oracle-seeded', 'NOMP'), ('NOMP+selfcal - NOMP', 'NOMP+selfcal', 'NOMP')] + \
        [(f'{m} - clean {m}', m, f'clean {m}') for m in METHODS]
KEYS3 = ['pd_paper', 'pd_strict', 'rmse']

_base, _clean = {}, {}


def get_base(snr):
    if snr not in _base:
        b = SS.base_scenes(N, 3, float(snr), SEEDS[snr]); _base[snr] = b
        Yc = SS.generate(b, 'clean'); out, _, _ = run_methods(Yc, b['psi'], b['phi'], 3)
        _clean[snr] = dict(Y=Yc, S={m: stats(out[m], b['psi'], b['phi'], 3) for m in METHODS})
    return _base[snr], _clean[snr]


def one_bank(kind, sev, snr):
    t0 = time.time(); b, cl = get_base(snr)
    Y = cl['Y'] if kind == 'clean' else SS.generate(b, kind, sev)
    G = cl['Y'] - b['noise']
    dpow = float(np.mean(np.abs(Y - cl['Y']) ** 2))     # impairment-to-signal power ratio (dB); None when Y is identical to clean
    dist_db = float(10 * np.log10(dpow / np.mean(np.abs(G) ** 2))) if dpow > 0 else None
    out, heat, Hc = run_methods(Y, b['psi'], b['phi'], 3)
    lab = 'clean' if kind == 'clean' else SS.sev_label(kind, sev)
    np.savez_compressed(f'{MYCACHE}/preds__{kind}_{lab}_snr{snr}.npz', **{f'{m}__{k}': v for m in METHODS for k, v in pack(out[m], 3).items()})
    S = {m: stats(out[m], b['psi'], b['phi'], 3) for m in METHODS}
    S.update({f'clean {m}': cl['S'][m] for m in METHODS})
    Bt = C.boot_metrics(S, b['snr'], B=2000, seed=0)
    res = dict(kind=kind, sev=sev, label=lab, snr=snr, n=N, seed=SEEDS[snr], distortion_db=dist_db, methods={}, diffs={})
    for m in METHODS:
        mm = C.metric_from_stats(S[m], np.arange(N))
        res['methods'][m] = dict({k: mm[k] for k in KEYS3}, dropped=mm['dropped'], ci={k: list(C.ci(Bt[m][:, j])[1:]) for j, k in enumerate(C.KEYS) if k in KEYS3})
    for nm, a_, b_ in DIFFS:
        d = Bt[a_] - Bt[b_]; ma = C.metric_from_stats(S[a_], np.arange(N)); mb = C.metric_from_stats(S[b_], np.arange(N))
        res['diffs'][nm] = {k: [ma[k] - mb[k], *C.ci(d[:, j])[1:]] for j, k in enumerate(C.KEYS) if k in KEYS3}
    r = res['methods']; d = res['diffs']
    print(f'{kind:5s} {lab:10s} SNR {snr:2d} dist {dist_db if dist_db is None else round(dist_db, 1)!s:>6s} dB | ' +
          ' '.join(f'{m.split()[0][:7]} {r[m]["pd_paper"]:.3f}' for m in METHODS) +
          f' | NOMP-clean {d["NOMP - clean NOMP"]["pd_paper"][0]:+.3f} heat-NOMP {d["IABR-Net heat - NOMP"]["pd_paper"][0]:+.3f} '
          f'[{d["IABR-Net heat - NOMP"]["pd_paper"][1]:+.3f},{d["IABR-Net heat - NOMP"]["pd_paper"][2]:+.3f}] | {time.time() - t0:.0f}s', flush=True)
    return res


for kind in kinds:
    sevs = [None] if kind == 'clean' else SS.SEVERITIES[kind]
    allres = [one_bank(kind, s, snr) for snr in [15, 0] for s in sevs]
    json.dump(dict(kind=kind, name='impairment-free' if kind == 'clean' else SS.NAMES[kind], banks=allres), open(f'{HERE}/parts/{kind}.json', 'w'), indent=1,
              default=float)
