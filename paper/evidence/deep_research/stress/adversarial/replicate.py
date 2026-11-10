"""Fresh-seed replication of the headline stress cells, with an independent forward model (adv_common; validated against stress_sim in
xcheck.log) and a fresh draw scheme. L = 3, N = 1000 scenes per bank (2x the original 500), base seeds 9000 (SNR 0) / 9015 (SNR 15);
the clean and every impaired bank of one SNR share angles, gains and noise (paired degradation).
Methods (all on the same Y, all given the true L = 3):
  NOMP                      nomp_greedy(Hant, 3, os 4, rc 3)
  NOMP+selfcal              selfcal(Hant, NOMP, lam 1, tau 5)            (the stress-test setting SC = (1, 5))
  NOMP+selfcal tau0         selfcal(Hant, NOMP, lam 1, tau 0)            (gate always open; a-priori variant, no tuning)
  NOMP+selfcal tuned        selfcal(Hant, NOMP, lam*, tau*, iters*)      (tuned on SEPARATE tuning banks, seeds 9900/9915, M3 gain 6 dB and
                                                                          phase 30 + gain 4 dB, 500 scenes each; never on the test banks)
  IABR-Net heat             decode(Hc, 3, heat, 'heat', SC)
  IABR-v2 dual              decode(Hc, 3, heat, 'dual', SC)
  NOMP on Hc                decode(Hc, 3, None, 'greedy', SC)            (attribution: network correction, no heat seeding)
Paired bootstrap: 2000 resamples, seed 0 (one SNR group per bank). usage: python replicate.py  (stdout -> replicate.log, writes replicate.json)"""
from adv_common import *

N = 1000; SEED = {0: 9000, 15: 9015}; TUNE_SEED = {0: 9900, 15: 9915}
ORIG = json.load(open('D:/ai_ml_project/paper/evidence/deep_research/stress/results.json'))['parts']
CELLS = [('clean', None), ('M3', (0.0, 6.0)), ('M3', (30.0, 4.0)), ('M3', (20.0, 0.0)), ('M3', (45.0, 0.0)), ('M4', 2), ('M4', 1),
         ('M2', 0.05), ('M2', 0.2), ('M1', 0.3), ('M5', 4), ('M5', 1), ('M6', 30.0), ('M8', 0.5)]


def label(kind, sev):
    return 'clean' if kind == 'clean' else (f'ph{sev[0]:g}_g{sev[1]:g}' if kind == 'M3' else f'{sev:g}')


def gate_frac(H, U, V, tau):
    x, A = C.ls_gains(H, U, V); Hm = np.einsum('bik,bk->bi', A, x).reshape(H.shape)
    se = (np.abs(Hm) ** 2).sum((1, 2)) / ((np.abs(H - Hm) ** 2).sum((1, 2)) + 1e-12)
    return float(np.mean(se > tau)), float(np.median(se))


def selfcal_it(H, U, V, lam, tau, iters):
    return C.selfcal(H, U, V, lam, tau, rc=2, iters=iters)


t00 = time.time()
# ---------------- tuning of the strengthened self-calibration baseline (separate banks)
GRID = [(lam, tau, it) for lam in [0.5, 1.0] for tau in [0.0, 1.0, 2.0, 5.0] for it in [2, 4]]
tb = []
for snr in [15, 0]:
    b = fresh_bank(500, snr, TUNE_SEED[snr])
    for sev in [(0.0, 6.0), (30.0, 4.0)]:
        H = to_ant(fresh_Y(b, 'M3', sev)); U, V = C.nomp_greedy(H, 3, 4, 3); tb.append((b, H, U, V))
acc = []
for lam, tau, it in GRID:
    acc.append(np.mean([pd(stats(C.uv_to_angles(*selfcal_it(H, U, V, lam, tau, it)), b)) for b, H, U, V in tb]))
    print(f'tune lam {lam} tau {tau} iters {it}: mean Pd over 4 tuning banks {acc[-1]:.4f}', flush=True)
LAM_T, TAU_T, IT_T = GRID[int(np.argmax(acc))]
print(f'tuned self-calibration: lam {LAM_T}, tau {TAU_T}, iters {IT_T} (default (1,5,2): {acc[GRID.index((1.0, 5.0, 2))]:.4f}) [{time.time() - t00:.0f}s]', flush=True)

METH = ['NOMP', 'NOMP+selfcal', 'NOMP+selfcal tau0', 'NOMP+selfcal tuned', 'IABR-Net heat', 'IABR-v2 dual', 'NOMP on Hc']
DIFFS = [('NOMP', 'clean NOMP'), ('IABR-Net heat', 'clean IABR-Net heat'), ('IABR-v2 dual', 'clean IABR-v2 dual')] + \
        [(m, r) for m in ['IABR-Net heat', 'IABR-v2 dual', 'NOMP on Hc'] for r in ['NOMP', 'NOMP+selfcal', 'NOMP+selfcal tau0', 'NOMP+selfcal tuned']] + \
        [('NOMP+selfcal tau0', 'NOMP'), ('NOMP+selfcal tuned', 'NOMP')]
ORIGKEY = {('NOMP', 'clean NOMP'): 'NOMP - clean NOMP', ('IABR-Net heat', 'clean IABR-Net heat'): 'IABR-Net heat - clean IABR-Net heat',
           ('IABR-v2 dual', 'clean IABR-v2 dual'): 'IABR-v2 dual - clean IABR-v2 dual',
           ('IABR-Net heat', 'NOMP'): 'IABR-Net heat - NOMP', ('IABR-v2 dual', 'NOMP'): 'IABR-v2 dual - NOMP',
           ('IABR-Net heat', 'NOMP+selfcal'): 'IABR-Net heat - NOMP+selfcal', ('IABR-v2 dual', 'NOMP+selfcal'): 'IABR-v2 dual - NOMP+selfcal'}
out = dict(n=N, seeds=SEED, tune_seeds=TUNE_SEED, tuned=dict(lam=LAM_T, tau=TAU_T, iters=IT_T), tune_grid=[list(g) + [a] for g, a in zip(GRID, acc)], banks=[])
for snr in [15, 0]:
    b = fresh_bank(N, snr, SEED[snr]); clean = None
    for kind, sev in CELLS:
        t0 = time.time(); lab = label(kind, sev)
        Y = fresh_Y(b, kind, sev); H = to_ant(Y); heat, Hc = net(Y)
        U, V = C.nomp_greedy(H, 3, 4, 3)
        P = {'NOMP': C.uv_to_angles(U, V), 'NOMP+selfcal': C.uv_to_angles(*C.selfcal(H, U, V, *C.SC)),
             'NOMP+selfcal tau0': C.uv_to_angles(*C.selfcal(H, U, V, 1.0, 0.0)),
             'NOMP+selfcal tuned': C.uv_to_angles(*selfcal_it(H, U, V, LAM_T, TAU_T, IT_T)),
             'IABR-Net heat': C.decode(Hc, 3, heat, 'heat', C.SC), 'IABR-v2 dual': C.decode(Hc, 3, heat, 'dual', C.SC),
             'NOMP on Hc': C.decode(Hc, 3, None, 'greedy', C.SC)}
        Sd = {m: stats(P[m], b) for m in METH}
        if kind == 'clean':
            clean = Sd
        Sd.update({f'clean {m}': clean[m] for m in METH})
        Bt = C.boot_metrics(Sd, b['snr'], B=2000, seed=0)
        gf, gmed = gate_frac(H, U, V, 5.0); gfc, gmedc = gate_frac(Hc, *C.nomp_greedy(Hc, 3, 4, 3), 5.0)
        rec = dict(kind=kind, sev=sev, label=lab, snr=snr, seed=SEED[snr], pd={m: pd(Sd[m]) for m in METH},
                   pd_ci={m: list(C.ci(Bt[m][:, 0])[1:]) for m in METH}, diffs={}, gate_open_frac_Hant=gf, snr_est_median_Hant=gmed,
                   gate_open_frac_Hc=gfc, snr_est_median_Hc=gmedc)
        ob = [x for x in ORIG.get(kind, {'banks': []})['banks'] if x['label'] == lab and x['snr'] == snr]
        for a_, r_ in DIFFS:
            d = Bt[a_][:, 0] - Bt[r_][:, 0]; e = [pd(Sd[a_]) - pd(Sd[r_]), *C.ci(d)[1:]]
            o = ob[0]['diffs'][ORIGKEY[(a_, r_)]]['pd_paper'] if ob and (a_, r_) in ORIGKEY else None
            rec['diffs'][f'{a_} - {r_}'] = dict(fresh=e, orig=o, orig_in_fresh_ci=(None if o is None else bool(e[1] <= o[0] <= e[2])))
        out['banks'].append(rec)
        dd = rec['diffs']
        f = lambda k: f'{dd[k]["fresh"][0]:+.3f} [{dd[k]["fresh"][1]:+.3f},{dd[k]["fresh"][2]:+.3f}]' + \
                      (f' (orig {dd[k]["orig"][0]:+.3f})' if dd[k]['orig'] is not None else '')
        print(f'{kind:5s} {lab:8s} SNR {snr:2d} | ' + ' '.join(f'{m} {rec["pd"][m]:.3f}' for m in METH) + f' | gate open (tau 5) Hant {gf:.2f} Hc {gfc:.2f} | {time.time() - t0:.0f}s', flush=True)
        print(f'      NOMP-clean {f("NOMP - clean NOMP")} | heat-NOMP {f("IABR-Net heat - NOMP")} | dual-NOMP {f("IABR-v2 dual - NOMP")}', flush=True)
        print(f'      heat-SC {f("IABR-Net heat - NOMP+selfcal")} | dual-SC {f("IABR-v2 dual - NOMP+selfcal")} | heat-SCtau0 {f("IABR-Net heat - NOMP+selfcal tau0")} | '
              f'dual-SCtau0 {f("IABR-v2 dual - NOMP+selfcal tau0")} | heat-SCtuned {f("IABR-Net heat - NOMP+selfcal tuned")} | dual-SCtuned {f("IABR-v2 dual - NOMP+selfcal tuned")}', flush=True)
        print(f'      NOMPonHc-NOMP {f("NOMP on Hc - NOMP")} | NOMPonHc-SCtuned {f("NOMP on Hc - NOMP+selfcal tuned")} | SCtau0-NOMP {f("NOMP+selfcal tau0 - NOMP")} | '
              f'SCtuned-NOMP {f("NOMP+selfcal tuned - NOMP")} | heat-clean heat {f("IABR-Net heat - clean IABR-Net heat")}', flush=True)
json.dump(out, open(f'{HERE}/replicate.json', 'w'), indent=1, default=float)
print(f'total {time.time() - t00:.0f}s')
