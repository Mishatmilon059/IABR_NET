"""Re-evaluation of the replicated imperfection-stage wins (finetune/imp/REPORT.md) on NEW seeds 8760 (0 dB) / 8775 (15 dB), 1000 scenes
per cell, stress_sim.base_scenes + generate (same generator as the test banks of that stage), true L = 3, all methods on the same Y,
plus simple classical fixes. Learned pipelines re-implemented exactly as finetune/imp/evaluate.py run() (same helpers, imp_ft.pt).
Cells: M6 30 deg 15 dB, M6 20 deg 15 dB, M6 30 deg 0 dB, M2 sigma 0.2 15 dB, M3 (45 deg, 0 dB) 0 dB.
Classical (imp-stage set): NOMP, NOMP+SC, NOMP+SC tuned, classical dual(+SC), NOMP os8 rc6, FFT parabolic os16, EB-ML-g5.
Classical fixes added here (fixed a priori, no tuning on these seeds):
   FFT-peak seeds+SC        = the exact classical analogue of the learned-seed path: top-3 local maxima of the 4x |FFT2(Ht)| as seeds ->
                              Newton -> cyclic (rc 3) on Ht -> self-cal (default 1, 5) [the imp stage had no classical heat-only path]
   FFT-peak seeds+SC tuned  = same with the tuned self-cal (1, 2, 4 it)
   multi-start K6 (+SC tuned) = acommon.multistart (top-6 FFT maxima, all 3-subsets + NOMP, lowest residual)
   NOMP+SC tuned it8        = tuned self-cal with 8 instead of 4 iterations
   NOMP os16 rc6+SC tuned   = larger oversampling and more cyclic rounds, then tuned self-cal
Learned: FT seeds on Ht+SC ('learned seeds'), FT heat+SC, FT dual+SC, orig dual+SC; M3 cell also v2b+NOMP+SC tuned, FT v2b heat+SC.
usage: python s4_imp.py  (stdout -> s4_imp.log; writes s4_imp.json; per-scene stats -> PCACHE)"""
from advc import *
sys.path.insert(0, 'D:/ai_ml_project/paper/evidence/deep_research/finetune/imp')
import icom as I                        # noqa: E402

SMOKE = int(os.environ.get('SMOKE', 0))
N, CH, L, RC = (SMOKE if SMOKE else 1000), 400, 3, 3
SFX = '_smoke' if SMOKE else ''
ft = I.load_cnn(I.FT_CKPT); orig = I.load_cnn(I.ORIG_CKPT)
CELLS = [('M6', 30.0, 15), ('M6', 20.0, 15), ('M2', 0.2, 15), ('M6', 30.0, 0), ('M3', (45.0, 0.0), 0)]
CLASSICAL = ['NOMP', 'NOMP+SC', 'NOMP+SC tuned', 'classical dual', 'classical dual+SC', 'NOMP os8 rc6', 'FFT parabolic os16', 'EB-ML-g5']
FIXES = ['FFT-peak seeds+SC', 'FFT-peak seeds+SC tuned', 'multi-start K6', 'multi-start K6+SC tuned', 'NOMP+SC tuned it8', 'NOMP os16 rc6+SC tuned']
LEARNED = ['FT seeds on Ht+SC', 'FT heat+SC', 'FT dual+SC', 'orig dual+SC']
M3X = ['v2b+NOMP+SC tuned', 'FT v2b heat+SC']


def heat_ref(H, heat):
    Uh, Vh = C.g['heat_peaks'](heat, L)
    Uh, Vh = C.newton(H[:, None].repeat(L, 1).reshape(-1, 16, 16), Uh.ravel(), Vh.ravel())
    return C.cyclic(H, Uh.reshape(-1, L), Vh.reshape(-1, L), RC)


def dual(H, h, g):
    pick = C.residual_energy(H, *h) < C.residual_energy(H, *g)
    return np.where(pick[:, None], h[0], g[0]), np.where(pick[:, None], h[1], g[1])


def scd(H, UV, sc=A.SC):
    return C.uv_to_angles(*C.selfcal(H, UV[0], UV[1], *sc))


def sct(H, UV, it=4):
    return C.uv_to_angles(*C.selfcal(H, UV[0], UV[1], 1.0, 2.0, rc=2, iters=it))


def run(Y, m3):
    H = SS.to_ant(Y); P = {}
    g = C.nomp_greedy(H, L, 4, RC)
    P['NOMP'] = C.uv_to_angles(*g); P['NOMP+SC'] = scd(H, g); P['NOMP+SC tuned'] = sct(H, g)
    fh = heat_ref(H, A.fft_heat(H, 4)); cd = dual(H, fh, g)
    P['classical dual'] = C.uv_to_angles(*cd); P['classical dual+SC'] = scd(H, cd)
    P['NOMP os8 rc6'] = C.uv_to_angles(*C.nomp_greedy(H, L, 8, 6))
    P['FFT parabolic os16'] = [p for s in range(0, len(H), 100) for p in A.as_list(A.parabolic_heat_decode(A.fft_heat(H[s:s + 100], 16), L))]
    P['EB-ML-g5'] = C.uv_to_angles(*I.ebml_g5(H, L, g))
    P['FFT-peak seeds+SC'] = scd(H, fh); P['FFT-peak seeds+SC tuned'] = sct(H, fh)
    ms = A.multistart(H, L, 6)[:2]; P['multi-start K6'] = C.uv_to_angles(*ms); P['multi-start K6+SC tuned'] = sct(H, ms)
    P['NOMP+SC tuned it8'] = sct(H, g, 8)
    P['NOMP os16 rc6+SC tuned'] = sct(H, C.nomp_greedy(H, L, 16, 6))
    hf, Hcf, _ = I.net_forward(ft, Y); ho, Hco, _ = I.net_forward(orig, Y)
    gf = C.nomp_greedy(Hcf, L, 4, RC); hrf = heat_ref(Hcf, hf)
    P['FT heat+SC'] = scd(Hcf, hrf); P['FT dual+SC'] = scd(Hcf, dual(Hcf, hrf, gf))
    P['FT seeds on Ht+SC'] = scd(H, heat_ref(H, hf))
    P['orig dual+SC'] = scd(Hco, dual(Hco, heat_ref(Hco, ho), C.nomp_greedy(Hco, L, 4, RC)))
    if m3:
        al = I.AL(); Hv = al.apply_corr(H, I.v2b_est(H, L)); gv = C.nomp_greedy(Hv, L, 4, RC)
        P['v2b+NOMP+SC tuned'] = sct(Hv, gv)
        P['FT v2b heat+SC'] = scd(Hv, heat_ref(Hv, I.trunk_heat(ft, I.to_beam(Hv))))
    return P


out = {}
jp = f'{HERE}/s4_imp{SFX}.json'
if os.path.exists(jp):
    out = json.load(open(jp))
for kind, sev, snr in CELLS:
    name = f'{kind}_{SS.sev_label(kind, sev)}_snr{snr}'
    if name in out:
        print('skip', name, flush=True); continue
    t0 = time.time()
    base = SS.base_scenes(N, 3, float(snr), IMP_SEEDS[snr]); Y = SS.generate(base, kind, sev)
    m3 = kind == 'M3'; P = {}
    for s0 in range(0, N, CH):
        Pc = run(Y[s0:s0 + CH], m3)
        for k, v in Pc.items():
            P.setdefault(k, []).extend(A.as_list(v))
    meths = CLASSICAL + FIXES + LEARNED + (M3X if m3 else [])
    S = {m: A.stats(P[m], base) for m in meths}
    np.savez_compressed(f'{PCACHE}/s4_stats_{name}{SFX}.npz', snr=base['snr'], **S)
    pd0 = {m: A.pd(S[m]) for m in meths}
    best_old = max(CLASSICAL, key=lambda m: pd0[m]); best_new = max(CLASSICAL + FIXES, key=lambda m: pd0[m])
    learned = LEARNED + (M3X if m3 else [])
    pairs = [(m, r) for m in learned for r in sorted({best_old, best_new, 'NOMP', 'NOMP+SC tuned', 'FFT-peak seeds+SC'}) if m != r] + \
            [(f, best_old) for f in FIXES if f != best_old] + [('FT dual+SC', 'orig dual+SC')]
    pdm, ci_, D = boot(S, base['snr'], pairs)
    out[name] = dict(pd=pdm, ci=ci_, diffs=D, best_classical_imp_set=best_old, best_classical_with_fixes=best_new, sec=time.time() - t0)
    print(f'=== {name} (n={N}) best classical (imp-stage set) {best_old} {pdm[best_old]:.4f}; with simple fixes {best_new} {pdm[best_new]:.4f} [{time.time() - t0:.0f}s]', flush=True)
    for m in meths:
        print(f'   {m:26s} Pd {pdm[m]:.4f} [{ci_[m][0]:.4f},{ci_[m][1]:.4f}]', flush=True)
    for k, d in D.items():
        print(f'   {k:56s} {fmt(d)}', flush=True)
    json.dump(out, open(jp, 'w'), indent=1, default=float)
print('done', flush=True)
