"""Pre-registered gate G0 (DESIGN.md section 3; validation seeds only, 71_200_100+). 200 scenes per cell b1/15, b2/15, b1/0, b2/0
(stress_sim M5 peak AGC, L = 3, true SNR known to the classical estimators).
 G0a: share of FAILING scenes (13-start ML + EM not perfect) in which ML started from the TRUE angles reaches a lower exact NLL than
      the 13-start ML solution (= the truth-region optimum fits better -> search loss) ;
 G0b: classical annealed ML (same 13 starts + EM start; LM at kappa 3 (15 it) -> 1.7 (15 it) -> 1 (40 it); lowest NLL at kappa 1);
 G0c: oracle Pd over the classical hypothesis pool H1..H6 (ucore classical configuration: EM-NOMP greedy / heat-seeded (adc_ft CNN) /
      2nd-3rd peak branches, then exact-likelihood LM polish), and the pool's lowest-NLL pick.
Decision (pre-registered): if G0a < 10% in all four cells AND (G0c oracle - ML) < 0.02 in all four cells -> drop A's learned selector."""
import os
os.environ.setdefault('THREADS', '6')
from ucore import *
sys.path.insert(0, f'{DR}/finetune/adversarial')
import advc  # noqa: E402  (ml_quant; sets its own thread env but torch threads reset below)
torch.set_num_threads(int(TH))
N = int(os.environ.get('G0N', 200))
CELLS = [(1, 15, 71_200_101), (2, 15, 71_200_102), (1, 0, 71_200_103), (2, 0, 71_200_104)]


def annealed_ml(Yq, bits, snr, L=3, K=5):
    lo, hi = mlq.cells_ri(Yq, bits, A.adc_cells); s1 = sigma_of(snr)[:, None] * np.ones((len(Yq), 1)); H = A.to_ant(Yq)
    starts = {}
    Uc, Vc, _ = A.consistent_nomp(Yq, bits, L, 8); starts['cNOMP'] = (Uc, Vc); starts['NOMP'] = C.nomp_greedy(H, L, 4, 3)
    ps = A.parabolic_heat_decode(A.fft_heat(H, 16), L)
    U0 = np.array([np.pi * np.cos(p[0][:L]) for p in ps]); V0 = np.array([np.pi * np.cos(p[1][:L]) for p in ps])
    starts['cpar'] = A.consistent_nomp(Yq, bits, L, 8, U=U0, V=V0)[:2]
    Uk, Vk = mlq.fft_cands(H, K, 4)
    Uk, Vk = C.newton(H[:, None].repeat(K, 1).reshape(-1, 16, 16), Uk.ravel(), Vk.ravel()); Uk = Uk.reshape(-1, K); Vk = Vk.reshape(-1, K)
    for sb in itertools.combinations(range(K), L):
        starts[f'fft{sb}'] = (Uk[:, list(sb)], Vk[:, list(sb)])
    X = mlq.init_gains(MO, Uc, Vc, Yq, lo, hi, s1); Ue, Ve, Xe, _ = mlq.em(MO, Uc, Vc, X, lo, hi, s1, L, iters=6); starts['EM'] = (Ue, Ve)
    res = []
    for nm, (U, V) in starts.items():
        X = mlq.init_gains(MO, U, V, Yq, lo, hi, 3.0 * s1)
        for kap, it in [(3.0, 15), (1.7, 15), (1.0, 40)]:
            U, V, X, F = mlq.lm(MO, U, V, X, lo, hi, kap * s1, iters=it)
        res.append((U, V, X, F))
    U, V, X, F, _ = mlq.pick(res)
    return U, V


out = {}
for bits, snr, seed in CELLS:
    t0 = time.time(); name = f'b{bits}_snr{snr}'
    base = SS.base_scenes(N, 3, float(snr), seed); Y = SS.generate(base, 'M5', bits)
    Ut = np.pi * np.cos(base['psi']); Vt = np.pi * np.cos(base['phi'])
    o = advc.ml_quant(Y, bits, base['snr'], 3, kappa=1.0, extra={'truth': (Ut, Vt)})
    Um, Vm, Fm, _ = o['ML (all starts)']; Ut2, Vt2, Ft, _ = o['ML from truth']
    Sm = stats(C.uv_to_angles(Um, Vm), base); St = stats(C.uv_to_angles(Ut2, Vt2), base)
    fail = Sm[:, 0] < 6
    g0a = float(((Ft < Fm - 1e-6) & fail).sum() / max(fail.sum(), 1))
    t1 = time.time(); Ua, Va = annealed_ml(Y, bits, base['snr']); Sa = stats(C.uv_to_angles(Ua, Va), base); t2 = time.time()
    m = UNOMP('A'); m.cfg.update(classical=True, polish=True, sc=False)
    m.eval(); Shyp = {}
    with torch.no_grad():
        bt = prep_batch(Y, np.full(N, bits), base['snr']); oo = m(bt, 3)
    for h in oo['hyps']:
        Shyp[h] = stats(C.uv_to_angles(oo['res'][h]['U'].numpy(), oo['res'][h]['V'].numpy()), base)
    hits = np.stack([Shyp[h][:, 0] for h in oo['hyps']], 1); best = hits.argmax(1)
    Sor = np.stack([Shyp[h] for h in oo['hyps']], 1)[np.arange(N), best]
    Spk = stats(C.uv_to_angles(oo['U'].numpy(), oo['V'].numpy()), base)
    t3 = time.time()
    S = {'ML': Sm, 'ML truth': St, 'annealed ML': Sa, 'pool oracle': Sor, 'pool lowest NLL': Spk, **{f'pool {h}': Shyp[h] for h in oo['hyps']}}
    pdm, ci_, D = boot_pairs(S, base['snr'], [('annealed ML', 'ML'), ('pool oracle', 'ML'), ('pool lowest NLL', 'ML')])
    rec = dict(n=N, seed=seed, failing=int(fail.sum()), G0a=g0a, pd=pdm, diffs=D, sec=dict(ml=t1 - t0, annealed=t2 - t1, pool=t3 - t2))
    out[name] = rec
    print(f'=== {name} (n={N}, seed {seed}) failing scenes {fail.sum()}; G0a truth-better share {g0a:.3f}', flush=True)
    for k_, v in pdm.items():
        print(f'   {k_:18s} Pd {v:.4f}', flush=True)
    for k_, v in D.items():
        print(f'   {k_:34s} {fmt(v)}', flush=True)
    print(f'   time ML {t1 - t0:.0f}s annealed {t2 - t1:.0f}s pool {t3 - t2:.0f}s', flush=True)
    save_json(out, f'{HERE}/g0_gate.json')
a_ok = all(out[c]['G0a'] < 0.10 for c in out); c_ok = all(out[c]['diffs']['pool oracle - ML'][0] < 0.02 for c in out)
dec = 'DROP learned selector (A = B + step MLP)' if (a_ok and c_ok) else 'KEEP learned selector'
out['decision'] = dict(G0a_all_below_10pct=a_ok, G0c_all_below_0p02=c_ok, decision=dec)
save_json(out, f'{HERE}/g0_gate.json')
print('G0 decision:', dec, flush=True)
