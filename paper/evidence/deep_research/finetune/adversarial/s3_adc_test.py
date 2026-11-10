"""PRIORITY test: does the ADC-fine-tuned CNN read out directly ('FT network only') still beat the best classical method once a
quantisation-aware maximum-likelihood estimator is in the classical set?  FRESH test banks: stress_sim.base_scenes(1000, L=3, SNR, seed)
+ generate(.., 'M5', b) (peak AGC), seeds 8700 (0 dB) / 8715 (15 dB), b = 1, 2. All methods on the same Y, all given the true L = 3.
Methods
  classical (no learning): NOMP (os 4, rc 3); consistent NOMP (ADC stage, 8 it); consistent parabolic; FFT parabolic os16;
     ML (all starts)          = advc.ml_quant: exact quantiser likelihood, LM from 13 classical starts + EM, lowest NLL kept, kappa tuned
                                on seeds 8730/8745 (tune.json); this is the PRE-SPECIFIED quantisation-aware classical baseline;
     ML from consistent NOMP  = single start; EM + ML = EM from consistent NOMP then LM;
     ML (all starts) kappa 1  = untuned a-priori version (only if the tuned kappa != 1);
     ML snr+5 / ML snr-5      = ML from consistent NOMP with the noise variance mis-specified by +-5 dB (robustness, not a candidate for 'best')
  learned: FT network only (finetune/adc/cache/adc_ft.pt, parabolic read-out); orig network only (final.pt); FT dual+SC (decoder path);
     FT-CNN + adapted learned decoder (nn_decoder_adapt/cache/ld_adc_final_run.pt, StagedSC(RefinerV3) as in ld_replicate.py);
     hybrid 'ML from FT network seeds' (network output used as an extra start of the ML refinement; reported separately)
  reference: 'ML from truth' (true (u, v) start; shows the ML objective's own accuracy limit)
Paired bootstrap 2000 within SNR groups, seed 0 (one SNR per bank). Resumable per chunk of 200 scenes (PCACHE/s3_<bank>_<chunk>.npz).
usage: python s3_adc_test.py  (stdout -> s3_adc_test.log; writes s3_adc_test.json)"""
from advc import *
import itertools

LD = 'D:/ai_ml_project/paper/evidence/learned_decoder'
sys.path.insert(0, LD)
from exp5 import StagedSC, RefinerV3, heat_peaks_np, uv_to_angles_np      # noqa: E402  (exp.py loads the original CNN at import)
torch.set_num_threads(6)
KAPPA = json.load(open(f'{HERE}/tune.json'))['kappa']
SMOKE = int(os.environ.get('SMOKE', 0))
N, CH = (SMOKE, SMOKE) if SMOKE else (1000, 200)
SFX = '_smoke' if SMOKE else ''
ft = A.load_cnn(A.FT_CKPT); orig = A.load_cnn(A.ORIG_CKPT)
ld = StagedSC(RefinerV3(), pre=6, rounds=3, inner=2)
ld.load_state_dict(torch.load('D:/ai_ml_project/paper/evidence/deep_research/nn_decoder_adapt/cache/ld_adc_final_run.pt', map_location='cpu'))
ld.eval(); ld.use_sc = True
CLASSICAL = ['NOMP', 'consistent NOMP', 'consistent parabolic', 'FFT parabolic os16', 'ML (all starts)', 'ML from consistent NOMP', 'EM + ML'] + \
            (['ML (all starts) kappa 1'] if KAPPA != 1.0 else [])
ROBUST = ['ML snr+5', 'ML snr-5']
LEARNED = ['FT network only', 'orig network only', 'FT dual+SC', 'FT-CNN + adapted learned decoder', 'ML from FT network seeds']
REF = ['ML from truth']
ALLM = CLASSICAL + ROBUST + LEARNED + REF


def angles_to_uv(pl, L=3):
    return np.array([np.pi * np.cos(p[0][:L]) for p in pl]), np.array([np.pi * np.cos(p[1][:L]) for p in pl])


def chunk(b, Y, bits):
    P, T = {}, {}
    H = A.to_ant(Y); L = 3; snr = b['snr']

    def tm(k, t0):
        T[k] = T.get(k, 0.0) + time.time() - t0
    t0 = time.time(); P['NOMP'] = C.uv_to_angles(*C.nomp_greedy(H, L, 4, 3)); tm('NOMP', t0)
    t0 = time.time(); P['FFT parabolic os16'] = A.parabolic_heat_decode(A.fft_heat(H, 16), L); tm('FFT parabolic os16', t0)
    t0 = time.time(); P['consistent NOMP'] = C.uv_to_angles(*A.consistent_nomp(Y, bits, L, 8)[:2]); tm('consistent NOMP', t0)
    U0, V0 = angles_to_uv(P['FFT parabolic os16'])
    P['consistent parabolic'] = C.uv_to_angles(*A.consistent_nomp(Y, bits, L, 8, U=U0, V=V0)[:2])
    t0 = time.time(); hf, Hcf = A.cnn_forward(ft, Y); P['FT network only'] = A.parabolic_heat_decode(hf, L); tm('FT network only', t0)
    ho, _ = A.cnn_forward(orig, Y); P['orig network only'] = A.parabolic_heat_decode(ho, L)
    P['FT dual+SC'] = C.decode(Hcf, L, hf, 'dual', A.SC)
    with torch.no_grad():
        Uh, Vh = heat_peaks_np(hf, L)
        Ul, Vl = ld(torch.from_numpy(Hcf.astype(np.complex64)), torch.tensor(Uh, dtype=torch.float32), torch.tensor(Vh, dtype=torch.float32))
    p = uv_to_angles_np(Ul.numpy().astype(np.float64), Vl.numpy().astype(np.float64)); P['FT-CNN + adapted learned decoder'] = p
    Un, Vn = angles_to_uv(P['FT network only'])
    Ut = np.pi * np.cos(b['psi'][:, :L]); Vt = np.pi * np.cos(b['phi'][:, :L])
    t0 = time.time()
    o = ml_quant(Y, bits, snr, L, kappa=KAPPA, extra={'FT network seeds': (Un, Vn), 'truth': (Ut, Vt)})
    T['ML (all starts) + 2 extra starts'] = time.time() - t0
    for k in ['ML (all starts)', 'ML from consistent NOMP', 'EM + ML', 'ML from FT network seeds', 'ML from truth']:
        P[k] = C.uv_to_angles(o[k][0], o[k][1])
    nchosen = np.bincount(o['ML (all starts)'][3], minlength=len(o['_start_names']))
    if KAPPA != 1.0:
        o1 = ml_quant(Y, bits, snr, L, kappa=1.0, em_iters=6)
        P['ML (all starts) kappa 1'] = C.uv_to_angles(o1['ML (all starts)'][0], o1['ML (all starts)'][1])
    lo, hi = mlq.cells_ri(Y, bits, A.adc_cells)
    Uc, Vc = angles_to_uv(A.as_list(P['consistent NOMP']))
    for nm, dsnr in [('ML snr+5', 5.0), ('ML snr-5', -5.0)]:
        s = (KAPPA * sigma_of(snr + dsnr))[:, None]
        X = mlq.init_gains(MO, Uc, Vc, Y, lo, hi, s); r = mlq.lm(MO, Uc, Vc, X, lo, hi, s)
        P[nm] = C.uv_to_angles(r[0], r[1])
    return P, T, nchosen, o['_start_names']


out = dict(kappa=KAPPA, seeds=TEST_SEEDS, n=N, banks={})
for snr in [15, 0]:
    base = SS.base_scenes(N, 3, float(snr), TEST_SEEDS[snr])
    for bits in [1, 2]:
        name = f'b{bits}_snr{snr}'; Y = SS.generate(base, 'M5', bits); t00 = time.time()
        S = {m: [] for m in ALLM}; T = {}; nch = 0
        for c0 in range(0, N, CH):
            f = f'{PCACHE}/s3_{name}_{c0}{SFX}.npz'
            sl = slice(c0, c0 + CH); bc = {k: (v[sl] if isinstance(v, np.ndarray) and v.ndim and len(v) == N else v) for k, v in base.items()}
            if os.path.exists(f):
                z = np.load(f, allow_pickle=True); Sc = {m: z[m] for m in ALLM}; Tc = z['T'].item(); ncc = z['nch']; names = list(z['names'])
            else:
                P, Tc, ncc, names = chunk(bc, Y[sl], bits)
                Sc = {m: A.stats(P[m], bc) for m in ALLM}
                np.savez(f, T=np.array(Tc, dtype=object), nch=ncc, names=np.array(names), **Sc)
                print(f'  {name} chunk {c0}: done ({time.time() - t00:.0f}s since bank start)', flush=True)
            for m in ALLM:
                S[m].append(Sc[m])
            for k, v in Tc.items():
                T[k] = T.get(k, 0.0) + v
            nch = nch + ncc
        S = {m: np.concatenate(v) for m, v in S.items()}
        cls = [m for m in CLASSICAL]; pd0 = {m: A.pd(S[m]) for m in ALLM}
        best = max(cls, key=lambda m: pd0[m]); bestold = max(['NOMP', 'consistent NOMP', 'consistent parabolic', 'FFT parabolic os16'], key=lambda m: pd0[m])
        pairs = [(m, 'FT network only') for m in ALLM if m != 'FT network only'] + [('FT-CNN + adapted learned decoder', best),
                 ('FT-CNN + adapted learned decoder', bestold), ('ML from FT network seeds', 'ML (all starts)'), ('ML (all starts)', 'consistent NOMP')]
        pdm, ci_, D = boot(S, base['snr'], pairs)
        rec = dict(pd=pdm, ci=ci_, diffs=D, best_classical=best, best_classical_old_set=bestold, sec_per_scene={k: v / N for k, v in T.items()},
                   ml_start_chosen=dict(zip(names, np.asarray(nch).tolist())))
        out['banks'][name] = rec
        print(f'=== {name} (n={N}, kappa {KAPPA}) best classical (incl. ML) = {best} {pdm[best]:.4f}; best of ADC-stage classical set = {bestold} {pdm[bestold]:.4f}', flush=True)
        for m in ALLM:
            d = D.get(f'{m} - FT network only')
            print(f'   {m:34s} Pd {pdm[m]:.4f} [{ci_[m][0]:.4f},{ci_[m][1]:.4f}]' + (f'   minus FT network only {fmt(d)}' if d else ''), flush=True)
        for k in [f'FT-CNN + adapted learned decoder - {best}', f'FT-CNN + adapted learned decoder - {bestold}', 'ML from FT network seeds - ML (all starts)',
                  'ML (all starts) - consistent NOMP']:
            print(f'   {k:60s} {fmt(D[k])}', flush=True)
        print('   ML start chosen (counts):', rec['ml_start_chosen'], flush=True)
        print('   sec/scene (6 threads, batch 200):', {k: round(v, 3) for k, v in rec['sec_per_scene'].items()}, flush=True)
        json.dump(out, open(f'{HERE}/s3_adc_test{SFX}.json', 'w'), indent=1, default=float)
print('done', flush=True)
