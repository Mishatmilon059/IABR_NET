"""Steps 1 + 3: classical baselines, original CNN, fine-tuned CNN and the oracle-seeded reference on one bank set; all methods on the same Y
and given the true L = 3. Paired bootstrap (2000 resamples within SNR groups, seed 0) of every network method against the BEST classical
method of the cell (chosen per cell on the test bank itself, which favours the classical side) and against NOMP.
Bank sets:
  m5        stress M5 banks, b = 4,3,2,1, SNR 15 / 0, seeds 5015 / 5000 (identical Y to stress/run_stress.py, see diag.log), 500 scenes
  fresh     same generator, NEW seeds 8315 / 8300, 1000 scenes per cell (replicate)
  clean     stress clean banks (no ADC), seeds 5015 / 5000, 500 scenes
  official  frozen official bank (8000 scenes, L = 3, SNR -10..25)
Methods
  classical : NOMP (os 4, rc 3); NOMP+SC (default lam 1 tau 5); NOMP+SC tuned (lam 1, tau 2, 4 iters; frozen from the adversarial
              study, tuned on seeds 9900/9915); FFT-peak seeds (top-3 local maxima of the 4x |FFT2(Hant)| as the heatmap of
              decode(...,'heat')) without / with SC; classical dual (greedy NOMP vs FFT-peak seeds, lowest residual) without / with SC;
              NOMP os 8 rc 6; multi-start K6 (top-6 FFT local maxima, all 20 3-subsets + greedy, Newton/cyclic, lowest residual);
              consistent NOMP (ADC banks only: quantisation-cell-consistent re-estimation, 8 iterations, fixed a priori, see acommon.py)
  network   : orig heat+SC, orig dual+SC (final.pt); FT heat+SC, FT dual+SC, FT NOMP on Hc (+SC), FT network only (parabolic, no physics),
              FT heat + consistent (ADC banks only: FT heat+SC solution as the start of consistent NOMP; hybrid)
  reference : oracle-seeded (true (u,v) -> 3 cyclic rounds on Hant; sees the ground truth)
usage: python evaluate.py m5|fresh|clean|official   (stdout -> eval_<set>.log; writes res_<set>.json, cache/preds_<set>__<bank>.npz,
       cache/stats_<set>__<bank>.npz)"""
from acommon import *
import iabr2_sim as SIM

torch.set_num_threads(6)
SET = sys.argv[1]
CLASSICAL = ['NOMP', 'NOMP+SC', 'NOMP+SC tuned', 'FFT-peak seeds', 'FFT-peak seeds+SC', 'classical dual', 'classical dual+SC', 'NOMP os8 rc6',
             'multi-start K6', 'consistent NOMP']
NETWORK = ['orig heat+SC', 'orig dual+SC', 'FT heat+SC', 'FT dual+SC', 'FT NOMP on Hc', 'FT network only', 'FT heat + consistent']
FT_METHODS = ['FT heat+SC', 'FT dual+SC', 'FT NOMP on Hc', 'FT network only', 'FT heat + consistent']
orig = load_cnn(ORIG_CKPT); ft = load_cnn(os.environ.get('FT_PATH', FT_CKPT))
PCACHE = 'C:/Users/Milon/AppData/Local/Temp/claude/D--thesis/a5db9a29-d75e-4561-834b-cd0638a16d91/scratchpad/dr_cache/finetune_adc'   # D: is full
os.makedirs(PCACHE, exist_ok=True)
SMOKE = int(os.environ.get('SMOKE', 0))      # >0: first SMOKE scenes only, outputs suffixed _smoke (code test only)


def banks():
    if SET in ('m5', 'fresh'):
        seeds, n = (STRESS_SEEDS, 500) if SET == 'm5' else (FRESH_SEEDS, 1000)
        for snr in [15, 0]:
            for bits in BITS:
                b, Y = m5_bank(n, snr, seeds[snr], bits)
                yield f'b{bits}_snr{snr}', b, Y, bits
    elif SET == 'clean':
        for snr in [15, 0]:
            b, Y = m5_bank(500, snr, STRESS_SEEDS[snr], None)
            yield f'clean_snr{snr}', b, Y, None
    elif SET == 'official':
        z = np.load('D:/ai_ml_project/IABR_Net_TestSuite/data/frozen_banks/eval_bank.npz')
        Y16 = SIM.downsample16(z['data']); Y = Y16[..., 0].astype(np.float64) + 1j * Y16[..., 1].astype(np.float64)
        b = dict(psi=z['feat'][:, 0, :].astype(np.float64), phi=z['feat'][:, 1, :].astype(np.float64), L=z['meta'][:, 0].astype(int),
                 snr=z['meta'][:, 1].astype(np.float32))
        assert (b['L'] == 3).all()
        yield 'official', b, Y, None


def run(b, Y, bits, timing):
    H = to_ant(Y); L = 3; P = {}

    def T(name, f):
        t0 = time.time(); P[name] = f(); timing[name] = time.time() - t0
    U, V = C.nomp_greedy(H, L, 4, 3)
    P['NOMP'] = C.uv_to_angles(U, V)
    T('NOMP+SC', lambda: C.uv_to_angles(*C.selfcal(H, U, V, *SC)))
    T('NOMP+SC tuned', lambda: C.uv_to_angles(*selfcal_tuned(H, U, V)))
    fh = fft_heat(H, 4)
    T('FFT-peak seeds', lambda: C.decode(H, L, fh, 'heat', None))
    T('FFT-peak seeds+SC', lambda: C.decode(H, L, fh, 'heat', SC))
    T('classical dual', lambda: C.decode(H, L, fh, 'dual', None))
    T('classical dual+SC', lambda: C.decode(H, L, fh, 'dual', SC))
    T('NOMP os8 rc6', lambda: C.uv_to_angles(*C.nomp_greedy(H, L, 8, 6)))
    T('multi-start K6', lambda: C.uv_to_angles(*multistart(H, L, 6)[:2]))
    if bits is not None:
        T('consistent NOMP', lambda: C.uv_to_angles(*consistent_nomp(Y, bits, L, 8)[:2]))
    ho, Hco = cnn_forward(orig, Y); hf, Hcf = cnn_forward(ft, Y)
    T('orig heat+SC', lambda: C.decode(Hco, L, ho, 'heat', SC))
    T('orig dual+SC', lambda: C.decode(Hco, L, ho, 'dual', SC))
    T('FT heat+SC', lambda: C.decode(Hcf, L, hf, 'heat', SC))
    T('FT dual+SC', lambda: C.decode(Hcf, L, hf, 'dual', SC))
    T('FT NOMP on Hc', lambda: C.decode(Hcf, L, None, 'greedy', SC))
    T('FT network only', lambda: parabolic_heat_decode(hf, L))
    if bits is not None:
        def hyb():
            ps, ph = P['FT heat+SC']
            return C.uv_to_angles(*consistent_nomp(Y, bits, L, 8, U=np.pi * np.cos(ps), V=np.pi * np.cos(ph))[:2])
        T('FT heat + consistent', hyb)
    U0 = np.pi * np.cos(b['psi'][:, :L]); V0 = np.pi * np.cos(b['phi'][:, :L])
    P['oracle-seeded'] = C.uv_to_angles(*C.cyclic(H, U0, V0, 3))
    return P


out = dict(set=SET, banks={})
t00 = time.time()
for name, b, Y, bits in banks():
    if SMOKE:
        b = {k: (v[:SMOKE] if isinstance(v, np.ndarray) and v.ndim and len(v) == len(Y) else v) for k, v in b.items()}; Y = Y[:SMOKE]; name += '_smoke'
    t0 = time.time(); timing = {}
    # memory rule: decoders on chunks of at most CH scenes (nomp_greedy allocates (B,64,64) float64 arrays); CNN batches of 250
    CH = 400; P = {}
    for s0 in range(0, len(Y), CH):
        sl = slice(s0, s0 + CH); bc = {k: (v[sl] if isinstance(v, np.ndarray) and v.ndim and len(v) == len(Y) else v) for k, v in b.items()}
        tc = {}; Pc = run(bc, Y[sl], bits, tc)
        for m, v in Pc.items():
            P.setdefault(m, []).extend(as_list(v))
        for m, v in tc.items():
            timing[m] = timing.get(m, 0.0) + v
        del Pc
    meths = [m for m in CLASSICAL + NETWORK + ['oracle-seeded'] if m in P]
    S = {m: stats(P[m], b) for m in meths}
    np.savez_compressed(f'{PCACHE}/preds_{SET}__{name}.npz', **{f'{m}__{k}': v for m in meths for k, v in pack(P[m], 3).items()})
    np.savez_compressed(f'{PCACHE}/stats_{SET}__{name}.npz', snr=b['snr'], **{m: S[m] for m in meths})
    Bt = C.boot_metrics(S, b['snr'], B=2000, seed=0)
    pdm = {m: C.metric_by_snr(S[m], b['snr'])['mean']['pd_paper'] for m in meths}
    cls = [m for m in CLASSICAL if m in P]
    best = max(cls, key=lambda m: pdm[m])
    rec = dict(n=len(Y), bits=bits, pd=pdm, pd_ci={m: list(C.ci(Bt[m][:, 0])[1:]) for m in meths}, best_classical=best, diffs={},
               sec_per_bank=timing, by_snr=None)
    if SET == 'official':
        rec['by_snr'] = {m: {str(k): v['pd_paper'] for k, v in C.metric_by_snr(S[m], b['snr']).items()} for m in meths}
    for a_ in [m for m in NETWORK + ['oracle-seeded'] + cls if m in P]:
        for key in ['BEST', 'NOMP', 'orig dual+SC', 'orig heat+SC']:
            r_ = best if key == 'BEST' else key
            if a_ == r_:
                continue
            d = Bt[a_][:, 0] - Bt[r_][:, 0]
            rec['diffs'][f'{a_} - {key}'] = [pdm[a_] - pdm[r_], float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))]
    out['banks'][name] = rec
    print(f'=== {SET} {name} (n={len(Y)}) best classical = {best} ({pdm[best]:.4f})  [{time.time() - t0:.0f}s]', flush=True)
    for m in meths:
        dd = rec['diffs'].get(f'{m} - BEST')
        print(f'   {m:24s} Pd {pdm[m]:.4f} [{rec["pd_ci"][m][0]:.4f},{rec["pd_ci"][m][1]:.4f}]' +
              (f'   vs best classical {dd[0]:+.4f} [{dd[1]:+.4f},{dd[2]:+.4f}]' if dd and m != best else '') +
              (f'   vs NOMP {rec["diffs"][f"{m} - NOMP"][0]:+.4f} [{rec["diffs"][f"{m} - NOMP"][1]:+.4f},{rec["diffs"][f"{m} - NOMP"][2]:+.4f}]'
               if f'{m} - NOMP' in rec['diffs'] else '') + (f'   {timing[m]:.1f}s' if m in timing else ''), flush=True)
    save_json(out, f'{HERE}/res_{SET}{"_smoke" if SMOKE else ""}.json')
print(f'total {time.time() - t00:.0f}s')
