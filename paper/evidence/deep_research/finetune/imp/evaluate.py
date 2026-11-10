"""Steps 2 + 3: evaluation of the fine-tuned CNN (cache/imp_ft.pt) on severe array imperfections, all methods on the SAME Y, given the
true L = 3. Paired bootstrap (2000 resamples within SNR groups, seed 0). Every bank is resumable (per-bank stats on C:, per-bank json here).
Bank sets (usage: python evaluate.py SET [SET ...]):
  stress     stress banks M1 (rho .05/.1/.2/.3), M2 (sigma .02/.05/.1/.2), M3 (all five settings), M6 (5/10/20/30 deg), M8 (2/1/0.5 m,
             out-of-distribution control) at SNR 15 / 0, seeds 5015 / 5000, 500 scenes; Y identity with stress/run_stress.py is checked by
             comparing our NOMP estimates with stress/cache/preds__<kind>_<sev>_snr<s>.npz (max |diff| logged; must be 0)
  clean      stress clean banks (seeds 5015 / 5000, 500 scenes)
  official   frozen official bank, all 8000 scenes (regression)
  fresh:K    fresh-seed replicate of family K (all its severities), seeds 8415 / 8400, 1000 scenes per cell (new; grep-checked unused)
Methods (decoder = notebook decode(): heat peaks -> Newton -> cyclic (rc 3) / greedy NOMP (os 4, rc 3) / dual = lower residual):
  classical  NOMP; NOMP+SC (default lam 1 tau 5); NOMP+SC tuned (lam 1 tau 2 it 4); classical dual (greedy vs FFT-peak seeds) +-SC;
             NOMP os8 rc6; FFT parabolic os4 / os16 (classical counterpart of the network-only read-out); EB-ML-g5 (iabc_v2 adversarial
             classical joint-ML + shrinkage self-cal, gated at blind SNR 5 dB, then NOMP)
  earlier learned   v2b+NOMP, v2b+NOMP+SC tuned (IABC-v2b corrector, iabc_v2/cache, read only)
  original CNN      orig heat+SC, orig dual+SC, orig network only (final.pt)
  fine-tuned CNN    FT heat+SC, FT dual+SC, FT network only (own fine-tuned IABC front)
                    FT nocorr heat+SC / dual+SC / network only : FT trunk on the RAW beamspace Y, decoder on raw Ht (no correction)
                    FT v2b heat+SC / dual+SC / network only    : IABC-v2b correction in place of the IABC (Hv = corr(Ht)), FT trunk on
                                                                 W^H Hv F, decoder on Hv
  ablation   (a) 'FT seeds on Ht+SC'  FT heat (own IABC) as seeds, decoder on RAW Ht    (learned seeds only)
             (b) 'FT NOMP on Hc+SC'   greedy NOMP on the FT-corrected Hc               (learned correction only)
             (c) 'FT heat+SC'         FT heat seeds on FT Hc                           (both)       reference: NOMP+SC (neither)
  reference  oracle-seeded (true (u,v) -> 3 cyclic rounds on Ht; sees the truth)
Best classical per cell = max over the classical set on the test bank itself (favours the classical side)."""
from icom import *
import iabr2_sim as SIM2

CLASSICAL = ['NOMP', 'NOMP+SC', 'NOMP+SC tuned', 'classical dual', 'classical dual+SC', 'NOMP os8 rc6', 'FFT parabolic os4',
             'FFT parabolic os16', 'EB-ML-g5']
LEARNED_PRIOR = ['v2b+NOMP', 'v2b+NOMP+SC tuned']
ORIGM = ['orig heat+SC', 'orig dual+SC', 'orig network only']
FTM = ['FT heat+SC', 'FT dual+SC', 'FT network only', 'FT nocorr heat+SC', 'FT nocorr dual+SC', 'FT nocorr network only',
       'FT v2b heat+SC', 'FT v2b dual+SC', 'FT v2b network only', 'FT seeds on Ht+SC', 'FT NOMP on Hc+SC']
ALLM = CLASSICAL + LEARNED_PRIOR + ORIGM + FTM + ['oracle-seeded']
SMOKE = int(os.environ.get('SMOKE', 0))
FT_PATH = os.environ.get('FT_PATH', FT_CKPT)
orig = load_cnn(ORIG_CKPT); ft = load_cnn(FT_PATH)
RC = 3


def heat_ref(H, heat, L):
    Uh, Vh = C.g['heat_peaks'](heat, L)
    Uh, Vh = C.newton(H[:, None].repeat(L, 1).reshape(-1, 16, 16), Uh.ravel(), Vh.ravel())
    return C.cyclic(H, Uh.reshape(-1, L), Vh.reshape(-1, L), RC)


def dual(H, h, g):
    pick = C.residual_energy(H, *h) < C.residual_energy(H, *g)
    return np.where(pick[:, None], h[0], g[0]), np.where(pick[:, None], h[1], g[1])


def scd(H, UV, sc=SC):
    return C.uv_to_angles(*C.selfcal(H, UV[0], UV[1], *sc))


def fft_par(H, os_):
    out = []
    for s in range(0, len(H), 100):
        out.extend(as_list(A.parabolic_heat_decode(A.fft_heat(H[s:s + 100], os_), 3)))
    return out


def run(b, Y, L=3):
    """all methods on one chunk (<= 400 scenes). Returns dict method -> list of (psi, phi)."""
    H = SS.to_ant(Y); P = {}
    g = C.nomp_greedy(H, L, 4, RC)
    P['NOMP'] = C.uv_to_angles(*g)
    P['NOMP+SC'] = scd(H, g)
    P['NOMP+SC tuned'] = C.uv_to_angles(*A.selfcal_tuned(H, *g))
    hf_ = heat_ref(H, A.fft_heat(H, 4), L); cd = dual(H, hf_, g)
    P['classical dual'] = C.uv_to_angles(*cd); P['classical dual+SC'] = scd(H, cd)
    P['NOMP os8 rc6'] = C.uv_to_angles(*C.nomp_greedy(H, L, 8, 6))
    P['FFT parabolic os4'] = fft_par(H, 4); P['FFT parabolic os16'] = fft_par(H, 16)
    P['EB-ML-g5'] = C.uv_to_angles(*ebml_g5(H, L, g))
    al = AL(); ev = v2b_est(H, L); Hv = al.apply_corr(H, ev); gv = C.nomp_greedy(Hv, L, 4, RC)
    P['v2b+NOMP'] = C.uv_to_angles(*gv); P['v2b+NOMP+SC tuned'] = C.uv_to_angles(*A.selfcal_tuned(Hv, *gv))
    ho, Hco, _ = net_forward(orig, Y)
    go = C.nomp_greedy(Hco, L, 4, RC); hro = heat_ref(Hco, ho, L)
    P['orig heat+SC'] = scd(Hco, hro); P['orig dual+SC'] = scd(Hco, dual(Hco, hro, go)); P['orig network only'] = A.parabolic_heat_decode(ho, L)
    hf, Hcf, _ = net_forward(ft, Y)
    gf = C.nomp_greedy(Hcf, L, 4, RC); hrf = heat_ref(Hcf, hf, L)
    P['FT heat+SC'] = scd(Hcf, hrf); P['FT dual+SC'] = scd(Hcf, dual(Hcf, hrf, gf)); P['FT network only'] = A.parabolic_heat_decode(hf, L)
    P['FT NOMP on Hc+SC'] = scd(Hcf, gf)
    P['FT seeds on Ht+SC'] = scd(H, heat_ref(H, hf, L))
    hn = trunk_heat(ft, Y); hrn = heat_ref(H, hn, L)
    P['FT nocorr heat+SC'] = scd(H, hrn); P['FT nocorr dual+SC'] = scd(H, dual(H, hrn, g)); P['FT nocorr network only'] = A.parabolic_heat_decode(hn, L)
    hv = trunk_heat(ft, to_beam(Hv)); hrv = heat_ref(Hv, hv, L)
    P['FT v2b heat+SC'] = scd(Hv, hrv); P['FT v2b dual+SC'] = scd(Hv, dual(Hv, hrv, gv)); P['FT v2b network only'] = A.parabolic_heat_decode(hv, L)
    U0 = np.pi * np.cos(b['psi'][:, :L]); V0 = np.pi * np.cos(b['phi'][:, :L])
    P['oracle-seeded'] = C.uv_to_angles(*C.cyclic(H, U0, V0, 3))
    if SMOKE:   # equivalence checks of the re-used decoder pieces against the notebook decode() and the CNN forward
        def mx(a, c):
            return max(float(np.nanmax(np.abs(np.asarray(x) - np.asarray(y)))) for x, y in zip(a, c))
        print('  check FT dual+SC == decode(Hcf, dual, SC):', mx(P['FT dual+SC'], C.decode(Hcf, L, hf, 'dual', SC)),
              '| classical dual == decode(H, fft_heat, dual):', mx(P['classical dual'], C.decode(H, L, A.fft_heat(H, 4), 'dual', None)),
              '| trunk_heat(ft, W^H Hcf F) vs forward heat:', float(np.abs(trunk_heat(ft, to_beam(Hcf)) - hf).max()), flush=True)
    return P


def stress_banks(kinds):
    for snr in [15, 0]:
        b = SS.base_scenes(500, 3, float(snr), STRESS_SEEDS[snr])
        for kind in kinds:
            for sev in SS.SEVERITIES[kind]:
                yield f'stress_{kind}_{SS.sev_label(kind, sev)}_snr{snr}', b, SS.generate(b, kind, sev), (kind, SS.sev_label(kind, sev), snr)


def banks(SET):
    if SET == 'stress':
        yield from stress_banks(['M1', 'M2', 'M3', 'M6', 'M8'])
    elif SET == 'clean':
        for snr in [15, 0]:
            b = SS.base_scenes(500, 3, float(snr), STRESS_SEEDS[snr]); yield f'clean_snr{snr}', b, SS.generate(b, 'clean'), ('clean', 'clean', snr)
    elif SET == 'official':
        z = np.load('D:/ai_ml_project/IABR_Net_TestSuite/data/frozen_banks/eval_bank.npz')
        Y16 = SIM2.downsample16(z['data']); Y = Y16[..., 0].astype(np.float64) + 1j * Y16[..., 1].astype(np.float64)
        b = dict(psi=z['feat'][:, 0, :].astype(np.float64), phi=z['feat'][:, 1, :].astype(np.float64), L=z['meta'][:, 0].astype(int),
                 snr=z['meta'][:, 1].astype(np.float32))
        assert (b['L'] == 3).all(); del z, Y16
        yield 'official', b, Y, None
    elif SET.startswith('fresh:'):
        kind = SET.split(':')[1]
        for snr in [15, 0]:
            b = SS.base_scenes(1000, 3, float(snr), FRESH_SEEDS[snr])
            sevs = [None] if kind == 'clean' else SS.SEVERITIES[kind]
            for sev in sevs:
                lab = 'clean' if kind == 'clean' else SS.sev_label(kind, sev)
                yield f'fresh_{kind}_{lab}_snr{snr}', b, SS.generate(b, kind, sev), None


def evaluate_bank(name, b, Y, tag):
    jpath = f'{HERE}/parts/{name}{"_smoke" if SMOKE else ""}.json'
    if os.path.exists(jpath):
        print(f'skip {name} (done)', flush=True); return json.load(open(jpath))
    if SMOKE:
        b = {k: (v[:SMOKE] if isinstance(v, np.ndarray) and v.ndim and len(v) == len(Y) else v) for k, v in b.items()}; Y = Y[:SMOKE]
    t0 = time.time(); P = {}
    for s0 in range(0, len(Y), 400):          # memory rule: <= 400 scenes per decoder chunk, CNN batches of 250
        sl = slice(s0, s0 + 400)
        bc = {k: (v[sl] if isinstance(v, np.ndarray) and v.ndim and len(v) == len(Y) else v) for k, v in b.items()}
        Pc = run(bc, Y[sl])
        for m, v in Pc.items():
            P.setdefault(m, []).extend(as_list(v))
        del Pc
    S = {m: stats(P[m], b) for m in ALLM}
    np.savez_compressed(f'{PCACHE}/stats_{name}{"_smoke" if SMOKE else ""}.npz', snr=b['snr'], **S)
    rec = dict(n=len(Y), sec=time.time() - t0)
    if tag is not None and tag[0] != 'clean':
        p = f'D:/ai_ml_project/paper/evidence/deep_research/stress/cache/preds__{tag[0]}_{tag[1]}_snr{tag[2]}.npz'
        if os.path.exists(p):
            z = np.load(p); pk = pack(P['NOMP'], 3); n = len(Y)
            rec['Y_check_nomp_maxdiff_vs_stress_cache'] = float(max(np.nanmax(np.abs(pk['psi'] - z['NOMP__psi'][:n])), np.nanmax(np.abs(pk['phi'] - z['NOMP__phi'][:n]))))
    Bt = C.boot_metrics(S, b['snr'], B=2000, seed=0)
    pdm = {m: C.metric_by_snr(S[m], b['snr'])['mean']['pd_paper'] for m in ALLM}
    pds = {m: C.metric_by_snr(S[m], b['snr'])['mean']['pd_strict'] for m in ALLM}
    best = max(CLASSICAL, key=lambda m: pdm[m])
    rec.update(pd=pdm, pd_strict=pds, pd_ci={m: list(C.ci(Bt[m][:, 0])[1:]) for m in ALLM}, best_classical=best, diffs={})
    if name == 'official':
        rec['by_snr'] = {m: {str(k): v['pd_paper'] for k, v in C.metric_by_snr(S[m], b['snr']).items()} for m in ALLM}
    refs = {'BEST': best, 'NOMP': 'NOMP', 'NOMP+SC tuned': 'NOMP+SC tuned', 'EB-ML-g5': 'EB-ML-g5', 'orig dual+SC': 'orig dual+SC',
            'orig heat+SC': 'orig heat+SC', 'v2b+NOMP+SC tuned': 'v2b+NOMP+SC tuned', 'NOMP+SC': 'NOMP+SC', 'FT heat+SC': 'FT heat+SC'}
    for a_ in ALLM:
        for key, r_ in refs.items():
            if a_ != r_:
                d = Bt[a_][:, 0] - Bt[r_][:, 0]
                rec['diffs'][f'{a_} - {key}'] = [pdm[a_] - pdm[r_], float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))]
    for x, y in [('FT seeds on Ht+SC', 'NOMP+SC'), ('FT NOMP on Hc+SC', 'NOMP+SC'), ('FT heat+SC', 'FT seeds on Ht+SC'), ('FT heat+SC', 'FT NOMP on Hc+SC'),
                 ('FT NOMP on Hc+SC', 'FT seeds on Ht+SC'), ('FT v2b heat+SC', 'FT heat+SC'), ('FT nocorr heat+SC', 'FT heat+SC'),
                 ('FT v2b dual+SC', 'FT dual+SC'), ('FT nocorr dual+SC', 'FT dual+SC'), ('FT network only', 'FFT parabolic os16'),
                 ('FT network only', 'orig network only'), ('FT dual+SC', 'orig dual+SC'), ('FT heat+SC', 'orig heat+SC')]:
        d = Bt[x][:, 0] - Bt[y][:, 0]; rec['diffs'][f'{x} - {y}'] = [pdm[x] - pdm[y], float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))]
    save_json(rec, jpath)
    yc = rec.get('Y_check_nomp_maxdiff_vs_stress_cache')
    print(f'=== {name} (n={len(Y)}) best classical {best} {pdm[best]:.4f} [{rec["sec"]:.0f}s]' + ('' if yc is None else f'  Y-check NOMP max|diff| vs stress cache {yc:.1e}'), flush=True)
    for m in ALLM:
        dd = rec['diffs'].get(f'{m} - BEST'); dn = rec['diffs'].get(f'{m} - NOMP')
        print(f'   {m:24s} Pd {pdm[m]:.4f}' + (f'  vs best {dd[0]:+.4f} [{dd[1]:+.4f},{dd[2]:+.4f}]' if dd else '') +
              (f'  vs NOMP {dn[0]:+.4f} [{dn[1]:+.4f},{dn[2]:+.4f}]' if dn else ''), flush=True)
    for k in ['FT seeds on Ht+SC - NOMP+SC', 'FT NOMP on Hc+SC - NOMP+SC', 'FT heat+SC - FT seeds on Ht+SC', 'FT heat+SC - FT NOMP on Hc+SC']:
        d = rec['diffs'][k]; print(f'   ablation {k:34s} {d[0]:+.4f} [{d[1]:+.4f},{d[2]:+.4f}]', flush=True)
    return rec


if __name__ == '__main__':
    os.makedirs(f'{HERE}/parts', exist_ok=True)
    print(f'FT weights: {FT_PATH}', flush=True)
    for SET in sys.argv[1:]:
        t00 = time.time()
        for name, b, Y, tag in banks(SET):
            evaluate_bank(name, b, Y, tag)
            del Y
        print(f'set {SET} done in {time.time() - t00:.0f}s', flush=True)
    print('done', flush=True)
