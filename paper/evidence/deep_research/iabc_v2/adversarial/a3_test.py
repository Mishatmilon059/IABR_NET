"""Adversarial re-measurement of the IABC-v2 headline numbers on FRESH seeds (never used before by any stage):
  cells: L=3, SNR -10..25 step 5, N=200 scenes per SNR, SNR index k -> seed 9300+k (angles/alpha default_rng(seed), noise [seed,1],
  impairment uniforms [seed,3]; v2common.make_scenes) -- paired across cells and methods.
  Cells: clean, ph8, g3, c30_6, ph45, c8_3, g6, ph20.
  official bank (8000, ideal hardware): Ht from outputs/cache/net__official.npz, true L, chunks of 200 scenes per L.
Estimators (identifiable error): v2b | v2b+SC tuned | SC tuned (lam 1 tau 2, 4 it) | SC tuned 10it | SC open (tau 0, 4 it) |
  SC redetect (2 rounds) | EB-ML | EB-ML-g5 | EB-ML-g10 | EB-ML-g5+SC tuned (a_lib.py; constants fixed before this run).
Pd (true L, paper Pd): NOMP | NOMP+SC tuned | NOMP+SC tuned 10it | NOMP+SC redetect | v2b+NOMP | v2b+NOMP+SC tuned | EB-ML+NOMP |
  EB-ML-g5+NOMP | EB-ML-g10+NOMP | EB-ML-g5+NOMP+SC tuned | CEILING (cells only).
Results -> scratchpad dr_cache/iabc_v2_adv/ (cell__*.npz, bank chunks *.pkl; resume: existing files skipped). stdout -> a3_test.log"""
import json, time, pickle
from a_lib import *

OUT = 'C:/Users/Milon/AppData/Local/Temp/claude/D--thesis/a5db9a29-d75e-4561-834b-cd0638a16d91/scratchpad/dr_cache/iabc_v2_adv'
os.makedirs(OUT, exist_ok=True)
N, L0 = 200, 3
CELLS = [('clean', 0, 0), ('ph8', 8, 0), ('g3', 0, 3), ('c30_6', 30, 6), ('ph45', 45, 0), ('c8_3', 8, 3), ('g6', 0, 6), ('ph20', 20, 0)]
PLAN = os.path.dirname(os.path.abspath(__file__)) + '/a3_plan.json'   # (bug in the run: HERE was shadowed by icommon.HERE, plan landed in ../../iabc/ and was moved here)
if not os.path.exists(PLAN):
    json.dump(dict(written=time.strftime('%Y-%m-%dT%H:%M:%S'), note='written by a3_test.py before its first test computation',
                   test_seeds='9300..9307 (+[s,1],[s,3]); official bank', proto_seeds='9340..9347 (a0_proto)', unit_seeds='9350-9352 (a2_unit)',
                   crb_seeds='9330-9332 (a1_crb_check)', classical_constants='SC tuned (1,2,4) from stress/adversarial; EB-ML none; gates 5/10 dB fixed after a0_proto',
                   rule='paired bootstrap 2000 within SNR, seed 0; claim needs 95% CI excluding 0'), open(PLAN, 'w'), indent=1)


def run(Ht, L, tgt=None):
    E, A = {}, {}
    Ug, Vg = C.nomp_greedy(Ht, L, 4, 3); A['NOMP'] = (Ug, Vg)
    e, A['NOMP+SC tuned'] = sc_iter(Ht, L, 4, UV=(Ug, Vg)); E['SC tuned'] = e
    e, A['NOMP+SC tuned 10it'] = sc_iter(Ht, L, 10, UV=(Ug, Vg)); E['SC tuned 10it'] = e
    E['SC open'] = sc_iter(Ht, L, 4, tau=0.0, UV=(Ug, Vg))[0]
    e, A['NOMP+SC redetect'] = sc_redetect(Ht, L); E['SC redetect'] = e
    eB = v2b_est(Ht, L); E['v2b'] = eB; HB = apply_corr(Ht, eB); A['v2b+NOMP'] = C.nomp_greedy(HB, L, 4, 3)
    e, A['v2b+NOMP+SC tuned'] = sc_iter(HB, L, 4, UV=A['v2b+NOMP']); E['v2b+SC tuned'] = eB + e
    eE, UVE, eml, snr_hat = eb_ml(Ht, L); E['EB-ML'] = eE; A['EB-ML+NOMP'] = UVE
    for g in (5, 10):
        eg, UVg = gate(eE, UVE, (Ug, Vg), snr_hat, g); E[f'EB-ML-g{g}'] = eg; A[f'EB-ML-g{g}+NOMP'] = UVg
    e, A['EB-ML-g5+NOMP+SC tuned'] = sc_iter(apply_corr(Ht, E['EB-ML-g5']), L, 4, UV=A['EB-ML-g5+NOMP'])
    E['EB-ML-g5+SC tuned'] = E['EB-ML-g5'] + e
    if tgt is not None:
        A['CEILING'] = C.nomp_greedy(apply_corr(Ht, tgt), L, 4, 3)
    else:
        tgt = np.zeros((len(Ht), 32, 2))
    rec = {'snr_hat': snr_hat}
    for k, e in [('none', np.zeros_like(tgt))] + list(E.items()):
        ph, gg, mp, mg = err_rms(tgt, e); rec[f'mp_{k}'] = mp; rec[f'mg_{k}'] = mg
    return A, rec


def save(fn, parts, psi, phi, Ls, snr, extra={}):
    AP = [p[0] for p in parts]; RP = [p[1] for p in parts]
    rec = {k: np.concatenate([r[k] for r in RP]) for k in RP[0]}
    ST = {}
    for m in AP[0]:
        U = np.concatenate([a[m][0] for a in AP]); V = np.concatenate([a[m][1] for a in AP])
        ST[m] = stats(C.uv_to_angles(U, V), psi, phi, Ls)
    np.savez_compressed(fn, snr=snr, methods=np.array(list(ST)), **{f'S_{m}': s for m, s in ST.items()}, **rec, **extra)
    return ST, rec


t00 = time.time()
for name, pdeg, gdb in CELLS:
    fn = f'{OUT}/cell__{name}.npz'
    if os.path.exists(fn):
        print(f'{name}: cached', flush=True); continue
    t0 = time.time(); parts, PS, PH, SN = [], [], [], []
    for k, snr in enumerate(SNRS):
        d = make_scenes(9300 + k, N, L0, snr, pdeg, gdb); Ht = to_ant(d['Y'])
        parts.append(run(Ht, L0, d['tgt'])); PS.append(d['psi']); PH.append(d['phi']); SN.append(d['snr'])
    snr_all = np.concatenate(SN)
    ST, rec = save(fn, parts, np.concatenate(PS), np.concatenate(PH), np.full(len(snr_all), L0), snr_all)
    hi = snr_all >= 5
    pd = {m: C.metric_by_snr(s, snr_all)['mean']['pd_paper'] for m, s in ST.items()}
    print(f'{name} ({pdeg} deg, {gdb} dB) {time.time() - t0:.0f}s [{time.time() - t00:.0f}s total]', flush=True)
    print('   pooled RMS SNR>=5 (deg/dB): ' + ' | '.join(f'{k[3:]} {np.rad2deg(np.sqrt(rec[k][hi].mean())):.2f}/{DB * np.sqrt(rec["mg_" + k[3:]][hi].mean()):.2f}'
                                                    for k in rec if k.startswith('mp_')), flush=True)
    print('   blind SNR estimate median per SNR: ' + ' '.join(f'{s}:{np.median(rec["snr_hat"][snr_all == s]):.1f}' for s in SNRS), flush=True)
    print('   Pd ' + ' | '.join(f'{m} {v:.4f}' for m, v in pd.items()), flush=True)

# ---------------- official bank
fn = f'{OUT}/bank__official.npz'
if not os.path.exists(fn):
    b = C.load_bank('official'); z = np.load(f'{C.CACHE}/net__official.npz')
    parts, IX = [], []
    for Lv in np.unique(b['L']):
        ix = np.where(b['L'] == Lv)[0]
        for s0 in range(0, len(ix), 200):
            q = ix[s0:s0 + 200]; cf = f'{OUT}/bankchunk__official_L{int(Lv)}_{s0}.pkl'
            if os.path.exists(cf):
                with open(cf, 'rb') as f:
                    r = pickle.load(f)
            else:
                r = run(z['Ht'][q].astype(complex), int(Lv))
                with open(cf + '.tmp', 'wb') as f:
                    pickle.dump(r, f)
                os.replace(cf + '.tmp', cf)
            parts.append(r); IX.append(q)
        print(f'   bank L={int(Lv)} ({len(ix)} scenes) done [{time.time() - t00:.0f}s total]', flush=True)
    q = np.concatenate(IX)
    ST, rec = save(fn, parts, b['psi'][q], b['phi'][q], b['L'][q], b['snr'][q].astype(float), dict(idx=q))
    pd = {m: C.metric_by_snr(s, b['snr'][q].astype(float))['mean']['pd_paper'] for m, s in ST.items()}
    print('official bank paper Pd: ' + ' | '.join(f'{m} {v:.4f}' for m, v in pd.items()), flush=True)
print(f'DONE {time.time() - t00:.0f}s', flush=True)
