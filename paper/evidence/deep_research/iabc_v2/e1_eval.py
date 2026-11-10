"""Evaluation of IABC-v2 vs the old IABC and classical comparators on the SAME scenes (paired).
Test scenes (fresh, disjoint from all training / tuning / earlier test seeds): L=3, SNR -10..25 step 5, 250 scenes per SNR,
SNR index k -> seed 8900+k (angles / alpha: default_rng(seed); noise [seed,1]; impairment uniforms [seed,3]) -- identical angles,
noise and uniform draws in every severity cell (only scaled), so all comparisons across methods and cells are paired.
Cells: clean; phase {5,8,20,30,45} deg; gain {3,6} dB; combined (8,3), (20,4), (30,6).
Estimators (identifiable residual error): none | old IABC (CNN) | IABC-v2a (blind transformer) | IABC-v2b (v2a + NOMP + ALS + learned
refiner) | Toeplitz (classical blind structure-based) | SC tuned (ALS from NOMP angles, lam 1, tau 2, 4 it) | SC open (tau 0, 4 it) |
SC snrgate (SC open only where nominal SNR >= 10 dB, else none; threshold from the earlier 7700-seed study) | oracle ALS (true
angles, lam 1, tau 0, 4 it, angles fixed) | v2a+SC tuned | v2b+SC tuned.
Pd (true L): NOMP | old IABC+NOMP | v2a+NOMP | v2b+NOMP | Toeplitz+NOMP | NOMP+SC tuned | NOMP+SC snrgate | v2a+NOMP+SC tuned |
v2b+NOMP+SC tuned | CEILING (NOMP on Ht corrected by the TRUE identifiable impairment) | FULL (true distortion incl. the
unidentifiable mean/ramp removed; reference only) | IABR-Net heat (original; decode 'heat', sc (1,5)) | IABR-Net heat with v2a /
v2b swapped in (CNN trunk frozen: trunk on W^H Hc F).
Per-scene results -> cache/eval__<cell>.npz (resume: existing cells skipped). Chunks = one SNR group (250 scenes).
Also the official bank (8000, ideal hardware, cached Ht/Hc/heat from outputs/cache/net__official.npz) and some cached v1 banks
(Pd only: truth impairments not stored) -> cache/eval__bank_<name>.npz."""
from v2common import *
from iabc2_model import load_v2, load_ref, predict

N = 250; L = 3
CELLS = [('clean', 0, 0), ('ph8', 8, 0), ('g3', 0, 3), ('c8_3', 8, 3), ('ph20', 20, 0), ('g6', 0, 6), ('c20_4', 20, 4), ('c30_6', 30, 6),
         ('ph45', 45, 0), ('ph30', 30, 0), ('ph5', 5, 0)]
BANKS = ['official', 'sep_3deg_snr15', 'sep_5deg_snr15', 'sep_10deg_snr15', 'tail_snr35', 'tail_snr30', 'L1_snr15', 'L8_snr15',
         'gain_g4_snr15', 'oodphase_d15_snr15', 'L5_snr0']
v2a = load_v2(f'{V2CACHE}/v2a_best.pt'); ref = load_ref(f'{V2CACHE}/v2b_best.pt')


def refine(tok, glob, eA):
    with torch.no_grad():
        return ref(torch.from_numpy(tok), torch.from_numpy(glob), torch.from_numpy(eA.astype(np.float32))).double().numpy()


def heat_decode(Hc, heat, L):
    Uh, Vh = heat_seed(Hc, heat, L)
    return C.selfcal(Hc, Uh, Vh, 1.0, 5.0)


def run_chunk(Y, Ht, heat_o, Hc_o, imp_o, L, snr, tgt=None, D_r=None, D_t=None, U0=None, V0=None, check=False):
    eA = predict(v2a, Ht)
    tok, glob, (U1, V1) = hybrid_inputs(Ht, L, eA)
    eB = refine(tok, glob, eA)
    eT = toeplitz_est(Ht)
    HA, HB, HT = apply_corr(Ht, eA), apply_corr(Ht, eB), apply_corr(Ht, eT)
    Ug, Vg = C.nomp_greedy(Ht, L, 4, 3)
    ANG = {'NOMP': (Ug, Vg), 'oldIABC+NOMP': C.nomp_greedy(Hc_o, L, 4, 3), 'v2a+NOMP': (U1, V1), 'v2b+NOMP': C.nomp_greedy(HB, L, 4, 3),
           'Toeplitz+NOMP': C.nomp_greedy(HT, L, 4, 3)}
    Ut, Vt_, e_tun, _, _ = selfcal_ret(Ht, Ug, Vg, 1.0, 2.0, iters=4)
    Uo, Vo, e_open, _, _ = selfcal_ret(Ht, Ug, Vg, 1.0, 0.0, iters=4)
    hi = (snr >= 10)[:, None]
    ANG['NOMP+SC tuned'] = (Ut, Vt_)
    ANG['NOMP+SC snrgate'] = (np.where(hi, Uo, Ug), np.where(hi, Vo, Vg))
    UaS, VaS, e_aS, _, _ = selfcal_ret(HA, U1, V1, 1.0, 2.0, iters=4)
    UbS, VbS, e_bS, _, _ = selfcal_ret(HB, *ANG['v2b+NOMP'], 1.0, 2.0, iters=4)
    ANG['v2a+NOMP+SC tuned'] = (UaS, VaS); ANG['v2b+NOMP+SC tuned'] = (UbS, VbS)
    ANG['IABR-Net heat'] = heat_decode(Hc_o, heat_o, L)
    ANG['IABR-Net heat (v2a swapped)'] = heat_decode(HA, trunk_heat(to_beam(HA)), L)
    ANG['IABR-Net heat (v2b swapped)'] = heat_decode(HB, trunk_heat(to_beam(HB)), L)
    rec = {}
    if check:
        h2 = trunk_heat(to_beam(Hc_o)); print(f'  check trunk_heat(W^H Hc_old F) vs CNN heat: max|d| {np.abs(h2 - heat_o).max():.1e}', flush=True)
    if tgt is not None:
        ANG['CEILING'] = C.nomp_greedy(apply_corr(Ht, tgt), L, 4, 3)
        ANG['FULL'] = C.nomp_greedy(full_true_corr(Ht, D_r, D_t), L, 4, 3)
        e_or = selfcal_ret(Ht, U0, V0, 1.0, 0.0, iters=4, fix_angles=True)[2]
        EST = {'none': np.zeros_like(tgt), 'oldIABC': imp_o, 'v2a': eA, 'v2b': eB, 'Toeplitz': eT, 'SC tuned': e_tun, 'SC open': e_open,
               'SC snrgate': np.where(hi[:, :, None], e_open, 0.0), 'oracle ALS': e_or, 'v2a+SC tuned': eA + e_aS, 'v2b+SC tuned': eB + e_bS}
        for k, e in EST.items():
            ph, g, mp, mg = err_rms(tgt, e); rec[f'rp_{k}'] = ph; rec[f'rg_{k}'] = g; rec[f'mp_{k}'] = mp; rec[f'mg_{k}'] = mg
    else:
        for k, e in {'oldIABC': imp_o, 'v2a': eA, 'v2b': eB, 'Toeplitz': eT}.items():      # ideal hardware: any estimate is pure error
            ph, g, mp, mg = err_rms(np.zeros_like(e), e); rec[f'rp_{k}'] = ph; rec[f'rg_{k}'] = g
    return ANG, rec


def finish(ANG_parts, rec_parts, psi, phi, Ls, snr_all, fn, extra):
    rec = {k: np.concatenate([r[k] for r in rec_parts]) for k in rec_parts[0]}
    ST = {}
    for m in ANG_parts[0]:
        U = np.concatenate([a[m][0] for a in ANG_parts]); V = np.concatenate([a[m][1] for a in ANG_parts])
        ST[m] = stats(C.uv_to_angles(U, V), psi, phi, Ls)
    np.savez_compressed(fn, snr=snr_all, methods=np.array(list(ST)), **{f'S_{m}': s for m, s in ST.items()}, **rec, **extra)
    return ST, rec


t00 = time.time()
for name, pdeg, gdb in CELLS:
    fn = f'{V2CACHE}/eval__{name}.npz'
    if os.path.exists(fn):
        print(f'{name}: cached', flush=True); continue
    t0 = time.time(); AP, RP, PS, PH, SN = [], [], [], [], []
    for k, snr in enumerate(SNRS):
        d = make_scenes(8900 + k, N, L, snr, pdeg, gdb); Ht = to_ant(d['Y'])
        heat_o, Hc_o, imp_o = net(d['Y'])
        U0, V0 = true_uv(d['psi'][:, :L], d['phi'][:, :L])
        a, r = run_chunk(d['Y'], Ht, heat_o, Hc_o, imp_o, L, d['snr'], d['tgt'], d['D_r'], d['D_t'], U0, V0, check=(k == 7 and name == 'clean'))
        AP.append(a); RP.append(r); PS.append(d['psi']); PH.append(d['phi']); SN.append(d['snr'])
    psi = np.concatenate(PS); phi = np.concatenate(PH); snr_all = np.concatenate(SN)
    ST, rec = finish(AP, RP, psi, phi, np.full(len(psi), L), snr_all, fn, dict(phase_deg=pdeg, gain_db=gdb))
    pd = {m: C.metric_from_stats(s, np.arange(len(s)))['pd_paper'] for m, s in ST.items()}
    print(f'{name} ({pdeg} deg, {gdb} dB) {time.time() - t0:.0f}s | RMS phase deg: ' + ' '.join(f'{k} {rec["rp_" + k].mean():.2f}' for k in ['none', 'oldIABC', 'v2a', 'v2b', 'Toeplitz', 'SC tuned', 'oracle ALS']) +
          ' | gain dB: ' + ' '.join(f'{k} {rec["rg_" + k].mean():.2f}' for k in ['none', 'oldIABC', 'v2a', 'v2b', 'Toeplitz', 'SC tuned', 'oracle ALS']), flush=True)
    print('   Pd ' + ' | '.join(f'{m} {v:.3f}' for m, v in pd.items()), flush=True)

for bank in BANKS:
    fn = f'{V2CACHE}/eval__bank_{bank}.npz'
    if os.path.exists(fn) or not C.bank_exists(bank) or not os.path.exists(f'{C.CACHE}/net__{bank}.npz'):
        print(f'bank {bank}: {"cached" if os.path.exists(fn) else "missing -> skip"}', flush=True); continue
    t0 = time.time(); b = C.load_bank(bank); z = np.load(f'{C.CACHE}/net__{bank}.npz')
    AP, RP, IX = [], [], []
    for Lv in np.unique(b['L']):
        ix = np.where(b['L'] == Lv)[0]
        for s0 in range(0, len(ix), 400):
            q = ix[s0:s0 + 400]; Ht = z['Ht'][q].astype(complex); Hc_o = z['Hc'][q].astype(complex)
            with torch.no_grad():
                imp_o = cnn.iabc(torch.from_numpy(Ht.astype(np.complex64)), torch.from_numpy(to_beam(Ht).astype(np.complex64))).double().numpy()
            a, r = run_chunk(None, Ht, z['heat'][q].astype(float), Hc_o, imp_o, int(Lv), b['snr'][q].astype(float))
            AP.append(a); RP.append(r); IX.append(q)
    q = np.concatenate(IX)
    Lm = b['psi'].shape[1]
    ST, rec = finish(AP, RP, b['psi'][q], b['phi'][q], b['L'][q], b['snr'][q].astype(float), fn, dict(idx=q))
    pd = {m: C.metric_from_stats(s, np.arange(len(s)))['pd_paper'] for m, s in ST.items()}
    print(f'bank {bank} N={len(q)} {time.time() - t0:.0f}s | ' + ' | '.join(f'{m} {v:.4f}' for m, v in pd.items()), flush=True)
    print(f'   added RMS (ideal hw unless bank impaired): ' + ' '.join(f'{k} {rec["rp_" + k].mean():.2f} deg/{rec["rg_" + k].mean():.2f} dB' for k in ['oldIABC', 'v2a', 'v2b', 'Toeplitz']), flush=True)
print(f'done {time.time() - t00:.0f}s', flush=True)
