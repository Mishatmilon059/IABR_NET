"""Supplement to evaluate.py, added after the first M5 results showed that the fine-tuned network's ONLY large win is the network-only
read-out (heatmap peaks + parabolic interpolation, no physics refinement) and that every LS-refined method (incl. the oracle start at
b >= 2) is pulled to the quantiser-biased LS optimum. Fairness requires the classical counterpart of that read-out, i.e. no refinement:
  'FFT parabolic os4'   top-3 local maxima of the 4x  |FFT2(Hant)| (64x64) + parabolic sub-bin interpolation, no Newton / cyclic
  'FFT parabolic os16'  same on the 16x grid (256x256)
  'consistent parabolic' (ADC banks) the FFT-parabolic os16 seeds as start of consistent NOMP (8 it) - classical hybrid
and the ORIGINAL CNN read the same way ('orig network only') to separate the effect of fine-tuning from the read-out.
Network-only method counts: number of estimates per scene is checked (paper Pd ignores missing estimates).
Reads cache stats_<set>__<bank>.npz written by evaluate.py, adds the new methods, recomputes the BEST classical method per cell over the
extended classical set, and the paired bootstrap (2000, within SNR groups, seed 0). usage: python supp.py m5|fresh|clean|official
-> supp_<set>.log, res2_<set>.json (same format as res_<set>.json, extended)."""
from acommon import *
import iabr2_sim as SIM
torch.set_num_threads(6)
SET = sys.argv[1]
PCACHE = 'C:/Users/Milon/AppData/Local/Temp/claude/D--thesis/a5db9a29-d75e-4561-834b-cd0638a16d91/scratchpad/dr_cache/finetune_adc'
CLASSICAL = ['NOMP', 'NOMP+SC', 'NOMP+SC tuned', 'FFT-peak seeds', 'FFT-peak seeds+SC', 'classical dual', 'classical dual+SC', 'NOMP os8 rc6',
             'multi-start K6', 'consistent NOMP', 'FFT parabolic os4', 'FFT parabolic os16', 'consistent parabolic']
NETWORK = ['orig heat+SC', 'orig dual+SC', 'orig network only', 'FT heat+SC', 'FT dual+SC', 'FT NOMP on Hc', 'FT network only', 'FT heat + consistent']
orig = load_cnn(ORIG_CKPT)
src = open(f'{HERE}/evaluate.py', encoding='utf-8').read()
exec('def banks():' + src.split('def banks():')[1].split('def run(')[0])


def new_methods(b, Y, bits):
    H = to_ant(Y); P = {}
    P['FFT parabolic os4'] = parabolic_heat_decode(fft_heat(H, 4), 3)
    P['FFT parabolic os16'] = parabolic_heat_decode(fft_heat(H, 16), 3)
    if bits is not None:
        ps = P['FFT parabolic os16']; U0 = np.array([np.pi * np.cos(p[0][:3]) for p in ps]); V0 = np.array([np.pi * np.cos(p[1][:3]) for p in ps])
        P['consistent parabolic'] = C.uv_to_angles(*consistent_nomp(Y, bits, 3, 8, U=U0, V=V0)[:2])
    ho, _ = cnn_forward(orig, Y)
    P['orig network only'] = parabolic_heat_decode(ho, 3)
    return P


out = dict(set=SET, banks={})
for name, b, Y, bits in banks():
    t0 = time.time(); P = {}
    for s0 in range(0, len(Y), 200):
        sl = slice(s0, s0 + 200); Pc = new_methods(b, Y[sl], bits)
        for m, v in Pc.items():
            P.setdefault(m, []).extend(as_list(v))
    z = np.load(f'{PCACHE}/stats_{SET}__{name}.npz'); S = {k: z[k] for k in z.files if k != 'snr'}
    assert np.array_equal(z['snr'], b['snr'])
    for m in P:
        S[m] = stats(P[m], b)
    zp = np.load(f'{PCACHE}/preds_{SET}__{name}.npz')
    nest = {m: np.bincount([len(p[0]) for p in P[m]], minlength=4).tolist() for m in P}
    nest['FT network only'] = np.bincount(zp['FT network only__n'], minlength=4).tolist()
    np.savez_compressed(f'{PCACHE}/stats2_{SET}__{name}.npz', snr=b['snr'], **S)
    meths = [m for m in CLASSICAL + NETWORK + ['oracle-seeded'] if m in S]
    Bt = C.boot_metrics(S, b['snr'], B=2000, seed=0)
    pdm = {m: C.metric_by_snr(S[m], b['snr'])['mean']['pd_paper'] for m in meths}
    pds = {m: C.metric_by_snr(S[m], b['snr'])['mean']['pd_strict'] for m in meths}
    cls = [m for m in CLASSICAL if m in S]; best = max(cls, key=lambda m: pdm[m])
    rec = dict(n=len(Y), bits=bits, pd=pdm, pd_strict=pds, pd_ci={m: list(C.ci(Bt[m][:, 0])[1:]) for m in meths}, best_classical=best,
               n_estimates_hist=nest, diffs={})
    if SET == 'official':
        rec['by_snr'] = {m: {str(k): v['pd_paper'] for k, v in C.metric_by_snr(S[m], b['snr']).items()} for m in meths}
    for a_ in meths:
        for key in ['BEST', 'NOMP', 'orig dual+SC', 'orig heat+SC', 'orig network only', 'consistent NOMP']:
            r_ = best if key == 'BEST' else key
            if a_ == r_ or r_ not in S:
                continue
            d = Bt[a_][:, 0] - Bt[r_][:, 0]
            rec['diffs'][f'{a_} - {key}'] = [pdm[a_] - pdm[r_], float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))]
    out['banks'][name] = rec
    print(f'=== {SET} {name} (n={len(Y)}) best classical (extended) = {best} ({pdm[best]:.4f})  [{time.time() - t0:.0f}s]', flush=True)
    print(f'   n-estimates histogram (0..3): {nest}', flush=True)
    for m in meths:
        dd = rec['diffs'].get(f'{m} - BEST')
        print(f'   {m:24s} Pd {pdm[m]:.4f} [{rec["pd_ci"][m][0]:.4f},{rec["pd_ci"][m][1]:.4f}] strict {pds[m]:.4f}' +
              (f'   vs best classical {dd[0]:+.4f} [{dd[1]:+.4f},{dd[2]:+.4f}]' if dd else ''), flush=True)
    save_json(out, f'{HERE}/res2_{SET}.json')
print('done')
