"""NEW MAIN METHOD (rewrite requested by the owner): IABR-Net = CNN heatmap -> heatmap peaks seed the decoder (no greedy FFT path, no dual initialisation)
-> Newton + cyclic refinement -> self-calibration, on the IABC-corrected channel Hc. Uses the cached CNN outputs net__<bank>.npz (heat = sigmoid heatmap, Hc, Ht).
Writes preds_heat/<bank>.npz and bank_json_heat/<bank>.json with metrics() quantities (paper-style Pd, strict Pd, pd_source, conditional RMSE),
per-SNR values for the 8-SNR banks and paired bootstrap (2000 resamples within SNR groups, seed 0) of: IABR-Net - U-Net (all banks) and, on the 8 ablation banks,
IABR-Net - {without self-calibration, decoder on the uncorrected channel Ht, FFT-initialised decoder (no network) + self-calibration}.
usage: python heat_main.py bank1,bank2,...   (or 'all')"""
import sys; sys.path.insert(0, 'D:/ai_ml_project/paper/evidence/new_results')
from common import *
os.makedirs(f'{OUT}/preds_heat', exist_ok=True); os.makedirs(f'{OUT}/bank_json_heat', exist_ok=True)
ABL = ['official', 'T2_d5', 'gain_g4_snr15', 'gain_g2_snr0', 'oodphase_d15_snr15', 'L8_snr0', 'L8_snr15', 'sep_5deg_snr15']
names = sys.argv[1].split(',')
if names == ['all']:
    names = sorted(f[:-5] for f in os.listdir(f'{OUT}/bank_json'))


def run_heat(Hant, heat, Ls, sc):
    psi = np.full((len(Ls), int(Ls.max())), np.nan); phi = psi.copy()
    for L in sorted(set(Ls.tolist())):
        ii = np.where(Ls == L)[0]
        for s in range(0, len(ii), 1000):
            jj = ii[s:s + 1000]
            a, b = decode(Hant[jj].astype(np.complex128), L, heat[jj].astype(np.float64), 'heat', sc)
            psi[jj, :L], phi[jj, :L] = a, b
    return dict(psi=psi, phi=phi, n=np.array(Ls))


def stats_of(p, b):
    return scene_stats([(p['psi'][i, :p['n'][i]], p['phi'][i, :p['n'][i]]) for i in range(len(p['n']))], b)


for name in names:
    t0 = time.time(); b = load_bank(name); Ls = b['L']; net = np.load(f'{CACHE}/net__{name}.npz')
    assert len(net['heat']) == len(Ls)
    main = run_heat(net['Hc'], net['heat'], Ls, SC)
    np.savez_compressed(f'{OUT}/preds_heat/{name}.npz', **main)
    S = {'IABR-Net': stats_of(main, b)}
    zu = np.load(f'{CACHE}/pred__U-Net_paper__{name}.npz'); S['U-Net'] = scene_stats([(zu['psi'][i, :zu['n'][i]], zu['phi'][i, :zu['n'][i]]) for i in range(len(zu['n']))], b)
    diffs = [('IABR-Net - U-Net', 'IABR-Net', 'U-Net')]
    if name in ABL:
        S['no self-calibration'] = stats_of(run_heat(net['Hc'], net['heat'], Ls, None), b)
        S['uncorrected channel'] = stats_of(run_heat(net['Ht'], net['heat'], Ls, SC), b)
        zf = np.load(f'{OUT}/preds/selfcal_pred__{name}.npz'); S['FFT init, no network'] = scene_stats([(zf['psi'][i, :zf['n'][i]], zf['phi'][i, :zf['n'][i]]) for i in range(len(zf['n']))], b)
        zn = f'{CACHE}/pred__v2_network_only_no_physics__{name}.npz'
        if os.path.exists(zn):
            z = np.load(zn); S['network only'] = scene_stats([(z['psi'][i, :z['n'][i]], z['phi'][i, :z['n'][i]]) for i in range(len(z['n']))], b)
        diffs += [('IABR-Net - no self-calibration', 'IABR-Net', 'no self-calibration'), ('IABR-Net - uncorrected channel', 'IABR-Net', 'uncorrected channel'),
                  ('IABR-Net - FFT init, no network', 'IABR-Net', 'FFT init, no network')]
    B = boot_metrics(S, b['snr'], B=2000, seed=0); snrs = sorted(set(b['snr'].astype(int).tolist()))
    res = dict(bank=name, n=len(Ls), L=sorted(set(Ls.tolist())), snr_groups=snrs, methods={}, diffs={})
    for m in S:
        per = metric_by_snr(S[m], b['snr'])
        res['methods'][m] = dict(mean=per['mean'], ci={k: ci(B[m][:, j])[1:] for j, k in enumerate(KEYS)}, per_snr={str(s): per[s] for s in snrs} if len(snrs) > 1 else None)
    for nm, a_, b_ in diffs:
        d = B[a_] - B[b_]; ma, mb = res['methods'][a_]['mean'], res['methods'][b_]['mean']
        res['diffs'][nm] = {k: [ma[k] - mb[k], *ci(d[:, j])[1:]] for j, k in enumerate(KEYS)}
    json.dump(res, open(f'{OUT}/bank_json_heat/{name}.json', 'w'), indent=1, default=float)
    r = res['methods']['IABR-Net']['mean']; u = res['methods']['U-Net']['mean']; d = res['diffs']['IABR-Net - U-Net']['pd_paper']
    print(f'{name:22s} N={len(Ls):5d} IABR-Net pd_paper {r["pd_paper"]:.4f} strict {r["pd_strict"]:.4f} | U-Net {u["pd_paper"]:.4f} strict {u["pd_strict"]:.4f} | diff {d[0]:+.4f} [{d[1]:+.4f},{d[2]:+.4f}] | {time.time()-t0:.0f}s', flush=True)
