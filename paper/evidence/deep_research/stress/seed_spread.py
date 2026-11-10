"""Follow-up to sanity_clean.py (B failed narrowly at SNR 15: the new clean bank, seed 5015, is ~0.02 below phase_d0_snr15 for all four
decoders while the pipeline re-run on the regenerated phase_d0 banks matches exactly). Seed-to-seed spread of the impairment-free L=3 bank:
20 further seeds (7000..7019) x 500 scenes per SNR, NOMP and IABR-Net heat-seeded, compared with the existing phase_d0 banks (seed 1000/1015) and
the stress clean banks (seed 5000/5015); plus the pooled 10,000-scene Pd. usage: python seed_spread.py (stdout -> seed_spread.log)"""
from scommon import *

res = {}
for snr in [15, 0]:
    pn, ph = [], []; Sn_all, Sh_all = [], []
    for s in range(7000, 7020):
        b = SS.base_scenes(500, 3, float(snr), s); Y = SS.generate(b, 'clean'); H = SS.to_ant(Y)
        U, V = C.nomp_greedy(H, 3, 4, 3); Sn = stats(C.uv_to_angles(U, V), b['psi'], b['phi'], 3)
        heat, Hc = cnn_forward(Y); Sh = stats(C.decode(Hc, 3, heat, 'heat', C.SC), b['psi'], b['phi'], 3)
        pn.append(C.metric_from_stats(Sn, np.arange(500))['pd_paper']); ph.append(C.metric_from_stats(Sh, np.arange(500))['pd_paper'])
        Sn_all.append(Sn); Sh_all.append(Sh)
    Sn_all = np.concatenate(Sn_all); Sh_all = np.concatenate(Sh_all)
    pooled_n = C.metric_from_stats(Sn_all, np.arange(len(Sn_all)))['pd_paper']; pooled_h = C.metric_from_stats(Sh_all, np.arange(len(Sh_all)))['pd_paper']
    san = json.load(open(f'{HERE}/sanity_clean.json'))[f'phase_d0_snr{snr}']
    res[snr] = dict(nomp=pn, heat=ph, pooled_nomp=pooled_n, pooled_heat=pooled_h, pooled_heat_minus_nomp=pooled_h - pooled_n)
    for nm, v, ex, new in [('NOMP', pn, san['NOMP']['pd_existing'], san['NOMP']['pd_new_clean_bank']),
                           ('IABR-Net heat', ph, san['IABR-Net heat']['pd_existing'], san['IABR-Net heat']['pd_new_clean_bank'])]:
        v = np.array(v)
        print(f'SNR {snr:2d} {nm:14s}: 20 seeds x 500: mean {v.mean():.4f}, sd {v.std(ddof=1):.4f}, min {v.min():.4f}, max {v.max():.4f}; '
              f'existing phase_d0 {ex:.4f} (rank {int((v < ex).sum())}/20 below), stress clean {new:.4f} (rank {int((v < new).sum())}/20 below)', flush=True)
    print(f'SNR {snr:2d} pooled 10,000 scenes: NOMP {pooled_n:.4f}, IABR-Net heat {pooled_h:.4f}, heat - NOMP {pooled_h - pooled_n:+.4f}', flush=True)
json.dump(res, open(f'{HERE}/seed_spread.json', 'w'), indent=1)
