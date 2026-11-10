"""Light check supporting the literature survey (recommendation R2, multi-hypothesis selection).
Upper bound of a per-scene SELECTOR between cached candidate decoders on the official bank (8000 scenes):
an oracle that, per scene, keeps the candidate with the most correct (|err|<=1 deg) components (ties -> NOMP).
This is NOT an achievable method (it uses the truth); it bounds what any learned candidate selector could gain
from these candidate sets. Reads cached predictions only; writes oracle_select.json next to this script."""
import sys, json, numpy as np
sys.path.insert(0, 'D:/ai_ml_project/paper/evidence/new_results')
from common import load_bank, load_preds, scene_stats, metric_by_snr, CACHE

bank = load_bank('official'); snr = bank['snr']
names = ['NOMP_no_network', 'v2_heatmap_init_only', 'IABR-v2', 'U-Net_paper', 'v2_network_only_no_physics']
S = {n: scene_stats(load_preds(f'{CACHE}/pred__{n}__official.npz'), bank) for n in names}

def select(cands):
    base = S[cands[0]].copy()
    for c in cands[1:]:
        Sc = S[c]
        # candidate wins only if it has strictly more correct components and is not dropped
        win = (Sc[:, 0] > base[:, 0]) & (Sc[:, 2] == 0)
        base[win] = Sc[win]
    return base

out = {}
for n in names:
    out[n] = metric_by_snr(S[n], snr)
sets = {'oracle{NOMP,heat}': ['NOMP_no_network', 'v2_heatmap_init_only'],
        'oracle{NOMP,heat,dual}': ['NOMP_no_network', 'v2_heatmap_init_only', 'IABR-v2'],
        'oracle{NOMP,heat,dual,UNet,netonly}': names}
for k, c in sets.items():
    out[k] = metric_by_snr(select(c), snr)

snrs = sorted(int(s) for s in set(snr.astype(int).tolist()))
print('paper-style Pd by SNR (official bank, 8000 scenes); oracle rows are truth-based UPPER BOUNDS, not methods')
print('%-40s' % 'method' + ''.join('%8d' % s for s in snrs) + '    mean')
for k, v in out.items():
    print('%-40s' % k + ''.join('%8.4f' % v[s]['pd_paper'] for s in snrs) + '  %7.4f' % v['mean']['pd_paper'])
json.dump({k: {str(s): v[s] for s in v} for k, v in out.items()},
          open('D:/ai_ml_project/paper/evidence/deep_research/unfolded_v2/lit/oracle_select.json', 'w'), indent=1, default=float)
