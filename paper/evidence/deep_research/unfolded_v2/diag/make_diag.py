"""Merge the summary results of the diag study into diag.json (per-scene count arrays stay in analyze.json)."""
import json
H = 'D:/ai_ml_project/paper/evidence/deep_research/unfolded_v2/diag'
J = lambda f: json.load(open(f'{H}/{f}'))
an = J('analyze.json')
d = dict(
    note='Unfolded-decoder headroom diagnostics (decoder family unfolded from NOMP, Mamandipoor et al. 2016). Pd = fraction of '
         'components with |err|<=1 deg, mean over SNR groups. CIs: paired bootstrap 2000 within SNR groups, seed 0.',
    task1_anatomy_official8000=J('anatomy.json'),
    task3_map_ceiling_official8000=J('map_decomp.json'),
    task2_4_selection_per_bank=an['selection'],
    task2_4_pooled_official_val=J('supp.json'),
    task2_oracle_decomposition=J('oracle_decomp.json'),
    task5_adaptivity_official_subset=J('adapt.json'),
    method_level_oracle_official8000=json.load(open(f'{H}/../lit/oracle_select.json')))
json.dump(d, open(f'{H}/diag.json', 'w'), indent=1, default=str)
print('diag.json keys', list(d))
for s, v in d['task1_anatomy_official8000']['per_snr'].items():
    print(s, 'bias median err', round(v['bias_median_err_deg'], 2), 'missed q25/75', [round(x) for x in v['missed_to_spurious_deg_q25_q75']])
