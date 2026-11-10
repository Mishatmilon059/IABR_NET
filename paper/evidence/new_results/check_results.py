"""Re-reads the key numbers quoted in RESULTS_NEW.md and compares them with the raw JSON / text outputs."""
import json
OUT = 'D:/ai_ml_project/paper/evidence/new_results'
md = open(f'{OUT}/RESULTS_NEW.md', encoding='utf-8').read(); bad = 0


def chk(name, cond):
    global bad
    print(('OK   ' if cond else 'FAIL ') + name); bad += (not cond)


J = lambda b: json.load(open(f'{OUT}/bank_json/{b}.json'))
o = J('official')
for m, k, v in [('U-Net', 'pd_paper', 0.7366), ('hybrid', 'pd_paper', 0.7551), ('NOMP', 'pd_paper', 0.7581), ('NOMP+selfcal', 'pd_paper', 0.7576), ('U-Net', 'pd_strict', 0.6940), ('U-Net', 'pd_source', 0.6035),
                ('hybrid', 'pd_source', 0.6681), ('NOMP', 'pd_source', 0.6741), ('NOMP+selfcal', 'pd_source', 0.6720), ('U-Net', 'rmse', 0.3611), ('hybrid', 'rmse', 0.3337), ('NOMP', 'rmse', 0.3277), ('NOMP+selfcal', 'rmse', 0.3288)]:
    chk(f'official {m} {k} = {v}', round(o['methods'][m]['mean'][k], 4) == v)
for d, v in {'NOMP - U-Net': (0.0215, 0.0181, 0.0248), 'hybrid - NOMP': (-0.0030, -0.0043, -0.0016), 'NOMP+selfcal - hybrid': (0.0025, 0.0014, 0.0036), 'NOMP+selfcal - NOMP': (-0.0005, -0.0014, 0.0005)}.items():
    chk(f'official diff {d}', all(abs(a - b) < 6e-5 for a, b in zip(o['diffs'][d]['pd_paper'], v)))
chk('strict NOMP-UNet', [round(x, 4) for x in o['diffs']['NOMP - U-Net']['pd_strict']] == [0.0641, 0.0602, 0.068])
chk('dropped official 538', o['methods']['U-Net']['dropped'] == 538)
for b, v in {'T2_d1': 476, 'T2_d2': 533, 'T2_d5': 427}.items():
    chk(f'dropped {b}', J(b)['methods']['U-Net']['dropped'] == v)
g = J('gain_g4_snr15')
chk('g4 values', [round(g['methods'][m]['mean']['pd_paper'], 4) for m in ['hybrid', 'NOMP+selfcal', 'NOMP']] == [0.9123, 0.8950, 0.8697])
chk('g4 hybrid-NOMP', [round(x, 4) for x in g['diffs']['hybrid - NOMP']['pd_paper']] == [0.0427, 0.0327, 0.0530])
chk('g4 sc-hybrid', [round(x, 4) for x in g['diffs']['NOMP+selfcal - hybrid']['pd_paper']] == [-0.0173, -0.0257, -0.0100])
chk('g4 sc-NOMP', [round(x, 4) for x in g['diffs']['NOMP+selfcal - NOMP']['pd_paper']] == [0.0253, 0.0177, 0.0330])
for b, v in {'gain_g2_snr15': 0.0140, 'oodphase_d15_snr15': 0.0097}.items():
    chk(b + ' sc-NOMP', round(J(b)['diffs']['NOMP+selfcal - NOMP']['pd_paper'][0], 4) == v)
sc_nomp = {b: round(J(b)['diffs']['NOMP+selfcal - NOMP']['pd_paper'][0], 4) for b in ['sep_1deg_snr15', 'sep_2deg_snr15', 'sep_3deg_snr15', 'sep_5deg_snr15']}
chk('sep 1-5 sc-NOMP range 0.046..0.113', min(sc_nomp.values()) == 0.046 and max(sc_nomp.values()) == 0.113)
# counts
R = {b: J(b) for b in ['official']}
import glob, os
R = {os.path.basename(f)[:-5]: json.load(open(f)) for f in glob.glob(f'{OUT}/bank_json/*.json')}
chk('51 banks', len(R) == 51)
exp = {'NOMP - U-Net': (39, 0, 12), 'hybrid - NOMP': (1, 17, 33), 'NOMP+selfcal - hybrid': (19, 1, 31), 'NOMP+selfcal - NOMP': (8, 1, 42)}
for d, e in exp.items():
    pos = sum(r['diffs'][d]['pd_paper'][1] > 0 for r in R.values()); neg = sum(r['diffs'][d]['pd_paper'][2] < 0 for r in R.values())
    chk(f'counts {d}', (pos, neg, 51 - pos - neg) == e)
chk('strict NOMP-UNet counts 48/0/3', (sum(r['diffs']['NOMP - U-Net']['pd_strict'][1] > 0 for r in R.values()), sum(r['diffs']['NOMP - U-Net']['pd_strict'][2] < 0 for r in R.values())) == (48, 0))
# E4
bl = ['sep_1deg_snr15', 'sep_2deg_snr15', 'sep_3deg_snr15', 'sep_5deg_snr15', 'sep_10deg_snr15', 'official']
E = {b: json.load(open(f'{OUT}/e4/e4__{b}.json')) for b in bl}
p = lambda b, v: round(E[b]['variants'][v]['point']['pd_paper'], 4)
d = lambda b, k: [round(x, 4) for x in E[b]['diffs'][k]['pd_paper']]
chk('sep5 greedy/sc/sc0/rc8', [p('sep_5deg_snr15', v) for v in ['greedy', 'sc(1,5)', 'sc(0,5)', 'rc8']] == [0.897, 0.943, 0.9595, 0.964])
chk('ii-iii sep5', d('sep_5deg_snr15', 'ii-iii') == [-0.0165, -0.025, -0.009])
chk('ii-iii 1,2,3', [d(b, 'ii-iii')[0] for b in bl[:3]] == [-0.012, -0.048, -0.04])
chk('ii-iii 1,2,3 exclude 0', all(d(b, 'ii-iii')[2] < 0 for b in bl[:3]))
chk('os8 sep5', d('sep_5deg_snr15', 'os8-i') == [0.0005, 0.0, 0.0015])
chk('rc8-i 5,3,2,1 (0.067,0.149,0.179,0.117 as written; rounding of x.xx5)', [d(b, 'rc8-i')[0] for b in ['sep_5deg_snr15', 'sep_3deg_snr15', 'sep_2deg_snr15', 'sep_1deg_snr15']] == [0.067, 0.1485, 0.1785, 0.1165])
chk('rc8>rc5>greedy up to 5deg', all(p(b, 'rc8') > p(b, 'rc5') > p(b, 'greedy') for b in bl[:4]))
chk('official E4', [p('official', v) for v in ['greedy', 'sc(1,5)', 'sc(0,5)', 'rc8']] == [0.7581, 0.7576, 0.7595, 0.7598] and d('official', 'ii-iii') == [-0.0019, -0.0027, -0.0011])
chk('sep10 all 0.995..0.9955', all(0.995 <= p('sep_10deg_snr15', v) <= 0.9955 for v in ['greedy', 'sc(1,5)', 'sc(0,5)', 'rc5', 'rc8', 'os8']))
chk('E4 greedy==cache all', all(e['recompute_greedy_equals_cache'] for e in E.values()))
# E5
r = json.load(open(f'{OUT}/e5_results.json')); t = open(f'{OUT}/e5_output.txt', encoding='utf-8').read()
chk('E5 iter', [round(r['iteration_control'][k], 4) for k in ['greedy', 'greedy_10x', 'greedy_rc15', 'truth_start', 'selfcal', 'selfcal_3x5x', 'truth_start_selfcal']] == [0.7548, 0.7568, 0.7572, 0.8186, 0.7542, 0.7551, 0.8166])
tg = r['taxonomy']['greedy']; ts = r['taxonomy']['greedy+selfcal']
chk('E5 tax greedy', (tg['fail'], tg['better'], tg['inband'], tg['miss'], tg['c_miss']) == (1120, 1100, 8, 12, 14))
chk('E5 tax selfcal', (ts['fail'], ts['better'], ts['miss']) == (1137, 1129, 1))
ef = r['endfire']['end-fire comps']; ot = r['endfire']['other comps']
chk('E5 endfire', (ef['n'], round(ef['obs_greedy'], 3), round(ef['obs_selfcal'], 3), round(ef['joint_wrap'], 3), round(ef['joint_clamp'], 3), ef['fold_greedy'], ef['fold_selfcal']) == (819, 0.319, 0.309, 0.331, 0.409, 222, 218))
chk('E5 other', (ot['n'], round(ot['obs_greedy'], 3), round(ot['joint_wrap'], 3)) == (11181, 0.787, 0.833))
chk('E5 fold', (r['fold']['fold_greedy'], r['fold']['fold_selfcal'], r['fold']['wrong_greedy']) == (387, 385, 2943))
a = [c for c in r['crb_prediction'] if c['snr'] == 'all'][0]
chk('E5 all', (round(a['obs_greedy'], 3), round(a['obs_selfcal'], 3), round(a['joint_wrap'], 3), round(a['joint_clamp'], 3), round(a['single_wrap'], 3), round(a['single_clamp'], 3)) == (0.755, 0.754, 0.798, 0.804, 0.802, 0.807))
gaps = {c['snr']: round(c['joint_wrap'] - c['obs_greedy'], 3) for c in r['crb_prediction'] if c['snr'] != 'all'}
chk('E5 gaps', (gaps[-10], gaps[-5], gaps[0], gaps[25]) == (0.162, 0.075, 0.044, 0.011))
chk('E5 single>joint all SNR (wrap)', all(c['single_wrap'] > c['joint_wrap'] for c in r['crb_prediction']))
print('   single-joint wrap per SNR', [round(c['single_wrap'] - c['joint_wrap'], 3) for c in r['crb_prediction'] if c['snr'] != 'all'])
chk('E5 paper-style same scenes', (round(r['paper_style_pd_same_scenes']['greedy'], 4), round(r['paper_style_pd_same_scenes']['greedy+selfcal'], 4)) == (0.7565, 0.7552))
cv = r['crb_validation']
chk('E5 crb validation', (round(cv['mc_ratio_median'], 3), [round(x, 3) for x in cv['mc_ratio_p10_p90']], round(cv['cond_median']), round(cv['cond_max'] / 1e6, 1), f"{cv['fd_rel_err']:.1e}") == (0.994, [0.897, 1.085], 579, 2.5, '3.2e-10'))
chk('E5 null scale', [round(x, 2) for x in r['null_scale_check']['T_quantiles_5_50_95_99']] == [1.45, 5.45, 12.8, 16.55])
st = r['stratification']
chk('E5 strat', (round(st['greedy']['no end-fire path'][1], 3), st['greedy']['no end-fire path'][0], round(st['greedy']['has an end-fire path (|cos|>0.995)'][1], 3), round(st['greedy+selfcal']['no end-fire path'][1], 3), round(st['greedy+selfcal']['has an end-fire path (|cos|>0.995)'][1], 3)) == (0.88, 648, 0.463, 0.873, 0.438))
w = r['weakest_rank']
chk('E5 rank', (round(w['weakest']['obs_greedy'], 3), round(w['middle']['obs_greedy'], 3), round(w['strongest']['obs_greedy'], 3), round(w['weakest']['obs_selfcal'], 3), round(w['middle']['obs_selfcal'], 3), round(w['strongest']['obs_selfcal'], 3)) == (0.644, 0.771, 0.849, 0.642, 0.772, 0.849))
for th in [1, 6]:
    j = json.load(open(f'{OUT}/e8_latency_threads{th}.json'))
    for nm in ['greedy (os=4, rc=3)', 'greedy + selfcal (1.0, 5.0)']:
        chk(f'E8 {th} {nm}', f"{j[nm]['median_ms']:.2f} ({j[nm]['p5_ms']:.2f}-{j[nm]['p95_ms']:.2f})" in md)
chk('E5 raw verbatim in md', t in md)
chk('E13 tables verbatim in md', open(f'{OUT}/e13_e6_tables.md', encoding='utf-8').read() in md)
print('FAILS:', bad)
