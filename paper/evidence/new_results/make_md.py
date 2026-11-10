"""Aggregates bank_json/*.json into the E13/E6 part of RESULTS_NEW.md (written to e13_e6_tables.md)."""
import json, os, glob, sys
OUT = 'D:/ai_ml_project/paper/evidence/new_results'
order = ['official', 'T2_d1', 'T2_d2', 'T2_d5'] + [f'phase_d{d}_snr{s}' for s in [0, 15] for d in [0, 1, 2, 5]] + [f'gain_g{g}_snr{s}' for s in [0, 15] for g in [0.5, 1, 2, 4]] + \
        [f'oodphase_d{d}_snr{s}' for s in [0, 15] for d in [10, 15]] + [f'L{L}_snr{s}' for s in [0, 15] for L in [1, 2, 4, 5, 6, 7, 8, 10]] + \
        [f'sep_{d}deg_snr15' for d in [30, 20, 10, 5, 3, 2, 1]] + [f'tail_snr{s}' for s in [-25, -20, 30, 35]]
R = {n: json.load(open(f'{OUT}/bank_json/{n}.json')) for n in order if os.path.exists(f'{OUT}/bank_json/{n}.json')}
M = ['U-Net', 'hybrid', 'NOMP', 'NOMP+selfcal']; D = ['NOMP - U-Net', 'hybrid - NOMP', 'NOMP+selfcal - hybrid', 'NOMP+selfcal - NOMP']


def label(n, r):
    snr = r['snr_groups']; s = 'SNR -10..25 step 5 (mean of 8)' if len(snr) > 1 else f'SNR {snr[0]} dB'
    return f'{n} | {r["n"]} | {",".join(map(str, r["L"]))} | {s}'


def cell(r, m, k, d=4):
    v = r['methods'][m]['mean'][k]; lo, hi = r['methods'][m]['ci'][k]
    return f'{v:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]'


def dcell(r, nm, k, d=4):
    v, lo, hi = r['diffs'][nm][k]; star = '*' if (lo > 0 or hi < 0) else ''
    return f'{v:+.{d}f} [{lo:+.{d}f}, {hi:+.{d}f}]{star}'


out = []
for key, title, d in [('pd_paper', 'Paper-style Pd (samples with fewer than L detections are dropped; mean over SNR groups)', 4), ('pd_strict', 'Strict Pd (dropped samples count as misses)', 4),
                      ('pd_source', 'pd_source (both components of a path within 1 degree; dropped samples count as misses)', 4), ('rmse', 'Conditional RMSE in degrees (over components within 1 degree; conditional on detection, not comparable across methods with different Pd)', 4)]:
    out.append(f'\n### Table: {title}\n\nBracket = 95 percent bootstrap percentile interval of the metric itself (2000 resamples of scenes within each SNR group).\n')
    out.append('| bank | n | L | SNR | ' + ' | '.join(M) + ' |\n|---|---|---|---|' + '---|' * len(M))
    for n, r in R.items():
        out.append(f'| {n} | {r["n"]} | {",".join(map(str, r["L"]))} | ' + ('8 levels' if len(r['snr_groups']) > 1 else f'{r["snr_groups"][0]} dB') + ' | ' + ' | '.join(cell(r, m, key, d) for m in M) + ' |')
for key, title in [('pd_paper', 'paper-style Pd'), ('pd_strict', 'strict Pd'), ('pd_source', 'pd_source'), ('rmse', 'conditional RMSE (degrees)')]:
    out.append(f'\n### Table: paired differences, {title}\n\nEntry = point difference [95 percent paired-bootstrap interval]; `*` = interval excludes 0. Scenes resampled within each SNR group, same indices for all methods.\n')
    out.append('| bank | ' + ' | '.join(D) + ' |\n|---|' + '---|' * len(D))
    for n, r in R.items():
        out.append(f'| {n} | ' + ' | '.join(dcell(r, nm, key) for nm in D) + ' |')
out.append('\n### Table: per-SNR paper-style Pd (point values) for the 8-SNR banks\n')
for n, r in R.items():
    if len(r['snr_groups']) > 1:
        out.append(f'\n**{n}** (1000 scenes per SNR)\n\n| SNR | ' + ' | '.join(M) + ' |\n|---|' + '---|' * len(M))
        for s in r['snr_groups']:
            out.append(f'| {s} | ' + ' | '.join(f'{r["methods"][m]["per_snr"][str(s)]["pd_paper"]:.4f}' for m in M) + ' |')
        out.append('| dropped scenes (all SNR) | ' + ' | '.join(str(r['methods'][m]['dropped']) for m in M) + ' |')
dr = [(n, r['methods']['U-Net']['dropped']) for n, r in R.items() if r['methods']['U-Net']['dropped']]
out.append('\nU-Net dropped scenes (fewer than L detections): ' + (', '.join(f'{n}: {k}' for n, k in dr) if dr else 'none') + '\n')
open(f'{OUT}/e13_e6_tables.md', 'w', encoding='utf-8').write('\n'.join(out))
print(len(R), 'banks written')
