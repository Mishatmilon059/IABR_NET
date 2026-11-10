"""Pairs missing from s4_imp.log, computed from its cached per-scene stats (PCACHE/s4_stats_*.npz): does the fine-tuned trunk add anything
on top of the IABC-v2b corrector at M3 (45 deg, 0 dB), and learned-seed path vs the old dual. usage: THREADS=1 python s4b_m3pair.py > s4b_m3pair.log"""
import os
os.environ['THREADS'] = '1'
from advc import *
z = np.load(f'{PCACHE}/s4_stats_M3_ph45_g0_snr0.npz'); S = {k: z[k] for k in z.files if k != 'snr'}
pdm, ci_, D = boot(S, z['snr'], [('FT v2b heat+SC', 'v2b+NOMP+SC tuned'), ('FT dual+SC', 'v2b+NOMP+SC tuned'), ('FT heat+SC', 'v2b+NOMP+SC tuned')])
for k, d in D.items():
    print(f'M3_ph45_g0_snr0  {k:45s} {fmt(d)}')
