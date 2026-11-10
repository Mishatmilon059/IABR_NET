"""Step 1: rerun the exp5 evaluation (learned_decoder/exp5.py, ev()) on its 2000-scene official subset (250 per SNR, exp.load_test)
and on the other three exp5 banks, with lurd.pt; compare with results5.json. Also checks that ndcommon.learned_decode (used at full scale)
gives the same predictions as exp.run_learned on that subset.
usage: python s1_reproduce.py > s1_reproduce.log"""
import json
from ndcommon import *

model = load_learned()
print('learned decoder params', sum(p.numel() for p in model.parameters()), 'lam', float(model.lam), 'torch threads', torch.get_num_threads(), flush=True)
ref = json.load(open(f'{LD}/results5.json'))['results']
banks = ['official', 'T2_d5', 'gain_g4_snr15', 'oodphase_d15_snr15']
tests = {n: E5.load_any(n) for n in banks}
out = {}
for use_sc, key in [(True, 'learned_with_sc'), (False, 'learned_no_sc')]:
    model.use_sc = use_sc
    for n, T in tests.items():
        U, V = E5.run_learned(model, T); m = E5.score(U, V, T)[1]; out.setdefault(key, {})[n] = m
        print(f'{key:16s} {n:20s} rerun {m:.4f}  results5.json {ref[key][n]:.4f}  diff {m - ref[key][n]:+.4f}', flush=True)
        if n == 'official':
            a, b = learned_decode(model, T['Hc'], T['heat'], 3, use_sc); a0, b0 = E5.uv_to_angles_np(U, V)
            print(f'   ndcommon.learned_decode vs exp.run_learned on the subset: max |d psi| {np.abs(a - a0).max():.2e} rad, max |d phi| {np.abs(b - b0).max():.2e} rad', flush=True)
json.dump(dict(rerun=out, results5=ref), open(f'{HERE}/s1_reproduce.json', 'w'), indent=1)
