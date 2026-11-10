"""Step 2: predictions of the ALL-NEURAL pipeline on every bank with a net__ cache and truth (s0_inventory.json):
cached CNN outputs (heat = sigmoid heatmap, Hc) -> top-L heatmap local maxima (notebook heat_peaks) -> learned decoder StagedSC(RefinerV3, pre=6,
rounds=3, inner=2) from lurd.pt, with use_sc=True ('allneural_sc') and use_sc=False ('allneural_nosc'). No FFT search, no classical Newton step.
Also the classical heat-seeded decoder (heat_main.py recipe) for banks without a cached new_results/preds_heat or verify/cache/heat__ file
('heatcalc'); for one bank that has preds_heat (L8_snr0) it is recomputed as a recipe check.
Deterministic (no randomness). Resumable: each (bank, method) is written to DRC/<bank>__<method>.npz and skipped if present.
Chunks: learned decoder 250 scenes, classical decoder 1000 -> reduced to 400 here (memory rule). Threads 6.
usage: python -u s2_predict.py > s2_predict.log"""
import os, sys, json, time
os.environ['NDTHREADS'] = '6'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ndcommon import *
DRC = 'C:/Users/Milon/AppData/Local/Temp/claude/D--thesis/a5db9a29-d75e-4561-834b-cd0638a16d91/scratchpad/dr_cache/nn_decoder'
os.makedirs(DRC, exist_ok=True)
inv = json.load(open(f'{HERE}/s0_inventory.json'))
banks = [n for n, v in inv.items() if v is not None]
order = ['official', 'T2_d1', 'T2_d2', 'T2_d5'] + sorted(n for n in banks if n not in ('official', 'T2_d1', 'T2_d2', 'T2_d5'))
model = load_learned()
print('learned decoder params', sum(p.numel() for p in model.parameters()), 'torch threads', torch.get_num_threads(), flush=True)


def heat_classical(Hc, heat, Ls):
    def fn(ii, L):
        oa, ob = [], []
        for s in range(0, len(ii), 400):
            jj = ii[s:s + 400]
            a, b = C.decode(Hc[jj].astype(np.complex128), L, heat[jj].astype(np.float64), 'heat', C.SC); oa.append(a); ob.append(b)
        return np.concatenate(oa), np.concatenate(ob)
    return run_by_L(fn, Ls)


for n in order:
    b = C.load_bank(n); Ls = b['L']; z = np.load(f'{CACHE}/net__{n}.npz'); Hc, heat = z['Hc'], z['heat']; assert len(Hc) == len(Ls)
    for meth, use_sc in [('allneural_sc', True), ('allneural_nosc', False)]:
        f = f'{DRC}/{n}__{meth}.npz'
        if os.path.exists(f):
            continue
        t0 = time.time()
        p = run_by_L(lambda ii, L: learned_decode(model, Hc[ii], heat[ii], L, use_sc, bs=250), Ls)
        np.savez_compressed(f, **p); print(f'{n:22s} {meth:15s} N={len(Ls)} {time.time()-t0:.1f}s', flush=True)
    has_heat = os.path.exists(f'{NEWR}/preds_heat/{n}.npz') or os.path.exists(f'{VER}/cache/heat__{n}.npz')
    f = f'{DRC}/{n}__heatcalc.npz'
    if (not has_heat or n == 'L8_snr0') and not os.path.exists(f):
        t0 = time.time(); p = heat_classical(Hc, heat, Ls); np.savez_compressed(f, **p)
        msg = ''
        if n == 'L8_snr0':
            q = load_predfile(f'{NEWR}/preds_heat/{n}.npz'); msg = f' recipe check vs preds_heat: max |d| {max(np.nanmax(np.abs(p["psi"]-q["psi"])), np.nanmax(np.abs(p["phi"]-q["phi"]))):.2e} rad'
        print(f'{n:22s} heatcalc        N={len(Ls)} {time.time()-t0:.1f}s{msg}', flush=True)
    del Hc, heat, z
print('done', flush=True)
