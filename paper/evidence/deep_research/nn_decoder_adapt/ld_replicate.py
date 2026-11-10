"""Replicate on the ADC stage's FRESH banks (seeds 8300 / 8315, 1000 scenes, b = 4,3,2,1; generator = stress_sim.base_scenes + M5, as in
finetune/adc/acommon.m5_bank): FT-CNN + adapted learned decoder (cache/ld_adc_final_run.pt, no retraining) and FT-CNN + original learned
decoder (lurd.pt), paired against the ADC stage's per-scene stats on the same scenes (NOMP stats asserted identical).
Decoders on chunks of 400 scenes. Paired bootstrap 2000 within SNR groups, seed 0. Writes ld_replicate.json (stdout -> ld_replicate.log)."""
import os, sys, json
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[k] = '6'
os.environ['IABR2_SUITE'] = 'D:/ai_ml_project'
HERE = os.path.dirname(os.path.abspath(__file__))
DR = 'D:/ai_ml_project/paper/evidence/deep_research'; LD = 'D:/ai_ml_project/paper/evidence/learned_decoder'
for p in (LD, f'{DR}/stress', 'D:/ai_ml_project/paper/evidence/new_results'):
    sys.path.insert(0, p)
import numpy as np, torch
from exp5 import *
import common as C
import stress_sim as SS
torch.set_num_threads(6)
PC = 'C:/Users/Milon/AppData/Local/Temp/claude/D--thesis/a5db9a29-d75e-4561-834b-cd0638a16d91/scratchpad/dr_cache'


def load_cnn(path):
    m = gm['IABRv2'](32, 10, 32); s = torch.load(path, map_location='cpu')
    if isinstance(s, dict) and 'model' in s and isinstance(s['model'], dict):
        s = s['model']
    m.load_state_dict(s); return m.eval()


cnn_ft = load_cnn(f'{DR}/finetune/adc/cache/adc_ft.pt')
ld_orig = StagedSC(RefinerV3(), pre=6, rounds=3, inner=2); ld_orig.load_state_dict(torch.load(f'{LD}/lurd.pt', map_location='cpu'))
ld = StagedSC(RefinerV3(), pre=6, rounds=3, inner=2); ld.load_state_dict(torch.load(f'{HERE}/cache/ld_adc_final_run.pt', map_location='cpu'))
for m in (ld, ld_orig):
    m.eval(); m.use_sc = True


def forward(model, Y, bs=250):
    X = np.ascontiguousarray(sim.to_ri(Y).transpose(0, 3, 1, 2)); heat, Hc = [], []
    with torch.no_grad():
        for s in range(0, len(X), bs):
            o = model(torch.from_numpy(X[s:s + bs]))
            heat.append(torch.sigmoid(o['heat']).numpy().astype(np.float64)); Hc.append(o['Hc'].numpy().astype(np.complex128))
    return np.concatenate(heat), np.concatenate(Hc)


def run_ld(model, heat, Hc):
    U0, V0 = heat_peaks_np(heat, 3)
    with torch.no_grad():
        U, V = model(torch.from_numpy(Hc.astype(np.complex64)), torch.tensor(U0, dtype=torch.float32), torch.tensor(V0, dtype=torch.float32))
    p = uv_to_angles_np(U.numpy().astype(np.float64), V.numpy().astype(np.float64))
    return [(p[0][i], p[1][i]) for i in range(len(p[0]))]


CLASSICAL = ['NOMP', 'NOMP+SC', 'NOMP+SC tuned', 'FFT-peak seeds', 'FFT-peak seeds+SC', 'classical dual', 'classical dual+SC', 'NOMP os8 rc6',
             'multi-start K6', 'consistent NOMP', 'FFT parabolic os4', 'FFT parabolic os16', 'consistent parabolic']
out = {}
for snr, seed in ((15, 8315), (0, 8300)):
    base = SS.base_scenes(1000, 3, float(snr), seed)
    for bits in (4, 3, 2, 1):
        name = f'b{bits}_snr{snr}'; Y = SS.generate(base, 'M5', bits)
        P = {'NOMP': [], 'FT-CNN + adapted learned decoder': [], 'FT-CNN + original learned decoder': []}
        for s0 in range(0, len(Y), 400):
            y = Y[s0:s0 + 400]; hf, Hf = forward(cnn_ft, y)
            U, V = C.nomp_greedy(SS.to_ant(y), 3, 4, 3); p = C.uv_to_angles(U, V)
            P['NOMP'] += [(p[0][i], p[1][i]) for i in range(len(p[0]))]
            for i0 in range(0, len(y), 200):
                P['FT-CNN + adapted learned decoder'] += run_ld(ld, hf[i0:i0 + 200], Hf[i0:i0 + 200])
                P['FT-CNN + original learned decoder'] += run_ld(ld_orig, hf[i0:i0 + 200], Hf[i0:i0 + 200])
        b = dict(psi=base['psi'][:, :3], phi=base['phi'][:, :3], L=np.full(1000, 3))
        S = {k: C.scene_stats(v, b) for k, v in P.items()}
        za = np.load(f'{PC}/finetune_adc/stats2_fresh__{name}.npz'); snr_v = base['snr']
        assert np.array_equal(S['NOMP'], za['NOMP']), name
        for k in za.files:
            if k not in ('snr', 'NOMP'):
                S['ADC:' + k] = za[k]
        pdm = {k: C.metric_by_snr(v, snr_v)['mean']['pd_paper'] for k, v in S.items()}
        best = max([k for k in CLASSICAL if k == 'NOMP' or 'ADC:' + k in S], key=lambda k: pdm[k if k == 'NOMP' else 'ADC:' + k])
        bk = best if best == 'NOMP' else 'ADC:' + best
        Bt = C.boot_metrics(S, snr_v, B=2000, seed=0)
        rec = dict(n=1000, best_classical=best, pd={k.replace('ADC:', ''): v for k, v in pdm.items()}, diffs={})
        print(f'=== fresh {name}: NOMP identical True; best classical {best} {pdm[bk]:.4f}; FT network only {pdm["ADC:FT network only"]:.4f}')
        for a in ('FT-CNN + adapted learned decoder', 'FT-CNN + original learned decoder'):
            line = f'   {a:36s} Pd {pdm[a]:.4f}'
            for r, rk in (('NOMP', 'NOMP'), ('BEST classical', bk), ('FT network only', 'ADC:FT network only'), ('FT dual+SC', 'ADC:FT dual+SC'),
                          ('FT-CNN + original learned decoder', 'FT-CNN + original learned decoder')):
                if rk == a:
                    continue
                d = Bt[a][:, 0] - Bt[rk][:, 0]; v = [pdm[a] - pdm[rk], float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))]
                rec['diffs'][f'{a} - {r}'] = v
                line += f' | vs {r} {v[0]:+.4f} [{v[1]:+.4f},{v[2]:+.4f}]{"*" if v[1] > 0 or v[2] < 0 else ""}'
            print(line, flush=True)
        out[name] = rec
        json.dump(out, open(f'{HERE}/ld_replicate.json', 'w'), indent=1, default=float)
print('done')
