"""Shared helpers for nn_decoder/ (all-neural IABR-Net = CNN heatmap peaks -> learned decoder StagedSC from lurd.pt).
Read-only use of existing files. Thread limits are set before numpy/torch import (default 2; NDTHREADS env overrides)."""
import os, sys
TH = os.environ.get('NDTHREADS', '2')
for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[k] = TH
LD = 'D:/ai_ml_project/paper/evidence/learned_decoder'
NEWR = 'D:/ai_ml_project/paper/evidence/new_results'
VER = 'D:/ai_ml_project/paper/evidence/deep_research/verify'
HERE = os.path.dirname(os.path.abspath(__file__))
MYCACHE = f'{HERE}/cache'
os.makedirs(MYCACHE, exist_ok=True)
sys.path.insert(0, LD)
import numpy as np, torch
import exp5 as E5          # imports exp3/exp (loads the CNN, sets torch threads 6 -> reset below)
torch.set_num_threads(int(TH))
sys.path.insert(0, NEWR)
import common as C          # load_bank, scene_stats, metric_by_snr, boot_metrics, ci, decode, heat_peaks, uv_to_angles, CACHE

CACHE = C.CACHE
DEV = E5.DEV


def load_learned():
    torch.manual_seed(0)
    m = E5.StagedSC(E5.RefinerV3(), pre=6, rounds=3, inner=2).to(DEV)
    m.load_state_dict(torch.load(f'{LD}/lurd.pt', map_location=DEV)); m.eval()
    return m


def learned_decode(model, Hc, heat, L, use_sc, bs=500):
    """Hc (B,16,16) complex, heat (B,32,32) sigmoid heatmap, all with the same L -> (psi, phi) radians (B,L).
    Seeds = top-L local maxima of the heatmap (notebook heat_peaks); then the learned decoder only."""
    model.use_sc = use_sc
    U0, V0 = C.g['heat_peaks'](heat.astype(np.float64), L)
    Us, Vs = [], []
    with torch.no_grad():
        for i in range(0, len(Hc), bs):
            H = torch.from_numpy(np.ascontiguousarray(Hc[i:i + bs]).astype(np.complex64)).to(DEV)
            u = torch.tensor(U0[i:i + bs], dtype=torch.float32, device=DEV); v = torch.tensor(V0[i:i + bs], dtype=torch.float32, device=DEV)
            U, V = model(H, u, v); Us.append(U.cpu().numpy()); Vs.append(V.cpu().numpy())
    return C.uv_to_angles(np.concatenate(Us).astype(np.float64), np.concatenate(Vs).astype(np.float64))


def run_by_L(fn, Ls):
    """fn(idx, L) -> (psi, phi) (len(idx), L); returns preds dict(psi, phi, n) like preds_heat."""
    psi = np.full((len(Ls), int(Ls.max())), np.nan); phi = psi.copy()
    for L in sorted(set(Ls.tolist())):
        ii = np.where(Ls == L)[0]
        a, b = fn(ii, L); psi[ii, :L], phi[ii, :L] = a, b
    return dict(psi=psi, phi=phi, n=np.array(Ls))


def classical_heat(Hc, heat, Ls):
    """heat_main.py recipe: decode(Hc, L, heat, 'heat', SC=(1.0, 5.0)) in chunks of 1000."""
    def fn(ii, L):
        out_a, out_b = [], []
        for s in range(0, len(ii), 1000):
            jj = ii[s:s + 1000]
            a, b = C.decode(Hc[jj].astype(np.complex128), L, heat[jj].astype(np.float64), 'heat', C.SC); out_a.append(a); out_b.append(b)
        return np.concatenate(out_a), np.concatenate(out_b)
    return run_by_L(fn, Ls)


def stats_of(p, b):
    return C.scene_stats([(p['psi'][i, :p['n'][i]], p['phi'][i, :p['n'][i]]) for i in range(len(p['n']))], b)


def load_predfile(path):
    z = np.load(path); return dict(psi=z['psi'], phi=z['phi'], n=z['n'])
