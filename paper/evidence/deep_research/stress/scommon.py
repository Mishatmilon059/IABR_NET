"""Shared runner for the stress tests: the six methods on one bank (all on the same Y), metrics, paired bootstrap.
Decoder = notebook_src.py slice [527:647] via new_results/common.py (CFG nomp_os=4, nomp_rc=3, SC=(1.0,5.0)); parabolic_heat_decode =
slice [649:668]; CNN = slice [419:522], IABRv2(32,10,32) + outputs/checkpoints/IABRv2/final.pt on CPU."""
import os, sys, json, time
os.environ.setdefault('IABR2_SUITE', 'D:/ai_ml_project')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, 'D:/ai_ml_project/paper/evidence/new_results')
import numpy as np, scipy, scipy.ndimage, torch, torch.nn as nn, torch.nn.functional as TF
import common as C
import stress_sim as SS

MYCACHE = f'{HERE}/cache'; os.makedirs(MYCACHE, exist_ok=True)
ROOT = C.ROOT
_src = open(f'{ROOT}/notebook_src.py', encoding='utf-8').read().split('\n')
_g = dict(np=np, scipy=scipy, uv_to_angles=C.uv_to_angles); exec('\n'.join(_src[649:668]), _g)
parabolic_heat_decode = _g['parabolic_heat_decode']
_gm = dict(np=np, torch=torch, nn=nn, TF=TF, W_IDEAL=SS.W_IDEAL, F_IDEAL=SS.F_IDEAL, P=16, Q=16); exec('\n'.join(_src[419:522]), _gm)
torch.set_num_threads(8)
cnn = _gm['IABRv2'](32, 10, 32); cnn.load_state_dict(torch.load(f'{ROOT}/outputs/checkpoints/IABRv2/final.pt', map_location='cpu')); cnn.eval()

METHODS = ['NOMP', 'NOMP+selfcal', 'IABR-Net heat', 'IABR-v2 dual', 'network only', 'oracle-seeded']
NN_METHODS = ['IABR-Net heat', 'IABR-v2 dual', 'network only']


def cnn_forward(Y, bs=250):
    heat, Hc = [], []
    X = np.ascontiguousarray(SS.S.to_ri(Y).transpose(0, 3, 1, 2))
    with torch.no_grad():
        for s in range(0, len(X), bs):
            o = cnn(torch.from_numpy(X[s:s + bs]))
            heat.append(torch.sigmoid(o['heat']).numpy().astype(np.float64)); Hc.append(o['Hc'].numpy().astype(np.complex128))
    return np.concatenate(heat), np.concatenate(Hc)


def run_methods(Y, psi, phi, L):
    """all scenes share the same L. Returns {method: (psi_hat (B,L'), phi_hat)} and the network outputs."""
    Hant = SS.to_ant(Y)
    heat, Hc = cnn_forward(Y)
    out = {}
    U, V = C.nomp_greedy(Hant, L, C.CFG['nomp_os'], C.CFG['nomp_rc'])
    out['NOMP'] = C.uv_to_angles(U, V)
    out['NOMP+selfcal'] = C.uv_to_angles(*C.selfcal(Hant, U, V, *C.SC))
    out['IABR-Net heat'] = C.decode(Hc, L, heat, 'heat', C.SC)
    out['IABR-v2 dual'] = C.decode(Hc, L, heat, 'dual', C.SC)
    out['network only'] = parabolic_heat_decode(heat, L)
    U0 = np.pi * np.cos(psi[:, :L]); V0 = np.pi * np.cos(phi[:, :L])
    out['oracle-seeded'] = C.uv_to_angles(*C.cyclic(Hant, U0, V0, C.CFG['nomp_rc']))
    return out, heat, Hc


def as_list(p):
    if isinstance(p, list):
        return p
    return [(p[0][i], p[1][i]) for i in range(len(p[0]))]


def stats(pred, psi, phi, L):
    b = dict(psi=psi, phi=phi, L=np.full(len(psi), L) if np.isscalar(L) else L)
    return C.scene_stats(as_list(pred), b)


def pack(pred, Lm):
    pl = as_list(pred); n = np.array([len(p[0]) for p in pl]); a = np.full((len(pl), Lm), np.nan); c = a.copy()
    for i, p in enumerate(pl):
        a[i, :n[i]] = p[0][:Lm]; c[i, :n[i]] = p[1][:Lm]
    return dict(psi=a, phi=c, n=n)
