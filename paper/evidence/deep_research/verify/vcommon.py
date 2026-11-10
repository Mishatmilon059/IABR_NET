"""Paths and loaders for the verify/ scripts (read-only access to existing files)."""
import os
import numpy as np

ROOT = 'D:/ai_ml_project/IABR_v2_extracted/IABR_v2'
CACHE = f'{ROOT}/outputs/cache'
GEN = 'D:/ai_ml_project/IABR_Net_TestSuite/data/generated_banks'
FROZEN = 'D:/ai_ml_project/IABR_Net_TestSuite/data/frozen_banks/eval_bank.npz'
NEWR = 'D:/ai_ml_project/paper/evidence/new_results'
HERE = os.path.dirname(os.path.abspath(__file__))
MYCACHE = f'{HERE}/cache'
os.makedirs(MYCACHE, exist_ok=True)
BANKS = ['official', 'T2_d5', 'gain_g4_snr15', 'oodphase_d15_snr15', 'L8_snr0', 'L8_snr15', 'sep_5deg_snr15', 'sep_2deg_snr15',
         'phase_d0_snr0', 'phase_d0_snr15']


def load_truth(name):
    """psi, phi (N, Lmax), L (N,), snr (N,) -- same sources as the notebook's load_bank."""
    if name == 'official':
        b = np.load(FROZEN)
        return dict(psi=b['feat'][:, 0, :], phi=b['feat'][:, 1, :], L=b['meta'][:, 0].astype(int), snr=b['meta'][:, 1].astype(np.float32))
    p = f'{ROOT}/data/banks/{name}.npz'
    z = np.load(p if os.path.exists(p) else f'{GEN}/{name}.npz')
    return dict(psi=z['psi'], phi=z['phi'], L=z['L'].astype(int), snr=z['snr'].astype(np.float32))


def load_pred(path):
    z = np.load(path)
    return z['psi'], z['phi'], z['n']
