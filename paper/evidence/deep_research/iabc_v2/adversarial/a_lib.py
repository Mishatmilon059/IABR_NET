"""Shared helpers for the adversarial re-test (iabc_v2/adversarial). Classical comparators (no learning):
  sc_iter     : NOMP + tuned self-cal with more ALS iterations (lam 1, tau 2, `iters`)
  sc_redetect : `rounds` x {NOMP on the currently corrected channel -> tuned self-cal (lam 1, tau 2, 4 it)}; corrections accumulate.
                Same structure as the v2b pipeline (correct -> re-detect -> self-cal) with the networks removed.
  eb_ml       : a_classical.py (joint ML by Levenberg-Marquardt + positive-part James-Stein shrinkage, no tuned constant).
All constants are the previously tuned SC constants (stress/adversarial/replicate.py) or none; nothing is tuned on 9300-9307."""
import os, sys
for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[k] = '6'
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, 'D:/ai_ml_project/paper/evidence/deep_research/iabc_v2'); sys.path.insert(0, HERE)
from v2common import *            # noqa
from iabc2_model import load_v2, load_ref, predict
import a_classical as AC

v2a = load_v2(f'{V2CACHE}/v2a_best.pt'); ref = load_ref(f'{V2CACHE}/v2b_best.pt')


def v2b_est(Ht, L):
    eA = predict(v2a, Ht); tok, glob, _ = hybrid_inputs(Ht, L, eA)
    with torch.no_grad():
        return ref(torch.from_numpy(tok), torch.from_numpy(glob), torch.from_numpy(eA.astype(np.float32))).double().numpy()


def sc_iter(Ht, L, iters, tau=2.0, UV=None):
    U, V = UV if UV is not None else C.nomp_greedy(Ht, L, 4, 3)
    U, V, e, _, _ = selfcal_ret(Ht, U, V, 1.0, tau, iters=iters)
    return e, (U, V)


def sc_redetect(Ht, L, rounds=2, tau=2.0, iters=4):
    e = np.zeros((len(Ht), 32, 2))
    for r in range(rounds):
        H1 = apply_corr(Ht, e); U, V = C.nomp_greedy(H1, L, 4, 3)
        U, V, de, _, _ = selfcal_ret(H1, U, V, 1.0, tau, iters=iters)
        e = ident_vec(e[..., 0] + de[..., 0], e[..., 1] + de[..., 1])
    return e, (U, V)


def eb_ml(Ht, L):
    """returns shrunk estimate, NOMP angles on the corrected channel, unshrunk ML estimate, blind SNR estimate (dB per entry)."""
    e, (U, V), e_ml, s2 = AC.eb_ml(Ht, L, rounds=2, shrink=True, lm_iters=10, return_s2=True)
    p = (np.abs(Ht) ** 2).mean((1, 2)); snr_hat = 10 * np.log10(np.maximum(p - s2, 1e-6 * p) / s2)
    return e, (U, V), e_ml, snr_hat


def gate(e, UV, UV0, snr_hat, g_db):
    """EB-ML applied only where the BLIND SNR estimate >= g_db (else no correction, NOMP angles on raw Ht).
    Gates fixed after the prototype (a0_proto.log: EB-ML harmful on clean scenes at <= 0 dB), before any test run: 5 dB and
    10 dB (10 dB = the nominal-SNR threshold of the earlier 7700-seed 'SC snrgate' study)."""
    ok = snr_hat >= g_db
    return np.where(ok[:, None, None], e, 0.0), (np.where(ok[:, None], UV[0], UV0[0]), np.where(ok[:, None], UV[1], UV0[1]))
