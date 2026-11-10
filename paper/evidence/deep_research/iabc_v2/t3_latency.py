"""CPU latency per scene (6 threads, batch 400, L=3, 10 dB, cell 20 deg / 4 dB, seed 61030; timing only)."""
from v2common import *
from iabc2_model import load_v2, load_ref, predict, n_params
v2a = load_v2(f'{V2CACHE}/v2a_best.pt'); ref = load_ref(f'{V2CACHE}/v2b_best.pt')
print('params: v2a', n_params(v2a), ' refiner', n_params(ref), ' total v2b', n_params(v2a) + n_params(ref), ' old IABC', n_params(cnn.iabc))
d = make_scenes(61030, 400, 3, 10.0, 20, 4); Ht = to_ant(d['Y']); n = len(Ht)
def tm(f, rep=2):
    f(); t = time.time()
    for _ in range(rep): f()
    return 1000 * (time.time() - t) / rep / n
eA = predict(v2a, Ht)
t_a = tm(lambda: predict(v2a, Ht))
t_h = tm(lambda: hybrid_inputs(Ht, 3, eA), 1)
tok, glob, _ = hybrid_inputs(Ht, 3, eA)
t_r = tm(lambda: ref(torch.from_numpy(tok), torch.from_numpy(glob), torch.from_numpy(eA.astype(np.float32))))
t_n = tm(lambda: C.nomp_greedy(Ht, 3, 4, 3), 1)
Ug, Vg = C.nomp_greedy(Ht, 3, 4, 3)
t_s = tm(lambda: selfcal_ret(Ht, Ug, Vg, 1.0, 2.0, iters=4), 1)
t_c = tm(lambda: net(d['Y']))
t_t = tm(lambda: toeplitz_est(Ht))
print(f'ms/scene: v2a {t_a:.2f} | v2b stage-2 inputs (NOMP+ALS) {t_h:.2f} | refiner {t_r:.2f} | v2b total {t_a + t_h + t_r:.2f} | NOMP {t_n:.2f} | SC tuned (4 it) {t_s:.2f} | CNN+old IABC {t_c:.2f} | Toeplitz {t_t:.2f}')
print(f'pipelines ms/scene: NOMP {t_n:.1f}; NOMP+SC tuned {t_n + t_s:.1f}; v2a+NOMP {t_a + t_n:.1f}; v2b+NOMP {t_a + t_h + t_r + t_n:.1f}; v2b+NOMP+SC tuned {t_a + t_h + t_r + t_n + t_s:.1f}')
