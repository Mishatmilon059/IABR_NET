"""Bin counts of the end-fire histogram (Fig. VII-5): absolute angle errors of the 12,000 components of the 2,000 scenes of e5_perscene.npz, end-fire = |cos| > 0.995 for the true angle (as in e5_persnr_taxonomy.py). Written by the fix agent; reproduces 819 end-fire components, 387 fold errors, 222 on end-fire."""
import sys; sys.path.insert(0, 'D:/ai_ml_project/paper/evidence/new_results')
from common import *
d = np.load('D:/ai_ml_project/paper/evidence/new_results/e5_perscene.npz')
b = load_bank('official'); idx = d['idx']
PT, FT = b['psi'][idx][:, :3].astype(float), b['phi'][idx][:, :3].astype(float)
ef = np.concatenate([np.abs(np.cos(PT)) > 0.995, np.abs(np.cos(FT)) > 0.995], 1)
err = np.abs(d['errA'])
print('endfire comps', ef.sum(), 'of', ef.size, 'fold(>90)', (err>90).sum(), 'fold on endfire', ((err>90)&ef).sum())
print('okA frac endfire', d['okA'][ef].mean(), 'others', d['okA'][~ef].mean())
edges=[0,1,2,5,10,20,45,90,180.001]
for name,mask in [('end-fire',ef),('other',~ef)]:
    h,_=np.histogram(err[mask],bins=edges)
    print(name, mask.sum(), list(h))
