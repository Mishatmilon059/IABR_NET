"""Round-1 revision of Section VII-F: per-SNR objective-vs-truth taxonomy of the GREEDY network-free decoder, recomputed from e5_perscene.npz
(UA, VA, okA, snr) plus the bank/channel used by e5_diag.py. Same definitions as e5_diag.py section 2 (RA, RT = residual_energy; fail = not all 6 comps ok;
corrected scale T = 2(RA-RT)/(RT/253), chi2(6) 99% band)."""
import sys; sys.path.insert(0, 'D:/ai_ml_project/paper/evidence/new_results')
from common import *
from scipy.stats import chi2
d = np.load('D:/ai_ml_project/paper/evidence/new_results/e5_perscene.npz')
b = load_bank('official'); n = np.load(f'{CACHE}/net__official.npz'); snr_all = b['snr']
idx = d['idx']
Ht = n['Ht'][idx].astype(np.complex128); PT, FT = b['psi'][idx][:, :3].astype(np.float64), b['phi'][idx][:, :3].astype(np.float64)
UT, VT = np.pi*np.cos(PT), np.pi*np.cos(FT); snr = snr_all[idx]; assert np.allclose(snr, d['snr'])
RA = residual_energy(Ht, d['UA'], d['VA']); RT = residual_energy(Ht, UT, VT)
fail = ~d['okA'].all(1); chi99 = float(chi2.ppf(0.99, 6))
dE = (RA-RT)/(RT/247); T = 2*(RA-RT)/(RT/253)
better = fail & (RA <= RT); band = fail & (T > 0) & (T <= chi99); miss = fail & (T > chi99)
print('greedy, pooled: failing %d, energy<=truth %d, band %d, miss(corrected) %d' % (fail.sum(), better.sum(), band.sum(), miss.sum()))
print('SNR  n  failing  fail%  energy<=truth(n, %%of failing)  band  miss(corrected)  miss%%ofall')
for s in sorted(set(snr.tolist())):
    m = snr == s; f = (fail & m).sum()
    print('%4d %4d %5d %6.1f   %4d %6.1f   %3d  %3d  %5.1f' % (s, m.sum(), f, 100*f/m.sum(), (better&m).sum(), 100*(better&m).sum()/max(f,1), (band&m).sum(), (miss&m).sum(), 100*(miss&m).sum()/m.sum()))
m = snr >= 15; f = (fail & m).sum()
print('SNR>=15 pooled: failing %d, energy<=truth %d (%.1f%%), band %d, miss %d' % (f, (better&m).sum(), 100*(better&m).sum()/f, (band&m).sum(), (miss&m).sum()))
m = snr >= 10; f = (fail & m).sum()
print('SNR>=10 pooled: failing %d, energy<=truth %d (%.1f%%), band %d, miss %d' % (f, (better&m).sum(), 100*(better&m).sum()/f, (band&m).sum(), (miss&m).sum()))
# end-fire (|cos| > 0.995 for any true psi or phi of the scene), as in e5_diag.py stratification
endf = np.concatenate([np.abs(np.cos(PT)) > 0.995, np.abs(np.cos(FT)) > 0.995], 1).any(1)
print('\nend-fire share among failing scenes, and among failing scenes with energy<=truth (objective-limited):')
for lo in (-10, 10, 15, 20, 25):
    m = (snr >= lo) if lo in (10, 15) else (snr == lo)
    if lo == -10: m = snr >= -10
    f = fail & m; o = better & m
    print('SNR %s%d: scenes with end-fire path %d of %d (%.1f%%); failing %d, of which end-fire %d (%.1f%%); objective-limited %d, of which end-fire %d (%.1f%%)' % (
        '>=' if lo in (-10, 10, 15) else '=', lo, (endf&m).sum(), m.sum(), 100*(endf&m).sum()/m.sum(), f.sum(), (f&endf).sum(), 100*(f&endf).sum()/f.sum(), o.sum(), (o&endf).sum(), 100*(o&endf).sum()/max(o.sum(),1)))
