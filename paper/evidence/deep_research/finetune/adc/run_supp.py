"""Detached driver for supp.py over m5, clean, fresh, official (sequential, resumable via supp_<set>.done)."""
import os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
env = dict(os.environ, OMP_NUM_THREADS='6', OPENBLAS_NUM_THREADS='6', MKL_NUM_THREADS='6')
for st in ['m5', 'clean', 'fresh', 'official']:
    if os.path.exists(f'{HERE}/supp_{st}.done'):
        print('skip', st, flush=True); continue
    with open(f'{HERE}/supp_{st}.log', 'w') as f:
        r = subprocess.run([sys.executable, '-u', f'{HERE}/supp.py', st], cwd=HERE, env=env, stdout=f, stderr=subprocess.STDOUT)
    print(st, 'exit', r.returncode, flush=True)
    if r.returncode == 0:
        open(f'{HERE}/supp_{st}.done', 'w').write('ok')
print('ALL DONE', flush=True)
