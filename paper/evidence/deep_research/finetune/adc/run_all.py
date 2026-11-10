"""Detached driver: runs evaluate.py for m5, clean, fresh, official, then latency.py, sequentially (one heavy process at a time).
Skips a stage whose output json already exists (resume after interruption). Each stage's stdout -> its own log."""
import os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
env = dict(os.environ, OMP_NUM_THREADS='6', OPENBLAS_NUM_THREADS='6', MKL_NUM_THREADS='6')
for st in ['m5', 'clean', 'fresh', 'official']:
    if os.path.exists(f'{HERE}/res_{st}.json') and os.path.exists(f'{HERE}/eval_{st}.done'):
        print('skip', st, flush=True); continue
    print('run', st, flush=True)
    with open(f'{HERE}/eval_{st}.log', 'w') as f:
        r = subprocess.run([sys.executable, '-u', f'{HERE}/evaluate.py', st], cwd=HERE, env=env, stdout=f, stderr=subprocess.STDOUT)
    print(st, 'exit', r.returncode, flush=True)
    if r.returncode == 0:
        open(f'{HERE}/eval_{st}.done', 'w').write('ok')
if not os.path.exists(f'{HERE}/latency.json'):
    with open(f'{HERE}/latency.log', 'w') as f:
        r = subprocess.run([sys.executable, '-u', f'{HERE}/latency.py'], cwd=HERE, stdout=f, stderr=subprocess.STDOUT)
    print('latency exit', r.returncode, flush=True)
print('ALL DONE', flush=True)
