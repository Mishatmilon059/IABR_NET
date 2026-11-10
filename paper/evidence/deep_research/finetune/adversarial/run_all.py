"""Detached sequential driver (one heavy process at a time): s3_adc_test -> s4_imp -> s5_official -> s6_latency.
Each stage writes <stage>.log; a stage with <stage>.done is skipped (the stages themselves resume from their partial outputs)."""
import os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
env = dict(os.environ, OMP_NUM_THREADS='6', OPENBLAS_NUM_THREADS='6', MKL_NUM_THREADS='6')
env.pop('SMOKE', None)
for st in sys.argv[1:] or ['s3_adc_test', 's4_imp', 's5_official', 's6_latency']:
    if os.path.exists(f'{HERE}/{st}.done'):
        print('skip', st, flush=True); continue
    print(time.strftime('%H:%M:%S'), 'run', st, flush=True)
    mode = 'a' if os.path.exists(f'{HERE}/{st}.log') else 'w'
    with open(f'{HERE}/{st}.log', mode) as f:
        r = subprocess.run([sys.executable, '-u', f'{HERE}/{st}.py'], cwd=HERE, env=env, stdout=f, stderr=subprocess.STDOUT)
    print(time.strftime('%H:%M:%S'), st, 'exit', r.returncode, flush=True)
    if r.returncode == 0:
        open(f'{HERE}/{st}.done', 'w').write('ok')
print('ALL DONE', flush=True)
