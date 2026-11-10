"""Sequential evaluation chain (one heavy process at a time): stop-threshold tuning (validation), classical baselines on every test bank,
designs B and A, ablations of A, latency, analysis. Each step is resumable (ev.py caches per bank / method). Logs: ev_<step>.log."""
import os, sys, subprocess, shutil
HERE = os.path.dirname(os.path.abspath(__file__))
STEPS = [('stop_tune', ['stop_tune.py'], {}),
         ('official', ['ev.py', 'official'], {}), ('regime', ['ev.py', 'regime'], {}), ('unknownL', ['ev.py', 'unknownL'], {}),
         ('prior', ['ev.py', 'prior'], {}), ('ood', ['ev.py', 'ood'], {}),
         ('models_official', ['ev.py', 'official+models', 'regime+models'], dict(MODELS='B,A')),
         ('models_rest', ['ev.py', 'unknownL+models', 'prior+models', 'ood+models'], dict(MODELS='B,A')),
         ('ablate', ['ev.py', 'ablate+models'], dict(MODELS='A abl:kappa1,A abl:eta0,A abl:truesig,A abl:stepoff,A abl:selnll,A abl:beta0')),
         ('latency', ['latency.py'], dict(THREADS='1')),
         ('analyze', ['analyze.py'], {})]
only = sys.argv[1:]
for name, args, env in STEPS:
    if only and name not in only:
        continue
    if os.path.exists(f'{HERE}/ev_{name}.done'):
        continue
    e = dict(os.environ); e.update(env)
    with open(f'{HERE}/ev_{name}.log', 'a') as fo, open(f'{HERE}/ev_{name}.err', 'a') as fe:
        rc = subprocess.run([sys.executable, '-u', f'{HERE}/{args[0]}'] + args[1:], stdout=fo, stderr=fe, cwd=HERE, env=e).returncode
    if rc != 0:
        print(f'{name} failed rc={rc}', flush=True); sys.exit(rc)
    open(f'{HERE}/ev_{name}.done', 'w').write('ok')
    print(f'{name} done', flush=True)
