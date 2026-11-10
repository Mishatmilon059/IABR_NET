"""Runs the training stages sequentially after the running S1 process (pid argv[1]) exits: S2B 1000, S2A 2000 steps (S3 decided later).
Each stage: python -u train.py STAGE STEPS > train_STAGE.log 2> train_STAGE.err (resumable; re-running this script resumes)."""
import os, sys, time, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
pid = int(sys.argv[1]) if len(sys.argv) > 1 else 0
while pid:
    r = subprocess.run(['tasklist', '/FI', f'PID eq {pid}'], capture_output=True, text=True)
    if str(pid) not in r.stdout:
        break
    time.sleep(30)
for stage, steps in [('S2B', 1000), ('S2A', 2000)]:
    if os.path.exists(f'{HERE}/train_hist_{stage}.json'):
        continue
    with open(f'{HERE}/train_{stage}.log', 'a') as fo, open(f'{HERE}/train_{stage}.err', 'a') as fe:
        rc = subprocess.run([sys.executable, '-u', f'{HERE}/train.py', stage, str(steps)], stdout=fo, stderr=fe, cwd=HERE).returncode
    if rc != 0:
        print(f'{stage} failed rc={rc}', flush=True); sys.exit(rc)
    print(f'{stage} done', flush=True)
