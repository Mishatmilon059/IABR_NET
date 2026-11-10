"""Sequential job queue (one heavy process at a time, launched detached). Each job appends to its own log; every job is resumable
(m4_physics.py skips banks already in m4_physics.json, evaluate.py skips banks with parts/<bank>.json)."""
import subprocess, sys, os, time
HERE = os.path.dirname(os.path.abspath(__file__))
if sys.argv[1].startswith('waitpid:'):                 # optional: wait until an earlier queue (pid) has exited -> never two heavy jobs at once
    import ctypes
    pid = int(sys.argv.pop(1).split(':')[1]); k = ctypes.windll.kernel32
    while True:
        h = k.OpenProcess(0x1000 | 0x00100000, False, pid)   # QUERY_LIMITED_INFORMATION | SYNCHRONIZE
        if not h:
            break
        r = k.WaitForSingleObject(h, 30000); k.CloseHandle(h)
        if r == 0:
            break
    print(f'{time.ctime()} pid {pid} has exited, starting', flush=True)
JOBS = [a.split('=', 1) for a in sys.argv[1:]]     # args like  m4_physics.log=m4_physics.py  eval_stress.log=evaluate.py,clean,stress
for log, cmd in JOBS:
    parts = cmd.split(',')
    with open(f'{HERE}/{log}', 'a') as f:
        f.write(f'\n##### queue start {time.ctime()} : {parts}\n'); f.flush()
        r = subprocess.run([sys.executable, '-u', f'{HERE}/{parts[0]}'] + parts[1:], stdout=f, stderr=subprocess.STDOUT, cwd=HERE)
        f.write(f'##### queue end {time.ctime()} rc={r.returncode}\n')
    print(f'{time.ctime()} {log} rc={r.returncode}', flush=True)
print('queue done', flush=True)
