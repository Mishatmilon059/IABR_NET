"""Keeps the PC from idle-sleeping while deep-research jobs run (Windows SetThreadExecutionState; released when this process exits).
Exits on its own after no python process whose command line or working directory mentions 'deep_research' has been seen for 30 minutes, or after 24 hours.
Does not change any power setting. Usage: python keepawake.py"""
import ctypes, time, os, psutil
ES = 0x80000000 | 0x00000001        # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'keepawake.log')
me = os.getpid(); t0 = last = time.time()
ctypes.windll.kernel32.SetThreadExecutionState(ES)
open(LOG, 'a').write(time.strftime('%Y-%m-%d %H:%M:%S') + ' start\n')
while time.time() - t0 < 24 * 3600 and time.time() - last < 1800:
    for p in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if p.info['pid'] == me or 'python' not in (p.info['name'] or '').lower():
                continue
            if any('deep_research' in c for c in (p.info['cmdline'] or [])) or 'deep_research' in p.cwd().replace(chr(92), '/'):
                last = time.time(); break
        except Exception:
            pass
    ctypes.windll.kernel32.SetThreadExecutionState(ES)
    time.sleep(60)
open(LOG, 'a').write(time.strftime('%Y-%m-%d %H:%M:%S') + ' exit\n')
