import os as _os
from gurobi import GRB
import colorama
import re as _re
from warnings import warn
from pathlib import Path
TTYCOLORS = colorama.Fore
EPS = 1e-5
DATA_DIR = (Path(__file__).parent/'../../../data').resolve()
LOGS_DIR = (Path(__file__).parent/'../../../logs').resolve()
ITSRSP_TIME_INFINITY = (2 ** 32) - 1

def _parse_slurm_time(string):
    match = _re.match(r'((?P<days>\d+)-)?(?P<hours>\d+):(?P<mins>\d+):(?P<secs>\d+)', string)
    if match is None:
        warn(f'Environment variable SLURM_TIME_LIMIT is invalid (`{string}`); ignoring')
        return GRB.INFINITY
    match = match.groupdict(0)
    match = dict(zip(match.keys(), map(int, match.values())))
    return ((int(match['days'])*24 + match['hours'])*60 + match['mins'])*60 + match['secs']

if 'SLURM_CPUS_PER_TASK' in _os.environ:
    N_CPUS = int(_os.environ['SLURM_CPUS_PER_TASK'])
else:
    import multiprocessing as _mp
    N_CPUS = _mp.cpu_count()
