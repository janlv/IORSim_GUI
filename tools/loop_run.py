#!/usr/bin/env python3
#import sys
#sys.path.append('../IORSim_GUI')
from ior2ecl import runsim
from pathlib import Path

def mypath(dirs):
    return Path.home().joinpath(*(dirs.split()))

def loop_run(cases, times):
    iorexe = mypath('codes IORSim build bin IORSimX')
    cdir = mypath('github IORSim_GUI cases')
    n = 0
    for case,time in zip(cases, times):
        print(f'CASE {n} : {case}\n--------------------------')
        runsim(root=cdir/case/case, time=time, iorexe=iorexe, delete=False, verbose=3)
        n += 1

if __name__ == '__main__':
    cases = 10*['EKOFISK_HAUKAAS_2014_BACK', 'L18_OLD', 'KURS-07A', 'SNORRE-IORSIM-OLD', 'GEOCHEM_BACK', 'GEOCHEM_FWD']
    times = 10*[ 16375,                       1100 , 2000      , 3800               , 1000          , 1000]
    # cases = 10*['EKOFISK_HAUKAAS_2014_BACK']
    # times = 10*[16375]
    try:
        loop_run(cases, times)
    except KeyboardInterrupt:
        pass
