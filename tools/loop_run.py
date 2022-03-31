#!/usr/bin/env python3
#import sys
#sys.path.append('../IORSim_GUI')
from ior2ecl import runsim
from pathlib import Path

def mypath(dirs):
    return Path.home().joinpath(*(dirs.split()))

def loop_run(cases, times):
    iorexe = mypath('codes IORSim CVS build bin IORSimX')
    cdir = mypath('github IORSim_GUI cases')
    n = 0
    for case,time in zip(cases, times):
        print(f'CASE {n} : {case}\n--------------------------')
        runsim(root=cdir/case/case, time=time, iorexe=iorexe, to_screen=False, 
               verbose=3, ecl_alive=True, ior_alive=True)
        n += 1

if __name__ == '__main__':
    cases = 1*['L18']+['KURS-07A', 'SNORRE-IORSIM-OLD', 'GEOCHEM_BACK', 'GEOCHEM_FWD']
    times = 1*[1100 ]+[2000      , 3700               , 1000          , 1000]
    try:
        loop_run(cases, times)
    except KeyboardInterrupt:
        pass
