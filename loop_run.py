from ior2ecl import runsim
from pathlib import Path

def mypath(dirs):
    return Path.home().joinpath(*(dirs.split()))

def run(cases, times):
    iorexe = mypath('codes IORSim build bin IORSimX.exe')
    cdir = mypath('codes IORSim_GUI GUI cases')
    n = 0
    for case,time in zip(cases, times):
        print(f'CASE {n} : {case}\n--------------------------')
        runsim(root=cdir/case/case, time=time, iorexe=iorexe)
        n += 1

if __name__ == '__main__':
    run(10*['L18'], 10*[1100])

