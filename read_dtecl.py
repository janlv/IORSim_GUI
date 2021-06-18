#!/usr/bin/env python3

from mmap import ACCESS_READ, mmap
import re
from IORlib.utils import matches

#-----------------------------------------------------------------------
def dtecl_ior(root, name=False):
#-----------------------------------------------------------------------
#
#  Get dtecl from IORSim input file .trcinp
#  Assumed format:    
#    
#  *INTEGRATION
#  # tstart  tstop
#    0.0  1.e99
#  # dtmin dtmax 
#    0.0  1.e99
#  # dtecl dteclmax 
#    5      20 
#  # metnum
#    0
#
    file=f'{root}.trcinp'
    # Find end-position of *INTEGRATION 
    end = [m.span()[1] for m in matches(file=file, pattern=r'\*INTEGRATION\s*\n+')]
    num = '\d+\.?e?E?\d*'
    # Find uncommented lines with two numbers
    val = [float(s.decode()) for m in matches(file=file, pattern=fr'\n+\s*{num}\s*{num}', pos=end[0]) for s in m.group(0).split()]
    # Find commented lines with variable names
    if name:
        name = [s.decode() for m in matches(file=file, pattern=fr'(?<=#)\s*\D+\s*\D*', pos=end[0]) for s in m.group(0).split()]
    #print(val[:5])
    #print(name[:5])
    return val[4]

root = 'GUI/cases/SNURRE1/SNURRE1'
print(dtecl_ior(root))
