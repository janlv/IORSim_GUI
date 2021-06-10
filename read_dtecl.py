#!/usr/bin/env python3

from mmap import ACCESS_READ, mmap
import re
from IORlib.utils import matches

#-----------------------------------------------------------------------
def read_dtecl_ior(root):
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
    comm = '\s*#*.*\n+'
    num = '(?<!#)\s*\d+\.?e?E?\d*'
    two_num = f'{num}\s+{num}\s*\n+'
    #pattern = r'(\*INTEGRATION\s*\n+)((#+.*\s\n+)+)'
    #pattern = rf'(\*INTEGRATION\s*\n+)({comm})({two_num})({comm})({two_num})'
    #pattern = rf'(\*INTEGRATION\s*\n+)({comm})({nums})({comm})({nums})({comm})({nums})({comm})({nums})'
    pattern = r'\*INTEGRATION\s*\n+'
    m = [m for m in matches(file=file, pattern=pattern)]
    #print(m[0].group(0))
    end = m[0].span()[1]
    pattern = r'(?<!\s*#\s*)\s*\d+\.?e?E?\d*'
    m = [m.group(0) for m in matches(file=file, pattern=pattern, pos=end)]
    print(m)
    #with open(file) as f:
    #    with mmap(f.fileno(), length=0, access=ACCESS_READ) as data:
    #        m = re.match(pattern.encode(), data)
    #        print(m)

root = 'GUI/cases/SNURRE1/SNURRE1'
read_dtecl_ior(root)
