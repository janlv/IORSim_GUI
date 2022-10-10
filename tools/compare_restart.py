#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
sys.path.append('../IORSim_GUI')
from IORlib.ECL import UNRST_file, FUNRST_file
from pathlib import Path

case = Path(sys.argv[1])
ecl_key = sys.argv[2]
ior_key = sys.argv[3]
ecl = UNRST_file(case/(case.name+'_ECLIPSE')).data(ecl_key)
ior = FUNRST_file(case/(case.name+'_IORSim_PLOT')).data(ior_key)

for a,b in zip(ecl, ior):
    print('ECL:', a)
    print('IOR:', b)



