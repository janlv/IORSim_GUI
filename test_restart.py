#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import IORlib.ECL as ECL
import sys
from pathlib import Path

file = Path(sys.argv[1])

n = 0
for block in ECL.unfmt_file(file).blocks():
    if block.key() in ('SEQNUM'):#,'TEMP'):#,'STARTSOL','ENDSOL'):
        print('{}: {}'.format(n, block.data()[0]))
        n += 1
        #if n>3:
        #    break
        block.print()
        #print('{} : {}'.format(str(n),block.data()[0]))
    #block.print()


