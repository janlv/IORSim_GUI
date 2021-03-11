#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import IORlib.ECL as ECL
import sys
from pathlib import Path

#inf = Path(sys.argv[1])
#out = inf.parent/(inf.stem+'_TEST.UNRST')
#out = ECL.fmt_file(inf).convert()

n = 0
#for block in ECL.unfmt_file(sys.argv[1]).tail_blocks(datalist=('MINISTEP')):
for block in ECL.unfmt_file(sys.argv[1]).blocks():
    block.print()
    if block.key() == 'SEQNUM':
        n+=1
        if n>2:
            break
    #    #print('{}: {}'.format(n, block.data()[0]))
    #    print(block.data()[0])
    #if block.key() == 'MINISTEP':
    #    #print('{}: {}'.format(n, block.data()[0]))
    #    print(block.data()[0])
    #    break
    #print(block.data())
#print(n)
#print(out)
