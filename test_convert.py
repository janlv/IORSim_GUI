#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import IORlib.ECL as ECL
import sys
from pathlib import Path

inf = Path(sys.argv[1])
out = inf.parent/(inf.stem+'_TEST.UNRST')
out = ECL.fmt_file(inf).convert()

n = 0
for block in ECL.unfmt_file(out).blocks():
    if block.key() in ('SEQNUM'):
        print('{}: {}'.format(n, block.data()[0]))
        n += 1
    #block.print()

print(out)
