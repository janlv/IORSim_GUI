#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import IORlib.ECL as ECL
import sys
from pathlib import Path

file = Path(sys.argv[1])
n = 0
for block in ECL.unfmt_file(file).blocks():
    if block.key() in ('SEQNUM'):
        #block.print()
        print(f'\r{n} : {block.data()[0]}', end='')
        n += 1
print()

