#!/usr/bin/env python3

from pathlib import Path
from time import sleep
import sys

sec        = float(sys.argv[1])
update_dir =  Path(sys.argv[2]) 
file       =  Path(sys.argv[3])
#print(sec, update_dir, file)
sleep(sec)
if file.is_file():
    src = update_dir/file
    dst = update_dir.parent/file
    #print(f'Moving {src} over {dst}')
    src.replace(dst)
    #print(f'Deleting {update_dir}')
    update_dir.rmdir()

