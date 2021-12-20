#!/usr/bin/env python3

from pathlib import Path
from time import sleep
import sys

sec        = float(sys.argv[1])
update_dir =  Path(sys.argv[2]) 
file       =  Path(sys.argv[3])
echo = False
if len(sys.argv)==5:
    echo = int(sys.argv[4])  
if echo:
    print(f'  script:{sys.argv[0]}, sec:{sec}, update_dir:{update_dir}, file:{file}')
sleep(sec)
if file.is_file():
    src = update_dir/file.name
    dst = file
    if echo:
        print(f'Moving {src}(size: {src.stat().st_size}) over {dst}(size: {dst.stat().st_size})')
    src.replace(dst)
    if echo:
        print(f'{dst} size: {dst.stat().st_size}')
        print(f'Deleting {update_dir}')
    update_dir.rmdir()

