#!/usr/bin/env python3

from datetime import datetime
from time import sleep
from zipfile import ZipFile
from pathlib import Path
from tempfile import TemporaryDirectory
from IORlib.utils import kill_process
from shutil import copy, copytree
from subprocess import Popen

def unzip_and_copy(file, target):
    with TemporaryDirectory() as tmpdir:
        with ZipFile(file, 'r') as zip:
            zip.extractall(tmpdir)
        dir = next(Path(tmpdir).iterdir())  # Extracted dir
        backup = backup_dir(target)
        print('backup:', backup)
        ### Loop over files in extracted dir
        for item in dir.iterdir():
            dest = target/item.name
            if not dest.exist(): 
                continue
            print(f'{item} -> {dest}')
            if dest.is_file():
                copy(dest, backup)
                copy(item, dest)
            else:
                kwargs = {'dirs_exist_ok':True, 'copy_function':copy}
                copytree(dest, backup/dest.name, **kwargs)
                copytree(item, dest, **kwargs)

def backup_dir(path):
    p = path/f'upgrade_{datetime.today().strftime("%Y-%m-%d")}'
    p.mkdir(exist_ok=True)
    return p


def main(argv):
    pid = int(argv[1])
    file = Path(argv[2])
    cmd = argv[3:]
    target = Path.cwd()

    print('upgrader.py, target: ',target)
    print(f'pid = {pid}, file = {file}, cmd = {cmd}')

    ### Stop app
    kill_process(pid)

    ### Move new files over old ones (with backup)
    if file.suffix == '.zip':
        ### Upgrader called from script
        unzip_and_copy(file, target)
    else:
        ### Upgrader called from bundeled version (suffix is '.exe' or '' )
        dest = target/Path(cmd[0]).name
        #print('dest:', dest)
        if dest.exists():
            back = backup_dir(target)
            print(f'{dest} -> {back}')
            copy(dest, back)
            print(f'{file} -> {dest}')
            copy(file, dest)

    ### Restart app
    print(f'Starting {cmd}')
    Popen(cmd)

if __name__ == '__main__':
    import sys
    main(sys.argv)
