#!/usr/bin/env python3

from datetime import datetime
from time import sleep
from zipfile import ZipFile
from pathlib import Path
from tempfile import TemporaryDirectory
from IORlib.utils import kill_process
from shutil import copy as shcopy, copytree
from subprocess import Popen

DEBUG = False

#--------------------------------------------------------------------------------
def backup_dir(path: Path):
#--------------------------------------------------------------------------------
    date = datetime.today().strftime("%Y-%m-%d")
    p = path/f'upgrade_{date}'
    p.mkdir(exist_ok=True)
    return p

#--------------------------------------------------------------------------------
def copy(src: Path, dst: Path, backup=None): 
#--------------------------------------------------------------------------------
    '''
    Copy file or dir from src to dst with optional backup (if dst exists)
    '''
    if src.is_file():
        if backup and dst.exists():
            DEBUG and print(f'backup: {dst} -> {backup}')
            shcopy(dst, backup)
        DEBUG and print(f'file: {src} -> {dst}')
        shcopy(src, dst)
    else:
        kwargs = {'dirs_exist_ok':True, 'copy_function':shcopy}
        if backup and dst.exists():
            DEBUG and print(f'backup: {dst} -> {backup/dst.name}')
            copytree(dst, backup/dst.name, **kwargs)
        DEBUG and print(f'dir: {src} -> {dst}')
        copytree(src, dst, **kwargs)


#--------------------------------------------------------------------------------
def unzip_and_copy(file, target):
#--------------------------------------------------------------------------------
    with TemporaryDirectory() as tmpdir:
        with ZipFile(file, 'r') as zip:
            zip.extractall(tmpdir)
        dir = next(Path(tmpdir).iterdir())  # Extracted dir
        backup = backup_dir(target)
        ### Loop over files in extracted dir
        for item in dir.iterdir():
            dest = target/item.name
            copy(item, dest, backup=backup)


#--------------------------------------------------------------------------------
def copy_bundle(file, target, cmd, limit=100, pause=0.05):
#--------------------------------------------------------------------------------
    dest = target/Path(cmd[0]).name
    if dest.exists():
        backup = backup_dir(target)
        ### Keep trying to overwrite if PermissionError
        for i in range(limit):
            try:
                copy(file, dest, backup=backup)
                break
            except PermissionError as e:
                sleep(pause)
        if i==limit-1:
            raise SystemError(f'Unable to copy {file} over {dest} within {limit} tries during {limit*pause} seconds: {e}')


#--------------------------------------------------------------------------------
def main(argv):
#--------------------------------------------------------------------------------
    pid = int(argv[1])
    file = Path(argv[2])
    cmd = argv[3:]
    target = Path.cwd()

    DEBUG and print(f'pid = {pid}, file = {file}, cmd = {cmd}, target = {target}')

    ### Stop app
    kill_process(pid)

    ### Move new files over old ones (with backup)
    if file.suffix == '.zip':
        ### Upgrader called from script
        unzip_and_copy(file, target)
    else:
        ### Upgrader called from bundeled version (suffix is '.exe' or '' )
        copy_bundle(file, target, cmd)

    ### Restart app
    DEBUG and print(f'Starting {cmd}')
    Popen(cmd)

#################################################################################

if __name__ == '__main__':
    import sys
    main(sys.argv)
