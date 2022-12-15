#!/usr/bin/env python3

from datetime import datetime
from zipfile import ZipFile
from pathlib import Path
from tempfile import TemporaryDirectory
from shutil import copy as shcopy, copytree
from subprocess import Popen
from IORlib.utils import kill_process, try_except_loop

#====================================================================================
class Upgrader:
#====================================================================================

    LOG_FILE = Path().home()/'.iorsim'/'upgrader.log'

    #--------------------------------------------------------------------------------
    def __init__(self, argv):
    #--------------------------------------------------------------------------------
        self.pid = int(argv[1])
        self.new_file = Path(argv[2])
        self.cmd = argv[3:]
        self.target = Path.cwd()
        self._clear_log = True

    #--------------------------------------------------------------------------------
    def __str__(self) -> str:
    #--------------------------------------------------------------------------------
        return '\n'.join(f'{k}: {v}' for k,v in self.__dict__.items() if k[0] != '_')

    #--------------------------------------------------------------------------------
    def upgrade(self):
    #--------------------------------------------------------------------------------
        self.log(f'Time: {datetime.now()}\n{self}')
        ### Stop app
        kill_process(self.pid)
        ### Move new files over old ones (with backup)
        if self.new_file.suffix == '.zip':
            ### Upgrader called from python script
            self.unzip_and_copy()
        else:
            ### Upgrader called from bundeled version (suffix is '.exe' or '' )
            self.copy_bundle()
        ### Delete file
        try:
            self.new_file.unlink()
        except PermissionError as error:
            self.log(error)
        ### Restart app
        with Popen(self.cmd) as proc:
            self.log(f'Started {self.cmd} as {proc}')

    #--------------------------------------------------------------------------------
    def make_backup_dir(self):
    #--------------------------------------------------------------------------------
        backup = self.target/'upgrade_backup'
        backup.mkdir(exist_ok=True)
        self.log(f'Created {backup}')
        #time = datetime.today().strftime('%Y-%m-%d, %H:%M:%S')
        #(p/'.timestamp').write_text(time)
        return backup

    #--------------------------------------------------------------------------------
    def copy(self, src: Path, dst: Path, backup=None):
    #--------------------------------------------------------------------------------
        '''
        Copy file or dir from src to dst with optional backup (if dst exists)
        '''
        #self.log(f'Copy {src} to {dst}' + (f' with backup {backup}' if backup else ''))
        try:
            if src.is_file():
                if backup and dst.exists():
                    shcopy(dst, backup)
                    self.log(f'Backup: {dst} -> {backup}')
                shcopy(src, dst)
                self.log(f'Copy: {src} -> {dst}')
            else:
                kwargs = {'dirs_exist_ok':True, 'copy_function':shcopy}
                if backup and dst.exists():
                    copytree(dst, backup/dst.name, **kwargs)
                    self.log(f'backup: {dst} -> {backup/dst.name}')
                copytree(src, dst, **kwargs)
                self.log(f'dir: {src} -> {dst}')
        except Exception as error:
            self.log(error)
            raise error
        else:
            self.log('Copy complete!')

    #--------------------------------------------------------------------------------
    def unzip_and_copy(self):
    #--------------------------------------------------------------------------------
        self.log('Upgrade from python')
        with TemporaryDirectory() as tmpdir:
            with ZipFile(self.new_file, 'r') as zipfile:
                zipfile.extractall(tmpdir)
            backup = self.make_backup_dir()
            ### Loop over files in extracted dir
            extract = next(Path(tmpdir).iterdir())  # Extracted dir
            for item in extract.iterdir():
                dest = self.target/item.name
                self.copy(item, dest, backup=backup)


    #--------------------------------------------------------------------------------
    def copy_bundle(self, limit=100, pause=0.05):
    #--------------------------------------------------------------------------------
        self.log('Upgrade from bundle')
        dest = self.target/Path(self.cmd[0]).name
        if dest.exists():
            ### Keep trying to overwrite if PermissionError
            try_except_loop(self.new_file, dest, backup=self.make_backup_dir(),
                func=self.copy, limit=limit, pause=pause, error=PermissionError)

    #--------------------------------------------------------------------------------
    def log(self, text):
    #--------------------------------------------------------------------------------
        mode = 'a'
        if self._clear_log:
            self._clear_log = False
            mode = 'w'
        with open(self.LOG_FILE, mode) as file:
            file.write(text+'\n')



#################################################################################

if __name__ == '__main__':
    import sys
    Upgrader(sys.argv).upgrade()
