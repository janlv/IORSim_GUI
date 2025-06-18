from datetime import datetime
from pathlib import Path
from psutil import Popen
from IORSim_GUI import IORSIM_DIR
from iorsimlib.utils import try_except_loop, kill_process, copy_recursive

#====================================================================================
class Upgrader:
#====================================================================================

    LOG_FILE = IORSIM_DIR/'upgrader.log'

    #--------------------------------------------------------------------------------
    def __init__(self, argv):
    #--------------------------------------------------------------------------------
        self.pid = int(argv[0])
        self.new_file = Path(argv[1])
        self.cmd = argv[2:]
        self.target = Path.cwd()
        self._clear_log = True

    #--------------------------------------------------------------------------------
    def __str__(self) -> str:
    #--------------------------------------------------------------------------------
        return '\n'.join(f'{k}: {v}' for k,v in self.__dict__.items() if k[0] != '_')

    #--------------------------------------------------------------------------------
    def upgrade(self, limit=100, pause=0.05):
    #--------------------------------------------------------------------------------
        self.log(f'Time: {datetime.now()}\n{self}')
        # Stop app
        procs = kill_process(self.pid)
        self.log(f'Killed {self.pid}: {procs}')
        if self.new_file.is_dir():
            # Upgrader called from python script
            dest = self.target
        else:
            # Upgrader called from bundeled version (suffix is '.exe' or '' )
            dest = self.cmd[0]
        self.log(f'Copy {self.new_file} to {dest}')
        # Move new files over old ones
        # Keep trying to overwrite if PermissionError
        try_except_loop(self.new_file, dest, log=self.log, func=copy_recursive,
                        limit=limit, pause=pause, error=(PermissionError, OSError))
        # Start new version 
        with Popen(self.cmd) as proc:
            self.log(f'Started {self.cmd} as {proc}')
        self.log('Upgrade complete!')


    #--------------------------------------------------------------------------------
    def log(self, text):
    #--------------------------------------------------------------------------------
        mode = 'a'
        if self._clear_log:
            self._clear_log = False
            mode = 'w'
        with open(self.LOG_FILE, mode, encoding='utf-8') as file:
            file.write(f'{text}\n')

