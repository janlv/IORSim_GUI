#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from datetime import datetime
from subprocess import Popen, PIPE, STDOUT, call
#import atexit
#import signal
#import os
#import sys
import psutil
from shutil import which
from time import sleep
from pathlib import Path #, PurePath
from shutil import copy
#from struct import unpack
from .utils import loop_until, list2str, safeopen, Timer, silentdelete

#--------------------------------------------------------------------------------
def permission_error(func):
#--------------------------------------------------------------------------------
    def inner(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except PermissionError:
            print('PermissionError in ' + func.__qualname__)
    return inner


#====================================================================================
class Control_file:
#====================================================================================
    #--------------------------------------------------------------------------------
    def __init__(self, ext=None, root=None):
    #--------------------------------------------------------------------------------
        self._ext = ext
        self._name = Path(str(root) + '.' + ext)

    #--------------------------------------------------------------------------------
    def name(self):
    #--------------------------------------------------------------------------------
        return self._name.name
        
    #--------------------------------------------------------------------------------
    def print(self):
    #--------------------------------------------------------------------------------
        print(self._name)
        
    @permission_error
    #--------------------------------------------------------------------------------
    def create_empty(self, delete=False):
    #--------------------------------------------------------------------------------
        self._name.touch()
    
    @permission_error
    #--------------------------------------------------------------------------------
    def create_from_string(self, string):    
    #--------------------------------------------------------------------------------
        with open(self._name,'w') as f:
            f.write(string)

    @permission_error
    #--------------------------------------------------------------------------------
    def copy(self, src, delete=False):
    #--------------------------------------------------------------------------------
        copy(src, self._name)
        if delete:
            silentdelete(src)
            #src.unlink(missing_ok=True)
        
    @permission_error
    #--------------------------------------------------------------------------------
    def append(self, _ascii):
    #--------------------------------------------------------------------------------
        with open(self._name, 'a') as f:
            f.write('{}\n'.format(_ascii))

    @permission_error
    #--------------------------------------------------------------------------------
    def delete(self):
    #--------------------------------------------------------------------------------
        path = self._name.parent
        name = self._name.name
        for f in path.glob(name):
            #print('Deleting {}'.format(f))
            f.unlink()

    #--------------------------------------------------------------------------------
    def is_deleted(self):
    #--------------------------------------------------------------------------------
        try:
            if not self._name.is_file():
                return True
        except PermissionError:
            return None
        

#====================================================================================
class runner:                                                               # runner
#====================================================================================
    """

    Class for running programs using Popen. 
    Execution is controlled by interface- and OK-files

    Initialization:
    runner(case, exe, log=None, children=[], pipe=False, verbose=2, timer=False, runlog=None)

    Methods:
      start(cmd)
      suspend()
      resume()
      parent_is_not_running()
      parent_has_stopped()
      parent_is_sleeping()
      kill()
      assert_running()
      wait_until_sleeping()
      wait_until_not_running()
 
    """
    
    #--------------------------------------------------------------------------------
    def __init__(self, N=0, T=0, name=None, case=None, exe=None, cmd=None, pipe=False, echo=False,
                 verbose=3, timer=None, runlog=None, ext_iface=None, ext_OK=None,
                 keep_files=False, stop_children=False, **kwargs):           # runner
    #--------------------------------------------------------------------------------
        #print('runner.__init__: ',N,T,name,case,exe,cmd,ext_iface,ext_OK)
        self.name = name
        self.case = Path(case)
        self.exe = exe
        self.cmd = cmd
        self.log = safeopen( Path(case).parent / Path(name.lower()+'.log'), 'w' )
        #atexit.register(self.log.close)
        self.runlog = runlog
        self.ext_iface = ext_iface
        self.ext_OK = ext_OK
        self.parent = None
        self.children = []
        self.stop_children = stop_children
        #print(f'stop_children: {stop_children}')
        self.pipe = pipe
        self.verbose = verbose
        self.echo = echo
        #if verbose > 2:
        #    self.echo = True
        self.timer = timer
        if self.timer:
            self.timer = Timer(name.lower())
        #self._days = 0
        #self._days_old = 0
        self.keep_files = keep_files
        self.canceled = False
        self.t = 0  
        self.n = 0
        self.T = int(T)   # Max time
        self.N = int(N)   # Max number of steps
        self.starttime = None
                       
    #-----------------------------------------------------------------------
    def set_time(self, time):
    #-----------------------------------------------------------------------
        self.T = int(time)

    #--------------------------------------------------------------------------------
    def check_input(self):                                                   # runner
    #--------------------------------------------------------------------------------
        ### check if executables exist on the system
        if which(self.exe) is None:
            raise SystemError('WARNING Executable not found: ' + self.exe)


    #--------------------------------------------------------------------------------
    def interface_file(self, nr):
    #--------------------------------------------------------------------------------
        if isinstance(nr, int):
            return Control_file(ext=self.ext_iface.format(nr), root=self.case) 
        elif nr == 'all':
            ext, num = self.ext_iface.split('{')
            n = int(num.split('d')[0][-1])
            return Control_file(ext=ext+'?'*n, root=self.case)
                
        
    #--------------------------------------------------------------------------------
    def OK_file(self):
    #--------------------------------------------------------------------------------
        return Control_file(ext=self.ext_OK, root=self.case)


    #--------------------------------------------------------------------------------
    def start(self, check=None):                                            # runner
    #--------------------------------------------------------------------------------
        pid = None
        self.starttime = datetime.now()
        #cmd_str = ' '.join(self.cmd)
        if self.pipe:
            self._print(f"starting {self.name} in PIPE-mode", v=1)
            P = Popen(self.cmd, stdin=PIPE, stdout=self.log, stderr=STDOUT)
        else:
            self._print(f'starting {self.name}', v=1)
            P = Popen(self.cmd, stdout=self.log, stderr=STDOUT)      
        self.P = P
        pid = P.pid
        self.parent = psutil.Process(pid=pid)
        #print()
        #print(self.parent)
        # check if we need to look for child processes
        if not self.parent.name().lower().startswith(self.name.lower()):
            # looking for child-processes
            for i in range(100):
                sleep(0.5)
                self.children = self.parent.children(recursive=True)
                #print(self.children)
                if any([p.name().lower().startswith(self.name.lower()) for p in self.children]):
                    break
        #self.procs = self.children + [self.parent,]
        self.assert_running()
        self.cmdline = '\'' + ' '.join(self.parent.cmdline()) + '\''
        #atexit.register(self.kill)
        self._print(self.process_info())
        
        
    #--------------------------------------------------------------------------------
    def get_logfile(self):                                                   # runner
    #--------------------------------------------------------------------------------
        return self.log.name
    
    #--------------------------------------------------------------------------------
    def process_info(self, indent=2):                                        # runner
    #--------------------------------------------------------------------------------
        ind = ' ' * indent
        header = ind + 'Started {:s} ({:d})\n'.format(self.cmdline, self.parent.pid)
        if self.children:
            ch_names = [p.name() for p in self.children]
            ch_pids = [p.pid for p in self.children]
            header += ind + 'Child processes are {:s} ({:s})\n'.format(list2str(ch_names, sep='\''), list2str(ch_pids))
        header += ind + 'Log-file is {:s}'.format(self.log.name)
        return header
        
    #--------------------------------------------------------------------------------
    def suspend(self, check=False, print=True, v=1):          # runner
    #--------------------------------------------------------------------------------
        self._print(f'suspending {self.name} ({self.parent.pid})', v=v)
        procs = [self.parent,]
        if self.stop_children:
            procs = self.children + procs
        # Suspend processes
        [p.suspend() for p in procs]
        if check:
            N = [loop_until(func, error=f'{self.parent.name()} did not stop') for func in (self.parent_has_stopped, self.children_has_stopped)]
            self._print(f'check: {", ".join([str(n) for n in N])}', v=v)
        if print:
            self._print(', '.join([f'{p.name()} {p.status()}' for p in procs]), v=v)
        self.timer and self.timer.stop()
        
    #--------------------------------------------------------------------------------
    def resume(self, check=False, print=True, v=1):                       # runner
    #--------------------------------------------------------------------------------
        self._print(f'resuming {self.name} ({self.parent.pid})', v=v)
        procs = [self.parent,]
        if self.stop_children:
            procs = procs + self.children
        # Resume processes
        [p.resume() for p in procs]
        if check:
            N = [loop_until(func, error=f'{self.parent.name()} did not start') for func in (self.parent_is_running, self.children_is_running)]
            self._print(f'check: {", ".join([str(n) for n in N])}', v=v)
        if print:
            self._print(', '.join([f'{p.name()} {p.status()}' for p in procs]), v=v)
        self.timer and self.timer.start()
        
    #--------------------------------------------------------------------------------
    def parent_is_not_running(self):                                         # runner
    #--------------------------------------------------------------------------------
        try:
            if not self.parent.is_running() or self.parent.status() == psutil.STATUS_ZOMBIE:
                return True
        except psutil.NoSuchProcess:
            return True
            
    #--------------------------------------------------------------------------------
    def parent_is_running(self):                                                    # runner
    #--------------------------------------------------------------------------------
        try:
            if self.parent and self.parent.is_running() and self.parent.status() != psutil.STATUS_ZOMBIE:
                return True
        except psutil.NoSuchProcess:
            return False
            
    #--------------------------------------------------------------------------------
    def parent_has_stopped(self):                                            # runner
    #--------------------------------------------------------------------------------
        if self.parent.status() == psutil.STATUS_STOPPED: 
            return True

    #--------------------------------------------------------------------------------
    def children_has_stopped(self):                                            # runner
    #--------------------------------------------------------------------------------
        if all([p.status() == psutil.STATUS_STOPPED for p in self.children]): 
            return True
                
    #--------------------------------------------------------------------------------
    def children_is_running(self):                                            # runner
    #--------------------------------------------------------------------------------
        try:
            if all([p.is_running() and p.status() != psutil.STATUS_ZOMBIE for p in self.children]): 
                return True
        except psutil.NoSuchProcess:
            return False
                
    #--------------------------------------------------------------------------------
    def parent_is_sleeping(self):                                            # runner
    #--------------------------------------------------------------------------------
        if self.parent.status() == psutil.STATUS_SLEEPING: # S in top
            return True
                
    #--------------------------------------------------------------------------------
    def kill(self, v=1):                                                     # runner
    #--------------------------------------------------------------------------------
        # define local functions
        def name_pid(proc):
            return '\'{:s}\' ({:d})'.format(proc.name(), proc.pid)
        def terminate(proc):
            try:
                self._print('Killing {:s}'.format(name_pid(proc)), v=v)
                if proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE:
                    proc.kill()
                else:
                    if proc.status() == psutil.STATUS_ZOMBIE:
                        self._print('process {:s} is a zombie...'.format(name_pid(proc)), v=v)
                    else:
                        self._print('process {:s} already gone'.format(name_pid(proc)), v=v)
            except psutil.NoSuchProcess:
                pass
        # terminate children before parent
        procs = self.children
        if self.parent:
            procs += [self.parent,]
        for p in procs:
            terminate(p)

            
    #--------------------------------------------------------------------------------
    def assert_running(self):                                                # runner
    #--------------------------------------------------------------------------------
        try:
            if self.parent.is_running() and self.parent.status()!=psutil.STATUS_ZOMBIE: 
                return True
            self._print('', tag='')
            raise SystemError('ERROR ' + self.name + ' is not running, status is ' + self.parent.status())
        except psutil.NoSuchProcess:
            self._print('', tag='')
            raise SystemError('ERROR ' + self.name + ' stopped unexpectedly, check the log')        
        
    #--------------------------------------------------------------------------------
    def wait_until_sleeping(self, v=1):                                      # runner
    #--------------------------------------------------------------------------------
        self._print('waiting for process to sleep', v=v)
        loop_until( self.parent_is_sleeping, error='{} did not sleep'.format(self.parent.name()) )

        
    #--------------------------------------------------------------------------------
    def time_and_step(self):                                                 # runner
    #--------------------------------------------------------------------------------
        return 0, 0

    #--------------------------------------------------------------------------------
    def stop_if_canceled(self, step='days'):
    #--------------------------------------------------------------------------------
        if self.canceled:
            if not step in ('steps','step'):
                # Use time unit
                c = int(self.t)
                if c == 0:
                    c = self.time_and_step()[0]
            else:
                # Use step unit
                c = int(self.n)
                if c == 0:
                    c = self.time_and_step()[1]
            self._print('', tag='')
            raise SystemError('INFO Run stopped after ' + str(c) + ' ' + step)    

    #--------------------------------------------------------------------------------
    def assert_running_and_stop_if_canceled(self):
    #--------------------------------------------------------------------------------
        self.assert_running()
        self.stop_if_canceled()

    #--------------------------------------------------------------------------------
    def time_and_step(self):
    #--------------------------------------------------------------------------------
        return None, None


    #--------------------------------------------------------------------------------
    def run_time(self):
    #--------------------------------------------------------------------------------
        return datetime.now()-self.starttime

    #--------------------------------------------------------------------------------
    def complete_msg(self, run_time=None):
    #--------------------------------------------------------------------------------
        if not run_time:
            run_time = self.run_time()
        return 'INFO Simulation complete, run-time was ' + str(run_time).split('.')[0]

    #--------------------------------------------------------------------------------
    def stop_if_limit_reached(self, limit='time', value=None): 
    #--------------------------------------------------------------------------------
        if not value:
            t, n = self.time_and_step()
        if limit in ('step','steps'):
            lim = self.N
            value = value or n
        else:
            lim = self.T
            value = value or t
        #print(n, self.N)
        if value > lim:
            #print('Step limit reached')
            raise SystemError(self.complete_msg())
        return value

    #--------------------------------------------------------------------------------
    def wait_for(self, func, *args, error=None, limit=100000, pause=0.01, log=None, loop_func=0, v=3, **kwargs):
    #--------------------------------------------------------------------------------
        if loop_func==0:
            # Default checks during loop
            loop_func = self.assert_running_and_stop_if_canceled
        passed_args = ','.join([f'{k}={v}' for k,v in kwargs.items()])
        self._print(f'calling wait_for( {func.__qualname__}({passed_args}), limit={limit} )...', v=v, end='')
        n = loop_until(func, *args, **kwargs, error=error, pause=pause, limit=limit, loop_func=loop_func)
        if n<0:
            self._print('', tag='')    
            return False    
        self._print(str(n) + ' loops', v=3, tag='')
        if callable(log):
            self._print(log())
        return True


    #--------------------------------------------------------------------------------
    def wait_for_process_to_finish(self, v=1, limit=None, error=None, pause=None, loop_func=None):      # runner
    #--------------------------------------------------------------------------------
        self._print('waiting for process to finish', v=v)
        success = self.wait_for(self.parent_is_not_running, error=error, pause=pause, limit=limit, loop_func=loop_func)
        if not success:
             self._print('process did not finish within reasonable time and was killed', v=v)
             self.kill()
        self.parent = None
        
    #--------------------------------------------------------------------------------
    def quit(self, v=1):
    #--------------------------------------------------------------------------------
        self._print('quitting process', v=v)
        self.resume()
        self.wait_for_process_to_finish(limit=1000, pause=0.01, loop_func=None)
        self.log.close()


    #--------------------------------------------------------------------------------
    def clean_up(self):
    #--------------------------------------------------------------------------------
        if self.ext_iface and not self.keep_files:
            self.interface_file('all').delete()


    #--------------------------------------------------------------------------------
    def kill_and_clean(self):
    #--------------------------------------------------------------------------------
        self.kill()
        self.log.close()
        self.clean_up()


    #--------------------------------------------------------------------------------
    def write_to_stdin(self, i):                                             # runner
    #--------------------------------------------------------------------------------
        if not self.pipe:
            raise SystemError('STDIN is not piped, unable to write. Aborting...')
        self._print('writing {} to STDIN'.format(i))
        inp = '{:d}\n'.format(i)
        self.P.stdin.write(inp.encode())
        self.P.stdin.flush()

        
    #--------------------------------------------------------------------------------
    def _print(self, txt, v=1, tag=None, **kwargs):                         # runner
    #--------------------------------------------------------------------------------
        if v <= self.verbose:
            if tag==None:
                tag = self.name + ': '
            print(tag + txt, file=self.runlog, flush=True, **kwargs)


    #--------------------------------------------------------------------------------
    def _printerror(self, txt, **kwargs):                                     # runner
    #--------------------------------------------------------------------------------
        print()
        print('  ERROR: ' + txt, **kwargs)
        print('', flush=True)
    
    #--------------------------------------------------------------------------------
    def _printwarning(self, txt, **kwargs):                                     # runner
    #--------------------------------------------------------------------------------
        print()
        print('  WARNING: ' + txt, **kwargs)
        print('', flush=True)
    
