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
def catch_permission_error(func):
#--------------------------------------------------------------------------------
    def inner(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except PermissionError as e:
            raise SystemError(f'WARNING PermissionError in {func.__qualname__}()')
            #print('PermissionError in ' + func.__qualname__)
            #print('PermissionError in ' + func.__qualname__ + ': ',e)
    return inner

#--------------------------------------------------------------------------------
def ignore_process_error(func):
#--------------------------------------------------------------------------------
    def inner(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (psutil.NoSuchProcess, ProcessLookupError):
            return False
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
    def path(self):
    #--------------------------------------------------------------------------------
        return self._name

    #--------------------------------------------------------------------------------
    def print(self):
    #--------------------------------------------------------------------------------
        print(self._name)
        
    @catch_permission_error
    #--------------------------------------------------------------------------------
    def create_empty(self, delete=False):
    #--------------------------------------------------------------------------------
        self._name.touch()
    
    @catch_permission_error
    #--------------------------------------------------------------------------------
    def create_from_string(self, string):    
    #--------------------------------------------------------------------------------
        with open(self._name,'w') as f:
            f.write(string)

    @catch_permission_error
    #--------------------------------------------------------------------------------
    def copy(self, src, delete=False):
    #--------------------------------------------------------------------------------
        copy(src, self._name)
        if delete:
            silentdelete(src)
            #src.unlink(missing_ok=True)
        
    @catch_permission_error
    #--------------------------------------------------------------------------------
    def append(self, _ascii):
    #--------------------------------------------------------------------------------
        with open(self._name, 'a') as f:
            f.write(f'{_ascii}\n')

    @catch_permission_error
    #--------------------------------------------------------------------------------
    def delete(self):
    #--------------------------------------------------------------------------------
        path = self._name.parent
        name = self._name.name
        for f in path.glob(name):
            #print('deleting',f)
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
class Process:                                                              # Process
#====================================================================================
    
    #--------------------------------------------------------------------------------
    def __init__(self, proc, app_name=None):                                # Process
    #--------------------------------------------------------------------------------
        self.proc = proc
        self.pid = proc.pid
        self._name = proc.name()
        self.app_name = app_name
        self._suspend_errors = 0
        #self.name = proc.name()

    #--------------------------------------------------------------------------------
    def process(self):                                                      # Process
    #--------------------------------------------------------------------------------
        return self.proc

    #--------------------------------------------------------------------------------
    def name(self):                                                         # Process
    #--------------------------------------------------------------------------------
        return self._name

    @ignore_process_error
    #--------------------------------------------------------------------------------
    def status(self):                                                       # Process
    #--------------------------------------------------------------------------------
        return self.proc.status()

    @ignore_process_error
    #--------------------------------------------------------------------------------
    def cmdline(self):                                                      # Process
    #--------------------------------------------------------------------------------
        return self.proc.cmdline()

    #--------------------------------------------------------------------------------
    def kill(self):                                                         # Process
    #--------------------------------------------------------------------------------
        try:
            self.proc.kill()
            return True
        except (psutil.AccessDenied, psutil.NoSuchProcess, ProcessLookupError):
            return False

    #--------------------------------------------------------------------------------
    def suspend(self):                                                      # Process
    #--------------------------------------------------------------------------------
        try:
            self.proc.suspend()
            return True
        except psutil.AccessDenied:
            self._suspend_errors += 1
            return False
        except (psutil.NoSuchProcess, ProcessLookupError):
            return False

    #--------------------------------------------------------------------------------
    def resume(self):                                                       # Process
    #--------------------------------------------------------------------------------
        try:
            self.proc.resume()
            return True
        except (psutil.AccessDenied, psutil.NoSuchProcess, ProcessLookupError):
            return False


    @ignore_process_error
    #--------------------------------------------------------------------------------
    def current_status(self):                                               # Process
    #--------------------------------------------------------------------------------
        return f'{self.proc.name()} {self.proc.status()}'
    

    #--------------------------------------------------------------------------------
    def suspend_errors(self):                                               # Process
    #--------------------------------------------------------------------------------
        if self._suspend_errors > 0:
            return f'{self._name} failed to suspend {self._suspend_errors} times'
        return ''

    #--------------------------------------------------------------------------------
    def is_running(self, raise_error=False):                                # Process
    #--------------------------------------------------------------------------------
        try:
            if self.proc.is_running() and self.proc.status() != psutil.STATUS_ZOMBIE:
                return True
            if raise_error:
                raise SystemError(f'ERROR {self.app_name} is not running ({self._name} is {self.status()})')
        except (psutil.NoSuchProcess, ProcessLookupError):
            if raise_error:
                raise SystemError(f'ERROR {self.app_name} stopped unexpectedly (process disappeared), check the log')        
            else:
                return False
        except AttributeError:
            if raise_error:
                raise SystemError(f'ERROR {self.app_name} process is {self.proc}')        
            else:
                return True
            
    #--------------------------------------------------------------------------------
    def is_not_running(self):                                               # Process
    #--------------------------------------------------------------------------------
        try:
            if not self.proc or not self.proc.is_running() or self.status() == psutil.STATUS_ZOMBIE:
                return True
        except (psutil.NoSuchProcess, ProcessLookupError):
            return True
            
    #--------------------------------------------------------------------------------
    def is_sleeping(self):                                                  # Process
    #--------------------------------------------------------------------------------
        try:
            if self.status() in (psutil.STATUS_SLEEPING, psutil.STATUS_STOPPED): 
                return True
            elif not self.proc.is_running() or self.status() == psutil.STATUS_ZOMBIE:
                raise SystemError(f'ERROR Process {self.name()} disappeared while trying to sleep')
        except (psutil.NoSuchProcess, ProcessLookupError, AttributeError):
            raise SystemError(f'ERROR Process {self.proc} disappeared while trying to sleep')
                
    #--------------------------------------------------------------------------------
    def assert_running(self):                                                # Process
    #--------------------------------------------------------------------------------
        return self.is_running(raise_error=True)


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
      kill()
 
    """
    
    #--------------------------------------------------------------------------------
    def __init__(self, N=0, T=0, name=None, case=None, exe=None, cmd=None, pipe=False,
                 verbose=3, timer=None, runlog=None, ext_iface=None, ext_OK=None,
                 keep_files=False, stop_children=True, **kwargs):           # runner
    #--------------------------------------------------------------------------------
        #print('runner.__init__: ',N,T,name,case,exe,cmd,ext_iface,ext_OK)
        self.name = name
        self.case = Path(case)
        self.exe = exe
        self.cmd = cmd
        self.log = safeopen( Path(case).parent / Path(name.lower()+'.log'), 'w' )
        self.runlog = runlog
        self.ext_iface = ext_iface
        self.ext_OK = ext_OK
        self.parent = None
        self.children = []
        self.stop_children = stop_children
        #print(f'stop_children: {stop_children}')
        self.pipe = pipe
        self.verbose = verbose
        self.timer = timer
        if self.timer:
            self.timer = Timer(name.lower())
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
    def start(self, check=None, remove_cmdexe=False):                        # runner
    #--------------------------------------------------------------------------------
        pid = None
        self.starttime = datetime.now()
        if self.pipe:
            self._print(f"starting {self.name} in PIPE-mode", v=1)
            P = Popen(self.cmd, stdin=PIPE, stdout=self.log, stderr=STDOUT)
        else:
            self._print(f'starting {self.name}', v=1)
            P = Popen(self.cmd, stdout=self.log, stderr=STDOUT)      
        self.P = P
        pid = P.pid
        self.parent = Process(psutil.Process(pid=pid), app_name=self.name)
        # check if we need to look for child processes
        if not self.parent.name().lower().startswith(self.name.lower()):
            # looking for child-processes
            for i in range(100):
                sleep(0.5)
                self.children = self.parent.process().children(recursive=True)
                if any([p.name().lower().startswith(self.name.lower()) for p in self.children]):
                    break
        if remove_cmdexe:
            self.children = [child for child in self.children if child.name != 'cmd.exe']
        self.children = [Process(c, app_name=self.name) for c in self.children]
        self.parent.assert_running()
        self.cmdline = '\'' + ' '.join(self.parent.cmdline()) + '\''
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
    def suspend(self, check=True, print=True, v=1):          # runner
    #--------------------------------------------------------------------------------
        self._print(f'suspending {self.name} ({self.parent.pid})', v=v)
        success = True
        # Suspend children
        if self.stop_children:
            success = all([p.suspend() for p in self.children])
            if check:
                [self.wait_for(p.is_sleeping) for p in self.children]
        # Suspend parent
        self.parent.suspend()
        if check:
            self.wait_for(self.parent.is_sleeping)
        if print:
            self.print_process_status()
        self.timer and self.timer.stop()
        return success

    #--------------------------------------------------------------------------------
    def resume(self, check=False, print=True, v=1):                       # runner
    #--------------------------------------------------------------------------------
        self._print(f'resuming {self.name} ({self.parent.pid})', v=v)
        # Resume parent
        self.parent.resume()
        if check:
            self.wait_for(self.parent.is_running)
        # Resume children
        [p.resume() for p in self.children]
        if check:
            [self.wait_for(p.is_running) for p in self.children]
        if print:
            self.print_process_status()
        self.timer and self.timer.start()


    #--------------------------------------------------------------------------------
    def kill(self, check=True, v=1):                                                     # runner
    #--------------------------------------------------------------------------------
        # terminate parent before children
        self._print('\n', tag='')
        procs = (self.parent and [self.parent] or []) + self.children
        for p in procs:
            self._print(f'killing {p.name()}')
            p.kill()
            if check:
                self.wait_for(p.is_not_running, loop_func=lambda:None)
        self.parent = None
        self.children = []

    #--------------------------------------------------------------------------------
    def process_list(self):
    #--------------------------------------------------------------------------------
        return (self.parent and [self.parent] or []) + self.children

    #--------------------------------------------------------------------------------
    def print_process_status(self, v=1):
    #--------------------------------------------------------------------------------
        self._print(', '.join([p.current_status() for p in self.process_list() if p.current_status()]), v=v)

    #--------------------------------------------------------------------------------
    def print_suspend_errors(self, v=1):
    #--------------------------------------------------------------------------------
        self._print(', '.join([p.suspend_errors() for p in self.process_list() if p.suspend_errors()]), v=v)
        # val = []
        # for p in [self.parent]+self.children:
        #     stat = p.suspend_errors()
        #     if stat:
        #         val.append(stat)        
        # self._print(', '.join(val), v=v)

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
            self._print(' ', tag='')
            raise SystemError('INFO Run stopped after ' + str(c) + ' ' + step)    

    #--------------------------------------------------------------------------------
    def assert_running_and_stop_if_canceled(self):
    #--------------------------------------------------------------------------------
        self.parent.assert_running()
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
    def wait_for(self, func, *args, error=None, limit=100000, pause=0.01, log=None, loop_func=None, v=3, **kwargs):
    #--------------------------------------------------------------------------------
        if not loop_func:
            # Default checks during loop
            loop_func = self.assert_running_and_stop_if_canceled
        if not error:
            # Default error-message
            error = f'{func.__name__} failed'
        passed_args = ','.join([f'{k}={v}' for k,v in kwargs.items()])
        self._print(f'calling wait_for( {func.__qualname__}({passed_args}), limit={limit}, pause={pause} )... ', v=v, end='')
        n = loop_until(func, *args, **kwargs, error=error, pause=pause, limit=limit, loop_func=loop_func)
        if n<0:
            self._print(' ', tag='')    
            return False    
        self._print(str(n) + ' loops', v=3, tag='')
        if callable(log):
            self._print(log())
        return True


    #--------------------------------------------------------------------------------
    def wait_for_process_to_finish(self, v=1, limit=None, error=None, pause=None, loop_func=None, msg=None):      # runner
    #--------------------------------------------------------------------------------
        msg = msg or 'waiting for parent process to finish'
        self._print(msg, v=v)
        success = self.wait_for(self.parent.is_not_running, error=error, pause=pause, limit=limit, loop_func=loop_func)
        if not success:
            self._print('process did not finish within reasonable time and was killed', v=v)
            self.kill()
        else:
            self.parent = None
            self.children = []


    #--------------------------------------------------------------------------------
    def quit(self, v=1, loop_func=lambda:None):
    #--------------------------------------------------------------------------------
        self._print(f'\nquitting', v=v)
        self.resume()
        self.wait_for_process_to_finish(msg='waiting for process to quit', limit=10000, pause=0.01, loop_func=loop_func)
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
        if txt and v <= self.verbose:
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
    
