
# -*- coding: utf-8 -*-

from datetime import datetime
from subprocess import DEVNULL, Popen, PIPE, STDOUT
import psutil
from shutil import which
from time import sleep
from pathlib import Path 
from shutil import copy
from .utils import loop_until, safeopen, Timer, silentdelete

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
def ignore_permission_error(func):
#--------------------------------------------------------------------------------
    def inner(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except PermissionError as e:
            pass
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

#--------------------------------------------------------------------------------
def pass_KeyboardInterrupt(func):
#--------------------------------------------------------------------------------
    def inner(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except KeyboardInterrupt:
            pass
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

    @ignore_permission_error
    #--------------------------------------------------------------------------------
    def delete(self):
    #--------------------------------------------------------------------------------
        path = self._name.parent
        name = self._name.name
        for f in path.glob(name):
            #print('deleting',f)
            f.unlink()

    #@ignore_permission_error
    #--------------------------------------------------------------------------------
    def is_deleted(self):
    #--------------------------------------------------------------------------------
        # if not self._name.is_file():
        #     return True
        try:
            if not self._name.is_file():
                return True
        except PermissionError:
            return None

#====================================================================================
class Process:                                                              # Process
#====================================================================================
    
    #--------------------------------------------------------------------------------
    def __init__(self, proc, app_name=None, error_func=None):               # Process
    #--------------------------------------------------------------------------------
        self._process = proc
        self.pid = proc.pid
        self._name = proc.name()
        self.app_name = app_name
        self._suspend_errors = 0
        self._error_func = error_func or self.raise_error
        #self.name = proc.name()

    #--------------------------------------------------------------------------------
    def raise_error(self):                                                      # Process
    #--------------------------------------------------------------------------------
        raise SystemError(f'ERROR {self.app_name} stopped unexpectedly, check the log')

    #--------------------------------------------------------------------------------
    def process(self):                                                      # Process
    #--------------------------------------------------------------------------------
        return self._process

    #--------------------------------------------------------------------------------
    def kill(self):                                                         # Process
    #--------------------------------------------------------------------------------
        self._process.kill()

    #--------------------------------------------------------------------------------
    def name(self):                                                         # Process
    #--------------------------------------------------------------------------------
        return self._name

    #--------------------------------------------------------------------------------
    def name_pid(self):                                                     # Process
    #--------------------------------------------------------------------------------
        return f"\'{self._name}\' ({self.pid})"

    #--------------------------------------------------------------------------------
    def suspend(self):                                                      # Process
    #--------------------------------------------------------------------------------
        try:
            self._process.suspend()
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
            self._process.resume()
            return True
        except (psutil.AccessDenied, psutil.NoSuchProcess, ProcessLookupError):
            return False


    @ignore_process_error
    #--------------------------------------------------------------------------------
    def current_status(self):                                               # Process
    #--------------------------------------------------------------------------------
        return f'{self._process.name()} {self._process.status()}'
    

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
            if self._process.is_running() and self._process.status() != psutil.STATUS_ZOMBIE:
                return True
            if raise_error:
                raise SystemError(f'ERROR {self.app_name} is not running ({self._name} is {self._process.status()})')
        except (psutil.NoSuchProcess, ProcessLookupError):
            if raise_error:
                #msg = ', check the log'
                #if raise_error is not True:
                #    msg = raise_error
                #raise SystemError(f'ERROR {self.app_name} stopped unexpectedly' + msg)        
                self._error_func()
            else:
                return False
        except AttributeError:
            if raise_error:
                raise SystemError(f'ERROR {self.app_name} process is {self._process}')        
            else:
                return True
            
    #--------------------------------------------------------------------------------
    def is_not_running(self):                                               # Process
    #--------------------------------------------------------------------------------
        try:
            if not self._process or not self._process.is_running() or self._process.status() == psutil.STATUS_ZOMBIE:
                return True
        except (psutil.NoSuchProcess, ProcessLookupError):
            return True
            
    #--------------------------------------------------------------------------------
    def is_sleeping(self):                                                  # Process
    #--------------------------------------------------------------------------------
        try:
            if self._process.status() in (psutil.STATUS_SLEEPING, psutil.STATUS_STOPPED): 
                return True
            elif not self._process.is_running() or self._process.status() == psutil.STATUS_ZOMBIE:
                raise SystemError(f'ERROR Process {self.name()} disappeared while trying to sleep')
        except (psutil.NoSuchProcess, ProcessLookupError, AttributeError):
            raise SystemError(f'ERROR Process {self._process} disappeared while trying to sleep')
                
    #--------------------------------------------------------------------------------
    def assert_running(self, raise_error=True):                             # Process
    #--------------------------------------------------------------------------------
        return self.is_running(raise_error=raise_error)

    #--------------------------------------------------------------------------------
    def get_children(self, raise_error=True, log_file=None):                               # Process
    #--------------------------------------------------------------------------------
        # looking for child-processes with a name that match the app_name 
        if not self._process:
            if raise_error:
                raise SystemError('Parent-process missing, unable to look for child-processes')
            else:  
                return [] 
        name = self.app_name.lower()
        #print(self._process.name().lower(), name)
        if self._process.name().lower().startswith(name):
            return []
        log_file and print('child-search: ', file=log_file)
        for i in range(100):
            sleep(0.5)
            self.assert_running()
            children = self._process.children(recursive=True)
            log_file and print(children, file=log_file)
            #print(children)
            # Stop if named child process is found
            if any([p.name().lower().startswith(name) for p in children]):
                break
        return children


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
        self.popen = None
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
    def unexpected_stop_error(self):                                       # runner
    #--------------------------------------------------------------------------------
        raise SystemError(f'ERROR {self.name} stopped unexpectedly, check the log')

    #--------------------------------------------------------------------------------
    def start(self, error_func=None):                        # runner
    #--------------------------------------------------------------------------------
        self.starttime = datetime.now()
        if self.pipe:
            self._print(f"Starting in PIPE-mode", v=1)
            self.popen = Popen(self.cmd, stdin=PIPE, stdout=self.log, stderr=STDOUT)
            self.stdin = self.popen.stdin
        else:
            self._print(f"Starting \'{' '.join(self.cmd)}\'", v=1)
            self.popen = Popen(self.cmd, stdout=self.log, stderr=STDOUT)      
            #self.popen = Popen(self.cmd, stdin=DEVNULL, stdout=self.log, stderr=DEVNULL)      
        self.set_processes(error_func=error_func)

    #--------------------------------------------------------------------------------
    def set_processes(self, error_func=None):                                 # runner
    #--------------------------------------------------------------------------------
        if error_func is None:
            error_func = self.unexpected_stop_error
        # Parent process
        kwargs = {'app_name':self.name, 'error_func':error_func}
        self.parent = Process(psutil.Process(pid=self.popen.pid), **kwargs)
        self.parent.assert_running()
        self._print(f'Parent process: {self.parent.name_pid()}')
        # Child processes (if they exists)
        children = self.parent.get_children(log_file=self.runlog)
        self.children = [Process(c, **kwargs) for c in children]
        self._print(f'Child process{len(self.children)>1 and "es" or ""}: {", ".join([p.name_pid() for p in self.children])}')

    #--------------------------------------------------------------------------------
    def get_logfile(self):                                                   # runner
    #--------------------------------------------------------------------------------
        return self.log.name


    #--------------------------------------------------------------------------------
    def suspend(self, check=False, status=True, v=1):          # runner
    #--------------------------------------------------------------------------------
        self._print(f'Suspending {self.name} ({self.parent.pid})', v=v)
        # Suspend children
        if self.stop_children:
            [p.suspend() for p in self.children]
            if check:
                [self.wait_for(p.is_sleeping, limit=100) for p in self.children]
        # Suspend parent
        self.parent.suspend()
        if check:
            self.wait_for(self.parent.is_sleeping, limit=100)
        if status:
            self.print_process_status()
        self.timer and self.timer.stop()

    #--------------------------------------------------------------------------------
    def resume(self, check=False, status=True, v=1):                       # runner
    #--------------------------------------------------------------------------------
        self._print(f'Resuming {self.name} ({self.parent.pid})', v=v)
        # Resume parent
        self.parent.resume()
        if check:
            self.wait_for(self.parent.is_running)
        # Resume children
        [p.resume() for p in self.children]
        if check:
            [self.wait_for(p.is_running) for p in self.children]
        if status:
            self.print_process_status()
        self.timer and self.timer.start()


    #--------------------------------------------------------------------------------
    def kill(self):                                                     # runner
    #--------------------------------------------------------------------------------
        # terminate children before parent
        procs = self.children + (self.parent and [self.parent] or []) 
        try:
            for p in procs:
                self._print(f'Killing {p.name()}, ', end='')
                #p.proc.kill()
                p.kill()
                self._print('done', tag='')
        except (psutil.NoSuchProcess, ProcessLookupError):
                self._print('process already gone', tag='')            
        except psutil.AccessDenied:
                self._print('access denied!!!', tag='')            
        finally:
            self.children = []
            self.parent = None

    #--------------------------------------------------------------------------------
    def process_list(self):
    #--------------------------------------------------------------------------------
        return (self.parent and [self.parent] or []) + self.children

    #--------------------------------------------------------------------------------
    def print_process_status(self, v=1):
    #--------------------------------------------------------------------------------
        self._print(', '.join( [str(p.current_status()) for p in self.process_list()] ), v=v)

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
    def assert_running_and_stop_if_canceled(self, raise_error=True):
    #--------------------------------------------------------------------------------
        self.parent.assert_running(raise_error=raise_error)
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
    def wait_for(self, func, *args, limit=None, pause=0.01, v=3, error=None, raise_error=False, log=None, loop_func=None, **kwargs):
    #--------------------------------------------------------------------------------
        if not loop_func:
            # Default checks during loop
            loop_func = self.assert_running_and_stop_if_canceled
        passed_args = ','.join([f'{k}={v}' for k,v in kwargs.items()])
        self._print(f'Calling wait_for( {func.__qualname__}({passed_args}), limit={limit}, pause={pause} )... ', v=v, end='')
        n = loop_until(func, *args, pause=pause, limit=limit, loop_func=loop_func, **kwargs)
        if n<0:
            if raise_error:
                raise SystemError(error or f'wait_for({func.__qualname__}) reached loop-limit {limit}')
            self._print('loop limit reached!', tag='', v=v)
            return False    
        self._print(str(n) + ' loops', tag='', v=v)
        if callable(log):
            self._print(log())
        return True


    #--------------------------------------------------------------------------------
    def wait_for_process_to_finish(self, v=1, limit=None, pause=None, loop_func=None, msg=None):      # runner
    #--------------------------------------------------------------------------------
        msg = msg or 'waiting for parent process to finish'
        self._print(msg, v=v)
        success = self.wait_for(self.parent.is_not_running, raise_error=False, pause=pause, limit=limit, loop_func=loop_func)
        if not success:
            time = (limit or 0)*(pause or 0)/60
            self._print(' ', tag='')
            self._print(f'process did not finish within {time:.2f} minutes and will be killed', v=v)
            self._print([p.proc.name() for p in self.process_list()])
            self.kill()
        else:
            self.parent = None
            self.children = []


    #--------------------------------------------------------------------------------
    def quit(self, v=1, loop_func=lambda:None):
    #--------------------------------------------------------------------------------
        self._print(f' ', tag='', v=v)
        self._print('Quitting', v=v)
        self.resume()
        self.wait_for_process_to_finish(msg='waiting for process to quit', limit=6000, pause=0.01, loop_func=loop_func)
        self.log.close()

    #--------------------------------------------------------------------------------
    def clean_up(self):
    #--------------------------------------------------------------------------------
        if self.ext_iface and not self.keep_files:
            self.interface_file('all').delete()

    
    #@pass_KeyboardInterrupt
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
            print(tag, txt, file=self.runlog, flush=True, **kwargs)


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
    
