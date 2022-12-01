
# -*- coding: utf-8 -*-

# Constants
CHILD_SEARCH_WAIT = 0.5        # Seconds to sleep during child process search
CHILD_SEARCH_LIMIT = 500       # Total number of iterations in child process search 
SUSPEND_TIMER_PRECICION = 0.1  # Precision of the delayed-suspend-timer in seconds

DEBUG = False

from datetime import datetime
from subprocess import Popen, PIPE, STDOUT
import psutil
from shutil import SameFileError, which
from time import sleep
from pathlib import Path 
from shutil import copy
from .utils import loop_until, matches, safeopen, Timer, silentdelete, timer_thread

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
            return f'{args[0]._name} is missing'
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
    def __init__(self, *args, path=None, log=False) -> None:
    #--------------------------------------------------------------------------------
        args = [str(a) for a in args]  # Ensure Path is cast to str
        self._base = len(args)>=2 and ''.join(args[:2]) or '' 
        self._nr = len(args)>2 and (lambda x : f'{x:{args[2]}}') or (lambda x : '')
        self._path = path and Path(path) or self.__call__(0)._path
        self._log = log

    #--------------------------------------------------------------------------------
    def __repr__(self) -> str:
    #--------------------------------------------------------------------------------
        return f'<Control_file {self._path.name}>'

    #--------------------------------------------------------------------------------
    def __call__(self, n):
    #--------------------------------------------------------------------------------
        self._path = Path(self._base + self._nr(n))
        #print('__call__', self)
        return self

    #--------------------------------------------------------------------------------
    def glob(self):
    #--------------------------------------------------------------------------------
        if self._base == '':
            return ()
        nr = self._nr(0).replace('0','?')
        path = Path(self._base + nr)
        return (Control_file(path=f, log=self._log) for f in path.parent.glob(path.name))

    #--------------------------------------------------------------------------------
    def name(self):
    #--------------------------------------------------------------------------------
        return self._path.name
        
    #--------------------------------------------------------------------------------
    def path(self):
    #--------------------------------------------------------------------------------
        return self._path
        
    @catch_permission_error
    #--------------------------------------------------------------------------------
    def create(self):
    #--------------------------------------------------------------------------------
        self._log and self._log(f'Create empty {self}')
        self._path.touch()
    
    @catch_permission_error
    #--------------------------------------------------------------------------------
    def create_from(self, file=None, string=None, delete=False):
    #--------------------------------------------------------------------------------
        if file is None and string is None:
            raise SyntaxError("ERROR Both 'file' and 'string' argument missing in Control_file.create_from()!")
        file = file and Path(file)
        ### Create from file
        if file and file.is_file:
            self._log and self._log(f'Create {self} from file {file.name}')
            try:
                copy(file, self._path)
            except SameFileError:
                self._log and self._log(f'WARNING in {self}: trying to copy same files {file.name}!')
            if delete:
                silentdelete(file)
        ### Create from string
        elif string:
            self._log and self._log(f'Create {self} from text')
            self._path.write_text(string)

    @catch_permission_error
    #--------------------------------------------------------------------------------
    def append(self, string):
    #--------------------------------------------------------------------------------
        self._log and self._log(f'Append text to {self}')
        with open(self._path, 'a') as f:
            f.write(f'{string}\n')

    @ignore_permission_error
    #--------------------------------------------------------------------------------
    def delete(self):
    #--------------------------------------------------------------------------------
        if self._path.is_file():
            self._log and self._log(f'Delete {self}')
            self._path.unlink()

    #--------------------------------------------------------------------------------
    def delete_all(self):
    #--------------------------------------------------------------------------------
        [file.delete() for file in self.glob()]

    #--------------------------------------------------------------------------------
    def is_deleted(self):
    #--------------------------------------------------------------------------------
        try:
            return not self._path.is_file()
        except PermissionError:
            return False




#====================================================================================
class Process:                                                              # Process
#====================================================================================
    
    #--------------------------------------------------------------------------------
    def __init__(self, proc, app_name=None, error_func=None):               # Process
    #--------------------------------------------------------------------------------
        self._process = proc
        self._pid = proc.pid
        #self._user = proc.username()
        self._name = proc.name()
        self._app_name = app_name
        self._suspend_errors = 0
        self._error_func = error_func or self.raise_error
        #self.name = proc.name()
        DEBUG and print(f'Creating {self}')

    #--------------------------------------------------------------------------------
    def __str__(self):                                                      # Process
    #--------------------------------------------------------------------------------
        return f'<Process(name={self._name}, pid={self._pid}>'


    #--------------------------------------------------------------------------------
    def __del__(self):                                                      # Process
    #--------------------------------------------------------------------------------
        DEBUG and print(f'Deleting {self}')


    #--------------------------------------------------------------------------------
    def raise_error(self, log=None):                                        # Process
    #--------------------------------------------------------------------------------
        raise SystemError(f'ERROR {self._app_name} stopped unexpectedly' + (log and f', check {log} for details' or ''))

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

    # #--------------------------------------------------------------------------------
    # def name_pid(self):                                                     # Process
    # #--------------------------------------------------------------------------------
    #     return f"\'{self._name}\' ({self.pid}, {self._user})"

    #--------------------------------------------------------------------------------
    def info(self):                                                     # Process
    #--------------------------------------------------------------------------------
        #return f"\'{self._name}\' ({self._pid}, {self._user})"
        return f"\'{self._name}\' ({self._pid})"

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
    def is_running(self, raise_error=False, **kwargs):                                # Process
    #--------------------------------------------------------------------------------
        try:
            if self._process.is_running() and self._process.status() != psutil.STATUS_ZOMBIE:
                return True
            if raise_error:
                raise SystemError(f'ERROR {self._app_name} is not running ({self._name} is {self._process.status()})')
        except (psutil.NoSuchProcess, ProcessLookupError):
            if raise_error:
                #msg = ', check the log'
                #if raise_error is not True:
                #    msg = raise_error
                #raise SystemError(f'ERROR {self.app_name} stopped unexpectedly' + msg)        
                self._error_func(**kwargs)
            else:
                return False
        except AttributeError:
            if raise_error:
                raise SystemError(f'ERROR {self._app_name} process is {self._process}')        
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
    def assert_running(self, raise_error=True, **kwargs):                   # Process
    #--------------------------------------------------------------------------------
        return self.is_running(raise_error=raise_error, **kwargs)


    #--------------------------------------------------------------------------------
    def get_children(self, raise_error=True, log=False, wait=CHILD_SEARCH_WAIT, limit=CHILD_SEARCH_LIMIT):      # Process
    #--------------------------------------------------------------------------------
        ### Looking for child-processes with a name that match the app_name
        ### Only do search if app_name is different from the name of this process  
        if not self._process:
            if raise_error:
                raise SystemError('Parent-process missing, unable to look for child-processes')
            else:  
                return [], None
        name = self._app_name.lower()
        # Return if this is the main process
        if self._process.name().lower().startswith(name):
            return [], None
        #found = False
        time = None
        for i in range(limit):
            sleep(wait)
            if self.is_not_running():
                raise SystemError(f'ERROR {self.info()} disappeared while searching for child-processes!')
            children = self._process.children(recursive=True)
            log is not False and log(children and children or '  child-process search ...', v=3)
            # Stop if named child process is found
            if any(p.name().lower().startswith(name) for p in children):
                #found = True
                time = wait*i
                break
        if time is None and raise_error:
            raise SystemError(f'Unable to find child process of {self._name} in {wait*limit:.1f} seconds, aborting...')
        return children, time


#====================================================================================
class Runner:                                                               # Runner
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
    def __init__(self, T=0, n=0, t=0, name='', case='', exe='', cmd=None, pipe=False,
                 verbose=3, timer=None, runlog=None, ext_iface=(), ext_OK=(),
                 keep_files=False, stop_children=True, keep_alive=False, lognr=None, 
                 time_regex=None, **kwargs):           # Runner
    #--------------------------------------------------------------------------------
        #print('runner.__init__: ',keep_alive, N,T,name,case,exe,cmd,ext_iface,ext_OK)
        self.reset_processes()
        self.name = name
        self.case = Path(case)
        self.exe = exe
        self.cmd = cmd
        self.logname = self.case.parent/f'{name.lower()}{lognr and "_" or ""}{lognr or ""}.log'
        self.log = None
        self.runlog = runlog
        log4 = lambda x: self._print(x, v=4)
        self.interface_file = Control_file(self.case, *ext_iface, log=log4)
        self.OK_file = Control_file(self.case, *ext_OK, log=log4)
        self.popen = None
        self.stop_children = stop_children
        self.pipe = pipe
        self.verbose = verbose
        self.timer = timer
        if self.timer:
            self.timer = Timer(name.lower())
        self.keep_files = keep_files
        self.canceled = False
        self.t = t
        self.n = int(n)
        self.T = T   # Max time
        #self.N = int(N)   # Max number of steps
        self.starttime = None
        self.keep_alive = keep_alive
        self.suspend_timer = None
        self.time_regex = time_regex
        self.kwargs = kwargs
        self.stdin = None
        if DEBUG:
            print(f'Creating {self}')

    #--------------------------------------------------------------------------------
    def __str__(self):                                                       # Runner
    #--------------------------------------------------------------------------------
        return f'<Runner(name={self.name}, cmd={self.cmd}, verbose={self.verbose})>'

    #--------------------------------------------------------------------------------
    def __del__(self):                                                       # Runner
    #--------------------------------------------------------------------------------
        DEBUG and print(f'Deleting {self}')

    #--------------------------------------------------------------------------------
    def set_time(self, time):                                                # Runner
    #--------------------------------------------------------------------------------
        self.T = int(time)

    #--------------------------------------------------------------------------------
    def reset_processes(self):                                               # Runner
    #--------------------------------------------------------------------------------
        self.parent = None
        self.main = None
        self.children = []
        self.active = []


    #--------------------------------------------------------------------------------
    def check_input(self):                                                   # Runner
    #--------------------------------------------------------------------------------
        ### check if executables exist on the system
        if which(self.exe) is None:
            raise SystemError('WARNING Executable not found: ' + self.exe)
        return True


    #--------------------------------------------------------------------------------
    def unexpected_stop_error(self, **kwargs):                               # Runner
    #--------------------------------------------------------------------------------
        raise SystemError(f'ERROR {self.name} stopped unexpectedly' + (self.log and f', check {Path(self.log.name).name} for details' or '') )


    #--------------------------------------------------------------------------------
    def start(self, error_func=None):                                        # Runner
    #--------------------------------------------------------------------------------
        self.log = not self.kwargs.get('to_screen', False) and safeopen(self.logname, 'w') or None
        self.starttime = datetime.now()
        if self.pipe:
            self._print("Starting in PIPE-mode", v=1)
            self.popen = Popen(self.cmd, stdin=PIPE, stdout=self.log, stderr=STDOUT)
            self.stdin = self.popen.stdin
        else:
            self._print(f"Starting \'{' '.join(self.cmd)}\'", v=1)
            self.popen = Popen(self.cmd, stdout=self.log, stderr=STDOUT)      
            #self.popen = Popen(self.cmd, stdin=DEVNULL, stdout=self.log, stderr=DEVNULL)      
        self.set_processes(error_func=error_func)
        if self.keep_alive > 0:
            self.suspend_timer = timer_thread(limit=self.keep_alive, prec=SUSPEND_TIMER_PRECICION, func=self.suspend_active)


    #--------------------------------------------------------------------------------
    def set_processes(self, error_func=None):                                # Runner
    #--------------------------------------------------------------------------------
        if error_func is None:
            error_func = self.unexpected_stop_error
        # Parent process
        kwargs = {'app_name':self.name, 'error_func':error_func}
        self.parent = self.main = Process(psutil.Process(pid=self.popen.pid), **kwargs)
        self._print(f'Parent process : {self.parent.info()}, ')
        #self.parent.assert_running()
        # Child processes (if they exists)
        children, time = self.parent.get_children(log=self.verbose>3 and self._print)
        self.children = [Process(c, **kwargs) for c in children]
        self._print('Child process' + (len(self.children)>1 and 'es' or '') + (time is not None and f' ({time:.1f} sec)' or '') + f' : {", ".join([p.info() for p in self.children])}')
        # Set active and main processes
        if self.children:
            self.main = self.children[-1]
        self.active = [self.parent]
        if self.stop_children:
            self.active = self.children + [self.parent]


    #--------------------------------------------------------------------------------
    def get_logfile(self):                                                   # Runner
    #--------------------------------------------------------------------------------
        return self.log and self.log.name

    #--------------------------------------------------------------------------------
    def suspend_active(self):                                                # Runner
    #--------------------------------------------------------------------------------
        return all([p.suspend() for p in self.active])

    #--------------------------------------------------------------------------------
    def resume_active(self):                                                # Runner
    #--------------------------------------------------------------------------------
        return all([p.resume() for p in self.active])


    #--------------------------------------------------------------------------------
    def suspend(self, check=False, v=1):                                     # Runner
    #--------------------------------------------------------------------------------
        if self.keep_alive > 0:
            self._print('Delayed suspend', v=2)
            self.suspend_timer.start()
        elif self.keep_alive < 0:
            self._print('No suspend', v=2)
        else:
            self._print('Suspend', v=2)
            self.suspend_active()
            if check:
                [self.wait_for(p.is_sleeping, limit=100) for p in self.active]
            self.timer and self.timer.stop()
        self.print_process_status()


    #--------------------------------------------------------------------------------
    def resume(self, check=False, v=1):                                      # Runner
    #--------------------------------------------------------------------------------
        if self.keep_alive > 0 and self.suspend_timer.cancel_if_alive():
            self._print(f'No resume (suspend delayed {self.suspend_timer.endtime():.0f} sec)', v=2)
        elif self.keep_alive < 0:
            self._print('No resume (not suspended)', v=2)
        else:
            msg = 'Resume'
            if self.suspend_timer and not self.suspend_timer.is_alive():
                msg += f' (suspended {-self.suspend_timer.uptime():.0f} sec ago)'
            self._print(msg, v=2)
            self.resume_active()
            if check:
                [self.wait_for(p.is_running, limit=100) for p in self.active]
            self.timer and self.timer.start()
        self.print_process_status()


    #--------------------------------------------------------------------------------
    def print_process_status(self, v=2):                                     # Runner
    #--------------------------------------------------------------------------------
        self._print(', '.join( [str(p.current_status()) for p in self.active] ), v=v)


    #--------------------------------------------------------------------------------
    def print_suspend_errors(self, v=1):                                     # Runner
    #--------------------------------------------------------------------------------
        errors = [p.suspend_errors() for p in self.active if p]
        text = ', '.join([e for e in errors if e])
        text and self._print(text, v=v)


    #--------------------------------------------------------------------------------
    def time(self):                                                          # Runner
    #--------------------------------------------------------------------------------
        t = 0
        if self.log:
            self.log.flush() 
            match = matches(file=self.log.name, pattern=self.time_regex)
            time = [m.group(1) for m in match]
            t = time and time[-1] or 0           
        return float(t)


    #--------------------------------------------------------------------------------
    def stop_if_canceled(self, unit='days'):                                 # Runner
    #--------------------------------------------------------------------------------
        if self.canceled:
            self._print('', tag='')
            raise SystemError(f'INFO Run stopped after {self.time():.2f}'.rstrip('0').rstrip('.') + f' {unit}')    


    #--------------------------------------------------------------------------------
    def assert_running_and_stop_if_canceled(self, raise_error=True):         # Runner
    #--------------------------------------------------------------------------------
        #self.parent.assert_running(raise_error=raise_error)
        #self.main.assert_running(raise_error=raise_error)
        log = self.log and self.log.name
        [p.assert_running(raise_error=raise_error, log=log) for p in self.active]
        self.stop_if_canceled()


    #--------------------------------------------------------------------------------
    def run_time(self):                                                      # Runner
    #--------------------------------------------------------------------------------
        return datetime.now()-self.starttime


    #--------------------------------------------------------------------------------
    def complete_msg(self, run_time=None):                                   # Runner
    #--------------------------------------------------------------------------------
        if run_time is None:
            run_time = self.run_time()
        return 'INFO Simulation complete, run-time was ' + str(run_time).split('.')[0]


    #--------------------------------------------------------------------------------
    def get_time_and_stop_if_limit_reached(self):                          # Runner
    #--------------------------------------------------------------------------------
        time = self.time()
        # if time > self.T:
        if int(time) > int(self.T):
            #print(f'Time-limit reached, {time} > {self.T}!')
            raise SystemError(self.complete_msg())
        return time


    #--------------------------------------------------------------------------------
    def wait_for(self, func, *args, timer=False, limit=None, pause=0.01, v=2, error=None, raise_error=False, log=None, loop_func=None, **kwargs):
    #--------------------------------------------------------------------------------
        if timer:
            starttime = datetime.now()
        if not loop_func:
            # Default checks during loop
            loop_func = self.assert_running_and_stop_if_canceled
        passed_args = ','.join([f'{k}={v}' for k,v in kwargs.items()])
        self._print(f'Calling wait_for( {func.__qualname__}({passed_args}), limit={limit}, pause={pause} )... ', v=v, end='')
        n = loop_until(func, *args, pause=pause, limit=limit, loop_func=loop_func, **kwargs)
        time = timer and f' ({(datetime.now()-starttime).total_seconds():.2f} sec)' or ''
        if n<0:
            if raise_error:
                raise SystemError(error or f'wait_for({func.__qualname__}) reached loop-limit {limit}')
            self._print(f'loop limit reached!{time}' or '', tag='', v=v)
            return False    
        self._print(str(n) + f' loops{time}', tag='', v=v)
        if callable(log):
            self._print(log())
        return True


    #--------------------------------------------------------------------------------
    def wait_for_process_to_finish(self, v=2, limit=None, pause=None, loop_func=None, msg=None):      # Runner
    #--------------------------------------------------------------------------------
        msg = msg or 'Waiting for parent process to finish'
        self._print(msg, v=v)
        success = self.wait_for(self.parent.is_not_running, raise_error=False, pause=pause, limit=limit, loop_func=loop_func)
        if not success:
            time = (limit or 0)*(pause or 0)/60
            self._print('', tag='')
            self._print(f'process did not finish within {time:.2f} minutes and will be killed', v=v)
            self._print([p.proc.name() for p in self.active if p])
            self.kill()

    #--------------------------------------------------------------------------------
    def cancel(self):                                                        # Runner
    #--------------------------------------------------------------------------------
        self.canceled = True

    #--------------------------------------------------------------------------------
    def close(self):                                                         # Runner
    #--------------------------------------------------------------------------------
        self.reset_processes()
        # Close log-file
        self.log and self.log.close()
        self.log = None
        # Stop and delete the suspend-timer-thread
        self.suspend_timer and self.suspend_timer.close()
        self.suspend_timer = None # For garbage collector (__del__)
        # Delete interface-files 
        # if self.ext_iface and not self.keep_files:
        #     self.interface_file('all').delete()
        if not self.keep_files:
            self.interface_file.delete_all()
            self.OK_file.delete()


    #--------------------------------------------------------------------------------
    def quit(self, v=1, loop_func=lambda:None):                              # Runner
    #--------------------------------------------------------------------------------
        self._print('', tag='', v=v)
        self._print('Quitting', v=v)
        self.resume()
        self.wait_for_process_to_finish(msg='Waiting for process to quit', limit=6000, pause=0.01, loop_func=loop_func)
        self.close()
        self._print('Finished', v=v)


    #--------------------------------------------------------------------------------
    def kill(self, v=2):                                                     # Runner
    #--------------------------------------------------------------------------------
        # terminate children before parent
        #procs = self.children + (self.parent and [self.parent] or []) 
        for p in self.active:
            try:
                self._print(f'Killing {p.name()}...', end='', v=v)
                p.kill()
                self._print('done', tag='', v=v)
            except (psutil.NoSuchProcess, ProcessLookupError):
                self._print('process already gone', tag='', v=v)            
            except psutil.AccessDenied:
                self._print('access denied!!!', tag='', v=v)            
        self.close()

    
    #--------------------------------------------------------------------------------
    def write_to_stdin(self, i):                                             # Runner
    #--------------------------------------------------------------------------------
        if not self.pipe:
            raise SystemError('STDIN is not piped, unable to write. Aborting...')
        self._print(f'writing {i} to STDIN')
        inp = f'{i:d}\n'
        self.P.stdin.write(inp.encode())
        self.P.stdin.flush()

        
    #--------------------------------------------------------------------------------
    def _print(self, txt, v=1, tag=True, flush=True, **kwargs):              # Runner
    #--------------------------------------------------------------------------------
        if v <= self.verbose:
            if isinstance(txt, str):
                txt = [txt]
            txt = (str(t) for t in txt)
            if tag is True:
                txt = f'{self.name}: ' + f'\n{self.name}: '.join(txt)
            else:
                txt = '\n'.join(txt)
            #print(tag, txt, file=self.runlog, flush=flush, **kwargs)
            print(txt, file=self.runlog, flush=flush, **kwargs)


    #--------------------------------------------------------------------------------
    def _printerror(self, txt, **kwargs):                                    # Runner
    #--------------------------------------------------------------------------------
        print()
        print('  ERROR: ' + txt, **kwargs)
        print('', flush=True)


    #--------------------------------------------------------------------------------
    def _printwarning(self, txt, **kwargs):                                  # Runner
    #--------------------------------------------------------------------------------
        print()
        print('  WARNING: ' + txt, **kwargs)
        print('', flush=True)
    
