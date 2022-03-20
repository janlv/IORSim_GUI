
# -*- coding: utf-8 -*-

# Constants
CHILD_SEARCH_WAIT = 0.5    # Seconds to sleep during child process search
CHILD_SEARCH_LIMIT = 500   # Total number of iterations in child process search 
SUSPEND_TIMER_PRECICION = 0.1 # Precision of the delayed-suspend-timer in seconds

DEBUG = False

from datetime import datetime
from subprocess import DEVNULL, Popen, PIPE, STDOUT
import psutil
from shutil import which
from time import sleep
from pathlib import Path 
from shutil import copy
from .utils import loop_until, safeopen, Timer, silentdelete, timer_thread

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
    def raise_error(self):                                                  # Process
    #--------------------------------------------------------------------------------
        raise SystemError(f'ERROR {self._app_name} stopped unexpectedly, check the log')

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
    def is_running(self, raise_error=False):                                # Process
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
                self._error_func()
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
    def assert_running(self, raise_error=True):                             # Process
    #--------------------------------------------------------------------------------
        return self.is_running(raise_error=raise_error)


    #--------------------------------------------------------------------------------
    def get_children(self, raise_error=True, log=False, wait=CHILD_SEARCH_WAIT, limit=CHILD_SEARCH_LIMIT):      # Process
    #--------------------------------------------------------------------------------
        # looking for child-processes with a name that match the app_name 
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
            children = self._process.children(recursive=True)
            log is not False and log(children, v=3)
            # Stop if named child process is found
            if any([p.name().lower().startswith(name) for p in children]):
                #found = True
                time = wait*i
                break
        if time is None and raise_error:
            raise SystemError(f'Unable to find child process of {self._name} in {wait*limit:.1f} seconds, aborting...')
        return children, time


#====================================================================================
class Runner:                                                               # runner
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
                 keep_files=False, stop_children=True, keep_alive=False, **kwargs):           # runner
    #--------------------------------------------------------------------------------
        #print('runner.__init__: ',keep_alive, N,T,name,case,exe,cmd,ext_iface,ext_OK)
        self.reset_processes()
        self.name = name
        self.case = Path(case)
        self.exe = exe
        self.cmd = cmd
        self.log = safeopen( Path(case).parent / Path(name.lower()+'.log'), 'w' )
        self.runlog = runlog
        self.ext_iface = ext_iface
        self.ext_OK = ext_OK
        self.popen = None
        self.stop_children = stop_children
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
        self.keep_alive = keep_alive
        self.suspend_timer = None
        DEBUG and print(f'Creating {self}')

    #-----------------------------------------------------------------------
    def __str__(self):
    #-----------------------------------------------------------------------
        return f'<Runner(name={self.name}, cmd={self.cmd})>'

    #-----------------------------------------------------------------------
    def __del__(self):
    #-----------------------------------------------------------------------
        DEBUG and print(f'Deleting {self}')

    #-----------------------------------------------------------------------
    def set_time(self, time):
    #-----------------------------------------------------------------------
        self.T = int(time)

    #-----------------------------------------------------------------------
    def reset_processes(self):
    #-----------------------------------------------------------------------
        self.parent = None
        self.main = None
        self.children = []
        self.active = []


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
    def start(self, error_func=None):                                        # runner
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
        if self.keep_alive > 0:
            self.suspend_timer = timer_thread(limit=self.keep_alive, prec=SUSPEND_TIMER_PRECICION, func=self.suspend_active)


    #--------------------------------------------------------------------------------
    def set_processes(self, error_func=None):                                 # runner
    #--------------------------------------------------------------------------------
        if error_func is None:
            error_func = self.unexpected_stop_error
        # Parent process
        kwargs = {'app_name':self.name, 'error_func':error_func}
        self.parent = self.main = Process(psutil.Process(pid=self.popen.pid), **kwargs)
        self._print(f'Parent process : {self.parent.info()}, ')
        #self.parent.assert_running()
        # Child processes (if they exists)
        children, time = self.parent.get_children(log=self._print)
        self.children = [Process(c, **kwargs) for c in children]
        self._print('Child process' + (len(self.children)>1 and 'es' or '') + (time is not None and f' ({time:.1f} sec)' or '') + f' : {", ".join([p.info() for p in self.children])}')
        # Set active and main processes
        if self.children:
            self.main = self.children[-1]
        self.active = [self.parent]
        if self.stop_children:
            self.active = self.children + [self.parent]


    #--------------------------------------------------------------------------------
    def get_logfile(self):                                                   # runner
    #--------------------------------------------------------------------------------
        return self.log.name

    #--------------------------------------------------------------------------------
    def log_message(self, msg):                                              # runner
    #--------------------------------------------------------------------------------
        self._print(msg, v=1, end='')
        self._print(f' {", ".join([p.info() for p in self.active])}', v=2, end='', tag='')
        self._print('', v=1, tag='')

    #--------------------------------------------------------------------------------
    def suspend_active(self):                                                # runner
    #--------------------------------------------------------------------------------
        return all([p.suspend() for p in self.active])

    #--------------------------------------------------------------------------------
    def resume_active(self):                                                # runner
    #--------------------------------------------------------------------------------
        return all([p.resume() for p in self.active])


    #--------------------------------------------------------------------------------
    def suspend(self, check=False, v=1):                                     # runner
    #--------------------------------------------------------------------------------
        if self.suspend_timer:
            self.log_message('Delayed suspend')
            self.suspend_timer.start()
        else:
            self.log_message('Suspend')
            #[p.suspend() for p in self.active]
            self.suspend_active()
            if check:
                [self.wait_for(p.is_sleeping, limit=100) for p in self.active]
            self.print_process_status()
            self.timer and self.timer.stop()


    #--------------------------------------------------------------------------------
    def resume(self, check=False, v=1):                                      # runner
    #--------------------------------------------------------------------------------
        if self.suspend_timer and self.suspend_timer.cancel_if_alive():
            self.log_message(f'No resume (suspend delayed {self.suspend_timer.endtime():.0f} sec)')
        else:
            if self.suspend_timer and not self.suspend_timer.is_alive():
                self.log_message(f'Resume (delayed suspend expired by {self.suspend_timer.uptime():.0f} sec)')
            else:
                self.log_message('Resume')
            #[p.resume() for p in self.active]
            self.resume_active()
            if check:
                [self.wait_for(p.is_running, limit=100) for p in self.active]
            self.timer and self.timer.start()
        self.print_process_status()


    #--------------------------------------------------------------------------------
    def print_process_status(self, v=2):
    #--------------------------------------------------------------------------------
        self._print(', '.join( [str(p.current_status()) for p in self.active] ), v=v)


    #--------------------------------------------------------------------------------
    def print_suspend_errors(self, v=1):
    #--------------------------------------------------------------------------------
        errors = [p.suspend_errors() for p in self.active if p]
        text = ', '.join([e for e in errors if e])
        text and self._print(text, v=v)


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
    def assert_running_and_stop_if_canceled(self, raise_error=True):
    #--------------------------------------------------------------------------------
        #self.parent.assert_running(raise_error=raise_error)
        #self.main.assert_running(raise_error=raise_error)
        [p.assert_running(raise_error=raise_error) for p in self.active]
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
        if run_time is None:
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
    def wait_for(self, func, *args, limit=None, pause=0.01, v=2, error=None, raise_error=False, log=None, loop_func=None, **kwargs):
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
    def wait_for_process_to_finish(self, v=2, limit=None, pause=None, loop_func=None, msg=None):      # runner
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
    def close(self):
    #--------------------------------------------------------------------------------
        self.reset_processes()
        # Close log-file
        self.log.close()
        # Stop and delete the suspend-timer-thread
        self.suspend_timer and self.suspend_timer.close()
        self.suspend_timer = None # For garbage collector (__del__)
        # Delete interface-files 
        if self.ext_iface and not self.keep_files:
            self.interface_file('all').delete()


    #--------------------------------------------------------------------------------
    def quit(self, v=1, loop_func=lambda:None):
    #--------------------------------------------------------------------------------
        self._print('', tag='', v=v)
        self._print('Quitting', v=v)
        self.resume()
        self.wait_for_process_to_finish(msg='Waiting for process to quit', limit=6000, pause=0.01, loop_func=loop_func)
        self.close()
        self._print('Finished', v=v)


    #--------------------------------------------------------------------------------
    def kill(self, v=2):                                                     # runner
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


    # #@pass_KeyboardInterrupt
    # #--------------------------------------------------------------------------------
    # def kill_and_clean(self):
    # #--------------------------------------------------------------------------------
    #     self.kill()
    #     self.log.close()
    #     self.clean_up()

    # #--------------------------------------------------------------------------------
    # def clean_up(self):
    # #--------------------------------------------------------------------------------
    #     if self.ext_iface and not self.keep_files:
    #         self.interface_file('all').delete()

    
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
    def _print(self, txt, v=1, tag=True, flush=True, **kwargs):                         # runner
    #--------------------------------------------------------------------------------
        if v <= self.verbose:
            if tag is True:
                tag = f'{self.name}:'
            print(tag, txt, file=self.runlog, flush=flush, **kwargs)


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
    
