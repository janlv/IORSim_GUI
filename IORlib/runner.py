#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from subprocess import Popen, PIPE, STDOUT, call
import atexit
#import signal
#import os
#import sys
import psutil
from time import sleep
from pathlib import Path #, PurePath
from shutil import copy
#from struct import unpack
from .utils import loop_until, loop_until_2, list2str, tail_file, safeopen, Timer, silentdelete

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
        self._name = Path(root + '.' + ext)

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
            f.write(content)

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
    def __init__(self, name=None, case=None, cmd=None, pipe=False,
                 verbose=None, timer=False, runlog=None, ext_iface=None, ext_OK=None):           # runner
    #--------------------------------------------------------------------------------
        self.name = name
        self.case = case
        self.cmd = cmd
        self.log = safeopen( Path(case).parent / Path(name.lower()+'.log'), 'w' )
        atexit.register(self.log.close)
        self.runlog = runlog
        self.ext_iface = ext_iface
        self.ext_OK = ext_OK
        self.parent = None
        self.children = []
        self.pipe = pipe
        self.verbose = verbose
        self.echo = False
        if verbose > 2:
            self.echo = True
        self.timer = timer
        if self.timer:
            self.timer = Timer(name.lower())
        #self.is_killed = False
            
    # #-----------------------------------------------------------------------
    # def kill_func(self, t, step='days'):
    # #-----------------------------------------------------------------------
    #     if self.is_killed:
    #         raise SystemError('INFO ' + self.name + ' stopped after ' + str(t) + ' ' + step)
                       

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
        if self.pipe:
            self._print('starting in PIPE-mode', v=1)
            P = Popen(self.cmd, stdin=PIPE, stdout=self.log, stderr=STDOUT)
        else:
            self._print('trying to start {}'.format(self.cmd), v=1)
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
        atexit.register(self.kill)
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
    def suspend(self, check=False, v=1):                                     # runner
    #--------------------------------------------------------------------------------
        self._print('suspending {:s} ({:d})'.format(self.name, self.parent.pid), v=v)
        self.parent.suspend()
        if self.timer:
            self.timer.stop()
            #file.write('{:d}\t{:.3e}\n'.format(self.num_resume_called, time.time()-self.starttime))
        if check:
            loop_until(self.parent_has_stopped, error='{} did not stop'.format(self.parent.name()))
        
    #--------------------------------------------------------------------------------
    def resume(self, check=False, v=1):                                      # runner
    #--------------------------------------------------------------------------------
        self._print('resuming {:s} ({:d})'.format(self.name, self.parent.pid), v=v)
        self.parent.resume()
        if self.timer:
            self.timer.start()
        if check:
            loop_until(self.parent.is_running, error='{} is not running'.format(self.parent.name()))
        
    #--------------------------------------------------------------------------------
    def parent_is_not_running(self):                                         # runner
    #--------------------------------------------------------------------------------
        try:
            if not self.parent.is_running() or self.parent.status() == psutil.STATUS_ZOMBIE:
                return True
        except psutil.NoSuchProcess:
            return True
            
    #--------------------------------------------------------------------------------
    def is_running(self):                                                    # runner
    #--------------------------------------------------------------------------------
        try:
            if self.parent and self.parent.is_running() and self.parent.status() != psutil.STATUS_ZOMBIE:
                return True
        except psutil.NoSuchProcess:
            return False
            
    #--------------------------------------------------------------------------------
    def parent_has_stopped(self):                                            # runner
    #--------------------------------------------------------------------------------
        if self.parent.status() == psutil.STATUS_STOPPED: # T in top
            return True
                
    #--------------------------------------------------------------------------------
    def parent_is_sleeping(self):                                            # runner
    #--------------------------------------------------------------------------------
        if self.parent.status() == psutil.STATUS_SLEEPING: # S in top
            return True
                
    #--------------------------------------------------------------------------------
    def kill(self, v=2):                                                     # runner
    #--------------------------------------------------------------------------------
        # define local functions
        def name_pid(proc):
            return '\'{:s}\' ({:d})'.format(proc.name(), proc.pid)
        def terminate(proc):
            try:
                self._print('Terminating {:s}'.format(name_pid(proc)), v=v)
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
            #if all([p.is_running() and p.status()!=psutil.STATUS_ZOMBIE for p in self.procs]):
            if self.parent.is_running() and self.parent.status()!=psutil.STATUS_ZOMBIE: 
                return True
            raise SystemError('ERROR ' + self.name + ' is not running, status is ' + self.parent.status())
            #raise SystemError('assert_running: Process ' + self.name + ' is not running, status is ' + self.parent.status())
        except psutil.NoSuchProcess:
            raise SystemError('ERROR ' + self.name + ' stopped unexpectedly, check the log')
            #raise SystemError('assert_running: Process ' + self.name + ' has disappeared, log-file says:\n'
            #                  + (tail_file(self.log.name, nchars=300) or 'Log-file is missing') )
        
        
    #--------------------------------------------------------------------------------
    def wait_until_sleeping(self, v=1):                                      # runner
    #--------------------------------------------------------------------------------
        self._print('waiting for process to sleep', v=v)
        loop_until( self.parent_is_sleeping, error='{} did not sleep'.format(self.parent.name()) )
        #self._print('done', v=v)
        
    #--------------------------------------------------------------------------------
    def wait_for_process_to_finish(self, v=1, limit=None, error=None, sleep_sec=None, refresh_func=None, refresh=None, 
                                   assert_running=None, kill_func=None, kill_msg=None, progress=None, progress_limit=1):        # runner
    #--------------------------------------------------------------------------------
        self._print('waiting for process to finish', v=v)
        if not error:
            error = '{} did not quit'.format(self.parent.name())
        loop_until( self.parent_is_not_running, error=error,
                    sleep_sec=sleep_sec, limit=limit, wait_func=refresh_func, wait=refresh, assert_running=assert_running, 
                    kill_func=kill_func, kill_msg=kill_msg, progress=progress, progress_limit=progress_limit)
        self.parent = None
        
    #--------------------------------------------------------------------------------
    def wait_for_process_to_finish_2(self, v=1, limit=None, error=None, pause=None, loop_func=None):      # runner
    #--------------------------------------------------------------------------------
        self._print('waiting for process to finish', v=v)
        if not error:
            error = '{} did not quit'.format(self.parent.name())
        loop_until_2( self.parent_is_not_running, error=error, pause=pause, limit=limit, loop_func=loop_func)
        self.parent = None
        
    # #--------------------------------------------------------------------------------
    # def wait_for(self, func, *args, v=1, log_msg=None, error=None, limit=None, pause=None, loop_func=None, **kwargs):
    # #--------------------------------------------------------------------------------
    #     self._print('calling wait_for( ' + func.__qualname__ + ' )...', v=v, end='')
    #     try:
    #         n = loop_until_2(func, *args, **kwargs, limit=limit, error=error, pause=pause, loop_func=loop_func)
    #         self._print(' {:d} loops'.format(n), v=v, tag='')
    #         msg = None
    #         if callable(log_msg):
    #             msg = log_msg()
    #         elif log_msg:
    #             msg = log_msg
    #         if msg:
    #             self._print(msg)
    #     except SystemError as e:
    #         msg = str(e)
    #         if raise_error or msg.startswith('ERROR'):
    #             raise e
    #         else:
    #             n = msg.split('>')[-1].split()[0]
    #             runner._print(n + '  loops', v=3, tag='')            
    #             return False
    #     else:
    #         return True


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
    
