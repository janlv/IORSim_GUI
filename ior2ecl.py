#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import Counter, namedtuple
from itertools import accumulate, chain, dropwhile, takewhile
from locale import getpreferredencoding
from operator import itemgetter
from pathlib import Path
from sys import platform
from argparse import ArgumentParser
from datetime import datetime, timedelta
from time import sleep
from shutil import copy as shutil_copy 
from traceback import print_exc as trace_print_exc, format_exc as trace_format_exc
from re import compile as re_compile, MULTILINE
from os.path import relpath

from psutil import NoSuchProcess, __version__ as psutil_version

from IORlib.utils import (flatten, get_keyword, get_python_version, list2text, pairwise,
    print_dict, print_error, remove_comments, safeopen, Progress, silentdelete,
    delete_files_matching, tail_file)
from IORlib.runner import Runner
from IORlib.ECL import (FUNRST_file, DATA_file, File, RFT_file, UNRST_file, 
    UNSMRY_file, MSG_file, PRT_file)

__version__ = '2.32'
__author__ = 'Jan Ludvig Vinningland'

DEBUG = False

# Options
COPY_CHEMFILE = True
SCHEDULE_SKIP_EMPTY = False

# Constants
# Default sleep-time during file-flush checks. Too low value might lead to errors on some systems.
CHECK_PAUSE = 0.01
# Interface-file from IORSim with statements for next Eclipse run
IOR_SATNUM_FILE = 'satnum.dat'
# Signature from IORSim to mark flushed interface-file
IOR_SATNUM_ENDTAG = '-- IORSimX done.'
# Seconds to wait before Eclipse/IORSim is suspended (if option is on). Neg. value = never suspended
ECL_ALIVE_LIMIT = 90
IOR_ALIVE_LIMIT = -1
# Log-level of the ior2ecl.log
LOG_LEVEL_MAX = 4
LOG_LEVEL_MIN = 1
DEFAULT_LOG_LEVEL = 3
# Number of iterations before reducing number of expected RFT blocks
RFT_CHECK_ITER = 100
# To avoid re-merging merged UNRST-files, this file is created in the case-folder 
MERGE_OK_FILE = '.merge_OK'



#====================================================================================
class Eclipse(Runner):                                                      # eclipse
#====================================================================================
    ' Eclipse runner class '

    #--------------------------------------------------------------------------------
    def __init__(self, root=None, exe='eclrun', **kwargs):
    #--------------------------------------------------------------------------------
        #print('eclipse.__init__: ',root, exe, kwargs)
        root = str(root)        
        exe = str(exe)
        super().__init__(name='Eclipse', case=root, exe=exe, cmd=[exe, 'eclipse', root], 
                         time_regex=r'TIME=?\s+([0-9.]+)\s+DAYS', **kwargs)                        
        self.update = kwargs.get('update') or None
        self.unrst = UNRST_file(root, wait_func=self.wait_for, timer=self.verbose==LOG_LEVEL_MAX)
        self.rft = RFT_file(root, wait_func=self.wait_for, timer=self.verbose==LOG_LEVEL_MAX)
        self.unsmry = UNSMRY_file(root)
        self.msg = MSG_file(root)
        self.prt = PRT_file(root)
        self.data_file = DATA_file(root, check=True)
        self.is_iorsim = False
        self.is_eclipse = True

    #--------------------------------------------------------------------------------
    def time(self):                                                         # eclipse
    #--------------------------------------------------------------------------------
        return self.rft.last_day() or self.unrst.last_day() or super().time()

    #--------------------------------------------------------------------------------
    def delete_output_files(self):                                          # eclipse
    #--------------------------------------------------------------------------------
        ' Delete these output-files before starting Eclipse '
        file_ext = ('*UNRST','RFT','SMSPEC','UNSMRY','RTELOG','RTEMSG','MSG','session*','dbprtx.lock*')
        delete_files_matching( [f'{self.case}*.{ext}' for ext in file_ext] )
        delete_files_matching(self.case.parent/'fort??????')
        delete_files_matching(self.case.parent/'hostfile.*')


    #--------------------------------------------------------------------------------
    def unexpected_stop_error(self, **kwargs):                              # eclipse
    #--------------------------------------------------------------------------------
        error = 'unexpectedly' + (self.log and f', check {Path(self.log.name).name} for details' or '')
        ### Check for license failure
        if 'LICENSE FAILURE'.encode() in self.msg.binarydata():
            error = 'due to a license failure'
        raise SystemError(f'ERROR {self.name} stopped {error}')


    #--------------------------------------------------------------------------------
    def start(self, error_func=None):                                       # eclipse
    #--------------------------------------------------------------------------------
        if self.update:
            self.update.status(value=f'Starting {self.name}...')
        error_func = error_func or self.unexpected_stop_error
        super().start(error_func)
        if self.update:
            self.update.status(value=f'{self.name} running...')


#====================================================================================
class ForwardMixin:
#====================================================================================
    ' Common functions for forward runs '

    #--------------------------------------------------------------------------------
    def init_control_func(self, update=(), count=5):
    #--------------------------------------------------------------------------------
        ' Set up control for forward runs '
        self.update = update
        self.loop_count = 0
        #self.pause = pause
        self.count = count

    #--------------------------------------------------------------------------------
    def control_func(self):
    #--------------------------------------------------------------------------------
        ' Enable plotting, progress and stop during forward runs '
        self.stop_if_canceled()
        self.loop_count += 1
        if self.loop_count == self.count:
            self.loop_count = 0
            self.t = self.get_time_and_stop_if_limit_reached()
            for func in self.update:
                func(run=self)


#====================================================================================
class EclipseForward(ForwardMixin, Eclipse):                         # EclipseForward
#====================================================================================
    ' Eclipse forward runner'

    #--------------------------------------------------------------------------------
    def check_input(self):                                           # EclipseForward
    #--------------------------------------------------------------------------------
        super().check_input()
        ### Check root.DATA exists and that READDATA keyword is NOT present
        if 'READDATA' in self.data_file: #.data():
            raise SystemError('WARNING The current case cannot run in forward-mode: '+
                              'Eclipse input contains the READDATA keyword.')
        return True


#====================================================================================
class BackwardMixin:
#====================================================================================
    ' Common functions for backward runs '

    #--------------------------------------------------------------------------------
    def update_function(self, progress=True, plot=False):            # BackwardMixin
    #--------------------------------------------------------------------------------
        self.assert_running_and_stop_if_canceled()
        self.t = self.time()
        self.update.status(run=self)
        if progress:
            self.update.progress(run=self)
        if plot:
            self.update.plot()



#====================================================================================
class EclipseBackward(BackwardMixin, Eclipse):                      # EclipseBackward
#====================================================================================
    ' Eclipse backward mode runner '

    #--------------------------------------------------------------------------------
    def __init__(self, check_unrst=True, check_rft=True, keep_alive=False, schedule=None, **kwargs):
    #--------------------------------------------------------------------------------
        # super().__init__(ext_iface='I{:04d}', ext_OK='OK', keep_alive=keep_alive, **kwargs)
        super().__init__(ext_iface=('.I','04d'), ext_OK=('.OK',), keep_alive=keep_alive, **kwargs)
        self.tsteps = kwargs.get('tsteps') or self.data_file.get('TSTEP')
        self.delete_interface = kwargs.get('delete_interface') or True
        self.init_tsteps = len(self.tsteps) 
        self.check_unrst = check_unrst
        self.check_rft = check_rft
        self.schedule = schedule
        self.nwell = 0
        self.del_satnum = False


    #--------------------------------------------------------------------------------
    def check_input(self):                                             # EclipseBackward
    #--------------------------------------------------------------------------------
        def raise_error(msg):
            raise SystemError(f'ERROR To run the current case in backward-mode you need to {msg}')
        ### Check that root.DATA exists 
        super().check_input()
        if 'READDATA' not in self.data_file: #.data():
            raise_error(f"insert 'READDATA /' between 'TSTEP' and 'END' in {self.data_file}.")
        ### Check presence of RPTSOL RESTART>1
        regex = r"\bRPTSOL\b\s+[A-Z0-9=_'\s]*\bRESTART\b *= *[2-9]{1}"
        if not self.data_file.section('SOLUTION').search('RPTSOL', regex):
            raise_error(f"'RPTSOL \\n RESTART=2 /' is missing in the SOLUTION section of {self.data_file}.")
        return True


    #--------------------------------------------------------------------------------
    def start(self, error_func=None, restart=False):                  # EclipseBackward
    #--------------------------------------------------------------------------------
        # Start Eclipse in backward mode
        if self.n > 0 or self.t > 0:
            self._print(f'Starting at {self.t} days (step {self.n})')
        self.n += self.init_tsteps   # Use += and not = in case self.n is not 0 (RESTART option)
        self.interface_file.delete_all()
        self.interface_file(self.n).create()
        self.OK_file.delete()
        # Start Eclipse
        super().start()
        # Wait for flushed UNRST-file   
        nblocks = 1 + self.init_tsteps # Add 1 for 0'th SEQNUM
        for i in range(nblocks):
            if i > 0:
                self.update_function(progress=not restart, plot=True)
            self.unrst.check.data_saved(nblocks=1, pause=CHECK_PAUSE)
        # Get number of wells from UNRST-file
        self.nwell = self.unrst.get('nwell')[0][-1]
        # Wait for flushed RFT-file
        msg = self.rft.check.data_saved_maxmin(nblocks=nblocks*self.nwell, iter=RFT_CHECK_ITER, pause=CHECK_PAUSE)
        if msg:
            self._print(msg)
        while self.nwell < 1:
            ### Run Eclipse until at least one well is producing and the RFT-file is created
            self.schedule.update(tstep=self.T)
            self.run_one_step(self.schedule.ifacefile.file, start_stop=False, nwell=True)
            self._print(f' nwell = {self.nwell}')
            self.update_function(progress=True, plot=True)
            nblocks = 1
            self.init_tsteps += 1
            if self.t >= self.T:
                raise SystemError('ERROR Simulation stopped prematurely due to missing input to IORSim (missing RFT-file). Try increasing the number of days.')
        self.suspend()
        self.t = self.rft.check.data()[-1] or self.time()
        self._print(f'Days: {self.t}')


    #--------------------------------------------------------------------------------
    def run_one_step(self, satnum_file, log=True, start_stop=True, nwell=False):  # EclipseBackward
    #--------------------------------------------------------------------------------
        ' Advance Eclipse to next report step '
        self.interface_file(self.n).create_from(file=satnum_file, delete=self.del_satnum)
        self.OK_file.create()
        ### Create next interface-file to avoid Eclipse from reading END
        self.interface_file(self.n+1).create()
        if start_stop:
            self.resume()
        self.wait_for( self.OK_file.is_deleted, error=self.OK_file.name()+' not deleted' )
        if self.check_unrst:
            self.unrst.check.data_saved(nblocks=1, pause=CHECK_PAUSE) 
        if self.check_rft and self.rft.exists():
            nblocks = 1
            if nwell:
                self.nwell = nblocks = self.unrst.get('nwell')[0][-1]
            msg = self.rft.check.data_saved_maxmin(nblocks=nblocks, iter=RFT_CHECK_ITER, pause=CHECK_PAUSE)
            if msg:
                self._print(msg)
        if start_stop:
            self.suspend()
        if self.delete_interface:
            self.interface_file(self.n).delete()
        self.n += 1
        # self.t = (data := self.rft.check.data()) and data[-1] or self.time()
        # self.t = self.rft.last_day() or self.unrst.last_day() or self.time()
        self.t = self.time()
        if self.check_rft and self.rft.not_in_sync(self.t):
            self._print(f'WARNING Simulation time not in sync with RFT-time: {self.t}, {self.rft.check.data()}')
        if log:
            self._print(f' Date is {self.unrst.dates(N=-1).date()} ({self.t} days)')
        self._print(f'Days: {self.t}')
        #self._print(f'Days: log:{self.time()}, RFT:{(data:=self.rft.check.data()) and data[-1]}, UNRST:{self.unrst.get("time", N=-1)}')


    #--------------------------------------------------------------------------------
    def quit(self, v=1, loop_func=lambda:None):                     # EclipseBackward
    #--------------------------------------------------------------------------------
        self.print_suspend_errors()
        ### Append END to interface-file
        self.interface_file(self.n).create_from(string='END')
        self.OK_file.create()
        super().quit(v, loop_func)


#====================================================================================
class IORSim_input:                                                    # iorsim_input
#====================================================================================
    #================================================================================
    class keywords:
    #================================================================================
        # IORSim input-file keywords
        #   value == 0 : optional
        #   value > 0 : required
        #   value == 1 : required for all cases
        #   value == 2 : required for specie cases
        #   value == 3 : required for solution cases
        #   value == 4 : might be removed in new versions
        all_ = {'*WELLMODEL':0, '*TEMPERATURE':0, '*GRIDPLOT_WRITE':0, '*REACTING_SYSTEM':4, 
                '*TRACER_LGR':1, '*INTEGRATION':1,'*INTEGRATE_SPECIES':1, '*MODELTYPE':1, 
                '*SPECIES':2, '*SOLUTION':3, '*SATNUM':0, '*MODELTEMPLATE':1, '*TINIT':1, 
                '*COMP':3, '*CHEMFILE':0, '*CO2': 0, '*KVALWAT': 0, '*KVALOIL': 0, '*KVALGAS': 0, 
                '*MODELINSTANCE':1, '*WELLSPECIES':1, '*INJECTOR': 0, '*TIME': 0, '*OUTPUT':1,
                '*PRODUCER': 0, '*WELLPLOT_INTERVAL':1, '*PRINTLEVEL':1, '*END':1}
        keys = all_.keys()
        ignored = ('*GRIDPLOT_FILE', '*N_TRACER')
        required = tuple(k for k,v in all_.items() if v>0)
        optional = tuple(k for k,v in all_.items() if v==0)
        specie   = tuple(k for k,v in all_.items() if v>0 and v!=3)
        solution = tuple(k for k,v in all_.items() if v>0 and v!=2)
        specie_key = '*SPECIES'
        solution_key = '*SOLUTION'

    #--------------------------------------------------------------------------------
    def __init__(self, root, check=False, check_format=False):                        # iorsim_input
    #--------------------------------------------------------------------------------
        self.file = Path(root).with_suffix('.trcinp')
        self.check_format = check_format
        self.warnings = check and self.check() or ''


    #--------------------------------------------------------------------------------
    def check_keywords(self):                                          # iorsim_input
    #--------------------------------------------------------------------------------
        # Check if required keywords are used, and if the order is correct 
        def raise_error(error):
            raise SystemError(f'ERROR Error in IORSim input file: {error}')    
        text = remove_comments(self.file, comment='#')
        file_kw = [kw.upper() for kw in re_compile(r'(\*[A-Za-z_-]+)').findall(text)]
        # Remove repeated keywords, i.e. make unique list
        file_kw = list(dict.fromkeys(file_kw))
        # Is this a species- or a solutions-case
        if self.keywords.solution_key in file_kw:
            required_kw = self.keywords.solution
        elif self.keywords.specie_key in file_kw:
            required_kw = self.keywords.specie
        else:
            raise_error(f"it must contain one of the keywords '{self.keywords.specie_key}' or '{self.keywords.solution_key}'")
        # Check if required keyword is missing            
        missing = [kw for kw in required_kw if kw not in file_kw]
        if missing:
            pos = [required_kw.index(kw) for kw in missing]
            front = ['after '+required_kw[i-1] if i>0 else None for i in pos]
            back = ['before '+required_kw[i+1] if i<len(required_kw)-1 else None for i in pos]
            tips = [f"{required_kw[pos[i]]} ({front[i] or ''}{(', '+back[i]) or ''})" for i in range(len(pos))]
            raise_error('required keyword' + (len(missing)>1 and f's {list2text(tips)} are' or f' {tips[0]} is') + ' missing')
        # Remove ignored keywords
        file_kw = [kw for kw in file_kw if kw not in self.keywords.ignored]
        # Make ordered list of input-file keywords
        ordered_kw = [o for o in self.keywords.keys if o in file_kw]
        # Check keyword order
        for o,f in zip(ordered_kw, file_kw):
            if o != f:                        
                raise_error(f'expected keyword {o} but got {f}')    


    #--------------------------------------------------------------------------------
    def check(self, error_msg=''):                      # iorsim_input
    #--------------------------------------------------------------------------------
        msg = error_msg and error_msg+': ' or ''

        warn = ''
        ### Check if input-file exists
        if not self.file.is_file():
            raise SystemError(f'ERROR {msg}missing input file {self.file.name}')

        ### Check if included files exists
        if (missing := [f for f in self.include_files() if not f.is_file()]):
            raise SystemError(f"ERROR {msg}'{list2text([f.name for f in missing])}' included from {self.file.name} is missing in folder {missing[0].parent.resolve()}")

        ### Check if tstart == 0
        inte = get_keyword(self.file, '\*INTEGRATION', end='\*')
        if inte and (tstart := inte[0][0]) > 0:
            warn += f'WARNING The IORSim start-time should be 0 but is currently {tstart}. Update the *INTEGRATION keyword in {self.file.name} to avoid sync problems.'

        ### Check if required keywords are used, and if the order is correct 
        self.check_format and self.check_keywords()

        return warn


    #--------------------------------------------------------------------------------
    def include_files(self):
    #--------------------------------------------------------------------------------
        '''
        Return full path to files included in the IORSim .trcinp-file
        '''
        parent = self.file.parent
        files = flatten(get_keyword(self.file, '\*CHEMFILE', end='\*', comment='#'))
        # Use negative lookahead (?!) to ignore commented lines
        regex = re_compile(rb'^(?!#)\s*add_species[\s"\']*(.*?)[\s"\']*$', flags=MULTILINE)
        for file in set(files):
            yield parent/file
            for match in regex.finditer(File(parent/file,'').binarydata()):
                yield parent/match.group(1).decode()

        #return set(self.file.parent/Path(f) for f in files)


#====================================================================================
class Iorsim(Runner):                                                        # iorsim
#====================================================================================

    #--------------------------------------------------------------------------------
    def __init__(self, root=None, exe='IORSimX', args='', relative_root=True, **kwargs):     # iorsim
    #--------------------------------------------------------------------------------
        #print('iorsim.__init__: ',root, exe, args, kwargs)
        exe = str(exe)
        if '.exe' not in exe and 'win' in platform:
            exe += '.exe'
        # IORSim only accepts root relative to the current directory
        abs_root = str(root)
        if relative_root:
            root = Path(relpath(root))
        else:
            root = Path(root).absolute()
        cmd = [exe, str(root)] + args.split()
        super().__init__(name='IORSim', case=root, exe=exe, cmd=cmd, time_regex=r'\bTime\b:\s+([0-9.e+-]+)', **kwargs)
        self.update = kwargs.get('update') or None
        self.funrst = FUNRST_file(abs_root+'_IORSim_PLOT')
        self.unrst = UNRST_file(self.funrst.file, end='SATNUM')
        self.inputfile = IORSim_input(root, check=True, check_format=kwargs.get('check_input_kw') or False)
        (warn:=self.inputfile.warnings) and self.update and self.update.message(warn)
        self.is_iorsim = True
        self.is_eclipse = False
        self.copied_chemfiles = []

    #--------------------------------------------------------------------------------
    def time(self):                                                         # iorsim
    #--------------------------------------------------------------------------------
        time = None
        ### Find most recently modified file
        files = sorted(((f, f.stat().st_size) for f in self.case.parent.glob('*.trcconc')), key=itemgetter(1))
        file = files and files[-1][0] or None
        #file = next(self.case.parent.glob('*.trcconc'), None)
        if line := next(tail_file(file, n=1), None):
            line = line.strip()   # Remove leading and trailing space
            time = line and line.split()[0] 
            time = time and not time.startswith('#') and float(time)       
        return time or super().time()


    #--------------------------------------------------------------------------------
    def start(self):                                                         # iorsim
    #--------------------------------------------------------------------------------
        self.update and self.update.status(value=f'Starting {self.name}...')
        ### Copy chem-files to working dir 
        if COPY_CHEMFILE:
            for file in self.inputfile.include_files():
                dest = Path.cwd()/file.name
                if not dest.exists(): # and not dest.samefile(file):
                    # print(f'{file} -> {dest}')
                    shutil_copy(file, dest)
                    self.copied_chemfiles.append(dest)
        ### Check if the necessary Eclipse output files exist 
        files = [self.case.with_suffix(ext) for ext in ('.UNRST', '.RFT', '.EGRID', '.INIT')]
        missing = [f.name for f in files if not f.is_file()]
        if missing:
            raise SystemError(f'ERROR Unable to start IORSim: Eclipse output file {", ".join(missing)} is missing')
        super().start()
        self.update and self.update.status(value=f'{self.name} running...')


    #--------------------------------------------------------------------------------
    def delete_output_files(self, raise_error=False):                        # iorsim
    #--------------------------------------------------------------------------------
        ### Delete old output files before starting new run
        case = str(self.case)
        delete_files_matching(case+'*.trcconc', raise_error=raise_error)
        delete_files_matching(case+'*.trcprd', raise_error=raise_error)
        silentdelete(self.funrst.file, self.unrst.file)


    #--------------------------------------------------------------------------------
    def close(self):                                                         # iorsim
    #--------------------------------------------------------------------------------
        super().close()
        ### Delete chem-files copied to working directory
        silentdelete(*self.copied_chemfiles)



#====================================================================================
class Ior_forward(ForwardMixin, Iorsim):                               # ior_forward
#====================================================================================
    pass

#====================================================================================
class Ior_backward(BackwardMixin, Iorsim):                             # ior_backward
#====================================================================================
    #--------------------------------------------------------------------------------
    def __init__(self, keep_alive=False, schedule=None, **kwargs):
    #--------------------------------------------------------------------------------
        #keep_alive = keep_alive and IOR_ALIVE_LIMIT or False
        # super().__init__(args='-readdata', ext_iface='IORSimI{:04d}', ext_OK='IORSimOK', keep_alive=keep_alive, **kwargs)
        super().__init__(args='-readdata', ext_iface=('.IORSimI','04d'), ext_OK=('.IORSimOK',), keep_alive=keep_alive, **kwargs)
        self.tsteps = kwargs.get('tsteps') or DATA_file(self.case).tsteps()
        self.delete_interface = kwargs.get('delete_interface') or True
        self.init_tsteps = len(self.tsteps)
        #self.satnum = DATA_file(IOR_SATNUM_FILE, '', reread=True)   # Output-file from IORSim, read by Eclipse as an interface-file
        self.satnum = DATA_file(IOR_SATNUM_FILE)   # Output-file from IORSim, read by Eclipse as an interface-file
        self.endtag = IOR_SATNUM_ENDTAG
        self.schedule = schedule


    #--------------------------------------------------------------------------------
    def satnum_flushed(self):                                          # ior_backward
    #--------------------------------------------------------------------------------
        if not self.satnum.is_file() or self.satnum.size() < len(self.endtag)+3:
            return False
        if self.endtag.encode() in self.satnum.binarydata():
            return True
        return False


    #--------------------------------------------------------------------------------
    def satnum_dist(self, echo=False):                                 # ior_backward
    #--------------------------------------------------------------------------------
        '''
        Return the distribution of SATNUM numbers as a dict
        '''
        lines = remove_comments(self.satnum.file, comment='--') 
        values = re_compile(r'SATNUM\s+([0-9\s]+)').findall(lines) 
        if values:
            values = [int(v) for v in values[0].split('\n') if v.strip()]
            count = Counter(values)
            dist = {k:count[k] for k in sorted(count.keys())}
            if echo:
                print( f'n: {self.n}, SATNUM: ' + ', '.join([f'{k}:{v}' for k,v in dist.items()]) )
            else:
                return dist


    #--------------------------------------------------------------------------------
    def start(self, restart=False, tsteps=None):                     # ior_backward
    #--------------------------------------------------------------------------------
        # Start IORSim backward run
        self.n = 0 
        self.interface_file.delete_all()
        if tsteps is None:
            tsteps = self.init_tsteps
        self.run_steps(1+tsteps, start=True)


    #--------------------------------------------------------------------------------
    def run_one_step(self):                                            # ior_backward
    #--------------------------------------------------------------------------------
        self.run_steps(1)
        #self.satnum_dist(echo=True)


    #--------------------------------------------------------------------------------
    def run_steps(self, N, start=False, log=True):                     # ior_backward
    #--------------------------------------------------------------------------------
        ### run IORSim
        for n in range(N):
            if n > 0:
                self.update_function(progress=True, plot=True)
            self.n += 1
            self.interface_file(self.n).create()
            self.OK_file.create()
            # silentdelete(self.satnum)
            self.satnum.delete()
            if n == 0:
                if start:
                    super().start()
                else:
                    self.resume()
            self.wait_for( self.OK_file.is_deleted, error=self.OK_file.name()+' not deleted')
            self.t = self.time()
        self.wait_for(self.satnum_flushed, pause=CHECK_PAUSE)
        #warn_empty_file(self.satnum, comment='--')
        if self.satnum.is_empty():
            self.update and self.update.message(f'WARNING {self.satnum} is empty!')
        self.suspend()
        if self.delete_interface:
            [self.interface_file(self.n-n).delete() for n in range(N)] 
#        self.t = self.time()
        if log:
            self._print(f' {self.t:.3f}/{self.T} days')


    #--------------------------------------------------------------------------------
    def quit(self, v=1, loop_func=lambda:None):                        # ior_backward
    #--------------------------------------------------------------------------------
        self.interface_file(self.n+1).append('Quit')
        self.OK_file.create()
        super().quit(v, loop_func)


#====================================================================================
class Schedule:
#====================================================================================
    #--------------------------------------------------------------------------------
    def __init__(self, case, T=0, init_days=0, start=None, ext='.SCH', comment='--', 
                 interface_file=None, skip_empty=False): #, end='/', tag='TSTEP'):
    #--------------------------------------------------------------------------------
        '''
        Create schedule from a .SCH-file if it exists. 
        The TSTEP in the satnum-file (created by IORSim) is modified to 
        ensure the next report time coincides with the next schedule-events.  
        The schedule is also used to ensure that the simulation ends at the right time
        by adding the end-time twice at the end of the schedule. 
        If a .SCH-file is present, the first entry in the schedule is the start-time.  

        The schedule is a list of lists: [[start-time, ''],[days, 'KEYWORD'],[end-time, '']]
        '''
        self.case = Path(case)
        self.skip_empty = skip_empty
        self.comment = comment
        #self.ifacefile = interface_file and DATA_file(interface_file.file, reread=True) or None
        self.ifacefile = interface_file and DATA_file(interface_file.file, sections=False) or None
        self.days = init_days 
        self.start = start
        self.tstep = 0
        self._schedule = ()
        self.end = 0
        ### Ignore case in file extension
        #self.file = is_file_ignore_suffix_case( self.case.with_suffix(ext) )
        self.file = DATA_file(self.case.with_suffix(ext), sections=False, ignore_case=True)
        if self.file.exists():
            self._schedule = self.get_schedule()
            self.end = (len(self._schedule) > 0) and self._schedule[-1][0] or 0
        else:
            self.file = None
        ### Add simulation end time 
        self.insert(days=T, remove=True)
        DEBUG and print(f'Creating {self}')

    #--------------------------------------------------------------------------------
    def __repr__(self):                                                     # Schedule
    #--------------------------------------------------------------------------------
        return f'<Schedule(file={self.file}, start={self.start}, end={self.end}, init_days={self.days}, length={len(self._schedule)})>'

    #--------------------------------------------------------------------------------
    def __str__(self):                                                     # Schedule
    #--------------------------------------------------------------------------------
        return f'{self.file and self.file.name}'

    #--------------------------------------------------------------------------------
    def __del__(self):                                                     # Schedule
    #--------------------------------------------------------------------------------
        DEBUG and print(f'Deleting {self}')

    #--------------------------------------------------------------------------------
    def to_file(self, name):                                               # Schedule
    #--------------------------------------------------------------------------------
        ' Write schedule to file '
        with open(self.case.parent/name, 'w', encoding=getpreferredencoding()) as file:
            file.write(''.join([str(s) for s in flatten(self._schedule)]))

    # #--------------------------------------------------------------------------------
    # def next_tstep(self):                                                  # Schedule
    # #--------------------------------------------------------------------------------
    #     return self._schedule[0][0] - self.days


    #--------------------------------------------------------------------------------
    def now(self):                                                         # Schedule
    #--------------------------------------------------------------------------------
        return self.start+timedelta(days=self.days) #).strftime('%d %b %Y')

    #--------------------------------------------------------------------------------
    def info(self):                                                        # Schedule
    #--------------------------------------------------------------------------------
        s =   'Schedule\n'
        s += f'  file   : {self.file}\n'
        s += f'  start  : {self.start}\n'
        s += f'  days   : {self.days}\n'
        s += f'  length : {len(self._schedule)}\n'
        return s


    #--------------------------------------------------------------------------------
    def insert(self, days=0, action='', remove=False):      # Schedule
    #--------------------------------------------------------------------------------
        action = action and action+'\n' or ''
        lt_days = lambda x: x[0] < days
        before = takewhile(lt_days, self._schedule)
        after = not remove and dropwhile(lt_days, self._schedule) or ()
        inset = ((float(days), action),)
        self._schedule = list( chain(before, inset, after) )


    #--------------------------------------------------------------------------------
    def get_schedule(self):                                                # Schedule
    #--------------------------------------------------------------------------------
        '''
        Return a list of tuples with days at index 0 and actions at index 1, such as:
        schedule = [(2.0, "WCONHIST \r\n    'P-15P'      'OPEN' "), (9.0, "WCONHIST")]
        '''
        ### Split, accumulate tsteps, and then zip tsteps and pos  
        tstep, pos = zip(*self.file.tsteps(start=self.start, pos=True)) # + [('',(len(self.file),0),)]
        tstep_pos = list(zip(accumulate(tstep), pos))
        ### Append end to make pairwise pick up the last entry
        tstep_pos.append(tstep_pos[-1])
        #filedata = self.file.data()
        tstep_act = ((tstep, self.file.data[a:b]) for (tstep,(_,a)), (_,(b,_)) in pairwise(tstep_pos))
        if self.skip_empty:
            tstep_act = (x for x in tstep_act if x[1])
        return list(tstep_act)


    #--------------------------------------------------------------------------------
    def append(self, action=None, tstep=None):                             # Schedule
    #--------------------------------------------------------------------------------
        '''
        line : line number of TSTEP
        '''
        #print(f'append(action={action}, tstep={tstep})')
        if action is None and tstep is None:
            return
        if tstep == 0:
            raise SystemError(f'ERROR Schedule gave TSTEP 0 at {self.days} days, simulation stopped. Check {self.file}')
        new_action_and_tstep = (action and action or '') + f'TSTEP\n{tstep} /\n'
        self.ifacefile.replace_keyword('TSTEP', new_action_and_tstep)


    #--------------------------------------------------------------------------------
    def check(self):                                                       # Schedule
    #--------------------------------------------------------------------------------
        # just a check...
        print(f'{self.days}, {self.start+timedelta(days=self.days)}')
        with open(self.ifacefile.file, 'r') as f:
            lines = f.readlines()
        print(self.ifacefile.file, ': ', ''.join(lines[-10:]))
        print('Schedule:')
        print(self._schedule)


    #--------------------------------------------------------------------------------
    def update(self, tstep=None):                                          # Schedule
    #--------------------------------------------------------------------------------        
        action = new_tstep = None
        ### Update days from previous step
        self.days += self.tstep
        ### Append action if time is right
        if self.days >= self._schedule[0][0]:
            action = self._schedule.pop(0)[1]
        if tstep is None:
            ### Get tstep for next step from satnum.dat (written by IORSim)
            self.tstep = new_tstep = self.ifacefile.get('TSTEP', raise_error=True)[0]
        else:
            ### Write given tstep to empty satnum.dat (IORSim is not yet running) 
            self.ifacefile.file.write_text(f'TSTEP\n{tstep} /\n')
            self.tstep = new_tstep = tstep
        # print(f'START: tstep:{self.tstep}, days:{self.days}, schedule:{self._schedule[:2]}')
        ### Check arrival of next event and adjust tstep if neccessary
        if self._schedule and self.days + self.tstep + 1e-8 > self._schedule[0][0]:
            self.tstep = new_tstep = self._schedule[0][0] - self.days
        self.append(action=action, tstep=new_tstep)
        #self.check()
        # print(f'END: tstep:{self.tstep}, days:{self.days}, action:{action}, schedule:{self._schedule[:2]}')
        return self.days



#====================================================================================
class Simulation:                                                        # Simulation
#====================================================================================
    #--------------------------------------------------------------------------------
    def __init__(self, mode=None, root=None, pause=0, run_names=(), to_screen=False,
                 convert=True, merge=True, del_convert=False, del_merge=False, delete=False,
                 status=lambda **x:None, progress=lambda **x:None, plot=lambda **x:None,
                 message=lambda **x:None, **kwargs):
    #--------------------------------------------------------------------------------
        #print('mode',mode,'root',root,'pause',pause,'runs',runs,'to_screen',to_screen,
        #      'convert',convert,'merge',merge,'del_merge',del_merge,'del_convert',del_convert,
        #      'status',status,'progress',progress,'plot',plot,'kwargs',kwargs)
        self.logname = 'ior2ecl'
        self.root = Path(root).with_suffix('').resolve()
        self.ECL_inp = DATA_file(self.root, check=False)
        self.merge_OK = self.root.with_name(MERGE_OK_FILE)
        self.update = namedtuple('update',['status','progress','plot','message'])(status, progress, plot, message)
        self.pause = pause
        if delete:
            del_convert = del_merge = True
        self.output = namedtuple('output',['convert','merge','del_convert','del_merge'])(convert, merge, del_convert, del_merge)
        self.runlog = None
        if self.root and not to_screen:
            lognr = kwargs.get('lognr')
            self.runlog = safeopen(self.root.parent/f'{self.logname}{lognr and "_" or ""}{lognr or ""}.log', 'w')
        self.print2log = lambda txt, **kwargs: print(txt, file=self.runlog, flush=True, **kwargs)
        self.current_run = None
        self.run_names = run_names
        self.runs = ()
        self.run_sim = None
        self.ior = self.ecl = None
        self.T = 0
        self.tsteps = 0
        self.init_days = 0
        self.mode = mode
        self.schedule = None
        self.start = None
        self.restart = False
        self.restart_file = None
        self.restart_step = self.restart_days = 0
        self.skiprest = False
        kwargs.update({'root':str(self.root), 'runlog':self.runlog, 'update':self.update, 'to_screen':to_screen})
        self.kwargs = kwargs


    #--------------------------------------------------------------------------------
    def prepare(self):                                                     # Simulation
    #--------------------------------------------------------------------------------
        if not self.run_sim and self.root:
            try:
                self.run_sim = self.init_runs()
            except SystemError as error:
                self.update.message(f'{error}')
                #self.update.status(value=f'{e}')


    #--------------------------------------------------------------------------------
    def close(self):                                                     # Simulation
    #--------------------------------------------------------------------------------
        self.runs = self.run_names = ()
        self.ior = self.ecl = self.current_run = self.schedule = None
        if self.runlog:
            self.runlog.close()


    #--------------------------------------------------------------------------------
    def versions(self):                                                  # Simulation
    #--------------------------------------------------------------------------------
        txt = f'Python: {get_python_version()}\n'
        txt += f'psutil: {psutil_version}\n'
        txt += f'Application: {__version__}\n'
        return txt


    #--------------------------------------------------------------------------------
    def ready(self):                                                     # Simulation
    #--------------------------------------------------------------------------------
        return bool(self.run_sim)


    #--------------------------------------------------------------------------------
    def init_runs(self):                                                 # Simulation
    #--------------------------------------------------------------------------------
        'Read Eclipse and IORSim input files, run the init_func, and return the run_func'

        self.restart_days = 0
        ### Check if this is a restart-run
        file, step = self.ECL_inp.get('RESTART')
        if file and step:
            ### Get time and step from the restart-file
            self.restart_file = UNRST_file(file)
            if not self.restart_file.is_file():
                raise SystemError(f'ERROR Restart file {self.restart_file.file.relative_to(Path.cwd())} is missing')
            self.restart_step = step
            time, n = self.restart_file.get('time', 'step', stop=('step', step))
            if step > n[-1] or not step in n: 
                new_step = min(n, key=lambda x:abs(x-step))
                raise SystemError(f'ERROR Error in the Eclipse input-file ({self.ECL_inp}): Unable to restart from step {step}, use {new_step} instead')
            self.restart_days = time[n.index(step)]
            self.restart = True
        ### Simulation start date given by first entry of restart-file (UNRST-file) or START keyword of DATA-file
        self.start = self.restart_file and self.restart_file.dates(N=1) or self.ECL_inp.get('START')[0]
        self.mode = self.mode or (('READDATA' in self.ECL_inp) and 'backward' or 'forward')
        init_func = {'backward':self.init_backward_run, 'forward': self.init_forward_run}[self.mode]
        run_func  = {'backward':self.backward,          'forward': self.forward}[self.mode]
        check_ok = False
        self.runs = init_func(**self.kwargs)
        ### Check input
        check_ok = self.runs and all(run.check_input() for run in self.runs)
        return check_ok and run_func

    #--------------------------------------------------------------------------------
    def init_forward_run(self, iorexe=None, eclexe=None, **kwargs): # Simulation
    #--------------------------------------------------------------------------------
        start = self.start + timedelta(days=self.restart_days)
        self.tsteps = self.ECL_inp.tsteps(start=start, skiprest='SKIPREST' in self.ECL_inp)
        self.init_days = sum(self.tsteps) + self.restart_days
        self.T = self.init_days
        kwargs.update({'T':self.T})
        if not self.run_names:
            self.run_names = ('eclipse','iorsim')
        for name in self.run_names:
            if name=='eclipse':
                self.ecl = EclipseForward(exe=eclexe, **kwargs)
            if name=='iorsim':
                self.ior = Ior_forward(exe=iorexe, **kwargs)
        return [run for run in (self.ecl, self.ior) if run]


    #--------------------------------------------------------------------------------
    def init_backward_run(self, iorexe=None, eclexe=None, ior_keep_alive=False,
                          ecl_keep_alive=False, time=0, **kwargs):       # Simulation
    #--------------------------------------------------------------------------------
        self.tsteps = self.ECL_inp.tsteps()
        self.init_days = sum(self.tsteps)
        if time > self.init_days:
            self.T = time
        else:
            self.T = self.init_days + 1
            self.update.message(text=f'INFO Simulation time increased to advance past the READDATA keyword in {self.ECL_inp}')
        #self.T = time 
        kwargs.update({'T':self.T, 'tsteps':self.tsteps})
        # Init runs
        self.ecl = EclipseBackward(exe=eclexe, keep_alive=ecl_keep_alive, n=self.restart_step, t=self.restart_days, **kwargs)
        self.ior = Ior_backward(exe=iorexe, keep_alive=ior_keep_alive, **kwargs)
        # Simulation start date given by first entry of restart-file (UNRST-file) or START keyword of DATA-file
        #start = self.restart_file and self.restart_file.dates(N=1) or self.ECL_inp.get('START')[0]
        # Set up schedule of commands to pass to satnum-file
        self.schedule = Schedule(self.root, T=self.T, start=self.start, init_days=self.init_days, interface_file=self.ior.satnum, 
                                 skip_empty=self.kwargs.get('skip_empty', False))
        self.ecl.schedule = self.ior.schedule = self.schedule
        return [self.ecl, self.ior]


    #--------------------------------------------------------------------------------
    def forward(self):                                                   # Simulation
    #--------------------------------------------------------------------------------
        run_time = timedelta()
        ret = ''
        self.update.progress(value=-self.T, min=self.restart_days or 0)
        for run in self.runs:
            self.current_run = run.name.lower()
            run.delete_output_files()
            run.start()
            self.update.progress()  # Reset
            #self.update.progress(value=-run.T, min=run is self.ecl and self.restart_days or 0)
            ### Progress is updated after 25*0.2 = 5 sec
            ### Check for cancelled run every 0.2 sec
            run.init_control_func(update=self.update, count=15) 
            run.wait_for_process_to_finish(pause=0.2, loop_func=run.control_func)
            run.t = run.time()
            dec = min(len(str(t).split('.')[-1]) for t in (run.t, run.T))
            #print(run.name, dec, run.t, run.T)
            if round(run.t, dec) < round(run.T, dec):
                run.unexpected_stop_error()
            run_time += run.run_time()
            ret = run.complete_msg(run_time=run_time)
        return ret


    #--------------------------------------------------------------------------------
    def backward(self):                                                  # Simulation
    #--------------------------------------------------------------------------------
        self.update.progress(value=-self.T)
        ecl, ior = self.runs
        # Start runs
        #for run in self.runs:
        ecl.delete_output_files()
        ecl.start(restart=self.restart)
        ior.delete_output_files()
        ior.start(restart=self.restart, tsteps=ecl.init_tsteps)
        # The schedule appends keywords to the interface file (satnum.dat)
        # ecl.t = ior.t = self.schedule.update()
        self.schedule.update()
        # Update progress
        if self.restart:
            # Fix progress for restart runs
            self.update.progress(value=self.ecl.t, min=self.restart_days)
        else:
            # Reset progress-time for more accurate time-estimate   
            self.update.progress(value=ior.t, n0=ior.t)
        # Start timestep loop
        while ior.t < ior.T:
            self.print2log(f'\nStep {ecl.n+1}')
            self.update.progress(run=ecl)
            self.update.status(run=ecl, mode=self.mode)
            ecl.run_one_step(ior.satnum.file)
            # Run IORSim to prepare satnum input for the next Eclipse run
            self.update.progress(run=ior)
            self.update.status(run=ior, mode=self.mode)
            ior.run_one_step()
            # ecl.t = ior.t = self.schedule.update()
            self.schedule.update()
            self.print2log(f'Step {ecl.n} ({self.schedule.now().date()}) completed')
            #self.update.progress(value=ior.t)
            self.update.plot()
        self.update.progress(run=ior)
        self.update.status(run=ior, mode=self.mode)
        # Timestep loop finished
        for run in self.runs:
            self.update.status(value=f'Stopping {run.name}...')
            run.quit()
        return ecl.complete_msg()


    #--------------------------------------------------------------------------------
    def run(self):                                                       # Simulation
    #--------------------------------------------------------------------------------
        # Print header
        silentdelete(self.merge_OK)
        self.print2log(self.versions())
        self.print2log(self.info_header())
        msg = ''
        success = False
        try:
            if self.ready():
                msg = self.run_sim()
                success = True
            else:
                return False, ''
        except (SystemError, ProcessLookupError, NoSuchProcess) as error:
            msg = str(error)
            success = 'simulation complete' in msg.lower()
            if not isinstance(error, SystemError) and any(r.canceled for r in self.runs):
                msg = 'INFO Run stopped'
            #print(type(e).__name__, e, msg)
        except KeyboardInterrupt:
            self.cancel()
            msg = 'Simulation cancelled'
        except Exception as exception:  # Catch all other exceptions 
            self.print2log(f'\nAn exception occured:\n{trace_format_exc()}')
            #self.print2log(f'\nAn exception occured:\n{exception}')
            #error = exc_info()
            n = [run.n for run in self.runs if run] or [-1]
            #msg += f'(step {max(n)}) {error[0].__name__}: {error[1]}'
            msg += f'(step {max(n)}) {type(exception).__name__}: {exception}'
            #if r'\x00\x00\x00\x00' in fr'{error[1]}':
            if r'\x00\x00\x00\x00' in fr'{exception}':
                msg += f', try increasing the CHECK_PAUSE value ({CHECK_PAUSE}).'
            else:
                msg += self.runlog and f', check {Path(self.runlog.name).name} for details' or ''
        finally:
            # Kill possible remaining processes
            self.print2log('')
            [run.kill() for run in self.runs]
            self.print2log(f'\n=====  {msg.replace("INFO","")}  =====')
            self.current_run = None
            #self.update.progress(value=0)   # Reset progress time
            self.update.progress()   # Reset progress time
            self.update.plot()
            #self.update.status(value=msg, newline=True)
            conv_msg = ''
            try:
                if self.output.convert and success and any(run.is_iorsim for run in self.runs):
                    sleep(0.05)  # Need a short break here to make the GUI progressbar responsive
                    conv_msg = self.convert_and_merge(case=self.root)
            except SystemError as error:
                self.update.message(f'{error}')
            finally:
                self.close()
                self.update.status(value=msg+conv_msg, newline=True)
                return success, msg + ' ' + conv_msg


    #--------------------------------------------------------------------------------
    def convert_and_merge(self, case=None, only_convert=False, only_merge=False): # Simulation
    #--------------------------------------------------------------------------------
        func = (self.convert_restart, self.merge_restart)
        if only_convert:
            func = (self.convert_restart,)
        if only_merge:
            func = (self.merge_restart,)
        for f in func:
            success, msg = f(case=case)
            self.print2log(f'\n=====  {msg}  =====')
            self.update.status(value=msg, newline=True)
            if not success:
                return msg
        return ''


    #--------------------------------------------------------------------------------
    def convert_restart(self, case=None, check=False):         # Simulation
    #--------------------------------------------------------------------------------
        ### Convert from formatted (ascii) to unformatted (binary) restart file
        self.update.status(value='Converting restart file...')
        ior = self.ior or Iorsim(root=case)   
        if ior.unrst.is_file():
            return True, 'INFO Convert already complete!'
        if not ior.funrst.is_file():
            return False, f'ERROR Unable to convert IORSim output: {ior.funrst.file} is missing'
        start = datetime.now()
        try:
            ior.funrst.fast_convert(rename_duplicate=True, rename_key=('TEMP','TEMP_IOR'),
                                    progress=lambda n: self.update.progress(value=n), 
                                    cancel=ior.stop_if_canceled)
            if check:
                nblocks = ior.unrst.get('step', N=-1)[0][0]
                msg = ior.unrst.check.data_saved(nblocks=nblocks, limit=1, wait_func=ior.wait_for)
                if msg:
                    raise SystemError(f'ERROR Converted file {ior.unrst.file.name} did not pass the check: {msg}')
        except (Exception, KeyboardInterrupt) as e:
            silentdelete(ior.unrst.file)
            msg = str(e)
            if isinstance(e, KeyboardInterrupt) or 'run stopped' in msg.lower():
                return False, 'Convert cancelled'
            else:
                raise SystemError(f'ERROR Unable to convert IORSim restart: {e}')
        if self.output.del_convert:
            silentdelete(ior.funrst.file)
        return True, 'Convert complete, process-time was '+str(datetime.now()-start).split('.')[0]


    #--------------------------------------------------------------------------------
    def merge_restart(self, case=None, check=False):                     # Simulation
    #--------------------------------------------------------------------------------
        # Merge Eclipse and IORSim restart files
        #self.update.progress(value=0)   # Reset progress time
        self.update.progress()   # Reset progress time
        sleep(0.05) # Give progress-bar time to respond
        self.update.status(value='Merging Eclipse and IORSim restart files...')
        if self.merge_OK.exists():
            return True, 'Merge already complete!'
        case = Path(case)
        ecl = self.ecl or Eclipse(root=case)   
        ior = self.ior or Iorsim(root=case)   
        missing = [f.name for f in (ecl.unrst.file, ior.unrst.file) if not f.is_file()]
        if missing:
            return False, f'Unable to merge restart files due to missing files: {", ".join(missing)}'
        try:
            starttime = datetime.now()
            error_msg = 'ERROR Unable to merge Eclipse and IORSim restart files' 
            ecl_backup = Path(f'{case}_ECLIPSE.UNRST')
            ### The merged file ends with SATNUM (IORSim UNRST) instead of ENDSOL (Eclipse UNRST)
            merge_unrst = UNRST_file(f'{case}_MERGED.UNRST', end='SATNUM')
            ### Reset progress-bar
            end = min([x.unrst.get('step', N=-1)[0][0] for x in (ecl, ior)])
            self.update.progress(value=-end)
            ### Use the same start index for both files/sections
            start = max([x.unrst.get('step', N=1)[0][0] for x in (ecl, ior)])
            ### Define the sections in the restart file where the stitching is done
            ecl_sec = ecl.unrst.sections(start_before='SEQNUM',  end_before='SEQNUM', begin=start)
            ior_sec = ior.unrst.sections(start_after='DOUBHEAD', end_before='SEQNUM', begin=start)
            ### Create merged UNRST file
            merged_file = merge_unrst.create(sections=(ecl_sec, ior_sec), 
                                             progress=lambda n: self.update.progress(value=n), 
                                             cancel=ior.stop_if_canceled)
            if check:
                msg = merge_unrst.check.data_saved(nblocks=end, limit=1, wait_func=ior.wait_for)
                if msg:
                    raise SystemError(f'ERROR Merged file did not pass the test: {msg}')
            if merged_file and merged_file.is_file():
                ### Rename original Eclipse UNRST for backup
                ecl.unrst.file.replace(ecl_backup)
                ### Rename merged UNRST to original Eclipse UNRST
                merged_file.replace(ecl.unrst.file)
            else:
                return False, error_msg
        except (Exception, KeyboardInterrupt) as error:
            ### Delete merged file
            silentdelete(merge_unrst.file)
            if isinstance(error, KeyboardInterrupt):
                return False, 'Merge cancelled'
            DEBUG and trace_print_exc()
            raise SystemError(f'{error_msg}: {error}') from error
            #raise SystemError(f'{error_msg}: {exc_info()[1]}')
        if self.output.del_merge:
            silentdelete(ecl_backup, ior.unrst.file)
        ### Create this file to avoid re-merging the merged UNRST-file 
        self.merge_OK.touch()
        return True, 'Merge complete, process-time was '+str(datetime.now()-starttime).split('.')[0]


    #--------------------------------------------------------------------------------
    def info_header(self):                                               # Simulation
    #--------------------------------------------------------------------------------
        width = '10s'
        logfiles = [run.logname for run in self.runs]+[log.name for log in (self.runlog,) if log]
        case = Path(self.root).name
        s  = '\n'
        s += f'    {"Case":{width}}: {case}\n'
        mode = len(self.runs)<2 and self.runs[0].name or self.mode
        s += f'    {"Mode":{width}}: {mode.capitalize()}\n'
        s += f'    {"Days":{width}}: {self.T}' 
        # if self.mode=='forward':
        #     s += f' (edit TSTEP in the DATA-file to change days)'
        if self.restart:
            days = timedelta(days=self.restart_days)
            # s += f' (restart after {days.days} days, at  {self.schedule.start.date() + days})'
            s += f' (restart after {days.days} days, at {(self.start + days).date()})'
            s += self.skiprest and ' (SKIPREST)' or ''
        s += '\n'
        inte = get_keyword(f'{self.root}.trcinp', '\*INTEGRATION', end='\*')
        a, b = inte and (inte[0][4],inte[0][5]) or (0,0) 
        s += f'    {"Timestep":{width}}: {a}{(a!=b and f" - {b}" or "")} days\n'
        s += (self.schedule and self.schedule.file) and f'    {"Schedule":{width}}: start={self.schedule.start.date()}, days={self.schedule.end}{(self.schedule.skip_empty and ", skip empty entries" or "")}\n' or ''
        rundir = str(Path.cwd())
        s += f'    {"Run-dir":{width}}: {rundir}\n'
        casedir = str(Path(self.root).parent).replace(rundir, '<Run-dir>')
        s += f'    {"Case-dir":{width}}: {casedir}\n'
        s += f'    {"Log-files":{width}}: {", ".join([Path(file).name for file in logfiles])}\n'
        s += '\n'
        return s


    #--------------------------------------------------------------------------------
    def set_time(self, time):                                            # Simulation
    #--------------------------------------------------------------------------------
        self.T = time
        for run in self.runs:
            run.set_time(time)


    #--------------------------------------------------------------------------------
    def get_time(self):                                                  # Simulation
    #--------------------------------------------------------------------------------
        return self.T


    # #--------------------------------------------------------------------------------
    # def mode_from_case(self):                                            # Simulation
    # #--------------------------------------------------------------------------------
    #     # data = str(self.root)+'.DATA'
    #     # if Path(data).is_file():
    #     #     if file_contains(data, text='READDATA', comment='--', end='END'):
    #     return ('READDATA' in self.ECL_inp) and 'backward' or 'forward'
    #     # if 'READDATA' in self.ECL_inp:
    #     #     return 'backward'
    #     # else:
    #     #     return 'forward'
    #     # else:
    #     #     return None


    #--------------------------------------------------------------------------------
    def cancel(self):                                                    # Simulation
    #--------------------------------------------------------------------------------
        [run.cancel() for run in self.runs if isinstance(run, Runner)]


    #--------------------------------------------------------------------------------
    def compare_restart(self, ecl_keys=[], ior_keys=[], limit=None):     # Simulation
    #--------------------------------------------------------------------------------
        ecl_unrst = UNRST_file(f'{self.root}_ECLIPSE.UNRST')
        ior = self.ior or Iorsim(root=self.root)   
        try:
            if self.runlog:
                self.runlog = open(self.runlog.name, 'a')
            #with open(self.runlog.name, 'a') as self.runlog:   
            self.print2log(f'\n Comparing {ecl_unrst.name()} and {ior.funrst.name()}:')
            n = 0
            for a,b in zip(ecl_unrst.data(*ecl_keys), ior.funrst.data(*ior_keys)):
                self.print2log(f'  ECL: {print_dict(a)}')
                self.print2log(f'  IOR: {print_dict(b)}')
                self.print2log('')
                n += 1
                if limit and n > limit:
                    break
        finally:
            self.runlog and self.runlog.close()



#############################################################################
#                                                                           #
#                    Various utility functions                              #
#                                                                           #
#############################################################################


# #--------------------------------------------------------------------------------
# def case_from_casedir(case_dir, root):
# #--------------------------------------------------------------------------------
#     # Find case in casedir if given DATA-file is missing
#     case_dir = Path(case_dir)
#     if case_dir.is_dir() and (case_dir/root/(root+'.DATA')).is_file():
#         return case_dir/root/root
#     ### Return path to case not in the case-folder (if it exists)
#     if (path := Path.cwd()/(root+'.DATA')).is_file():
#         return path.with_suffix('').resolve()
#     raise SystemError('\n   '+root+'.DATA'+' not found in '+str(case_dir/root)+'\n')


#--------------------------------------------------------------------------------
#def parse_input(case_dir=None, settings_file=None):
def parse_input(settings_file=None):
#--------------------------------------------------------------------------------
    description = 'Script for running IORSim and Eclipse in backward and forward mode'
    parser = ArgumentParser(description=description)
    parser.add_argument('root',            help='Eclipse case folder or full path of the DATA-file')
    parser.add_argument('days',            help='Simulation time interval', type=float)
    parser.add_argument('-eclexe',         default='eclrun', help="Name of excecutable, default is 'eclrun'")
    parser.add_argument('-iorexe',         help="Name of IORSim executable, default is 'IORSimX'"                  )
    parser.add_argument('-no_unrst_check', help='Backward mode: do not check flushed UNRST-file', action='store_true')
    parser.add_argument('-no_rft_check',   help='Backward mode: do not check flushed RFT-file', action='store_true')
    parser.add_argument('-iorsim',         help="Run only IORSim", action='store_true')
    parser.add_argument('-eclipse',        help="Run only Eclipse", action='store_true')
    parser.add_argument('-v',              default=DEFAULT_LOG_LEVEL, help='Verbosity level, higher number increase verbosity, default is 3', type=int)
    if SCHEDULE_SKIP_EMPTY:
        parser.add_argument('-not_skip_empty', help='Do not skip empty schedule-file entries', action='store_true')
    else:
        parser.add_argument('-skip_empty', help='Skip schedule-file entries with no statements', action='store_true')
    parser.add_argument('-keep_files',     help='Interface-files are not deleted after completion', action='store_true')
    parser.add_argument('-to_screen',      help='Print program log to screen', action='store_true')
    parser.add_argument('-only_convert',   help='Only convert+merge and exit', action='store_true')
    parser.add_argument('-only_merge',     help='Only merge and exit', action='store_true')
    parser.add_argument('-delete',         help='Delete obsolete output files after convert and merge has finished', action='store_true')
    parser.add_argument('-ecl_alive',      help=f'Keep Eclipse alive at least {ECL_ALIVE_LIMIT} seconds', action='store_true')
    parser.add_argument('-ior_alive',      help=f'Keep IORSim alive', action='store_true')
    parser.add_argument('-check_input',    help='Check IORSim input file keywords', action='store_true', dest='check_input_kw')
    parser.add_argument('-lognr',          help='Add this number to the log-files', type=int)
    args = vars(parser.parse_args())
    if SCHEDULE_SKIP_EMPTY: 
        args['skip_empty'] = not args['not_skip_empty']
    # Look for case in case_dir if root is not a file
    # if case_dir and not Path(args['root']).is_file():
    #     args['root'] = case_from_casedir(case_dir, args['root'])
    # Remove suffix from root
    #args['root'] = Path(args['root']).parent/Path(args['root']).stem
    args['root'] = Path(args['root']).with_suffix('')
    #print(args['root'])
    # Read iorexe from settings if argument is missing
    if settings_file and not args['iorexe']:
        iorsim = get_keyword(settings_file, 'iorsim', end=' ')
        if any(iorsim):
            args['iorexe'] = iorsim[0][0]
        else:
            raise SystemError('IORSim executable is missing')    
    return args


@print_error
#--------------------------------------------------------------------------------
def runsim(root=None, time=None, iorexe=None, eclexe='eclrun', to_screen=False, 
           check_unrst=True, check_rft=True, keep_files=False, 
           only_convert=False, only_merge=False, convert=True, merge=True, delete=True,
           ecl_alive=False, ior_alive=False, only_eclipse=False, only_iorsim=False, check_input=False, 
           verbose=DEFAULT_LOG_LEVEL, lognr=None, skip_empty=SCHEDULE_SKIP_EMPTY):
#--------------------------------------------------------------------------------
    #----------------------------------------
    def status(value=None, **x):
    #----------------------------------------
        if not to_screen and value:
            value = value.replace('INFO','').strip()
            print('\r   '+value+(80-len(value))*' ', end=x.get('newline') and '\n' or '')

    prog = Progress(format='40#')
    #----------------------------------------
    def progress(run=None, value=None, min=None, n0=None):
    #----------------------------------------
        #print('progress in:', value, min, n0)
        if n0 is not None:
            prog.reset_time(n=n0)
        if min is not None:
            prog.set_min(min)
        if run:
            value = run.t
        if value is None:
            prog.reset_time(min=prog.min)
        else:
            if value<0:
                prog.reset(N=abs(value), min=min)
                return
            # elif value==0:
            #     prog.reset_time(min=prog.min)
            prog.print(value, text=run and f'({run.name})' or '')
            
        #print('progress out:', value, prog)
            
    #----------------------------------------
    def message(text=None, **x):
    #----------------------------------------
        text and print(f'\n\n     {text}\n')

    # Check if we only run eclipse or iorsim
    mode, runs = None, []
    if only_eclipse or only_iorsim:
        mode = 'forward'
        runs = only_eclipse and ['eclipse'] or ['iorsim']

    sim = Simulation(root=root, time=time, iorexe=iorexe, eclexe=eclexe, 
                     check_unrst=check_unrst, check_rft=check_rft, keep_files=keep_files, 
                     progress=progress, status=status, message=message, to_screen=to_screen,
                     convert=convert, merge=merge, delete=delete, ecl_keep_alive=ecl_alive,
                     ior_keep_alive=ior_alive, run_names=runs, mode=mode, check_input_kw=check_input, verbose=verbose,
                     lognr=lognr, skip_empty=skip_empty)
    sim.prepare()
    if not sim.ready():
        return 

    if only_convert or only_merge:
        sim.convert_and_merge(case=sim.root, only_merge=only_merge)
        return

    if not to_screen:
        print(sim.info_header())
    result, msg = sim.run()
    print()
    return sim


@print_error
#--------------------------------------------------------------------------------
#def main(case_dir='GUI/cases', settings_file='GUI/settings.txt'):
def main(settings_file=None):
#--------------------------------------------------------------------------------
    from os import _exit as os_exit
    args = parse_input(settings_file=settings_file)
    runsim(root=args['root'], time=args['days'], check_unrst=(not args['no_unrst_check']), check_rft=(not args['no_rft_check']),  
           to_screen=args['to_screen'], eclexe=args['eclexe'], iorexe=args['iorexe'],
           delete=args['delete'], keep_files=args['keep_files'], only_convert=args['only_convert'], only_merge=args['only_merge'],
           ecl_alive=args['ecl_alive'] and ECL_ALIVE_LIMIT, ior_alive=args['ior_alive'] and IOR_ALIVE_LIMIT, only_eclipse=args['eclipse'], only_iorsim=args['iorsim'],
           check_input=args['check_input_kw'], verbose=args['v'], lognr=args['lognr'], skip_empty=args['skip_empty'])
    os_exit(0)


######################################################################################

if __name__ == '__main__':

    main()
