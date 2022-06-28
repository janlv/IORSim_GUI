#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__version__ = '2.25'
__author__ = 'Jan Ludvig Vinningland'

DEBUG = False

# Constants
ECL_ALIVE_LIMIT = 90   # Seconds to wait before Eclipse is suspended (if option is on)
IOR_ALIVE_LIMIT = -1   # Negative value = never suspended
LOG_LEVEL_MAX = 4
LOG_LEVEL_MIN = 1
DEFAULT_LOG_LEVEL = 3      
CHECK_PAUSE = 0.01     # Default sleep-time during wait-loops
RFT_CHECK_ITER = 100   # Number of iterations before reducing number of expected RFT blocks


from collections import Counter, namedtuple
from mmap import ACCESS_READ, mmap
from pathlib import Path
from sys import exc_info, platform
from argparse import ArgumentParser
from datetime import datetime, timedelta
from time import sleep
from itertools import accumulate
from psutil import NoSuchProcess, __version__ as psutil_version
from shutil import copy as shutil_copy 
from traceback import print_exc as trace_print_exc, format_exc as trace_format_exc
from re import search, compile
from os.path import relpath

from IORlib.utils import flat_list, get_keyword, get_python_version, list2text, matches, print_error, is_file_ignore_suffix_case, number_of_blocks, remove_comments, safeopen, Progress, warn_empty_file, silentdelete, delete_files_matching, file_contains
from IORlib.runner import Runner
from IORlib.ECL import Input_file as ECL_input, RFT_file, UNRST_file, UNSMRY_file, unfmt_file, fmt_file, Section


#====================================================================================
class Eclipse(Runner):                                                      # eclipse
#====================================================================================
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
        #self.msg = MSG_file(root)
        #self.prt = PRT_file(root)
        self.inputfile = ECL_input(root)
        self.is_iorsim = False
        self.is_eclipse = True


    # #--------------------------------------------------------------------------------
    # def time(self):                                                         # eclipse
    # #--------------------------------------------------------------------------------
    #     t = 0
    #     if self.log:
    #         pattern = r'TIME=?\s+([0-9.]+)\s+DAYS'
    #         match = matches(file=self.log.name, pattern=pattern)
    #         time = [m.group(1) for m in match]
    #         t = time and time[-1] or 0           
    #     return float(t)


    #--------------------------------------------------------------------------------
    def delete_output_files(self):                                          # eclipse
    #--------------------------------------------------------------------------------
        delete_files_matching( [f'{self.case}*.{ext}' for ext in ('*UNRST','RFT','SMSPEC','UNSMRY','RTELOG','RTEMSG','MSG', 'session*', 'dbprtx.lock*')])
        delete_files_matching(self.case.parent/'fort??????')


    #--------------------------------------------------------------------------------
    def check_input(self):                                                  # eclipse
    #--------------------------------------------------------------------------------
        super().check_input()
        msg = f'ERROR Unable to start {self.name}:'

        # Check if root.DATA exists
        if not self.inputfile.is_file():
            raise SystemError(f'{msg} missing input file {self.inputfile}')

        # Check if included files exists
        # for file in self.inputfile.get('INCLUDE'):
        for file in self.inputfile.include_files():
            if not file.is_file():
                raise SystemError(f"{msg} '{file.name}' included from {self.inputfile.file.name} is missing")
        return True


    #--------------------------------------------------------------------------------
    def unexpected_stop_error(self):                                        # eclipse
    #--------------------------------------------------------------------------------
        error = 'unexpectedly, check the log'
        # Check for license failure
        with open(str(self.case)+'.MSG') as file:
            for line in file:
                if 'LICENSE FAILURE' in line:
                    error = 'due to a license failure'
                    break
        # ifile = self.interface_file('all').path()
        # ifiles = list(ifile.parent.glob(ifile.name))
        # if len(ifiles) == 1:
        #     error = f'due to missing interface-files (last file is {ifiles[-1].name})'
        raise SystemError(f'ERROR {self.name} stopped {error}')


    #--------------------------------------------------------------------------------
    def start(self):                                                        # eclipse
    #--------------------------------------------------------------------------------
        super().start(error_func=self.unexpected_stop_error)


#====================================================================================
class Forward_mixin:
#====================================================================================
    #--------------------------------------------------------------------------------
    def init_control_func(self, update=(), pause=0.01, count=5, **kwargs):
    #--------------------------------------------------------------------------------
        self.update = update
        self.loop_count = 0
        self.pause = pause
        self.count = count

    #--------------------------------------------------------------------------------
    def control_func(self):
    #--------------------------------------------------------------------------------
        self.stop_if_canceled()
        self.loop_count += 1
        if self.loop_count == self.count:
            self.loop_count = 0
            self.t = self.stop_if_timelimit_reached()
            for func in self.update:
                func(run=self)


#====================================================================================
class Ecl_forward(Forward_mixin, Eclipse):                              # ecl_forward
#====================================================================================

    #--------------------------------------------------------------------------------
    def check_input(self):                                             # ecl_forward
    #--------------------------------------------------------------------------------
        super().check_input()
        ### Check root.DATA exists and that READDATA keyword is NOT present
        if file_contains(str(self.case)+'.DATA', text='READDATA', comment='--', end='END'):
            raise SystemError('WARNING The current case cannot run in forward-mode: '+
                              'Eclipse input contains the READDATA keyword.')
        return True


#====================================================================================
class Backward_mixin:
#====================================================================================
    #--------------------------------------------------------------------------------
    def update_function(self, progress=True, plot=False):            # backward_mixin
    #--------------------------------------------------------------------------------
        #print(f'update_function(progress={progress}, plot={plot}')
        self.assert_running_and_stop_if_canceled()
        #if self.update:
        # self.t = self.time_and_step()[0]
        self.t = self.time()
        self.update.status(run=self)
        progress and self.update.progress(value=self.t)
        plot and self.update.plot()



#====================================================================================
class Ecl_backward(Backward_mixin, Eclipse):                           # ecl_backward
#====================================================================================
    #--------------------------------------------------------------------------------
    def __init__(self, check_unrst=True, check_rft=True, keep_alive=False, schedule=None, **kwargs):
    #--------------------------------------------------------------------------------
        super().__init__(ext_iface='I{:04d}', ext_OK='OK', keep_alive=keep_alive, **kwargs)
        self.tsteps = kwargs.get('tsteps') or self.inputfile.get('TSTEP')
        self.delete_interface = kwargs.get('delete_interface') or True
        self.init_tsteps = len(self.tsteps) 
        self.check_unrst = check_unrst
        self.check_rft = check_rft
        self.schedule = schedule
        self.nwell = 0
        self.del_satnum = False


    #--------------------------------------------------------------------------------
    def check_input(self):                                             # ecl_backward
    #--------------------------------------------------------------------------------
        def raise_error(msg):
            raise SystemError(f'ERROR To run the current case in backward-mode you need to {msg}')
        ### Check that root.DATA exists 
        super().check_input()
        kwargs = {'comment':'--', 'end':'END'}
        ### Check presence of READDATA
        DATA_file = str(self.case)+'.DATA'
        if not file_contains(DATA_file, text='READDATA', **kwargs):
            raise_error("insert 'READDATA /' in the DATA-file between 'TSTEP' and 'END'.")
        ### Check presence of RPTSOL RESTART>1
        if not file_contains(DATA_file, regex=r"\bRPTSOL\b\s+[A-Z0-9=_'\s]*\bRESTART\b *= *[2-9]{1}", **kwargs):
            raise_error("insert 'RPTSOL \\n RESTART=2 /' at the top of the SOLUTION section in the DATA-file.")
        return True


    #--------------------------------------------------------------------------------
    def start(self, restart=False):                                    # ecl_backward
    #--------------------------------------------------------------------------------
        # Start Eclipse in backward mode
        self.update.status(value=f'Starting {self.name}...')
        if self.n > 0 or self.t > 0:
            self._print(f'Starting at {self.t} days (step {self.n})')
        self.n += self.init_tsteps   # Use += and not = in case self.n is not 0 (RESTART option)
        self.interface_file('all').delete()
        self.interface_file(self.n).create_empty()
        self.OK_file().delete()
        # Start Eclipse
        super().start()
        self.update.status(value=f'{self.name} running...')
        # Wait for flushed UNRST-file   
        nblocks = 1 + self.init_tsteps # Add 1 for 0'th SEQNUM
        for i in range(nblocks):
            if i > 0:
                self.update_function(progress=not restart, plot=True)
            self.unrst.check.data_saved(nblocks=1, pause=CHECK_PAUSE)
        # Get number of wells from UNRST-file
        self.nwell = self.unrst.get(['nwell'])[0][-1]
        while self.nwell < 1:
            ### Run Eclipse until at least one well is producing and the RFT-file is created
            self.schedule.update(tstep=self.T)
            self.run_one_step(self.schedule.ifacefile.file, start_stop=False, nwell=True)
            self._print(f' nwell = {self.nwell}')
            self.update_function(progress=True, plot=True)
            nblocks = 1
            self.init_tsteps += 1
        # Wait for flushed RFT-file
        self.rft.check.data_saved_maxmin(nblocks=nblocks*self.nwell, iter=RFT_CHECK_ITER, pause=CHECK_PAUSE)
        self.suspend()
        self.t = self.time()


    #--------------------------------------------------------------------------------
    def run_one_step(self, satnum_file, log=True, start_stop=True, nwell=False):       # ecl_backward
    #--------------------------------------------------------------------------------
        self.interface_file(self.n).copy(satnum_file, delete=self.del_satnum)
        self.OK_file().create_empty()
        # Create next interface-file to avoid Eclipse from reading END
        self.interface_file(self.n+1).create_empty()
        start_stop and self.resume()
        self.wait_for( self.OK_file().is_deleted, error=self.OK_file().name()+' not deleted' )
        if self.check_unrst:
            self.unrst.check.data_saved(nblocks=1, pause=CHECK_PAUSE) 
        if self.check_rft and self.rft.exists():
            nblocks = 1
            if nwell:
                self.nwell = nblocks = self.unrst.get(['nwell'])[0][-1]
            msg = self.rft.check.data_saved_maxmin(nblocks=nblocks, iter=RFT_CHECK_ITER, pause=CHECK_PAUSE) 
            msg and self._print(msg)
        start_stop and self.suspend()
        if self.delete_interface:
            self.interface_file(self.n).delete()
        self.n += 1
        self.t = self.time()
        if self.check_rft and self.rft.not_in_sync(self.t):
            self._print(f'WARNING Simulation time not in sync with RFT-time: {self.t}, {self.rft.check.data()}')
        if log:
            self._print(f' Date is {self.unrst.date(N=-1)} ({self.t} days)')


    #--------------------------------------------------------------------------------
    def quit(self):                                                    # ecl_backward
    #--------------------------------------------------------------------------------
        #self._print(f'appending END to {self.interface_file(self.n).name()}')
        self.print_suspend_errors()
        self.interface_file(self.n).create_from_string('END')
        self.OK_file().create_empty()
        super().quit()


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
        ignored = ['*GRIDPLOT_FILE', '*N_TRACER']
        required = [k for k,v in all_.items() if v>0]
        optional = [k for k,v in all_.items() if v==0]
        specie = [k for k,v in all_.items() if v>0 and v!=3]
        solution = [k for k,v in all_.items() if v>0 and v!=2]
        specie_key = '*SPECIES'
        solution_key = '*SOLUTION'

    #--------------------------------------------------------------------------------
    def __init__(self, root):                                          # iorsim_input
    #--------------------------------------------------------------------------------
        self.file = Path(root).with_suffix('.trcinp')


    #--------------------------------------------------------------------------------
    def check_keywords(self):                                          # iorsim_input
    #--------------------------------------------------------------------------------
        # Check if required keywords are used, and if the order is correct 
        def raise_error(error):
            raise SystemError(f'ERROR Error in IORSim input file: {error}')    
        text = remove_comments(self.file, comment='#')
        file_kw = [kw.upper() for kw in compile(r'(\*[A-Za-z_-]+)').findall(text)]
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
    def check(self, error_msg='', check_kw=True):                      # iorsim_input
    #--------------------------------------------------------------------------------
        #super().check_input()
        #msg = f'WARNING Unable to start {self.name}:'

        # Check if input-file exists
        if not self.file.is_file():
            raise SystemError(f'ERROR {error_msg}: missing input file {self.file.name}')

        # Check if included files exists
        #include = flat_list(get_keyword(self.inputfile, '\*CHEMFILE', end='\*'))
        for file in self.include_files(): #include:
            #if not self.file.with_name(file).is_file():
            if not file.is_file():
                raise SystemError(f"ERROR {error_msg}: '{file.name}' included from {self.file.name} is missing in folder {file.parent}")

        # Check if required keywords are used, and if the order is correct 
        if check_kw:
            self.check_keywords()
        return True


    #--------------------------------------------------------------------------------
    def include_files(self):
    #--------------------------------------------------------------------------------
        '''
        Return full path to files included in the IORSim .trcinp-file
        '''
        files = flat_list(get_keyword(self.file, '\*CHEMFILE', end='\*', comment='#'))
        files = [self.file.parent/Path(f) for f in files]
        #print('IOR:', files) 
        return files


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
        self.funrst = Path(abs_root+'_IORSim_PLOT.FUNRST')
        self.unrst = self.funrst.with_suffix('.UNRST')
        #self.inputfile = Path(abs_root+'.trcinp')
        self.inputfile = IORSim_input(root)
        self.check_input_kw = kwargs.get('check_input_kw') or False
        self.is_iorsim = True
        self.is_eclipse = False


    #--------------------------------------------------------------------------------
    def check_input(self):                                                   # iorsim
    #--------------------------------------------------------------------------------
        super().check_input()
        self.inputfile.check(error_msg=f'Unable to start {self.name}:', check_kw=self.check_input_kw)
        return True


    #--------------------------------------------------------------------------------
    def start(self, copy_chem=True):                                         # iorsim
    #--------------------------------------------------------------------------------
        ### Copy chem-files to working dir 
        if copy_chem:
            for file in self.inputfile.include_files():
                #print(f'{file} -> {Path.cwd()/file.name}')
                shutil_copy(file, Path.cwd()/file.name)
        ### Check existence of necessary Eclipse output files 
        files = [self.case.with_suffix(ext) for ext in ('.UNRST', '.RFT', '.EGRID', '.INIT')]
        missing = [f.name for f in files if not f.is_file()]
        if missing:
            raise SystemError(f'ERROR Unable to start IORSim: Eclipse output file {", ".join(missing)} is missing')
        super().start()


    # #--------------------------------------------------------------------------------
    # def time(self):                                                 # iorsim
    # #--------------------------------------------------------------------------------
    #     t = 0
    #     if self.log:
    #         pattern = 
    #         match = matches(file=self.log.name, pattern=pattern)
    #         time = [m.group(1) for m in match]
    #         t = time and time[-1] or 0           
    #     return float(t)


    #--------------------------------------------------------------------------------
    def delete_output_files(self, raise_error=False):                        # iorsim
    #--------------------------------------------------------------------------------
        ### Delete old output files before starting new run
        case = str(self.case)
        delete_files_matching(case+'*.trcconc', raise_error=raise_error)
        delete_files_matching(case+'*.trcprd', raise_error=raise_error)
        silentdelete(self.funrst)

    #--------------------------------------------------------------------------------
    def close(self):                                                         # iorsim
    #--------------------------------------------------------------------------------
        super().close()
        ### Delete chem-files copied to working directory
        [silentdelete(Path.cwd()/file.name) for file in self.inputfile.include_files()]



#====================================================================================
class Ior_forward(Forward_mixin, Iorsim):                               # ior_forward
#====================================================================================
    pass

#====================================================================================
class Ior_backward(Backward_mixin, Iorsim):                             # ior_backward
#====================================================================================
    #--------------------------------------------------------------------------------
    def __init__(self, keep_alive=False, schedule=None, **kwargs):
    #--------------------------------------------------------------------------------
        #keep_alive = keep_alive and IOR_ALIVE_LIMIT or False
        super().__init__(args='-readdata', ext_iface='IORSimI{:04d}', ext_OK='IORSimOK', keep_alive=keep_alive, **kwargs)
        self.tsteps = kwargs.get('tsteps') or ECL_input(self.case).tsteps()
        self.delete_interface = kwargs.get('delete_interface') or True
        self.init_tsteps = len(self.tsteps)
        self.satnum = Path('satnum.dat')   # Output-file from IORSim, read by Eclipse as an interface-file
        self.endtag = '-- IORSimX done.'
        self.schedule = schedule


    #--------------------------------------------------------------------------------
    def delete_output_files(self):                                     # ior_backward
    #--------------------------------------------------------------------------------
        super().delete_output_files()
        silentdelete(self.satnum)


    #--------------------------------------------------------------------------------
    def satnum_flushed(self):                                          # ior_backward
    #--------------------------------------------------------------------------------
        file = Path(self.satnum)
        nchar = len(self.endtag) + 3
        endtag = self.endtag.encode()
        if not file.is_file() or file.stat().st_size < nchar:
            return False
        with open(file) as f:
            try:
                with mmap(f.fileno(), length=0, access=ACCESS_READ) as data:
                    if endtag in data[-nchar:]:
                        #print(data[-nchar:])
                        return True
            except ValueError:  # Catch 'cannot mmap empty file'
                return False
        return False


    #--------------------------------------------------------------------------------
    def satnum_dist(self, echo=False):                                 # ior_backward
    #--------------------------------------------------------------------------------
        '''
        Return the distribution of SATNUM numbers as a dict
        '''
        lines = remove_comments(self.satnum, comment='--') 
        values = compile(r'SATNUM\s+([0-9\s]+)').findall(lines) 
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
        self.interface_file('all').delete()
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
                self.update_function(plot=True)
            self.n += 1
            self.interface_file(self.n).create_empty()
            self.OK_file().create_empty()
            if n == 0:
                if start:
                    self.update and self.update.status(value=f'Starting {self.name}...')
                    super().start()
                    self.update and self.update.status(value=f'{self.name} running...')
                else:
                    self.resume()
            self.wait_for( self.OK_file().is_deleted, error=self.OK_file().name()+' not deleted')
        self.wait_for(self.satnum_flushed)
        warn_empty_file(self.satnum, comment='--')
        self.suspend()
        if self.delete_interface:
            [self.interface_file(self.n-n).delete() for n in range(N)] 
        self.t = self.time()
        if log:
            self._print(f' {self.t:.3f}/{self.T} days')


    #--------------------------------------------------------------------------------
    def quit(self):                                                    # ior_backward
    #--------------------------------------------------------------------------------
        #print('Quit: ',self.n+1)
        self.interface_file(self.n+1).append('Quit')
        self.OK_file().create_empty()
        super().quit()


#====================================================================================
class Schedule:
#====================================================================================
    #--------------------------------------------------------------------------------
    def __init__(self, case, T=0, init_days=0, start=None, ext='.SCH', comment='--', 
                 interface_file=None, file=None, merge_empty=False): #, end='/', tag='TSTEP'):
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
        self.comment = comment
        self.ifacefile = ECL_input(interface_file, read=False)
        self.days = init_days 
        self.start = start
        self.tstep = 0
        self._schedule = []
        self.file = file
        # Ignore case in file extension
        self.file = is_file_ignore_suffix_case( self.case.with_suffix(ext) )
        if self.file:
            self._schedule = self.days_and_actions(merge_empty=merge_empty)
        # Add end time 
        self.insert(days=T, remove=True)
        DEBUG and print(f'Creating {self}')


    #--------------------------------------------------------------------------------
    def __str__(self):                                                     # Schedule
    #--------------------------------------------------------------------------------
        return f'<Schedule(file={self.file}, start={self.start}, days={self.days}, length={len(self._schedule)})>'


    #--------------------------------------------------------------------------------
    def __del__(self):
    #--------------------------------------------------------------------------------
        DEBUG and print(f'Deleting {self}')


    #--------------------------------------------------------------------------------
    def next_tstep(self):                                                  # Schedule
    #--------------------------------------------------------------------------------
        return self._schedule[0][0] - self.days


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
    def insert(self, index=None, days=None, action='', remove=False):      # Schedule
    #--------------------------------------------------------------------------------
        if action:
            # Add newline
            action += '\n'
        index = [i for i,(d,a) in enumerate(self._schedule) if d>=days]
        if index:
            self._schedule.insert(index[0], [days, action])
            if remove:
                self._schedule = self._schedule[0:index[0]+1]
        else:
            self._schedule.append([days, action])


    #--------------------------------------------------------------------------------
    def get_type(self, string):                                            # Schedule
    #--------------------------------------------------------------------------------
        # Determine if schedule file contains DATES or TSTEP
        regex = compile(r'\n?\s*(\bDATES|TSTEP\b)')
        tstep_or_dates = [m.group(1) for m in regex.finditer(string)]
        #print(tstep_or_dates)
        N = len(tstep_or_dates)
        if tstep_or_dates.count('TSTEP') == N:
            use_dates = False
        elif tstep_or_dates.count('DATES') == N:
            use_dates = True
        else:
            raise SystemError('WARNING Schedule-file contains a mix of TSTEP and DATES')
        return use_dates


    #--------------------------------------------------------------------------------
    def days_and_actions(self, remove_end=True, merge_empty=False):               # Schedule
    #--------------------------------------------------------------------------------
        '''
        Use regexp to extract DATES or TSTEP values from the .SCH-file together
        with the file-positions (span) of these keywords. The file-positions are used
        to extract the scheduling commands contained between the DATES/TSTEP commands

        Returns: 
            A list of lists with days at index 0 and actions at index 1, such as:
            schedule = [[2.0, "WCONHIST \r\n    'P-15P'      'OPEN' "], [9.0, "WCONHIST"]]
        '''
        # Short Python regexp guide:
        #   \s : whitespace, [ \t\n\r\f\v]
        #   \w : alphanumeric, [a-zA-Z0-9_]
        #   \d : decimal digit, [0-9]
        #   \b : word-delimiter
        #    ? : 0 or 1 repetitions
        #    + : 1 or more rep.
        #    * : 0 or more rep.

        schfile = remove_comments(self.file)
        # Remove entries after END 
        if remove_end:
            found = search(r'\bEND\b', schfile)
            if found:
                schfile = schfile[0:found.start()]
        # Determine type, DATES or TSTEP
        use_dates = self.get_type(schfile)
        if use_dates:
            regex = compile(r'\n?\s*\bDATES\b\s+(\d+)\s+\'?(\w+)\'?\s+(\d+)\s*/\s+/')
        else:
            regex = compile(r'\n?\s*\bTSTEP\b\s+([0-9 *.]+)\s*/\s+')
        # Create list of (time, (start, end))-tuple
        date_span = [(' '.join(m.groups()), m.span()) for m in regex.finditer(schfile)]
        #print(date_span)
        pos = [s for d,s in date_span] + [(len(schfile), 0)]
        # Extract actions
        actions = [schfile[pos[i][1]:pos[i+1][0]].rstrip() for i in range(len(pos)-1)]
        # Calculate number of days from the simulation start (given by START in .DATA-file)
        if use_dates:
            days = [(datetime.strptime(d, '%d %b %Y').date()-self.start).days for d,s in date_span]
        else:   
            # Process x*y statements
            prod = lambda x, y : int(x)*float(y) 
            days = list(accumulate([sum([prod(*i.split('*')) if '*' in i else float(i) for i in d.split()]) for d,s in date_span]))
        if merge_empty:
            ### Return only non-empty [day, action] pairs        
            return [[float(d), a+'\n'] for (d,a) in zip(days, actions) if a]
        else:
            ### Keep all DATES/TSTEP entries
            return [[float(d), a+'\n'] for (d,a) in zip(days, actions)]


    #--------------------------------------------------------------------------------
    def append(self, action=None, tstep=None, append_line=-4):             # Schedule
    #--------------------------------------------------------------------------------
        '''
        line : line number of TSTEP
        '''
        #print(f'append(action={action}, tstep={tstep})')
        if action is None and tstep is None:
            return
        if tstep == 0:
            raise SystemError(f'ERROR Schedule gave TSTEP 0 at {self.days} days, simulation stopped. Check {self.file}')
        #lines = []
        #if self.ifacefile.is_file():
        match = self.ifacefile.get('TSTEP', pos=True)
        if match:
            file_tstep, pos = match[0]
        else:
            raise SystemError(f'ERROR Missing TSTEP in schedule file {self.ifacefile.file.name}')
        #print('UPDATE:', file_tstep, pos)
        with open(self.ifacefile.file, 'r') as f:
            lines = ''.join(f.readlines())
        out = lines[:pos[0]] + (action and action or '') + f'TSTEP\n{tstep} /\n' + lines[pos[1]:]
        #out = lines[:pos[0]] + '\n-- ACTION START --\n' + (action and action or '') + f'TSTEP\n{tstep} /\n' + '\n-- TSTEP END --\n' + lines[pos[1]:]
        #print(out)
        with open(self.ifacefile.file, 'w') as f:
            f.write(out)
        #n = max(0, len(lines) + append_line)
        #print('n',n)
        # if n > 0:
        #     if tstep is not None:
        #         # Replace TSTEP
        #         # +1 because we edit the value on the line after TSTEP
        #         lines[n+1] = f'{tstep} /\n'
        #     # Append action
        #     if action:
        #         lines.insert(n, action)
        # else:
        #     ### n == 0: empty file
        #     lines.extend(('TSTEP', f'{tstep}', '/', action))
        # #print('lines', lines)
        #with open(self.ifacefile.file, 'w') as f:
        #    f.write('\n'.join(lines))
        # with open(self.ifacefile.file, 'r') as f:
        #     print(self.ifacefile.file.name,'\n',''.join(f.readlines()))
        

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
        # Update days from previous step
        self.days += self.tstep
        # Append action if time is right
        if self.days >= self._schedule[0][0]:
            action = self._schedule.pop(0)[1]
        # Get tstep for next step (given by IORSim in satnum.dat)
        # with open(self.ifacefile.file) as f:
        #     print('SATNUM.DAT', '\n'.join(f.readlines))
        if tstep is None:
            self.tstep = new_tstep = self.ifacefile.get('TSTEP')[0]
        else:
            self.ifacefile.file.write_text(f'TSTEP\n{tstep} /\n')
            self.tstep = new_tstep = tstep
        #print(f'START: tstep:{self.tstep}, days:{self.days}, schedule:{self._schedule[:2]}')
        # Check arrival of next event and adjust tstep if neccessary
        if self._schedule and self.days + self.tstep + 1e-8 > self._schedule[0][0]:
            self.tstep = new_tstep = self._schedule[0][0] - self.days
        self.append(action=action, tstep=new_tstep)
        #self.check()
        #print(f'END: tstep:{self.tstep}, days:{self.days}, action:{action}, schedule:{self._schedule[:2]}')
        return self.days



#====================================================================================
class Simulation:                                                        # Simulation
#====================================================================================
    #--------------------------------------------------------------------------------
    def __init__(self, mode=None, root=None, pause=0, runs=[], to_screen=False, 
                 convert=True, merge=True, del_convert=False, del_merge=False, delete=False,
                 status=lambda **x:None, progress=lambda **x:None, plot=lambda **x:None, 
                 message=lambda **x:None, **kwargs):
    #--------------------------------------------------------------------------------
        #print('mode',mode,'root',root,'pause',pause,'runs',runs,'to_screen',to_screen,
        #      'convert',convert,'merge',merge,'del_merge',del_merge,'del_convert',del_convert,
        #      'status',status,'progress',progress,'plot',plot,'kwargs',kwargs)
        self.name = 'ior2ecl'
        self.root = root
        self.ECL_inp = ECL_input(root)
        self.update = namedtuple('update',['status','progress','plot','message'])(status, progress, plot, message)
        self.pause = pause
        if delete:
            del_convert = del_merge = True
        self.output = namedtuple('output',['convert','merge','del_convert','del_merge'])(convert, merge, del_convert, del_merge)
        self.runlog = None
        if root and not to_screen:
            lognr = kwargs.get('lognr')
            self.runlog = safeopen(Path(root).parent/f'{self.name}{lognr and "_" or ""}{lognr or ""}.log', 'w')
        self.print2log = lambda txt, **kwargs: print(txt, file=self.runlog, flush=True, **kwargs)
        self.current_run = None
        self.runs = runs
        self.run_sim = None
        self.ior = self.ecl = None
        self.T = 0
        self.dt = None
        self.mode = mode
        self.schedule = None
        self.restart = False
        self.restart_file = None
        self.restart_step = self.restart_days = 0
        kwargs.update({'root':str(root), 'runlog':self.runlog})
        self.kwargs = kwargs
        if self.root:
            try:
                self.run_sim = self.init_runs()
            except SystemError as e:
                self.update.message(f'{e}')


    #--------------------------------------------------------------------------------
    def close(self):                                                     # Simulation
    #--------------------------------------------------------------------------------
        self.runs = self.ior = self.ecl = self.current_run = self.schedule = None
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
        '''
        Read Eclipse and IORSim input files, run the init_func, and return the run_func
        '''
        # Check if this is a restart-run
        file, step = self.ECL_inp.get('RESTART')
        if file and step:
            # Get time and step from the restart-file
            self.restart_file = UNRST_file(file)
            self.restart_step = step
            time, n = self.restart_file.get(['time', 'step'], stop=('step', step))
            if step > n[-1] or not step in n: 
                new_step = min(n, key=lambda x:abs(x-step))
                self.update.message(f'ERROR Error in the Eclipse input-file ({self.ECL_inp.file.name}): Unable to restart from step {step}, use {new_step} instead')
                return False
            self.restart_days = time[n.index(step)]
            self.restart = True
        self.tsteps = ECL_input(self.root, include=True).tsteps()
        if self.tsteps == [0]:
            self.update.message(f'ERROR No TSTEP or DATES in {self.ECL_inp.file.name} or the included files, simulation stopped...')
            return False
        self.mode = self.mode or self.mode_from_case()
        init_func = {'backward':self.init_backward_run, 'forward': self.init_forward_run}[self.mode]
        run_func  = {'backward':self.backward,          'forward': self.forward}[self.mode]
        check_OK = False
        self.runs = init_func(**self.kwargs)
        # Check input
        check_OK = self.runs and all([run.check_input() for run in self.runs])
        return check_OK and run_func


    #--------------------------------------------------------------------------------
    def init_forward_run(self, iorexe=None, eclexe=None, time=0, **kwargs): # Simulation
    #--------------------------------------------------------------------------------
        time_ecl = sum(self.tsteps)
        if time != time_ecl:
            time = time_ecl
        self.T = time
        #print(f'time: {time}, time_ecl: {time_ecl}')
        kwargs.update({'T':self.T})
        if not self.runs:
            self.runs = ('eclipse','iorsim')
        for name in self.runs:
            if name=='eclipse':
                self.ecl = Ecl_forward(exe=eclexe, **kwargs)
            if name=='iorsim':
                self.ior = Ior_forward(exe=iorexe, **kwargs)
        return [run for run in (self.ecl, self.ior) if run]


    #--------------------------------------------------------------------------------
    def init_backward_run(self, iorexe=None, eclexe=None, ior_keep_alive=False, ecl_keep_alive=False, time=0, **kwargs): # Simulation
    #--------------------------------------------------------------------------------
        time_ecl = sum(self.tsteps) + self.restart_days
        if time < time_ecl:
            time = time_ecl + 1
            self.update.message(text=f'INFO Simulation time set to sum of TSTEP ({sum(self.tsteps)}) and RESTART ({self.restart_days}) in Eclipse input')
        self.T = time 
        self.dt = get_keyword(f'{self.root}.trcinp', '\*INTEGRATION', end='\*')[0][4]
        kwargs.update({'T':self.T, 'tsteps':self.tsteps, 'update':self.update})
        # Init runs
        self.ecl = Ecl_backward(exe=eclexe, keep_alive=ecl_keep_alive, n=self.restart_step, t=self.restart_days, **kwargs)
        self.ior = Ior_backward(exe=iorexe, keep_alive=ior_keep_alive, **kwargs)
        # Simulation start date given by first entry of restart-file (UNRST-file) or START keyword of DATA-file
        start = self.restart_file and self.restart_file.date(N=1) or self.ECL_inp.get('START')[0]
        # Set up schedule of commands to pass to satnum-file
        self.schedule = Schedule(self.root, T=self.T, start=start, init_days=time_ecl, interface_file=self.ior.satnum, 
                                 merge_empty=self.kwargs.get('merge_empty', False))
        self.ecl.schedule = self.ior.schedule = self.schedule
        return [self.ecl, self.ior]


    #--------------------------------------------------------------------------------
    def forward(self):                                                   # Simulation
    #--------------------------------------------------------------------------------
        run_time = timedelta()
        ret = ''
        for run in self.runs:
            self.current_run = run.name.lower()
            run.delete_output_files()
            self.update.status(value='Starting '+run.name, mode=self.mode)
            self.update.progress(value=-run.T)
            run.start()
            self.update.status(value=run.name+' running', mode=self.mode)
            run.init_control_func(update=self.update) 
            run.wait_for_process_to_finish(pause=0.2, loop_func=run.control_func)
            run.t = run.time()
            # print(run.name, run.t, run.T)
            if run.t < run.T:
                run.unexpected_stop_error()
            run_time += run.run_time()
            ret = run.complete_msg(run_time=run_time)
        return ret


    #--------------------------------------------------------------------------------
    def backward(self):                                                  # Simulation
    #--------------------------------------------------------------------------------
        self.update.progress(value=-self.runs[0].T)
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
            self.update.status(run=ecl, mode=self.mode)
            ecl.run_one_step(ior.satnum)
            # Run IORSim to prepare satnum input for the next Eclipse run
            self.update.status(run=ior, mode=self.mode)
            ior.run_one_step()
            # ecl.t = ior.t = self.schedule.update()
            self.schedule.update()
            self.print2log(f'Step {ecl.n} ({self.schedule.now()}) completed')
            self.update.progress(value=ior.t)
            self.update.plot()
        # Timestep loop finished
        for run in self.runs:
            self.update.status(value=f'Stopping {run.name}...')
            run.quit()
        return ecl.complete_msg()


    #--------------------------------------------------------------------------------
    def run(self):                                                       # Simulation
    #--------------------------------------------------------------------------------
        # Print header
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
        except (SystemError, ProcessLookupError, NoSuchProcess) as e:
            msg = str(e)
            success = 'simulation complete' in msg.lower()
            if not type(e) is SystemError and any([r.canceled for r in self.runs]):
                msg = 'INFO Run stopped'
            #print(type(e).__name__, e, msg)
        except KeyboardInterrupt:
            self.cancel()
            msg = 'Simulation cancelled' 
        except:  # Catch all other exceptions 
            self.print2log(f'\nAn exception occured:\n{trace_format_exc()}')
            e = exc_info()
            msg = f'{e[0].__name__}: {e[1]}, check the application log for details'
        finally:
            # Kill possible remaining processes
            self.print2log('')
            [run.kill() for run in self.runs]
            self.print2log(f'\n=====  {msg.replace("INFO","")}  =====')
            self.current_run = None
            self.update.progress(value=0)   # Reset progress time
            self.update.plot()
            #self.update.status(value=msg, newline=True)
            conv_msg = ''
            if self.output.convert and success and any([run.is_iorsim for run in self.runs]):
                sleep(0.05)  # Need a short break here to make the GUI progressbar responsive
                conv_msg = self.convert_and_merge(case=self.root)
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
    def convert_restart(self, case=None, fast=True):                     # Simulation
    #--------------------------------------------------------------------------------
        ### Convert from formatted (ascii) to unformatted (binary) restart file
        self.update.status(value='Converting restart file...')
        ior = self.ior or Iorsim(root=case)   
        if not ior.funrst.is_file():
            if ior.unrst.is_file():
                return True, f'{ior.unrst} already exists'
            else:
                return False, f'IORSim output {ior.funrst.name} is missing'
        start = datetime.now()
        try:
            infile = fmt_file(ior.funrst)
            if fast:
                convert = infile.fast_convert
            else:
                N = number_of_blocks(file=ior.funrst, blockstart='SEQNUM')
                self.update.progress(value=-(N-1))
                convert = infile.convert 
            convert(rename_duplicate=True, rename_key=('TEMP','TEMP_IOR'),
                    progress=lambda n: self.update.progress(value=n), 
                    cancel=ior.stop_if_canceled)
        except SystemError as e:
            silentdelete(ior.unrst)
            msg = str(e)
            if 'run stopped' in msg.lower():
                msg = 'Convert cancelled'
            return False, msg
        except KeyboardInterrupt:
            silentdelete(ior.unrst)
            return False, 'Convert cancelled'
        if self.output.del_convert:
            silentdelete(ior.funrst)
        return True, 'Convert complete, process-time was '+str(datetime.now()-start).split('.')[0]


    #--------------------------------------------------------------------------------
    def merge_restart(self, case=None):                                  # Simulation
    #--------------------------------------------------------------------------------
        # Merge Eclipse and IORSim restart files
        self.update.status(value='Merging Eclipse and IORSim restart files...')
        self.update.progress(value=0)   # Reset progress time
        sleep(0.05) # Give progress-bar time to respond
        ecl = self.ecl or Eclipse(root=case)   
        ior = self.ior or Iorsim(root=case)   
        if ecl.unrst.is_file() and ior.unrst.is_file():
            backup_ecl = Path(str(ecl.case)+'_ECLIPSE.UNRST')
            if backup_ecl.is_file():
                # This is a pure IORSim run and backup already exists; restore backup
                shutil_copy(backup_ecl, ecl.unrst.file)
            else:
                # No backup exists; create backup copy
                shutil_copy(ecl.unrst.file, backup_ecl)
        else:
            missing = [f.name for f in (ecl.unrst.file, ior.unrst) if not f.is_file()]
            return False, f'Unable to merge restart files due to missing files: {", ".join(missing)}'
        start = datetime.now()
        try:
            # Define the sections in the restart files where the stitching is done
            ecl_sec = Section(ecl.unrst.file, start_before='SEQNUM', end_before='SEQNUM', skip_sections=(0,))
            ior_sec = Section(ior.unrst, start_after='DOUBHEAD', end_before='SEQNUM')
            fname =  Path(str(ecl.case)+'_MERGED.UNRST') 
            merged_file = unfmt_file(fname).create(ecl_sec, ior_sec, 
                                                   progress=lambda n: self.update.progress(value=n),
                                                   cancel=ior.stop_if_canceled)
            # Rename merged UNRST-file to original Eclipse restart file
            if merged_file and merged_file.is_file():
                merged_file.replace(ecl.unrst.file)
            else:
                return False, f'Unable to merge {ecl.unrst.file} and {ior.unrst}'
        except OSError as e:
            return False, str(e)
        if self.output.del_merge:
            silentdelete((backup_ecl, ior.unrst))
        return True, 'Merge complete, process-time was '+str(datetime.now()-start).split('.')[0]


    #--------------------------------------------------------------------------------
    def info_header(self):                                               # Simulation
    #--------------------------------------------------------------------------------
        logfiles = [run.log.name for run in self.runs]+[log.name for log in (self.runlog,) if log]
        case = Path(self.root).name
        s  = '\n'
        s += f'    {"Case":10s}: {case}\n'
        s += f'    {"Mode":10s}: {self.mode.capitalize()}\n'
        s += f'    {"Days":10s}: {self.T}' 
        if self.mode=='forward':
            s += f' (edit TSTEP in {case}.DATA to change number of days)'
        if self.restart:
            days = timedelta(days=self.restart_days)
            s += f' (restart after {days.days} days, at {self.schedule.start + days})'
        s += '\n'
        s += self.dt and f'    {"Timestep":10s}: {self.dt} days\n' or ''
        s += f'    {"Folder":10s}: {Path(self.root).parent}\n'
        s += f'    {"Log-files":10s}: {", ".join([Path(file).name for file in logfiles])}\n'
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


    #--------------------------------------------------------------------------------
    def mode_from_case(self):                                            # Simulation
    #--------------------------------------------------------------------------------
        data = str(self.root)+'.DATA'
        if Path(data).is_file():
            if file_contains(data, text='READDATA', comment='--', end='END'):
                return 'backward'
            else:
                return 'forward'
        else:
            return None


    #--------------------------------------------------------------------------------
    def cancel(self):                                                    # Simulation
    #--------------------------------------------------------------------------------
        for run in self.runs:
            run.canceled = True




#############################################################################
#                                                                           #
#                    Various utility functions                              #
#                                                                           #
#############################################################################


#--------------------------------------------------------------------------------
def case_from_casedir(case_dir, root):
#--------------------------------------------------------------------------------
    # Find case in casedir if given DATA-file is missing
    case_dir = Path(case_dir)
    if case_dir.is_dir() and (case_dir/root/(root+'.DATA')).is_file():
        return case_dir/root/root
    raise SystemError('\n   '+root+'.DATA'+' not found in '+str(case_dir/root)+'\n')


#--------------------------------------------------------------------------------
def parse_input(case_dir=None, settings_file=None):
#--------------------------------------------------------------------------------
    description = 'Script for running IORSim and Eclipse in backward and forward mode'
    parser = ArgumentParser(description=description)
    parser.add_argument('root',            help='Eclipse case folder or full path of the DATA-file')
    parser.add_argument('days',            help='Simulation time interval', type=int)
    parser.add_argument('-eclexe',         default='eclrun', help="Name of excecutable, default is 'eclrun'")
    parser.add_argument('-iorexe',         help="Name of IORSim executable, default is 'IORSimX'"                  )
    parser.add_argument('-no_unrst_check', help='Backward mode: do not check flushed UNRST-file', action='store_true')
    parser.add_argument('-no_rft_check',   help='Backward mode: do not check flushed RFT-file', action='store_true')
    parser.add_argument('-iorsim',         help="Run only IORSim", action='store_true')
    parser.add_argument('-eclipse',        help="Run only Eclipse", action='store_true')
    parser.add_argument('-v',              default=DEFAULT_LOG_LEVEL, help='Verbosity level, higher number increase verbosity, default is 3', type=int)
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
    # Look for case in case_dir if root is not a file
    if case_dir and not Path(args['root']).is_file():
        args['root'] = case_from_casedir(case_dir, args['root'])
    # Remove suffix from root
    args['root'] = Path(args['root']).parent/Path(args['root']).stem
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
           verbose=DEFAULT_LOG_LEVEL, lognr=None):
#--------------------------------------------------------------------------------
    #----------------------------------------
    def status(value=None, **x):
    #----------------------------------------
        if not to_screen and value:
            value = value.replace('INFO','').strip()
            print('\r   '+value+60*' ', end=x.get('newline') and '\n' or '')

    prog = Progress(format='40#')
    #----------------------------------------
    def progress(run=None, value=None, min=None, n0=None):
    #----------------------------------------
        if n0 is not None:
            prog.reset_time(n=n0)
        if min is not None:
            prog.set_min(min)
        if run:
            value = run.t
        if value is not None:
            if value<0:
                prog.reset(N=abs(value))
                return
            elif value==0:
                prog.reset_time()
            prog.print(value)
            
    #----------------------------------------
    def message(text=None, **x):
    #----------------------------------------
        text and print('\n\n     ' + text + '\n')

    # Check if we only run eclipse or iorsim
    mode, runs = None, []
    if only_eclipse or only_iorsim:
        mode = 'forward'
        runs = only_eclipse and ['eclipse'] or ['iorsim']

    sim = Simulation(root=root, time=time, iorexe=iorexe, eclexe=eclexe, 
                     check_unrst=check_unrst, check_rft=check_rft, keep_files=keep_files, 
                     progress=progress, status=status, message=message, to_screen=to_screen,
                     convert=convert, merge=merge, delete=delete, ecl_keep_alive=ecl_alive,
                     ior_keep_alive=ior_alive, runs=runs, mode=mode, check_input_kw=check_input, verbose=verbose,
                     lognr=lognr)

    if not sim.ready():
        return 

    if only_convert or only_merge:
        sim.convert_and_merge(case=sim.root, only_merge=only_merge)
        return

    if not to_screen:
        print(sim.info_header())
    result, msg = sim.run()
    print()


@print_error
#--------------------------------------------------------------------------------
def main(case_dir='GUI/cases', settings_file='GUI/settings.txt'):
#--------------------------------------------------------------------------------
    from os import _exit as os_exit
    args = parse_input(case_dir=case_dir, settings_file=settings_file)
    runsim(root=args['root'], time=args['days'], check_unrst=(not args['no_unrst_check']), check_rft=(not args['no_rft_check']),  
           to_screen=args['to_screen'], eclexe=args['eclexe'], iorexe=args['iorexe'],
           delete=args['delete'], keep_files=args['keep_files'], only_convert=args['only_convert'], only_merge=args['only_merge'],
           ecl_alive=args['ecl_alive'] and ECL_ALIVE_LIMIT, ior_alive=args['ior_alive'] and IOR_ALIVE_LIMIT, only_eclipse=args['eclipse'], only_iorsim=args['iorsim'],
           check_input=args['check_input_kw'], verbose=args['v'], lognr=args['lognr'])
    os_exit(0)


######################################################################################

if __name__ == '__main__':

    main()
