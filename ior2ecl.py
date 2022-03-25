#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__version__ = '2.25'
__author__ = 'Jan Ludvig Vinningland'

# Constants
MAX_ITERATIONS = 1e5   # Iteration limit (avoid time consuming creation of interface-files)
ECL_ALIVE_LIMIT = 90   # Seconds to wait before Eclipse is suspended (if option is on)
IOR_ALIVE_LIMIT = -1   # Negative value = never suspended
LOG_LEVEL_MAX = 3
LOG_LEVEL_MIN = 1
LOG_LEVEL = 3          # Default log level
LOOP_PAUSE = 0.01

DEBUG = False

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
from traceback import print_exc as trace_print_exc, format_exc as  trace_format_exc
from re import search, compile
from os.path import relpath
from threading import Thread, Timer as th_timer

from IORlib.utils import flat_list, get_keyword, get_python_version, list2text, print_error, is_file_ignore_suffix_case, number_of_blocks, remove_comments, safeopen, Progress, warn_empty_file, silentdelete, delete_files_matching, file_contains
from IORlib.runner import Runner
from IORlib.ECL import RFT_file, UNRST_file, check_blocks, get_included_files, get_restart_file_step, get_start_UNRST, get_time_step_MSG, get_restart_time_step, get_start, get_time_step_UNRST, get_time_step_UNSMRY, get_tsteps, get_tsteps_from_schedule_files, unfmt_file, fmt_file, Section


#====================================================================================
class Eclipse(Runner):                                                      # eclipse
#====================================================================================
    #--------------------------------------------------------------------------------
    def __init__(self, root=None, exe='eclrun', **kwargs):
    #--------------------------------------------------------------------------------
        #print('eclipse.__init__: ',root, exe, kwargs)
        root = str(root)        
        exe = str(exe)
        super().__init__(name='Eclipse', case=root, exe=exe, cmd=[exe, 'eclipse', root], **kwargs)                        
        self.update = kwargs.get('update') or None
        self.unrst = UNRST_file(root+'.UNRST', wait_func=self.wait_for)
        self.rft = RFT_file(root+'.RFT', wait_func=self.wait_for)
        # self.unrst = Path(root+'.UNRST')
        # self.rft = Path(root+'.RFT')
        self.unsmry = Path(root+'.UNSMRY')
        self.msg = Path(root+'.MSG')
        self.inputfile = Path(root+'.DATA')
        self.is_iorsim = False
        self.is_eclipse = True

    #--------------------------------------------------------------------------------
    def time_and_step(self):                                                # eclipse
    #--------------------------------------------------------------------------------
        t = n = 0
        if self.unsmry.is_file():
            t, n = get_time_step_UNSMRY(file=self.unsmry)
        elif self.msg.is_file():
            t, n = get_time_step_MSG(file=self.msg) 
        elif self.unrst.is_file():
            t, n = get_time_step_UNRST(file=self.unrst.file, end=True)
        #print('{}: time = {}, step = {}'.format(self.name, t, n))
        return int(t), int(n)        

    #--------------------------------------------------------------------------------
    def delete_output_files(self):                                          # eclipse
    #--------------------------------------------------------------------------------
        silentdelete( [self.unrst.file, self.rft.file] )
        silentdelete( [str(self.case)+ext for ext in ('.SMSPEC','.UNSMRY','.RTELOG','.RTEMSG','_ECLIPSE.UNRST')] )
        delete_files_matching( [str(self.case)+fil for fil in ('*.session*', '*.dbprtx.lock', '*.dbprtx.lock-journal')] )

    #--------------------------------------------------------------------------------
    def check_input(self):                                                  # eclipse
    #--------------------------------------------------------------------------------
        super().check_input()
        msg = f'WARNING Unable to start {self.name}:'

        # Check if root.DATA exists
        inp_file = Path(str(self.case)+'.DATA')
        if not inp_file.is_file():
            raise SystemError(f'{msg} missing input file {inp_file}')

        # Check if included files exists
        for file in get_included_files(inp_file):
            if not inp_file.with_name(file).is_file():
                raise SystemError(f"{msg} '{file}' included from {inp_file} is missing")


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
        ifile = self.interface_file('all').path()
        ifiles = list(ifile.parent.glob(ifile.name))
        if len(ifiles) == 1:
            error = f'due to missing interface-files (last file is {ifiles[-1].name})'
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
            self.t = self.stop_if_limit_reached(limit='time')
            for func in self.update:
                func(run=self)


#====================================================================================
class Ecl_forward(Forward_mixin, Eclipse):                              # ecl_forward
#====================================================================================
    # #--------------------------------------------------------------------------------
    # def __init__(self, root, **kwargs):
    # #--------------------------------------------------------------------------------
    #     super().__init__(root, **kwargs)

    #--------------------------------------------------------------------------------
    def check_input(self):                                             # ecl_forward
    #--------------------------------------------------------------------------------
        super().check_input()
        ### Check root.DATA exists and that READDATA keyword is NOT present
        if file_contains(str(self.case)+'.DATA', text='READDATA', comment='--', end='END'):
            raise SystemError('WARNING The current case cannot run in forward-mode: '+
                              'Eclipse input contains the READDATA keyword.')


#====================================================================================
class Backward_mixin:
#====================================================================================
    #--------------------------------------------------------------------------------
    def update_function(self, progress=True, plot=False):            # backward_mixin
    #--------------------------------------------------------------------------------
        #print(f'update_function(progress={progress}, plot={plot}')
        self.assert_running_and_stop_if_canceled()
        #if self.update:
        self.t = self.time_and_step()[0]
        self.update.status(run=self)
        progress and self.update.progress(value=self.t)
        plot and self.update.plot()



#====================================================================================
class Ecl_backward(Backward_mixin, Eclipse):                           # ecl_backward
#====================================================================================
    #--------------------------------------------------------------------------------
    def __init__(self, check_unrst=True, check_rft=True, keep_alive=False, **kwargs):
    #--------------------------------------------------------------------------------
        super().__init__(ext_iface='I{:04d}', ext_OK='OK', keep_alive=keep_alive, **kwargs)
        self.tsteps = kwargs.get('tsteps') or get_tsteps(self.case.with_suffix('.DATA'))
        self.delete_interface = kwargs.get('delete_interface') or True
        self.init_tsteps = len(self.tsteps) 
        self.check_unrst = check_unrst
        self.check_rft = check_rft
        self.nwell = 0


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


    #--------------------------------------------------------------------------------
    def start(self, restart=False):                                    # ecl_backward
    #--------------------------------------------------------------------------------
        # Start Eclipse in backward mode
        self.update.status(value=f'Starting {self.name}...')
        # If RESTART in DATA, add time and step from restart-file
        self.t, self.n = get_restart_time_step(self.case.with_suffix('.DATA'))
        self.n += self.init_tsteps  
        self.interface_file('all').delete()
        # Need to create all interface files in advance to avoid Eclipse termination        
        [self.interface_file(i).create_empty() for i in range(self.n, self.N)] 
        self.OK_file().delete()
        # Start Eclipse
        super().start()
        self.update.status(value=f'{self.name} running...')
        # Wait for output-files to appear  
        self.unrst.wait_for_file(pause=LOOP_PAUSE)
        self.rft.wait_for_file(pause=LOOP_PAUSE)
        # Wait for flushed UNRST-file   
        nblocks = 1 + self.init_tsteps # Add 1 for 0'th SEQNUM
        for i in range(nblocks):
            if i > 0:
                self.update_function(progress=not restart, plot=True)
            self.unrst.wait_for_complete_file(nblocks=1, pause=LOOP_PAUSE)
        # Get number of wells from UNRST-file
        self.nwell, = self.unrst.var(['nwell'], end=True)
        self._print(f' nwell = {self.nwell}')
        # Wait for flushed RFT-file
        self.rft.wait_for_complete_file(nblocks=nblocks*self.nwell, pause=LOOP_PAUSE)
        self.suspend()


    #--------------------------------------------------------------------------------
    def run_one_step(self, satnum_file, log=True):                     # ecl_backward
    #--------------------------------------------------------------------------------
        self.interface_file(self.n).copy(satnum_file, delete=True)
        self.OK_file().create_empty()
        self.resume()
        self.wait_for( self.OK_file().is_deleted, error=self.OK_file().name()+' not deleted' )
        if self.check_unrst:
            self.unrst.wait_for_complete_file(nblocks=1, pause=LOOP_PAUSE) 
        if self.check_rft:
            self.rft.wait_for_complete_file(nblocks=self.nwell, pause=LOOP_PAUSE) 
        self.suspend()
        if self.delete_interface:
            self.interface_file(self.n).delete()
        self.n += 1
        if log:
            # y, m, d = self.unrst.var(['year','month','day'])
            # self._print(f' UNRST-file at {y}-{m:02d}-{d:02d}')
            self._print(f' UNRST-file now at {self.unrst.date(end=True)}')



    #--------------------------------------------------------------------------------
    def quit(self):                                                    # ecl_backward
    #--------------------------------------------------------------------------------
        #self._print(f'appending END to {self.interface_file(self.n).name()}')
        self.print_suspend_errors()
        self.interface_file(self.n).create_from_string('END')
        self.OK_file().create_empty()
        super().quit()



#====================================================================================
class Iorsim(Runner):                                                        # iorsim
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
        super().__init__(name='IORSim', case=root, exe=exe, cmd=cmd, **kwargs)
        self.update = kwargs.get('update') or None
        self.trcconc = None
        self.funrst = Path(abs_root+'_IORSim_PLOT.FUNRST')
        self.unrst = self.funrst.with_suffix('.UNRST')
        self.inputfile = Path(abs_root+'.trcinp')
        self.check_input_kw = kwargs.get('check_input_kw') or False
        self.is_iorsim = True
        self.is_eclipse = False

    #--------------------------------------------------------------------------------
    def check_keywords(self):                                                # iorsim
    #--------------------------------------------------------------------------------
        # Check if required keywords are used, and if the order is correct 
        def raise_error(error):
            raise SystemError(f'ERROR Error in IORSim input file: {error}')    
        text = remove_comments(self.inputfile, comment='#')
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
    def check_input(self):                                                   # iorsim
    #--------------------------------------------------------------------------------
        super().check_input()
        msg = f'WARNING Unable to start {self.name}:'

        # Check if input-file exists
        if not self.inputfile.is_file():
            raise SystemError(f'{msg} missing input file {self.inputfile.name}')

        # Check if included files exists
        include = flat_list(get_keyword(self.inputfile, '\*CHEMFILE', end='\*'))
        for file in include:
            if not self.inputfile.with_name(file).is_file():
                raise SystemError(f"{msg} '{file}' included from {self.inputfile.name} is missing")

        # Check if required keywords are used, and if the order is correct 
        if self.check_input_kw:
            self.check_keywords()


    #--------------------------------------------------------------------------------
    def start(self):                                                         # iorsim
    #--------------------------------------------------------------------------------
        ### check that Eclipse UNRST and RFT files exists
        for ext in ('.UNRST','.RFT'):
            fname = str(self.case)+ext
            if not Path(fname).is_file():
                raise SystemError(f'ERROR Unable to start IORSim: Eclipse output file {Path(fname).name} is missing')
        super().start()

    #--------------------------------------------------------------------------------
    def time_and_step(self):                                                 # iorsim
    #--------------------------------------------------------------------------------
        # Output file for reading days
        if not self.trcconc:
            for outfile in self.case.parent.glob(self.case.stem+'*.trcconc'):
                if outfile.is_file():
                    self.trcconc = outfile
                    break
        if not self.trcconc:
            return 0, 0
        ### Get time and step from IORSim output
        t, n = 0, 0
        with open(self.trcconc) as out:
           for line in out:
               line = line.strip()
               if line and not line.startswith('#'):
                   n += 1
        #print('line: |'+line+'|')
        if line:
            t = float(line.split()[0])
        #print('{}: time = {}, step = {}'.format(self.name, t, n))
        return int(t), int(n)

    #--------------------------------------------------------------------------------
    def delete_output_files(self):                                           # iorsim
    #--------------------------------------------------------------------------------
        case = str(self.case)
        delete_files_matching(case+'*.trcconc', raise_error=True)
        delete_files_matching(case+'*.trcprd', raise_error=True)
        silentdelete(self.funrst)


#====================================================================================
class Ior_forward(Forward_mixin, Iorsim):                               # ior_forward
#====================================================================================
    pass

#====================================================================================
class Ior_backward(Backward_mixin, Iorsim):                             # ior_backward
#====================================================================================
    #--------------------------------------------------------------------------------
    def __init__(self, keep_alive=False, **kwargs):
    #--------------------------------------------------------------------------------
        #keep_alive = keep_alive and IOR_ALIVE_LIMIT or False
        super().__init__(args='-readdata', ext_iface='IORSimI{:04d}', ext_OK='IORSimOK', keep_alive=keep_alive, **kwargs)
        self.tsteps = kwargs.get('tsteps') or get_tsteps(self.case.with_suffix('.DATA'))
        self.delete_interface = kwargs.get('delete_interface') or True
        self.init_tsteps = len(self.tsteps)
        self.satnum = Path('satnum.dat')   # Output-file from IORSim, read by Eclipse as an interface-file
        self.endtag = '-- IORSimX done.'


    #--------------------------------------------------------------------------------
    def delete_output_files(self):                                     # ior_backward
    #--------------------------------------------------------------------------------
        super().delete_output_files()
        silentdelete(self.satnum)

    #--------------------------------------------------------------------------------
    def satnum_flushed(self):
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
    def satnum_dist(self, echo=False):                                             # ior_backward
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
    def start(self, restart=False):                                    # ior_backward
    #--------------------------------------------------------------------------------
        # Start IORSim backward run
        self.n = 0 
        self.interface_file('all').delete()
        self.run_steps(1+self.init_tsteps, start=True)

    #--------------------------------------------------------------------------------
    def run_one_step(self):                                            # ior_backward
    #--------------------------------------------------------------------------------
        self.run_steps(1)
        #self.satnum_dist(echo=True)

    #--------------------------------------------------------------------------------
    def run_steps(self, N, start=False):                               # ior_backward
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
            [self.interface_file(n).delete() for n in range(N)] 


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
    def __init__(self, case, T=0, init_days=0, ext='.SCH', comment='--', interface_file=None): #, end='/', tag='TSTEP'):
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
        self.ifacefile = None
        self.comment = comment
        self.ifacefile = interface_file
        self.days = init_days 
        DATA_file = self.case.with_suffix('.DATA')
        self.start = get_start(DATA_file)
        # Read start-date from restart file if RESTART in DATA-file
        self.restart_file = get_restart_file_step(DATA_file)[0]
        if self.restart_file and self.restart_file.is_file():
            self.start = get_start_UNRST(file=self.restart_file)
        #print(self.start)
        self.tstep = 0
        self._schedule = []
        self.file = None
        # Ignore case in file extension
        self.file = is_file_ignore_suffix_case( self.case.with_suffix(ext) )
        if self.file:
            self._schedule = self.days_and_actions()
        # Add end time 
        self.insert(days=T, remove=True)
        #[print(s) for s in self._schedule]
        #print(self.info())
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
    def days_and_actions(self, remove_end=True):                           # Schedule
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
        # Return only non-empty [day, action] pairs
        return [[float(d), a+'\n'] for (d,a) in zip(days, actions) if a]

    #--------------------------------------------------------------------------------
    def append(self, action=None, tstep=None, append_line=-4):             # Schedule
    #--------------------------------------------------------------------------------
        '''
        line : line number of TSTEP
        '''
        if action is None and tstep is None:
            return
        if tstep == 0:
            raise SystemError(f'ERROR Simulation stopped because schedule gave TSTEP 0 at {self.days} days, check {self.file}')
        with open(self.ifacefile, 'r') as f:
            lines = f.readlines()
        n = len(lines) + append_line
        if n > 0:
            if tstep is not None:
                # Replace TSTEP
                # +1 because we edit the value on the line after TSTEP
                lines[n+1] = f'{tstep} /\n'
            # Append action
            if action:
                lines.insert(n, action)
            with open(self.ifacefile, 'w') as f:
                f.write(''.join(lines))

    #--------------------------------------------------------------------------------
    def check(self):                                                       # Schedule
    #--------------------------------------------------------------------------------
        # just a check...
        print(f'{self.days}, {self.start+timedelta(days=self.days)}')
        with open(self.ifacefile, 'r') as f:
            lines = f.readlines()
        print(self.ifacefile, ': ', ''.join(lines[-10:]))
        print('Schedule:')
        print(self._schedule)

    #--------------------------------------------------------------------------------
    def update(self):                                                      # Schedule
    #--------------------------------------------------------------------------------        
        action = new_tstep = None
        # Update days from previous step
        self.days += self.tstep
        # Append action if time is right
        if self.days >= self._schedule[0][0]:
            action = self._schedule.pop(0)[1]
        # Get tstep for next step (given by IORSim in satnum.dat)
        self.tstep = get_tsteps(self.ifacefile)[0]
        #print(f'START: tstep:{self.tstep}, days:{self.days}, schedule:{self._schedule[:2]}')
        # Check arrival of next event and adjust tstep if neccessary
        if self._schedule and self.days + self.tstep + 1e-8 > self._schedule[0][0]:
            self.tstep = new_tstep = self._schedule[0][0] - self.days
        self.append(action=action, tstep=new_tstep)
        #self.check()
        #print(f'END: tstep:{self.tstep}, days:{self.days}, schedule:{self._schedule[:2]}')
        return self.days


#====================================================================================
class Simulation:
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
        DATA_file = Path(str(self.root)+'.DATA') 
        if not DATA_file.is_file():
            raise SystemError(f'ERROR No DATA-file found in {DATA_file.parent}')
        self.update = namedtuple('update',['status','progress','plot','message'])(status, progress, plot, message)
        self.pause = pause
        if delete:
            del_convert = del_merge = True
        self.output = namedtuple('output',['convert','merge','del_convert','del_merge'])(convert, merge, del_convert, del_merge)
        self.runlog = None
        if root and not to_screen:
            self.runlog = safeopen(Path(root).parent/(self.name+'.log'), 'w')
        self.print2log = lambda txt, **kwargs: print(txt, file=self.runlog, flush=True, **kwargs)
        self.current_run = None
        self.runs = runs
        self.run_sim = None
        self.ior = self.ecl = None
        self.T = 0
        self.mode = mode
        self.schedule = None
        self.restart = False
        if root:
            kwargs.update({'root':str(root), 'runlog':self.runlog})
            self.prepare_mode(**kwargs)

    #-----------------------------------------------------------------------
    def close(self):
    #-----------------------------------------------------------------------
        self.runs = self.ior = self.ecl = self.current_run = self.schedule = None
        if self.runlog:
            self.runlog.close()


    #-----------------------------------------------------------------------
    def versions(self):
    #-----------------------------------------------------------------------
        txt = f'Python: {get_python_version()}\n'
        txt += f'psutil: {psutil_version}\n'
        txt += f'Application: {__version__}\n'
        return txt

    #-----------------------------------------------------------------------
    def ready(self):
    #-----------------------------------------------------------------------
        if self.run_sim:
            return True
        else:
            return False

    #-----------------------------------------------------------------------
    def prepare_mode(self, iorexe=None, eclexe=None, ior_keep_alive=False, ecl_keep_alive=False, time=0, tsteps=None, restart_days=None, **kwargs):
    #-----------------------------------------------------------------------
        DATA_file = Path(self.root).with_suffix('.DATA')
        try:
            self.restart_days = int( restart_days or get_restart_time_step(DATA_file)[0] )
        except SystemError as e:
            self.update.message(f'{e}')
            return
        self.restart = self.restart_days > 0
        self.tsteps = tsteps or get_tsteps(DATA_file)
        if self.tsteps == [0]:
            self.tsteps = get_tsteps_from_schedule_files(self.root)
        if not self.mode:
            self.mode = self.mode_from_case()
        # Backward simulation
        if self.mode=='backward':
            time_ecl = sum(self.tsteps) + self.restart_days
            ior_integration = get_keyword(str(self.root)+'.trcinp', '\*INTEGRATION', end='\*')[0]
            ior_dt_min = ior_integration[4] 
            if time < time_ecl:
                time = time_ecl + 1
                self.update.message(text=f'INFO Simulation time set to sum of TSTEP ({sum(self.tsteps)}) and RESTART ({self.restart_days}) in Eclipse input')
                #print(f'   INFO Simulation time increased to > {time_ecl} days, sum of TSTEP ({sum(self.tsteps)}) and RESTART ({self.restart_days}) in {DATA_file.name}')
            self.T = time 
            # No problem if N yields a too large t, the simulation automatically ends when t == T. 
            N = int(time/ior_dt_min) + 2 + len(self.tsteps)
            if N > MAX_ITERATIONS:
                self.update.message(f'ERROR Too many iterations: time/timestep = {time}/{ior_dt_min} = {time/ior_dt_min:.0f}')
                self.run_sim = None
                return
            kwargs.update({'N':N, 'T':self.T, 'tsteps':self.tsteps, 'update':self.update})
            # Init runs
            self.run_sim = self.backward
            self.runs = [Ecl_backward(exe=eclexe, keep_alive=ecl_keep_alive, **kwargs), Ior_backward(exe=iorexe, keep_alive=ior_keep_alive, **kwargs)]
            self.ecl, self.ior = self.runs
            # Set up schedule of commands to pass to satnum-file
            self.schedule = Schedule(self.root, T=self.T, init_days=time_ecl, interface_file=self.ior.satnum)
        # Forward simulation
        if self.mode=='forward':
            time_ecl = sum(self.tsteps)
            if time != time_ecl:
                time = time_ecl
            self.T = time
            #print(f'time: {time}, time_ecl: {time_ecl}')
            kwargs.update({'T':self.T})
            self.run_sim = self.forward
            if not self.runs:
                self.runs = ('eclipse','iorsim')
            for name in self.runs:
                if name=='eclipse':
                    self.ecl = Ecl_forward(exe=eclexe, **kwargs)
                if name=='iorsim':
                    self.ior = Ior_forward(exe=iorexe, **kwargs)
            self.runs = [run for run in (self.ecl, self.ior) if run]
        # Check neccessary input files
        try:
            for run in self.runs:
                run.check_input()
        except SystemError as e:
            self.run_sim = None  # => ready() returns False
            self.update.message(f'{e}')
            return

    #-----------------------------------------------------------------------
    def forward(self): 
    #-----------------------------------------------------------------------
        #print('Forward run')
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
            run.t = run.time_and_step()[0]
            #print(run.name, t, run.T)
            if run.t < run.T:
                run.unexpected_stop_error()
                #msg = 'ERROR ' + run.name + ' stopped unexpectedly, check the log'
                #raise SystemError(msg)
            run_time += run.run_time()
            ret = run.complete_msg(run_time=run_time)
        return ret


    #-----------------------------------------------------------------------
    def backward(self): 
    #-----------------------------------------------------------------------
        self.update.progress(value=-self.runs[0].T)
        ecl, ior = self.runs
        # Start runs
        for run in self.runs:
            run.delete_output_files()
            #run.start(update=self.update, restart=self.restart)
            run.start(restart=self.restart)
        # The schedule appends keywords to the interface file (satnum.dat)
        ecl.t = ior.t = self.schedule.update()
        # Update progress
        if self.restart:
            # Fix progress for restart runs
            self.update.progress(value=self.ecl.t, min=self.restart_days)
        else:
            # Reset progress-time for more accurate time-estimate   
            self.update.progress(value=ior.t, n0=ior.t)
        # Start timestep loop
        while ior.t < ior.T:
            self.print2log(f'\nStep {ecl.n+1}/{ecl.N}')
            self.update.status(run=ecl, mode=self.mode)
            ecl.run_one_step(ior.satnum)
            # Run IORSim to prepare satnum input for the next Eclipse run
            self.update.status(run=ior, mode=self.mode)
            ior.run_one_step()
            ecl.t = ior.t = self.schedule.update()
            self.print2log(f'days = {ior.t:.3f}/{ior.T} ({self.schedule.now()})')
            self.update.progress(value=ior.t)
            self.update.plot()
        # Timestep loop finished
        for run in self.runs:
            self.update.status(value=f'Stopping {run.name}...')
            run.quit()
        return ecl.complete_msg()

    #-----------------------------------------------------------------------
    def run(self):
    #-----------------------------------------------------------------------
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


    #-----------------------------------------------------------------------
    def convert_and_merge(self, case=None, only_convert=False, only_merge=False):
    #-----------------------------------------------------------------------
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

    #-----------------------------------------------------------------------
    def convert_restart(self, case=None, fast=True):
    #-----------------------------------------------------------------------
        # Convert from formatted (ascii) to unformatted (binary) restart file
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

    #-----------------------------------------------------------------------
    def merge_restart(self, case=None):
    #-----------------------------------------------------------------------
        # Merge Eclipse and IORSim restart files
        self.update.status(value='Merging Eclipse and IORSim restart files...')
        self.update.progress(value=0)   # Reset progress time
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

    #-----------------------------------------------------------------------
    def info_header(self):
    #-----------------------------------------------------------------------
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
        s += f'    {"Folder":10s}: {Path(self.root).parent}\n'
        s += f'    {"Log-files":10s}: {", ".join([Path(file).name for file in logfiles])}\n'
        s += '\n'
        return s

    #-----------------------------------------------------------------------
    def set_time(self, time):
    #-----------------------------------------------------------------------
        self.T = time
        for run in self.runs:
            run.set_time(time)

    #-----------------------------------------------------------------------
    def get_time(self):
    #-----------------------------------------------------------------------
        return self.T

    #-----------------------------------------------------------------------
    def mode_from_case(self):
    #-----------------------------------------------------------------------
        data = str(self.root)+'.DATA'
        if Path(data).is_file():
            if file_contains(data, text='READDATA', comment='--', end='END'):
                return 'backward'
            else:
                return 'forward'
        else:
            return None


    #-----------------------------------------------------------------------
    def cancel(self):
    #-----------------------------------------------------------------------
        for run in self.runs:
            run.canceled = True




#############################################################################
#                                                                           #
#                    Various utility functions                              #
#                                                                           #
#############################################################################



# #--------------------------------------------------------------------------------
# def iorexe_from_settings(settings_file, iorexe):
# #--------------------------------------------------------------------------------
#     iorsim = get_keyword(settings_file, 'iorsim', with_space=False)
#     print(iorsim)
#     # Find iorexe in settings.txt if missing
#     settings_file = Path(settings_file)
#     if settings_file.is_file():
#         with open(settings_file) as f:
#             for line in f:
#                 if line.strip().startswith('#'):
#                     continue
#                 try: var, val = line.split()
#                 except ValueError: break
#                 if var=='iorsim':
#                     break 
#         print(val)
#         return val
#     raise SystemError('\n   Missing IORSim executable: '+str(iorexe)+'\n')


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
    # parser.add_argument('-rft_size',       help='Backward mode: Only check size of RFT-file, default is full check', action='store_true')
    parser.add_argument('-iorsim',         help="Run only iorsim", action='store_true')
    parser.add_argument('-eclipse',        help="Run only eclipse", action='store_true')
    parser.add_argument('-v',              default=LOG_LEVEL, help='Verbosity level, higher number increase verbosity, default is 3', type=int)
    parser.add_argument('-keep_files',     help='Interface-files are not deleted after completion', action='store_true')
    parser.add_argument('-to_screen',      help='Print program log to screen', action='store_true')
    parser.add_argument('-only_convert',   help='Only convert+merge and exit', action='store_true')
    parser.add_argument('-only_merge',     help='Only merge and exit', action='store_true')
    parser.add_argument('-delete',         help='Delete obsolete output files after convert and merge has finished', action='store_true')
    parser.add_argument('-ecl_alive',      help=f'Keep Eclipse alive at least {ECL_ALIVE_LIMIT} seconds', action='store_true')
    parser.add_argument('-ior_alive',      help=f'Keep IORSim alive', action='store_true')
    parser.add_argument('-check_input',    help='Check IORSim input file keywords', action='store_true', dest='check_input_kw')
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
           verbose=LOG_LEVEL):
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
                     ior_keep_alive=ior_alive, runs=runs, mode=mode, check_input_kw=check_input, verbose=verbose)

    if not sim.ready():
        return 

    if only_convert or only_merge:
        sim.convert_and_merge(case=sim.root, only_merge=only_merge)
        return

    if sim.mode=='forward':
        sim.set_time(int(sum(get_tsteps(str(root)+'.DATA'))))
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
           check_input=args['check_input_kw'], verbose=args['v'])
    os_exit(0)


######################################################################################

if __name__ == '__main__':

    main()
