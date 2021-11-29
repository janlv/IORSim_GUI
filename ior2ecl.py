#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import Counter, namedtuple
from mmap import ACCESS_READ, mmap
import os
from pathlib import Path
import sys
from argparse import ArgumentParser
from datetime import datetime, timedelta
from time import sleep
from itertools import accumulate
from psutil import NoSuchProcess
import shutil
import traceback
from re import search, compile

from IORlib.utils import get_python_version, list2text, print_error, is_file_ignore_suffix_case, number_of_blocks, remove_comments, safeopen, Progress, warn_empty_file, silentdelete, delete_files_matching, file_contains
from IORlib.runner import runner
from IORlib.ECL import check_blocks, get_restart_file_step, get_start_UNRST, get_time_step_MSG, get_restart_time_step, get_start, get_time_step_UNRST, get_time_step_UNSMRY, get_tsteps, get_tsteps_from_schedule_files, unfmt_file, fmt_file, Section



#====================================================================================
class eclipse(runner):                                                      # eclipse
#====================================================================================
    #--------------------------------------------------------------------------------
    def __init__(self, root=None, exe='eclrun', **kwargs):
    #--------------------------------------------------------------------------------
        #print('eclipse.__init__: ',root, exe, kwargs)
        #root = kwargs.pop('root', None)
        root = str(root)        
        #exe = kwargs.pop('exe', None) or 'eclrun' # Default executable
        #print(exe)
        exe = str(exe)
        super().__init__(name='Eclipse', case=root, exe=exe, cmd=[exe, 'eclipse', root], **kwargs)                        
        self.unrst = Path(root+'.UNRST')
        #self.unsmry = unfmt_file(root+'.UNSMRY')
        self.unsmry = Path(root+'.UNSMRY')
        self.rft = Path(root+'.RFT')
        self.msg = Path(root+'.MSG')
        self.inputfile = Path(root+'.DATA')
        self.is_iorsim = False
        self.is_eclipse = True

    #--------------------------------------------------------------------------------
    def time_and_step(self):                                                # eclipse
    #--------------------------------------------------------------------------------
        #for block in self.unsmry.blocks(only_new=True):
        t = n = 0
        # 
        if self.unsmry.is_file():
            t, n = get_time_step_UNSMRY(file=self.unsmry)
        elif self.msg.is_file():
            t, n = get_time_step_MSG(file=self.msg) 
        elif self.unrst.is_file():
            t, n = get_time_step_UNRST(file=self.unrst, end=True)
        #print('{}: time = {}, step = {}'.format(self.name, t, n))
        return int(t), int(n)        

    #--------------------------------------------------------------------------------
    def delete_output_files(self):                                          # eclipse
    #--------------------------------------------------------------------------------
        silentdelete( [self.unrst, self.rft] )
        silentdelete( [str(self.case)+ext for ext in ('.SMSPEC','.UNSMRY','.RTELOG','.RTEMSG','_ECLIPSE.UNRST')] )
        delete_files_matching( [str(self.case)+fil for fil in ('*.session*', '*.dbprtx.lock', '*.dbprtx.lock-journal')] )

    #--------------------------------------------------------------------------------
    def check_input(self):                                                  # eclipse
    #--------------------------------------------------------------------------------
        super().check_input()
        msg = 'WARNING Unable to start ' + self.name + ': '
        ### check root.DATA exists
        inp_file = str(self.case)+'.DATA'
        if not Path(inp_file).is_file():
            raise SystemError(msg + 'missing input file ' + inp_file)


#====================================================================================
class forward_mixin:
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
class ecl_forward(forward_mixin, eclipse):                              # ecl_forward
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
class backward_mixin:
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
class ecl_backward(backward_mixin, eclipse):                           # ecl_backward
#====================================================================================
    #--------------------------------------------------------------------------------
    def __init__(self, check_unrst=True, check_rft=True, rft_size=False, **kwargs):
    #--------------------------------------------------------------------------------
        super().__init__(ext_iface='I{:04d}', ext_OK='OK', **kwargs)
        self.tsteps = kwargs.get('tsteps') or get_tsteps(self.case.with_suffix('.DATA'))
        self.update = kwargs.get('update') or None
        self.init_tsteps = len(self.tsteps) 
        self.check_unrst = check_unrst
        self.check_rft = check_rft
        self.rft_size = rft_size
        self.unrst_check = check_blocks(self.unrst, start='SEQNUM', end='ENDSOL', var='nwell')
        self.rft_check = check_blocks(self.rft, start='TIME', end='CONNXT')
        self.rft_start_size = 0


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
        super().start()  
        self.wait_for( self.unrst.exists, error=self.unrst.name+' not created')
        self.wait_for( self.rft.exists, error=self.rft.name+' not created')
        self.update.status(value=f'{self.name} running...')
        # Check if restart-file (UNRST) is flushed   
        nblocks = 1 + self.init_tsteps # 1 for 0'th SEQNUM
        for i in range(nblocks):
            if i > 0:
                self.update_function(progress=not restart, plot=True)
            self.check_UNRST_file() 
        self.nwell = self.unrst_check.var('nwell')
        nwell_max = nblocks*self.nwell
        rft_wells = self.check_RFT_file(nwell_max=nwell_max, nwell_min=self.nwell)
        self.suspend()
        if self.rft_size:
            self.init_RFT_size_check(rft_wells, nwell_max)


    #--------------------------------------------------------------------------------
    def run_one_step(self, satnum_file):                               # ecl_backward
    #--------------------------------------------------------------------------------
        if self.rft_size:
            self.rft_start_size = self.rft.stat().st_size
        ### run Eclipse
        self.interface_file(self.n).copy(satnum_file, delete=True)
        self.OK_file().create_empty()
        self.resume()
        self.wait_for( self.OK_file().is_deleted, error=self.OK_file().name()+' not deleted' )
        if self.check_unrst:
            self.check_UNRST_file()
        if self.check_rft:
            if self.rft_size:
                self.wait_for( self.check_RFT_size )
            else:
                self.check_RFT_file(nwell_max=self.nwell)
        self.suspend()
        self.n += 1

    #--------------------------------------------------------------------------------
    def check_unformatted_file(self, file, print_block=False):             # ecl_backward
    #--------------------------------------------------------------------------------
        print(f'{file} size: {file.stat().st_size}')
        for block in unfmt_file(file).blocks(only_new=True):
            if block.key() in ('SEQNUM','TIME'):
                print(f'{block.key()} : {block.data()}')
            print_block and block.print()


    #--------------------------------------------------------------------------------
    def check_UNRST_file(self, nblocks=1, pause=0.01):   # ecl_backward
    #--------------------------------------------------------------------------------
        self.wait_for( self.unrst_check.blocks_complete, nblocks=nblocks, log=self.unrst_check.info, pause=pause )

    #--------------------------------------------------------------------------------
    def check_RFT_file(self, nwell_max=0, nwell_min=0, limit=100):        # ecl_backward
    #--------------------------------------------------------------------------------
        ###
        ###  cannot always require nblocks=2*nwell in the initial RFT-check. In some situations
        ###  all wells may not be ready after the TSTEP in the DATA-file. The RFT-check
        ###  starts to look for 2*nwell blocks. If the check fails, the check is repeated
        ###  with nblocks-1, and so on until nblocks==nwell.
        ###
        for nblocks in range(nwell_max, nwell_min-1, -1):
            passed = self.wait_for( self.rft_check.blocks_complete, nblocks=nblocks, log=self.rft_check.info, limit=limit)
            if passed:
                break
            if nblocks==nwell_min:
                if nwell_min == 0:
                    self._print('WARNING! No TIME blocks found in the RFT-file. Are all wells closed?')
                else:
                    self._print(f'WARNING! Only {nwell_min} TIME blocks found in the RFT-file')
        return nblocks


    #--------------------------------------------------------------------------------
    def init_RFT_size_check(self, init_wells, total_wells):            # ecl_backward
    #--------------------------------------------------------------------------------
        #print(init_wells, total_wells)
        if init_wells != total_wells:
            # Turn off simple RFT-file size check if some wells are missing in initial RFT-file 
            self.rft_size = False
            info = 'Turned on full RFT-check due to missing wells in initial file' 
            self._print(info)
        if self.rft_size: 
            # Check size of initial RFT file
            self.rft_size = int(0.5*self.rft.stat().st_size)
            if 2*self.rft_size != self.rft.stat().st_size:
                self.print2log('\nWARNING! Initial size of RFT size not even!\n')


    #--------------------------------------------------------------------------------
    def check_RFT_size(self):                                          # ecl_backward
    #--------------------------------------------------------------------------------
        diff = self.rft.stat().st_size-self.rft_start_size
        if diff==self.rft_size:
            #if self.rft_check.file.tail_block_is('CONNXT'):
            return True
            #else:
            #    return False
        else:
            return False

    #--------------------------------------------------------------------------------
    def quit(self):                                                    # ecl_backward
    #--------------------------------------------------------------------------------
        #self._print(f'appending END to {self.interface_file(self.n).name()}')
        self.print_suspend_errors()
        self.interface_file(self.n).create_from_string('END')
        self.OK_file().create_empty()
        super().quit()



#====================================================================================
class iorsim(runner):                                                        # iorsim
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
    def __init__(self, root=None, exe='IORSimX', args='', relative_root=True, **kwargs):
    #--------------------------------------------------------------------------------
        #print('iorsim.__init__: ',root, exe, args, kwargs)
        exe = str(exe)
        if '.exe' not in exe and sys.platform == 'win32':
            exe += '.exe'
        # IORSim only accepts root relative to the current directory
        abs_root = str(root)
        cwd = Path().cwd()
        if relative_root:
            if str(cwd) in str(root):
                root = Path(root).relative_to(cwd)
        else:
            root = Path(root).absolute()
        #print(root)
        #cmd = [exe, '-root_name='+str(root)] + args.split()
        cmd = [exe, str(root)] + args.split()
        super().__init__(name='IORSim', case=root, exe=exe, cmd=cmd, **kwargs)
        self.trcconc = None
        self.funrst = Path(abs_root+'_IORSim_PLOT.FUNRST')
        self.unrst = self.funrst.with_suffix('.UNRST')
        self.inputfile = Path(abs_root+'.trcinp')
        self.check_keywords = kwargs.get('check_input_kw') or False
        self.is_iorsim = True
        self.is_eclipse = False

    #--------------------------------------------------------------------------------
    def check_input(self):                                                   # iorsim
    #--------------------------------------------------------------------------------
        def raise_error(error):
            raise SystemError(f'ERROR Error in IORSim input file: {error}')    

        super().check_input()

        # Check if input-file exists
        if not self.inputfile.is_file():
            raise SystemError(f'ERROR Missing IORSim input file {self.inputfile.name}')
        # Check if required keywords are used, and if the order is correct 
        if self.check_keywords:
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
                raise_error('required keyword' + (len(missing)>1 and f's {list2text(missing)} are' or f' {missing[0]} is') + ' missing')
            # Remove ignored keywords
            file_kw = [kw for kw in file_kw if kw not in self.keywords.ignored]
            # Make ordered list of input-file keywords
            ordered_kw = [o for o in self.keywords.keys if o in file_kw]
            # Check keyword order
            for o,f in zip(ordered_kw, file_kw):
                if o != f:                        
                    raise_error(f'expected keyword {o} but got {f}')    


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
class ior_forward(forward_mixin, iorsim):                               # ior_forward
#====================================================================================
    pass

#====================================================================================
class ior_backward(backward_mixin, iorsim):                             # ior_backward
#====================================================================================
    #--------------------------------------------------------------------------------
    def __init__(self, **kwargs):
    #--------------------------------------------------------------------------------
        # Call iorsim.__init__()
        super().__init__(args='-readdata', ext_iface='IORSimI{:04d}', ext_OK='IORSimOK', **kwargs)
        self.tsteps = kwargs.get('tsteps') or get_tsteps(self.case.with_suffix('.DATA'))
        self.update = kwargs.get('update') or None
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

    #-----------------------------------------------------------------------
    def info(self):
    #-----------------------------------------------------------------------
        s =   'Schedule\n'
        s += f'  file   : {self.file}\n'
        s += f'  start  : {self.start}\n'
        s += f'  days   : {self.days}\n'
        s += f'  length : {len(self._schedule)}\n'
        return s

    #-----------------------------------------------------------------------
    def insert(self, index=None, days=None, action='', remove=False):
    #-----------------------------------------------------------------------
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

    #-----------------------------------------------------------------------
    def get_type(self, string):
    #-----------------------------------------------------------------------
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

    #-----------------------------------------------------------------------
    def days_and_actions(self, remove_end=True):
    #-----------------------------------------------------------------------
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
            regex = compile(r'\n?\s*\bTSTEP\b\s+([0-9.* ]+)\s*/\s+')
        # Create list of (time, (start, end))-tuple
        date_span = [(' '.join(m.groups()), m.span()) for m in regex.finditer(schfile)]
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
    def append(self, action=None, tstep=None, append_line=-4):             # schedule
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
    def check(self):                                               # schedule
    #--------------------------------------------------------------------------------
        # just a check...
        print(f'{self.days}, {self.start+timedelta(days=self.days)}')
        with open(self.ifacefile, 'r') as f:
            lines = f.readlines()
        print(self.ifacefile, ': ', ''.join(lines[-10:]))
        #print('Schedule:')
        #print(self._schedule)

    #--------------------------------------------------------------------------------
    def update(self):                                               # schedule
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
        if self._schedule and self.days + self.tstep > self._schedule[0][0]:
            self.tstep = new_tstep = self._schedule[0][0] - self.days
        self.append(action=action, tstep=new_tstep)
        #self.check()
        #print(f'END: tstep:{self.tstep}, days:{self.days}, schedule:{self._schedule[:2]}')
        return self.days

#====================================================================================
class simulation:
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
        self.print2log = lambda txt: print(txt, file=self.runlog, flush=True)
        self.print2log(get_python_version())
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
    def ready(self):
    #-----------------------------------------------------------------------
        if self.run_sim:
            return True
        else:
            return False

    #-----------------------------------------------------------------------
    def prepare_mode(self, iorexe=None, eclexe=None, time=0, tsteps=None, restart_days=None, **kwargs):
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
            if time < time_ecl:
                time = time_ecl + 1
                self.update.message(text=f'INFO Simulation time set to sum of TSTEP ({sum(self.tsteps)}) and RESTART ({self.restart_days}) in Eclipse input')
                #print(f'   INFO Simulation time increased to > {time_ecl} days, sum of TSTEP ({sum(self.tsteps)}) and RESTART ({self.restart_days}) in {DATA_file.name}')
            self.T = time 
            # We assume 1 day timesteps, and set total steps N larger than what is needed.
            # The simulation is anyway ended when t == T. 
            N = int(time) #- self.restart_days
            kwargs.update({'N':N, 'T':self.T, 'tsteps':self.tsteps, 'update':self.update})
            # Init runs
            self.run_sim = self.backward
            self.runs = [ecl_backward(exe=eclexe, **kwargs), ior_backward(exe=iorexe, **kwargs)]
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
                    self.ecl = ecl_forward(exe=eclexe, **kwargs)
                if name=='iorsim':
                    self.ior = ior_forward(exe=iorexe, **kwargs)
            self.runs = [run for run in (self.ecl, self.ior) if run]
        # Check neccessary input files
        try:
            for run in self.runs:
                run.check_input()
        except SystemError as e:
            self.run_sim = None  # => ready() returns False
            self.update.message(f'{e}')
            return
        # Print simulation data
        self.print2log(self.info_header())
        #if self.schedule:
        #    self.print2log(self.schedule.info())            


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
                raise SystemError('ERROR ' + run.name + ' stopped unexpectedly, check the log')
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
            #self.update.progress(value=0)
            self.update.progress(value=ior.t, n0=ior.t)
        # Start timestep loop
        while ior.t < ior.T:
            self.print2log(f'\nLoop step {ecl.n}/{ecl.N}')
            self.update.status(run=ecl, mode=self.mode)
            ecl.run_one_step(ior.satnum)
            # Need a short stop after Eclipse has finished, otherwise IORSim sometimes stops 
            #sleep(self.pause)
            # Run IORSim to prepare satnum input for the next Eclipse run
            self.update.status(run=ior, mode=self.mode)
            ior.run_one_step()
            ecl.t = ior.t = self.schedule.update()
            self.print2log(f'days = {ior.t}/{ior.T}')
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
            traceback.print_exc()
            msg = traceback.format_exc()
        finally:
            # Kill possible remaining processes
            self.print2log('')
            [run.kill_and_clean() for run in self.runs]
            self.print2log('\n======  ' + msg + '  ======')
            self.current_run = None
            self.update.progress(value=0)   # Reset progress time
            self.update.plot()
            #self.update.status(value=msg, newline=True)
            conv_msg = ''
            if self.output.convert and success and any([run.is_iorsim for run in self.runs]):
                sleep(0.05)  # Need a short break here to make the GUI progressbar responsive
                conv_msg = self.convert_and_merge(case=self.root)
            self.runs = []
            if self.runlog:
                self.runlog.close()
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
            self.print2log('\n===== '+msg+' ======')
            self.update.status(value=msg, newline=True)
            if not success:
                return msg
        return ''

    #-----------------------------------------------------------------------
    def convert_restart(self, case=None, fast=True):
    #-----------------------------------------------------------------------
        # Convert from formatted (ascii) to unformatted (binary) restart file
        self.update.status(value='Converting restart file...')
        ior = self.ior or iorsim(root=case)   
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
        ecl = self.ecl or eclipse(root=case)   
        ior = self.ior or iorsim(root=case)   
        if ecl.unrst.is_file() and ior.unrst.is_file():
            backup_ecl = Path(str(ecl.case)+'_ECLIPSE.UNRST')
            if backup_ecl.is_file():
                # This is a pure IORSim run and backup already exists; restore backup
                shutil.copy(backup_ecl, ecl.unrst)
            else:
                # No backup exists; create backup copy
                shutil.copy(ecl.unrst, backup_ecl)
        else:
            missing = [f.name for f in (ecl.unrst, ior.unrst) if not f.is_file()]
            return False, f'Unable to merge restart files due to missing files: {", ".join(missing)}'
        start = datetime.now()
        try:
            # Define the sections in the restart files where the stitching is done
            ecl_sec = Section(ecl.unrst, start_before='SEQNUM', end_before='SEQNUM', skip_sections=(0,))
            ior_sec = Section(ior.unrst, start_after='DOUBHEAD', end_before='SEQNUM')
            fname =  Path(str(ecl.case)+'_MERGED.UNRST') 
            merged_file = unfmt_file(fname).create(ecl_sec, ior_sec, 
                                                   progress=lambda n: self.update.progress(value=n),
                                                   cancel=ior.stop_if_canceled)
            # Rename merged UNRST-file to original Eclipse restart file
            if merged_file and merged_file.is_file():
                merged_file.replace(ecl.unrst)
            else:
                return False, f'Unable to merge {ecl.unrst} and {ior.unrst}'
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



#--------------------------------------------------------------------------------
def iorexe_from_settings(settings_file, iorexe):
#--------------------------------------------------------------------------------
    # Find iorexe in settings.txt if missing
    settings_file = Path(settings_file)
    if settings_file.is_file():
        with open(settings_file) as f:
            for line in f:
                if line.strip().startswith('#'):
                    continue
                try: var, val = line.split()
                except ValueError: break
                if var=='iorsim':
                    break 
        return val
    raise SystemError('\n   Missing IORSim executable: '+str(iorexe)+'\n')


#--------------------------------------------------------------------------------
def case_from_casedir(case_dir, root):
#--------------------------------------------------------------------------------
    # Find case in casedir if given DATA-file is missing
    case_dir = Path(case_dir)
    if case_dir.is_dir() and (case_dir/root/(root+'.DATA')).is_file():
        return case_dir/root/root
    raise SystemError('\n   '+root+'.DATA'+' not found in '+str(case_dir/root)+'\n')
    #raise SystemExit

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
    parser.add_argument('-rft_size',       help='Backward mode: Only check size of RFT-file, default is full check', action='store_true')
    parser.add_argument('-iorsim',         help="Run only iorsim", action='store_true')
    parser.add_argument('-eclipse',        help="Run only eclipse", action='store_true')
    parser.add_argument('-v',              default=3, help='Verbosity level, higher number increase verbosity, default is 3', type=int)
    parser.add_argument('-keep_files',     help='Interface-files are not deleted after completion', action='store_true')
    parser.add_argument('-to_screen',      help='Print program log to screen', action='store_true')
    parser.add_argument('-only_convert',   help='Only convert+merge and exit', action='store_true')
    parser.add_argument('-only_merge',     help='Only merge and exit', action='store_true')
    parser.add_argument('-delete',         help='Delete obsolete output files after convert and merge has finished', action='store_true')
    parser.add_argument('-alive_children', help='Only stop parent-processes (approx. 5%% faster, but might be more unstable)', action='store_true')
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
        args['iorexe'] = iorexe_from_settings(settings_file, args['iorexe'])
    return args

@print_error
#--------------------------------------------------------------------------------
def runsim(root=None, time=None, iorexe=None, eclexe='eclrun', to_screen=False, 
           check_unrst=True, check_rft=True, rft_size=False, keep_files=False, 
           only_convert=False, only_merge=False, convert=True, merge=True, delete=True,
           stop_children=True, only_eclipse=False, only_iorsim=False, check_input=False):
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

    sim = simulation(root=root, time=time, iorexe=iorexe, eclexe=eclexe, 
                     check_unrst=check_unrst, check_rft=check_rft, rft_size=rft_size,  
                     keep_files=keep_files, progress=progress, status=status, message=message, to_screen=to_screen,
                     convert=convert, merge=merge, delete=delete, stop_children=stop_children,
                     runs=runs, mode=mode, check_input_kw=check_input)

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
    args = parse_input(case_dir=case_dir, settings_file=settings_file)
    runsim(root=args['root'], time=args['days'], check_unrst=(not args['no_unrst_check']), check_rft=(not args['no_rft_check']), rft_size=args['rft_size'], 
           to_screen=args['to_screen'], eclexe=args['eclexe'], iorexe=args['iorexe'],
           delete=args['delete'], keep_files=args['keep_files'], only_convert=args['only_convert'], only_merge=args['only_merge'],
           stop_children=(not args['alive_children']), only_eclipse=args['eclipse'], only_iorsim=args['iorsim'],
           check_input=args['check_input_kw'])
    os._exit(0)


######################################################################################

if __name__ == '__main__':

    main()
