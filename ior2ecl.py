#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import Counter, namedtuple
from itertools import accumulate, chain, dropwhile, repeat, takewhile
from operator import itemgetter
from os import environ
from pathlib import Path
from sys import platform
from argparse import ArgumentParser
from datetime import datetime, timedelta
from time import sleep
from shutil import copy as shutil_copy
from traceback import print_exc as trace_print_exc, format_exc as trace_format_exc
from re import compile as re_compile, MULTILINE, finditer
from os.path import relpath
from threading import Thread
from struct import error as struct_error

from numpy import prod, sum as npsum
from psutil import NoSuchProcess

from IORlib.utils import (batched, convert_float_or_str, dates_after,
    empty_folder, flat_list, flatten, get_terminal_environment, list2text, pairwise, print_error, remove_comments, safeopen, 
    Progress, silentdelete, delete_files_matching, tail_file, LivePlot, running_jupyter)
from IORlib.runner import Runner
from IORlib.ECL import (FUNRST_file, DATA_file, File, INIT_file, RFT_file, Restart, SMSPEC_file, UNRST_file,
    UNSMRY_file, MSG_file, PRT_file, IX_input, unfmt_file)

__version__ = '3.7.3'
__author__ = 'Jan Ludvig Vinningland'

DEBUG = False

# Options
COPY_CHEMFILE = True
SCHEDULE_SKIP_EMPTY = False

# Constants
IOR_RESTART_NAMETAG = '_IORSim_PLOT'
SLB_BACKUP_NAMETAG = '_SLB'
#IOR_RESTART_ENDKEY = 'SATNUM'
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
RFT_CHECK_NITER = 100
# To avoid re-merging merged UNRST-files, this file is created in the case-folder
MERGE_OK_FILE = '.merge_OK'



#====================================================================================
class SLBRunner(Runner):                                                  # SLBRunner
#====================================================================================
    """ Common runner class for Eclipse and Intersect """

    #--------------------------------------------------------------------------------
    def __init__(self, root=None, name='SLB', exe='eclrun', cmd=None, **kwargs):
    #--------------------------------------------------------------------------------
        #print('eclipse.__init__: ',root, exe, kwargs)
        root = str(root)
        exe = str(exe)
        #super().__init__(name=name, case=root, exe=exe, cmd=[exe, cmd, root], app_name=cmd, **kwargs)
        nproc = kwargs.get('nproc')
        nproc = ('--np', str(nproc)) if nproc else ()
        nice = kwargs.get('nice')
        nice = ('nice', f'-{nice}') if nice else ()
        super().__init__(name=name, case=root, exe=exe, cmd=[*nice, exe, *nproc, cmd, root], app_name=cmd, **kwargs)
        self.update = kwargs.get('update') or None
        self.unrst = UNRST_file(root, wait_func=self.wait_for, timer=self.verbose==LOG_LEVEL_MAX)
        self.rft = RFT_file(root, wait_func=self.wait_for, timer=self.verbose==LOG_LEVEL_MAX)
        self.unsmry = UNSMRY_file(root)
        self.msg = MSG_file(root)
        self.prt = PRT_file(root)
        self.input_file = None
        # self.unrst_backup_file = Path(f'{root}_{self.name.upper()}.UNRST')
        self.is_iorsim = False
        self.is_eclipse = False
        self.is_intersect = False

    #--------------------------------------------------------------------------------
    def time(self):                                                       # SLBRunner
    #--------------------------------------------------------------------------------
        #print('self', super().time(), 'PRT', self.prt.end_time(), 'UNRST', self.unrst.end_time(), 'RFT', self.rft.end_time())
        return max(self.rft.end_time(), super().time()) or self.prt.end_time() or self.unrst.end_time()

    #--------------------------------------------------------------------------------
    def delete_output_files(self):                                        # SLBRunner
    #--------------------------------------------------------------------------------
        """ 
        Delete case-specific output-files
        """
        file_ext = ('X????', '*UNRST', 'RFT', 'SMSPEC', 'UNSMRY', 'MSG', 'PRT', 'session*', 'dbprtx.lock*')
        delete_files_matching( *[f'{self.case}*.{ext}' for ext in file_ext] )
        delete_files_matching(self.case.parent/'fort??????')
        delete_files_matching(self.case.parent/'hostfile.*')

    #--------------------------------------------------------------------------------
    def unexpected_stop_error(self, **kwargs):                            # SLBRunner
    #--------------------------------------------------------------------------------
        error = 'unexpectedly' + (self.log and f', check {Path(self.log.name).name} for details' or '')
        # Check for license failure (uppercase for Eclipse, lowercase for Intersect)
        if self.msg.contains_any('LICENSE FAILURE', 'LICENSE ERROR', 'license failure',
                                 'Unable to checkout license'):
            error = 'due to a license failure'
        raise SystemError(f'ERROR {self.name} stopped {error}')


    #--------------------------------------------------------------------------------
    def start(self, error_func=None, host=None):                          # SLBRunner
    #--------------------------------------------------------------------------------
        # Get license file from terminal environment
        var = 'LM_LICENSE_FILE'
        environ[var] = get_terminal_environment(var)
        self._print(f'{var} = {environ.get(var)}')
        # Workaround for MPI suggested by IX output
        environ['I_MPI_SHM_LMT'] = 'shm'
        if self.update:
            self.update.status(value=f'Starting {self.name}...')
        error_func = error_func or self.unexpected_stop_error
        super().start(error_func)
        # self.wait_for(self.unrst.is_file, loop_func=error_func)
        if self.update:
            self.update.status(value=f'{self.name} running...')

    #--------------------------------------------------------------------------------
    def is_finished(self):                                                # SLBRunner
    #--------------------------------------------------------------------------------
        return not self.is_running() and self.end_time == self.unrst.end_time()

    #--------------------------------------------------------------------------------
    def has_aborted(self):                                                # SLBRunner
    #--------------------------------------------------------------------------------
        return not self.is_running() and self.end_time > self.unrst.end_time()

#====================================================================================
class Eclipse(SLBRunner):                                                   # Eclipse
#====================================================================================
    """ Eclipse runner class """

    #--------------------------------------------------------------------------------
    def __init__(self, root=None, **kwargs):
    #--------------------------------------------------------------------------------
        regex = r'TIME=?\s+([0-9.]+)\s+DAYS'
        super().__init__(name='Eclipse', root=root, cmd='eclipse', time_regex=regex, **kwargs)
        self.input_file = DATA_file(root) #, check=True)
        self.input_file.check(include=True, uppercase=True)
        self.is_eclipse = True


#====================================================================================
class Intersect(SLBRunner):                                               # Intersect
#====================================================================================
    """ Intersect runner class """

    #--------------------------------------------------------------------------------
    def __init__(self, root=None, **kwargs):
    #--------------------------------------------------------------------------------
        super().__init__(name='Intersect', root=root, cmd='ix',
                         time_regex=r'TIME(?:[ a-zA-Z\s/%-]+;|=) +([\d.]+)', **kwargs)
        #                 time_regex=r' (?:Rep |Init|HRep)   ;\s*([0-9.]+)\s+', **kwargs)
        # Necessary for IX backward mode (not yet implemented)
        #self.input_file = IX_input(root, check=True)
        self.is_intersect = True


#====================================================================================
class ForwardMixin:                                                    # ForwardMixin
#====================================================================================
    """ Common functions for forward runs """

    #--------------------------------------------------------------------------------
    def init_control_func(self, update=(), count=5):                   # ForwardMixin
    #--------------------------------------------------------------------------------
        """ Set up control for forward runs """
        self.update_funcs = update
        self.loop_count = 0
        self.count = count

    #--------------------------------------------------------------------------------
    def control_func(self):                                            # ForwardMixin
    #--------------------------------------------------------------------------------
        """ Enable plotting, progress and stop during forward runs """
        self.stop_if_canceled()
        self.loop_count += 1
        if self.loop_count == self.count:
            #print('\nCONTROL_FUNC')
            self.loop_count = 0
            self.t = self.get_time_and_stop_if_limit_reached()
            for func in self.update_funcs:
                # print('CALL UPDATE', func.__name__)
                func(run=self)
                # print('DONE!', func.__name__)


#====================================================================================
class EclipseForward(ForwardMixin, Eclipse):                         # EclipseForward
#====================================================================================
    """ Eclipse forward runner """

    #--------------------------------------------------------------------------------
    def check_input(self):                                           # EclipseForward
    #--------------------------------------------------------------------------------
        super().check_input()
        ### Check root.DATA exists and that READDATA keyword is NOT present
        if 'READDATA' in self.input_file:
            raise SystemError('WARNING The current case cannot run in forward-mode: '+
                              'Eclipse input contains the READDATA keyword.')
        return True

#====================================================================================
class IntersectForward(ForwardMixin, Intersect):                   # IntersectForward
#====================================================================================
    """ Intersect forward runner """

#====================================================================================
class BackwardMixin:                                                  # BackwardMixin
#====================================================================================
    """ Common functions for backward runs """

    #--------------------------------------------------------------------------------
    def update_function(self, progress=True, plot=False):             # BackwardMixin
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
    """ Eclipse backward mode runner """

    #--------------------------------------------------------------------------------
    def __init__(self, check_unrst=True, check_rft=True, keep_alive=False, schedule=None, init_tsteps=None, restart=None, **kwargs):
    #--------------------------------------------------------------------------------
        # super().__init__(ext_iface='I{:04d}', ext_OK='OK', keep_alive=keep_alive, **kwargs)
        super().__init__(ext_iface=('.I','04d'), ext_OK=('.OK',), keep_alive=keep_alive, **kwargs)
        self.n = restart.step if restart else 0
        self.t = restart.days if restart else 0
        self.init_tsteps = init_tsteps or len(self.input_file.get('TSTEP'))
        self.delete_interface = kwargs.get('delete_interface') or True
        self.check_unrst = check_unrst
        self.check_rft = check_rft
        self.schedule = schedule
        self.nwell = 0
        self.del_satnum = False


    #--------------------------------------------------------------------------------
    def check_input(self):                                             # EclipseBackward
    #--------------------------------------------------------------------------------
        def raise_error(msg):
            raise SystemError(f'ERROR To run the current case in backward-mode you need to insert {msg}')
        ### Check that root.DATA exists 
        super().check_input()
        if 'READDATA' not in self.input_file: #.data():
            raise_error(f"'READDATA /' between 'TSTEP' and 'END' in {self.input_file}.")
        ### Check presence of RPTSOL RESTART>1
        regex = r"\bRPTSOL\b\s+[A-Z0-9=_'\s]*\bRESTART\b *= *[2-9]{1}"
        if not self.input_file.section('SOLUTION').search('RPTSOL', regex):
            raise_error(f"'RPTSOL \\n RESTART=2 /' in the SOLUTION section of {self.input_file}.")
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
        #self.nwell = next(self.unrst.read('nwell', tail=True))[0]
        self.nwell = next(self.unrst.read('nwell', tail=True))
        # Wait for flushed RFT-file
        msg = self.rft.check.data_saved_maxmin(nblocks=nblocks*self.nwell, niter=RFT_CHECK_NITER, pause=CHECK_PAUSE)
        if msg:
            self._print(msg)
        while self.nwell < 1:
            ### Run Eclipse until at least one well is producing and the RFT-file is created
            self.schedule.update(tstep=self.end_time)
            self.run_one_step(self.schedule.ifacefile.path, start_stop=False, nwell=True)
            self._print(f' nwell = {self.nwell}')
            self.update_function(progress=True, plot=True)
            nblocks = 1
            self.init_tsteps += 1
            if self.t >= self.end_time:
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
                #self.nwell = nblocks = self.unrst.get('nwell')[0][-1]
                #self.nwell = nblocks = next(self.unrst.read('nwell', tail=True))[0]
                self.nwell = nblocks = next(self.unrst.read('nwell', tail=True))
                #print(nwell)
            msg = self.rft.check.data_saved_maxmin(nblocks=nblocks, niter=RFT_CHECK_NITER, pause=CHECK_PAUSE)
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
            #self._print(f' Date is {self.unrst.dates(N=-1).date()} ({self.t} days)')
            self._print(f' Date is {next(self.unrst.dates(tail=True)).date()} ({self.t} days)')
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
class IORSim_input(File):                                              # iorsim_input
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
    def __init__(self, file, check=False, check_format=False):         # iorsim_input
    #--------------------------------------------------------------------------------
        super().__init__(file, suffix='.trcinp', role='IORSim input-file', ignore_suffix_case=True)
        self.check_format = check_format
        self.warnings = self.check() if check else ''


    #--------------------------------------------------------------------------------
    def get(self, *keys, unpack_single=True):                          # iorsim_input
    #--------------------------------------------------------------------------------
        out = {k:[] for k in keys}
        # The regex is explained here: https://regex101.com/r/6H0pHT/1
        regex = r'^\s*\*(\b(?:' + '|'.join(keys) + r')\b)([^*]*)$'
        for match in finditer(regex, self.as_text(), flags=MULTILINE):
            key = match.group(1)
            # Split in non-empty lines
            lines = (line for l in match.group(2).split('\n') if (line:=l.strip()) )
            # Remove commented lines and in-line comments
            words = flatten(line.split() for l in lines if (line:=l.split('#')[0].strip()))
            # Convert number strings to numbers
            data = list(convert_float_or_str(words))
            if key == 'INTEGRATION':
                tupl = namedtuple(key, 'tstart tstop dtmin dtmax dtmin_ecl dtmax_ecl metnum')
                data = tupl(*data[:7])
            out[key].append(data)
        if unpack_single:
            # Extract element from single element lists
            out = [v[0] if len(v)==1 else v for v in out.values()]
        else:
            out = list(out.values())
        return out[0] if len(out)==1 else out

    #--------------------------------------------------------------------------------
    def check_keywords(self):                                          # iorsim_input
    #--------------------------------------------------------------------------------
        # Check if required keywords are used, and if the order is correct
        def raise_error(error):
            raise SystemError(f'ERROR Error in IORSim input file: {error}')
        text = remove_comments(self.path, comment='#')
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
    def check(self, error_msg='', warn_time=False):                     # iorsim_input
    #--------------------------------------------------------------------------------
        msg = error_msg and error_msg+': ' or ''
        warn = ''
        # Check if input-file exists
        self.exists(raise_error=True)
        # Check if included files exists
        missing = [f for f in self.include_files() if not f.is_file()]
        if missing:
            files = list2text([f.name for f in missing])
            folder = missing[0].parent.resolve()
            pl = len(missing) > 1
            raise SystemError(
                f"ERROR {msg}The file{'s' if pl else ''} '{files}' referenced by {self.path.name}"
                f" (or by CHEMFILE files) {'are' if pl else 'is'} missing in folder {folder}")
        # Warn if tstart > 0 (might not be important, so check is off by default)
        #if warn_time and (tstart := self.get('tstart')) and tstart > 0:
        if warn_time:
            ior = self.get('INTEGRATION')
            if ior.tstart > 0:
                warn += f'INFO It is recommended to start IORSim at time 0 instead of {ior.tstart}.'
        # Check if required keywords are used, and if the order is correct
        if self.check_format:
            self.check_keywords()
        return warn


    #--------------------------------------------------------------------------------
    def include_files(self, with_parent=False):                        # iorsim_input
    #--------------------------------------------------------------------------------
        def gen():
            regex = re_compile(r'^\s*add_species\s*[\"\']?([^\"\'\n\s]+)', flags=MULTILINE)
            # Regex explained here: https://regex101.com/r/XrBEmM/1
            files = flatten(self.get('CHEMFILE', unpack_single=False))
            if with_parent:
                files = chain([self.path], files)
            for file in files:
                chemfile = self.parent/file
                yield chemfile
                for match in regex.finditer(File(chemfile).as_text()):
                    yield self.parent/match.group(1)

        return tuple(set(gen()))

    #--------------------------------------------------------------------------------
    def solutions(self):                                               # iorsim_input
    #--------------------------------------------------------------------------------
        return {sol:dict(batched(val,2)) for sol, *val in self.get('solution')}


    #--------------------------------------------------------------------------------
    def species(self):                                                 # iorsim_input
    #--------------------------------------------------------------------------------
        if sol:=self.get('solution'):
            return sol[0][1::2]
        # Read old input format
        if spec:=self.get('SPECIES'):
            return [s[0] for s in spec]
        return []


    #--------------------------------------------------------------------------------
    def tracers(self):                                                 # iorsim_input
    #--------------------------------------------------------------------------------
        return [a[0] for a in self.get('NAME')]

    #--------------------------------------------------------------------------------
    def wells(self):                                                   # iorsim_input
    #--------------------------------------------------------------------------------
        wells = namedtuple('Wells', 'prod inj')
        prod, inj = self.get('PRODUCER', 'INJECTOR', unpack_single=False)
        if prod and inj:
            return wells(sorted(flatten(prod)), sorted(flatten(inj)))
            #return sorted(flatten(prod)), sorted(flatten(inj))
        # Read old input format
        prod, inj = self.get('OUTPUT', 'CONC_INJECTION', unpack_single=False)
        return wells(sorted(prod[0][1:] if prod else []), sorted(inj[0][1::6] if inj else []))
        #return sorted(prod[0][1:] if prod else []), sorted(inj[0][1::6] if inj else [])


#====================================================================================
class IORSim_output(File):                                              # iorsim_output
#====================================================================================
    #--------------------------------------------------------------------------------
    def __init__(self, root):     # iorsim_output
    #--------------------------------------------------------------------------------
        self.root = root
        self.start = None

    #--------------------------------------------------------------------------------
    def welldata(self, well, path='.', raise_error=True):
    #--------------------------------------------------------------------------------
        Data = namedtuple('Data','well days dates conc prod')
        self.start = self.start or SMSPEC_file(self.root).startdate()
        wellpath = (self.root.parent/path/self.root.stem).resolve()
        files = (Path(f'{wellpath}_W_{well}{ext}') for ext in ('.trcconc', '.trcprd'))
        data = [self._filedata(self.start, file) for file in files if file.is_file()]
        if data:
            conc, prod = data
            return Data(well, conc.days, conc.dates, conc.data, prod.data)
        if raise_error:
            raise ValueError(f"IORSim_output: Missing data for well '{well}' in {wellpath.parent}")

    #--------------------------------------------------------------------------------
    def _filedata(self, start, filename):
    #--------------------------------------------------------------------------------
        Filedata = namedtuple('Filedata', 'days dates data')
        with open(filename) as file:
            header = next(batched(file,5))
            names = header[2].split()[1:]
            first_line = next(file).split()
            ncol = len(first_line)
            values = batched((float(d) for d in first_line+''.join(file).split()), ncol)
            days, *data = zip(*values)
            # Add dates
            dates = [start + timedelta(days=int(day)) for day in days]
            return Filedata(days, dates, dict(zip(names[1:ncol], data)))


#====================================================================================
class Iorsim(Runner):                                                        # iorsim
#====================================================================================

    #--------------------------------------------------------------------------------
    def __init__(self, root=None, exe='IORSimX', args='', 
                 relative_root=True, check=True, **kwargs):                  # iorsim
    #--------------------------------------------------------------------------------
        #print('iorsim.__init__: ',root, exe, args, kwargs)
        app_name = Path(exe).stem
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
        regex = r'\bTime\b:\s+([0-9.e+-]+)'
        super().__init__(name='IORSim', case=root, exe=exe, app_name=app_name, cmd=cmd, time_regex=regex, **kwargs)
        self.update = kwargs.get('update') or None
        self.input_file = IORSim_input(root, check=check, check_format=kwargs.get('check_input_kw') or False)
        if self.update and (warn:=self.input_file.warnings):
            self.update.message(warn)
        self.is_iorsim = True
        self.is_eclipse = False
        self.copied_chemfiles = []

    #--------------------------------------------------------------------------------
    def time(self):                                                         # iorsim
    #--------------------------------------------------------------------------------
        # Find most recently modified file
        files = ((f, f.stat().st_mtime_ns) for f in File(self.case).glob('*.trcconc'))
        files = sorted(files, key=itemgetter(1))
        file = files[-1][0] if files else ''
        try:
            values = next(tail_file(file, size=1000),'').split('\n')[-2].strip().split()
            return float(values[0])
        except (ValueError, IndexError): # as e:
            #print(e)
            #print('IORSim time:',super().time())
            return super().time(tag='Time')


    #--------------------------------------------------------------------------------
    def start(self, error_func=None):                                        # iorsim
    #--------------------------------------------------------------------------------
        if self.update:
            self.update.status(value=f'Starting {self.name}...')
        # Copy chem-files to current working dir 
        if COPY_CHEMFILE:
            #for file in set(self.input_file.include_files()):
            for file in self.input_file.include_files():
                dest = Path.cwd()/file.name
                if not dest.exists() or (dest.exists() and not dest.samefile(file)):
                    shutil_copy(file, dest)
                    self.copied_chemfiles.append(dest)
                    self._print(f'Copied chemfile: {file} -> {dest}')
        # Check if the necessary Eclipse output files exist
        files = (self.case.with_suffix(ext) for ext in ('.UNRST', '.RFT', '.EGRID', '.INIT'))
        if missing := [f.name for f in files if not f.is_file()]:
            raise SystemError(f'ERROR Unable to start IORSim: Eclipse output file {", ".join(missing)} is missing')
        super().start(error_func)
        if self.update:
            self.update.status(value=f'{self.name} running...')

    #--------------------------------------------------------------------------------
    def delete_output_files(self, raise_error=False):                        # iorsim
    #--------------------------------------------------------------------------------
        # Delete old output files before starting new run
        case = str(self.case)
        delete_files_matching(case+'*.trcconc', raise_error=raise_error)
        delete_files_matching(case+'*.trcprd', raise_error=raise_error)
        restart = Path(str(self.case)+IOR_RESTART_NAMETAG)
        silentdelete( *(restart.with_suffix(x) for x in ('.FUNRST', '.UNRST')) )
        #silentdelete(self.funrst.path, self.unrst.path)

    #--------------------------------------------------------------------------------
    def close(self):                                                         # iorsim
    #--------------------------------------------------------------------------------
        super().close()
        # Delete chem-files copied to working directory
        self._print(f'Removing chemfiles: {self.copied_chemfiles}')
        silentdelete(*self.copied_chemfiles)

    #--------------------------------------------------------------------------------
    def is_finished(self):                                                   # iorsim
    #--------------------------------------------------------------------------------
        return self.OK_file.is_deleted()

#====================================================================================
class Ior_forward(ForwardMixin, Iorsim):                               # ior_forward
#====================================================================================
    pass

#====================================================================================
class Ior_tandem(Iorsim):                                                # ior_tandem
#====================================================================================
    rundir = 'iorsim'

    #--------------------------------------------------------------------------------
    def __init__(self, root=None, del_tandem=False, restart=None, **kwargs):  # ior_tandem                                            # ior_tandem
    #--------------------------------------------------------------------------------
        self.host_root = root = Path(root)
        # Change root to make IORSim run in a subfolder of the case-folder
        root = root.parent/self.rundir/root.name
        empty_folder(root.parent)
        super().__init__(root=root, args='-readdata', ext_iface=('.IORSimI','04d'),
                         ext_OK=('.IORSimOK',), check=False, **kwargs)
        self.delete = del_tandem
        self.host_rft = RFT_file(self.host_root)       # Unified RFT-file in host rundir
        self.inp_unrst = UNRST_file(self.case)         # Singlestep input UNRST in IORSim rundir
        self.inp_rft = RFT_file(self.case)             # Singlestep input RFT in IORSim rundir
        self.out_unrst = Output(self.case).ior_unrst   # Singlestep output UNRST in IORSim rundir
        self.copy_rft = False
        self.rft_slice = None
        self.rft_time = None
        self.host = None
        self.restart = restart
        self.start_step = 0      # Shift in numbering of .Xnnnn- and .Innnn-files for restart
        self.skip = 0            # If a restart-run fails and must be started at later step

    #--------------------------------------------------------------------------------
    def host_stopped(self):                                              # ior_tandem
    #--------------------------------------------------------------------------------
        if self.host.unexpected_stop or self.host.has_aborted():
            raise SystemError(f'ERROR {self.host.name} stopped unexpectedly')

    #--------------------------------------------------------------------------------
    def prepare_restart(self, step):
    #--------------------------------------------------------------------------------
        #filenum = (n for file in sorted(File(self.host_root).glob('*.X????')) if (n:=int(file.suffix[2:])) > step)
        #num = next(filenum, 0)
        #self.skip = num - step
        #self.n = step + self.skip
        self.start_step = self.n = step + 1
        self._print(f'***** Restart run starting at step {self.n}, skipping {self.skip} steps *****')
        self.host_rft_is_ready() # Skip first time slice

    #--------------------------------------------------------------------------------
    def xfile(self):                                                     # ior_tandem
    #--------------------------------------------------------------------------------
        return unfmt_file(self.host_root.with_suffix(f'.X{self.n:04d}'))

    #--------------------------------------------------------------------------------
    def host_rft_is_ready(self):                                         # ior_tandem
    #--------------------------------------------------------------------------------
        """
        Check if the RFT-file is ready by retrieving the next time slice.

        Returns:
            bool: True if a valid time slice is retrieved, False otherwise.
        """
        self.rft_time, self.rft_slice = next(self.host_rft.time_slice(), (None, None))
        if self.rft_time:
            return True
        return False

    #--------------------------------------------------------------------------------
    def copy_to_iorcase(self, *files, log=True):                         # ior_tandem
    #--------------------------------------------------------------------------------
        """
        Copies the specified files to the IOR case directory.
        """
        for file in files:
            dst = self.case.parent/file.name
            if log:
                self._print(f'Copy {file.resolve()} --> {dst.resolve()}')
            shutil_copy(file, dst)

    #--------------------------------------------------------------------------------
    def get_start_step(self):                                            # ior_tandem
    #--------------------------------------------------------------------------------
        """
        Get the number of the first X-file. Not 0 if a previous run failed.
        """
        xstart = sorted(File(self.host_root).glob('*.X????'))
        return int(xstart[0].suffix[2:]) if xstart else 0

    #--------------------------------------------------------------------------------
    def start(self, error_func=None, host=None):                                    # ior_tandem
    #--------------------------------------------------------------------------------
        self.host = host
        self.n = self.start_step = self.get_start_step()
        self._print(f'Start simulation at step {self.n}')
        if self.restart:
            self.prepare_restart(self.restart.step)
        # Wait for necessary host-files to appear
        host_files = [self.host_root.with_suffix(ext) for ext in ('.INIT', '.EGRID', '.RFT')]
        xfile = self.xfile()
        self.wait_for_files(*host_files, xfile.path, loop_func=error_func)
        # Copy host-files and IORSim input to IORSim case directory
        ior_inp = IORSim_input(self.host_root)
        self.copy_to_iorcase(*host_files, *ior_inp.include_files(with_parent=True))
        #self._print(f'rft.end_time: {self.inp_rft.end_time()}, end_time: {self.end_time}')
        # Copy RFT-file during the simulation if it has not yet reached the end-time
        if self.inp_rft.end_time() < self.end_time:
            self.copy_rft = True
        # Create UNRST-file
        self.inp_unrst.from_Xfile(xfile)
        super().start(error_func)
        # Run step 0 as part of the startup. Output is generated after step 1
        self._print('Run step 0 for 0 days')
        self.run_one_step()

    #--------------------------------------------------------------------------------
    def copy_rft_data(self):                                  # ior_tandem
    #--------------------------------------------------------------------------------
        """
        Waits for the RFT file to reach the same time as the UNRST file. 
        If a host is present, the host simulation is suspended while copying the RFT data. 
        """
        # Wait for RFT-file to reach the same time as the UNRST-file
        self._print(f'Wait for {self.host_rft} to get ready')
        self.wait_for(self.host_rft_is_ready, pause=0.1, loop_func=self.host_stopped)
        if self.host:
            # Suspend host simulation while copying RFT-data
            self.host.suspend()
        self._print(f'Copy RFT-data (time = {self.rft_time}) '
                    f'from {self.host_rft} --> {self.rundir}/{self.inp_rft}')
        self.inp_rft.write_bytes( self.host_rft.binarydata(pos=self.rft_slice) )
        if self.host:
            self.host.resume()

    #--------------------------------------------------------------------------------
    def run_one_step(self):                                   # ior_tandem
    #--------------------------------------------------------------------------------
        # Create UNRST-file from X-file
        xfile = self.xfile()
        # Wait for the X-file to appear
        self.wait_for(xfile.is_flushed, 'ENDSOL', func_name=f'unfmt_file({xfile}).is_flushed', loop_func=self.host_stopped)
        # Create single-step UNRST-file from X-file
        self._print(f'Creating {self.rundir}/{self.inp_unrst} from {xfile}')
        self.inp_unrst.from_Xfile(xfile)
        if self.n > self.start_step and self.copy_rft:
            self.copy_rft_data()
        # Append 'singlestep' command to the interface-file.
        # This instructs IORSim to read and write singlestep UNRST-files
        ifile = self.n + 1 - self.start_step
        self._print(f'Creating interface-file .I{ifile:04d}')
        self.interface_file(ifile).append('singlestep')
        # Tell IORSim to run a step
        self.OK_file.create()
        # Wait for IORSim to finish
        self.wait_for(self.is_finished)
        self.interface_file(ifile).delete()
        if self.delete:
            # Delete X-file
            xfile.delete()
        t = self.time()
        self._print(f'Current time: {self.t:.4f} + {t-self.t:.4f} = {t:.4f} days')
        self.t = t
        self.n += 1

    #--------------------------------------------------------------------------------
    def quit(self, v=1, loop_func=lambda:None):                          # ior_tandem
    #--------------------------------------------------------------------------------
        # Give 'quit' command to tell IORSim to terminate
        self.interface_file(self.n+1).append('quit')
        self.OK_file.create()
        super().quit(v, loop_func)


#====================================================================================
class Ior_backward(BackwardMixin, Iorsim):                             # ior_backward
#====================================================================================
    #--------------------------------------------------------------------------------
    def __init__(self, keep_alive=False, schedule=None, init_tsteps=None, **kwargs):
    #--------------------------------------------------------------------------------
        super().__init__(args='-readdata', ext_iface=('.IORSimI','04d'), ext_OK=('.IORSimOK',), keep_alive=keep_alive, **kwargs)
        self.init_tsteps = init_tsteps or len(DATA_file(self.case).timesteps())
        self.delete_interface = kwargs.get('delete_interface') or True
        self.satnum = DATA_file(IOR_SATNUM_FILE)   # Output-file from IORSim, read by Eclipse as an interface-file
        self.endtag = IOR_SATNUM_ENDTAG
        self.schedule = schedule


    #--------------------------------------------------------------------------------
    def delete_output_files(self, raise_error=False):                  # ior_backward
    #--------------------------------------------------------------------------------
        super().delete_output_files(raise_error)
        self.satnum.unlink(missing_ok=True)

    #--------------------------------------------------------------------------------
    def satnum_flushed(self):                                          # ior_backward
    #--------------------------------------------------------------------------------
        if self.endtag in self.satnum.tail(size=len(self.endtag)+3, size_limit=True):
            return True
        return False

    #--------------------------------------------------------------------------------
    def satnum_dist(self, echo=False):                                 # ior_backward
    #--------------------------------------------------------------------------------
        """
        Return the distribution of SATNUM numbers as a dict
        """
        lines = remove_comments(self.satnum.path, comment='--')
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
    def start(self, error_func=None, tsteps=None):                     # ior_backward
    #--------------------------------------------------------------------------------
        # Start IORSim backward run
        self.n = 0
        self.interface_file.delete_all()
        tsteps = tsteps or self.init_tsteps
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
            self.satnum.delete()
            if n == 0:
                if start:
                    super().start()
                else:
                    self.resume()
            self.wait_for(self.OK_file.is_deleted, error=self.OK_file.name()+' not deleted')
            self.t = self.time()
        self.wait_for(self.satnum_flushed, pause=CHECK_PAUSE)
        if self.satnum.is_empty() and self.update:
            self.update.message(f'WARNING {self.satnum} is empty!')
        self.suspend()
        if self.delete_interface:
            for n in range(N):
                self.interface_file(self.n-n).delete()
        if log:
            self._print(f' {self.t:.3f}/{self.end_time} days')


    #--------------------------------------------------------------------------------
    def quit(self, v=1, loop_func=lambda:None):                        # ior_backward
    #--------------------------------------------------------------------------------
        self.interface_file(self.n+1).append('quit')
        self.OK_file.create()
        super().quit(v, loop_func)

    #--------------------------------------------------------------------------------
    def close(self):                                                   # ior_backward
    #--------------------------------------------------------------------------------
        super().close()
        if not self.keep_files:
            self.satnum.unlink(missing_ok=True)
            Path('block0_firstsolutioninit.out').unlink(missing_ok=True)
        

#====================================================================================
class Schedule:
#====================================================================================
    #suffix = '.SCH'
    comment = '--'

    #--------------------------------------------------------------------------------
    def __init__(self, case, end_time=0, init_days=0, start=None, interface_file=None, skip_empty=False): 
    #--------------------------------------------------------------------------------
        """
        Create schedule from a .SCH-file if it exists. 
        The TSTEP in the satnum-file (created by IORSim) is modified to 
        ensure the next report time coincides with the next schedule-events.  
        The schedule is also used to ensure that the simulation ends at the right time
        by adding the end-time twice at the end of the schedule. 
        If a .SCH-file is present, the first entry in the schedule is the start-time.  

        The schedule is a list of lists: [[start-time, ''],[days, 'KEYWORD'],[end-time, '']]
        """
        self.skip_empty = skip_empty
        self.ifacefile = DATA_file(interface_file.path, sections=False) if interface_file else None
        self.days = init_days
        self.start = start
        self.tstep = 0
        self._schedule = ()
        self.end = 0
        # Ignore case in file extension
        self.sch_file = DATA_file(case, suffix='.SCH', ignore_suffix_case=True, sections=False)
        if self.sch_file.is_file(): # and self.file.exists():
            self._schedule = self.get_schedule()
            self.end = (len(self._schedule) > 0) and self._schedule[-1][0] or 0
        # Add simulation end time
        self.insert(days=end_time, remove=True)
        if DEBUG:
            print(f'Creating {self}')
        #self.to_file('schedule.txt')

    #--------------------------------------------------------------------------------
    def __repr__(self):                                                     # Schedule
    #--------------------------------------------------------------------------------
        return f'<Schedule(file={self.sch_file}, start={self.start}, end={self.end}, init_days={self.days}, length={len(self._schedule)})>'

    #--------------------------------------------------------------------------------
    def __str__(self):                                                     # Schedule
    #--------------------------------------------------------------------------------
        return f'{self.sch_file}' # and self.sch_file.name}'

    #--------------------------------------------------------------------------------
    def __del__(self):                                                     # Schedule
    #--------------------------------------------------------------------------------
        if DEBUG:
            print(f'Deleting {self}')

    #--------------------------------------------------------------------------------
    def to_file(self, name):                                               # Schedule
    #--------------------------------------------------------------------------------
        ' Write schedule to file '
        name = self.sch_file.with_name(name)
        if not name:
            return 
        print(f'  Writing {name}')
        with open(name, 'w', encoding='utf8') as file:
            file.write('\n'.join([str(s) for s in flat_list(self._schedule)]))

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
        s += f'  file   : {self.sch_file}\n'
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
        """
        Return a list of tuples with days at index 0 and actions at index 1, such as:
        schedule = [(2.0, "WCONHIST \r\n    'P-15P'      'OPEN' "), (9.0, "WCONHIST")]
        """
        # Split, accumulate tsteps, and then zip tsteps and pos  
        tstep, pos = zip(*self.sch_file.timesteps(start=self.start, pos=True)) # + [('',(len(self.file),0),)]
        tstep_pos = list(zip(accumulate(tstep), pos))
        #  Append end to make pairwise pick up the last entry
        tstep_pos.append(tstep_pos[-1])
        tstep_act = ((tstep, self.sch_file.data[a:b]) for (tstep,(_,a)), (_,(b,_)) in pairwise(tstep_pos))
        if self.skip_empty:
            tstep_act = (x for x in tstep_act if x[1])
        return list(tstep_act)


    #--------------------------------------------------------------------------------
    def append(self, action=None, tstep=None):                             # Schedule
    #--------------------------------------------------------------------------------
        """
        line : line number of TSTEP
        """
        #print(f'append(action={action}, tstep={tstep})')
        if action is None and tstep is None:
            return
        if tstep == 0:
            raise SystemError(f'ERROR Schedule gave TSTEP 0 at {self.days} days, simulation stopped. Check {self.sch_file}')
        new_action_and_tstep = (action and action or '') + f'TSTEP\n{tstep} /\n'
        self.ifacefile.replace_keyword('TSTEP', new_action_and_tstep)


    #--------------------------------------------------------------------------------
    def check(self):                                                       # Schedule
    #--------------------------------------------------------------------------------
        # just a check...
        print(f'{self.days}, {self.start+timedelta(days=self.days)}')
        with open(self.ifacefile.path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        print(self.ifacefile.path, ': ', ''.join(lines[-10:]))
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
            self.ifacefile.path.write_text(f'TSTEP\n{tstep} /\n')
            self.tstep = new_tstep = tstep
        # print(f'START: tstep:{self.tstep}, days:{self.days}, schedule:{self._schedule[:2]}')
        ### Check arrival of next event and adjust tstep if neccessary
        if self._schedule and self.days + self.tstep + 1e-8 > self._schedule[0][0]:
            self.tstep = new_tstep = self._schedule[0][0] - self.days
        self.append(action=action, tstep=new_tstep)
        #self.check()
        #print('===== SCHEDULE START =====')
        #print(f'===== SCHEDULE: tstep:{self.tstep}, days:{self.days}')
        #print(f'===== SCHEDULE: action:{action}')
        #print(f'===== SCHEDULE: schedule:{self._schedule[:2]}')
        #print('===== SCHEDULE END =====')
        return self.days



#====================================================================================
class Simulation:                                                        # Simulation
#====================================================================================
    #--------------------------------------------------------------------------------
    def __init__(self, mode=None, root=None, pause=0, run_names=(), to_screen=False,
                 status=lambda **x:None, progress=lambda **x:None,
                 plot=lambda **x:None, message=lambda **x:None, intersect=0, host_inp=None,
                 ior_inp=None, **kwargs):
    #--------------------------------------------------------------------------------
        # print('Simulation', 'mode',mode,'root',root,'pause',pause,'run_names',run_names,'to_screen',to_screen,
        #       'status',status,'progress',progress,'plot',plot,'kwargs',kwargs)

        self.logname = 'ior2ecl'
        self.root = Path(root).with_suffix('').resolve()
        self.ECL_inp = None
        self.IOR_inp = None
        if intersect or 'intersect' in run_names:
            self.ECL_inp = host_inp or IX_input(self.root, check=True, convert=intersect > 1)
        if 'eclipse' in run_names:
            self.ECL_inp = host_inp or DATA_file(self.root, check=True)
        if 'iorsim' in run_names:
            self.IOR_inp = ior_inp or IORSim_input(self.root, check=False)
        self.output = Output(self.root, progress=progress, status=status, **kwargs)
        self.update = namedtuple('update',['progress','plot','message','status'])(progress, plot, message, status)
        self.pause = pause
        self.runlog = None
        if self.root and not to_screen:
            logtag = kwargs.get('logtag')
            self.runlog = safeopen(self.root.parent/f'{self.logname}{logtag or ""}.log', 'w')
        self.print2log = lambda *txt, **kwargs: print(*txt, file=self.runlog, flush=True, **kwargs)
        self.current_run = None
        self.run_names = run_names
        self.runs = ()
        self.run_sim = None
        self.ior = self.ecl = None
        self.report_dates = None
        self.end_time = 0
        self.mode = mode
        self.schedule = None
        self.start = None
        self.restart = None
        self.skiprest = False
        self.only_iorsim = False
        kwargs.update({'root':str(self.root), 'runlog':self.runlog, 'update':self.update, 'to_screen':to_screen})
        self.kwargs = kwargs

    #--------------------------------------------------------------------------------
    def __repr__(self):                                                   # Simulation
    #--------------------------------------------------------------------------------
        return f'<Simulation root={self.root}, mode={self.mode}>'

    #--------------------------------------------------------------------------------
    def prepare(self):                                                   # Simulation
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


    # #--------------------------------------------------------------------------------
    # def versions(self):                                                  # Simulation
    # #--------------------------------------------------------------------------------
    #     txt = f'Python: {get_python_version()}\n'
    #     txt += f'psutil: {psutil_version}\n'
    #     txt += f'Application: {__version__}\n'
    #     return txt


    #--------------------------------------------------------------------------------
    def ready(self):                                                     # Simulation
    #--------------------------------------------------------------------------------
        return bool(self.run_sim)

    #--------------------------------------------------------------------------------
    def host_input(self):                                                # Simulation
    #--------------------------------------------------------------------------------
        inp_gen = (inp for file in (IX_input, DATA_file) if (inp:=file(self.root)).exists())
        if inp := next(inp_gen, None):
            return inp
        raise SystemError('ERROR Input-files for IX or ECL are missing')

    #--------------------------------------------------------------------------------
    def init_runs(self):                                                 # Simulation
    #--------------------------------------------------------------------------------
        """
        Read Eclipse and IORSim input files, run the init_func, and return the run_func
        """
        if self.ECL_inp:
            # Check if this is a restart-run
            self.restart = self.ECL_inp.restart()
            self.start = self.restart.start or self.ECL_inp.start()
            self.mode = self.mode or self.ECL_inp.mode()
        else:
            if self.mode == 'tandem':
                self.restart = self.host_input().restart()
            else:
                self.mode = 'forward'
                self.restart = Restart()

        init_func = {'backward' : self.init_backward_run,
                     'forward'  : self.init_forward_run,
                     'tandem'   : self.init_tandem_run   }[self.mode]
        self.runs = init_func(**self.kwargs)
        # Check input
        if all(run.check_input() for run in self.runs):
            return {'backward' : self.backward,
                    'forward'  : self.forward,
                    'tandem'   : self.tandem   }[self.mode]
        return None

    #--------------------------------------------------------------------------------
    def init_forward_run(self, iorexe=None, eclexe=None, **kwargs):      # Simulation
    #--------------------------------------------------------------------------------
        if self.ECL_inp:
            start = self.start + timedelta(days=self.restart.days)
            tsteps = self.ECL_inp.timesteps(start=start, skiprest='SKIPREST' in self.ECL_inp)
            self.end_time = sum(tsteps) + self.restart.days
            # Check if IX writes unified UNRST
            if not self.ECL_inp.write_unified_UNRST():
                raise SystemError(f"ERROR IX must be set to write unified UNRST. Modify {self.ECL_inp.UNRST_settings().file}")
        else:
            # This is a forward IORSim run
            self.end_time = min(UNRST_file(self.root).end_time(), self.IOR_inp.get('INTEGRATION').tstop)
        kwargs.update({'end_time':self.end_time})
        for name in self.run_names:
            if name == 'eclipse':
                self.ecl = EclipseForward(exe=eclexe, **kwargs)
            if name == 'intersect':
                self.ecl = IntersectForward(exe=eclexe, **kwargs)
            if name=='iorsim':
                self.ior = Ior_forward(exe=iorexe, **kwargs)
        self.only_iorsim = self.run_names in (['iorsim'],('iorsim',))
        return [run for run in (self.ecl, self.ior) if run]


    #--------------------------------------------------------------------------------
    def forward(self):                                                   # Simulation
    #--------------------------------------------------------------------------------
        do_merge = True
        if do_merge:
            # Delete the merge_OK-file to allow a new merge of Eclipse and IORSim output
            silentdelete(self.output.merge_OK)
        run_time = timedelta()
        ret = ''
        self.update.progress(value=-self.end_time, min=self.restart.days)
        for run in self.runs:
            self.current_run = run.name.lower()
            run.delete_output_files()
            run.start()
            self.update.progress()  # Reset
            # Progress is updated after 15*0.2 = 3 sec
            # Check for cancelled run every 0.2 sec
            run.init_control_func(update=(self.update.progress, self.update.plot), count=15)
            run.wait_for_process_to_finish(pause=0.2, loop_func=run.control_func)
            self.update.progress(run)
            self.update.plot()
            run.t = run.time()
            ndeci = min(len(str(t).split('.')[-1]) for t in (run.t, run.end_time))
            if round(run.t, ndeci) < round(run.end_time, ndeci):
                run.unexpected_stop_error()
            run_time += run.run_time()
            ret = run.complete_msg(run_time=run_time)
        return ret

    #--------------------------------------------------------------------------------
    def init_tandem_run(self, iorexe=None, eclexe=None, time=None, **kwargs):      # Simulation
    #--------------------------------------------------------------------------------
        if self.ECL_inp:
            start = self.start + timedelta(days=self.restart.days)
            tsteps = self.ECL_inp.timesteps(start=start, skiprest='SKIPREST' in self.ECL_inp)
            self.end_time = sum(tsteps) + self.restart.days
            # Check if IX writes unified UNRST
            if self.ECL_inp.write_unified_UNRST():
                raise SystemError(f"ERROR Tandem mode: IX must be set to write non-unified UNRST, "
                                  f"modify {self.ECL_inp.UNRST_settings().file}")
        else:
            # This is a forward tandem IORSim only run
            inp = self.host_input()
            self.start = inp.start()
            self.end_time = sum(inp.timesteps())
        self.end_time = time or self.end_time
        self.report_dates = IX_input(self.root).report_dates()
        restart = self.restart if self.restart.days else None
        if restart:
            # Copy root.trcinp to restart.trcinp
            basename = self.root.name.split('_restart')[0]
            src = self.root.parent/(basename+'.trcinp')
            dst = self.root.with_suffix('.trcinp')
            #self.print2log(f'Restart run: copy {src.resolve()} --> {dst.resolve()}')
            shutil_copy(src, dst)
        kwargs.update({'end_time':self.end_time, 'restart':restart})
        for name in self.run_names:
            if name == 'eclipse':
                self.ecl = EclipseForward(exe=eclexe, **kwargs)
            if name == 'intersect':
                self.ecl = IntersectForward(exe=eclexe, **kwargs)
            if name=='iorsim':
                self.ior = Ior_tandem(exe=iorexe, **kwargs)
        # Check output format in IORSim input
        ior_inp = IORSim_input(self.root)
        if ior_inp.get('GRIDPLOT_FILE')[0] != 'UNFORMATTED':
            raise SystemError(f"ERROR Tandem mode: IORSim output is FORMATTED, "
                              f"modify {inp}': *GRIDPLOT_FILE UNFORMATTED")
        return [run for run in (self.ecl, self.ior) if run]


    #--------------------------------------------------------------------------------
    def tandem(self):                                                   # Simulation
    #--------------------------------------------------------------------------------
        check = True
        do_merge = True
        if do_merge:
            # Delete the merge_OK-file to allow a new merge of Eclipse and IORSim output
            silentdelete(self.output.merge_OK)
        starttime = datetime.now()
        self.update.progress(value=-self.end_time, min=self.restart.days)
        # Start runs
        host = None
        for run in self.runs:
            run.delete_output_files()
            # if host:
            #     run.start(host=host)
            # else:
            #     run.start()
            run.start(host=host)
            self.update.progress()
            #error_func = run.assert_running_and_stop_if_canceled
            host = run
        ior = self.runs[-1]
        #slb = self.runs[0] if len(self.runs)>1 else None

        nwrite = 0
        out = Tandem_output(self.root, log=self.print2log)
        while ior.t < ior.end_time:
            text = f'  step {ior.n}, time = {ior.t:.2f}/{ior.end_time} days  '
            band = int(0.5*(80-len(text)))*'='
            self.print2log('\n' + band + text + band)
            self.update.progress(run=ior)
            try:
                ior.run_one_step()
            except SystemError as err:
                raise SystemError(f'Tandem simulation stopped: {err}') from err
            date = self.start + timedelta(days=ior.t)
            self.print2log(f'Current date: {date}')
            if check:
                out.check_chlorine(ior.out_unrst)
            if nwrite == 0:
                # Remove report-dates older than the initial simulation date (restart run)
                self.report_dates = dates_after(self.start + timedelta(days=ior.t), self.report_dates)
            if date >= self.report_dates[nwrite]:
                text = f'  report step {nwrite}/{len(self.report_dates)}  '
                band = int(0.5*(80-len(text)))*'='
                self.print2log('\n' + band + text + band)
                self.print2log(f'Date: {next(ior.inp_unrst.dates(resolution='sec'))}')
                # Append relevant blocks from host and ior unrst to unified unrst
                out.append_host(ior.inp_unrst)
                out.append_ior(ior.out_unrst)
                nwrite += 1
        ior.quit()
        return ior.complete_msg(run_time=datetime.now()-starttime)


    #--------------------------------------------------------------------------------
    def init_backward_run(self, iorexe=None, eclexe=None, ior_keep_alive=False,
                          ecl_keep_alive=False, time=0, **kwargs):       # Simulation
    #--------------------------------------------------------------------------------
        if time is None:
            raise SystemError("ERROR Missing 'time' argument")
        tsteps = self.ECL_inp.timesteps()
        init_days = sum(tsteps) + self.restart.days
        if time > init_days:
            self.end_time = time
        else:
            self.end_time = init_days + 1
            self.update.message(text='INFO Simulation time increased to advance past the READDATA keyword')
        kwargs.update({'end_time':self.end_time, 'init_tsteps':len(tsteps)})
        # Init runs
        self.ecl = EclipseBackward(exe=eclexe, keep_alive=ecl_keep_alive, restart=self.restart, **kwargs)
        self.ior = Ior_backward(exe=iorexe, keep_alive=ior_keep_alive, **kwargs)
        # Set up schedule of commands to pass to satnum-file
        self.schedule = Schedule(self.root, end_time=self.end_time, start=self.start, init_days=init_days, interface_file=self.ior.satnum,
                                 skip_empty=self.kwargs.get('skip_empty', False))
        self.ecl.schedule = self.ior.schedule = self.schedule
        return [self.ecl, self.ior]


    #--------------------------------------------------------------------------------
    def backward(self):                                                  # Simulation
    #--------------------------------------------------------------------------------
        silentdelete(self.output.merge_OK)
        self.update.progress(value=-self.end_time)
        ecl, ior = self.runs
        # Start runs
        ecl.delete_output_files()
        ior.delete_output_files()
        ecl.start(restart=self.restart.run)
        #ior.start(restart=self.restart.run, tsteps=ecl.init_tsteps)
        ior.start(tsteps=ecl.init_tsteps)
        # The schedule appends keywords to the interface file (satnum.dat)
        self.schedule.update()
        # Update progress
        if self.restart.run:
            # Fix progress for restart runs
            self.update.progress(value=self.ecl.t, min=self.restart.days)
        else:
            # Reset progress-time for more accurate time-estimate   
            self.update.progress(value=ior.t, n0=ior.t)
        # Start timestep loop
        while ior.t < ior.end_time:
            self.print2log(f'\nStep {ecl.n+1}')
            self.update.progress(run=ecl)
            ecl.run_one_step(ior.satnum.path)
            # Run IORSim to prepare satnum input for the next Eclipse run
            self.update.progress(run=ior)
            ior.run_one_step()
            self.schedule.update()
            self.print2log(f'Step {ecl.n} ({self.schedule.now().date()}) completed')
            self.update.plot()
        self.update.progress(run=ior)
        # Timestep loop finished
        for run in self.runs:
            self.update.status(value=f'Stopping {run.name}...')
            run.quit()
        return ecl.complete_msg()


    # #--------------------------------------------------------------------------------
    # def set_output(self):                                                       # Simulation
    # #--------------------------------------------------------------------------------
    #     # Create output for convert and merge if this is an IORSim run
    #     if ior := (run for run in self.runs if run.is_iorsim):
    #         self.output.set_update_functions(cancel=ior.stop_if_canceled, 
    #                                          progress=self.update.progress, 
    #                                          reset=self.update.progress, 
    #                                          status=self.update.status)

    #--------------------------------------------------------------------------------
    def run(self):                                                       # Simulation
    #--------------------------------------------------------------------------------
        # Print header
        self.print2log(self.info_header())
        msg = conv_msg = ''
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
        except KeyboardInterrupt:
            self.cancel()
            msg = 'Simulation cancelled'
        except Exception as exception:  # Catch all other exceptions
            self.print2log(f'\nAn exception occured:\n{trace_format_exc()}')
            n = [run.n for run in self.runs if run] or [-1]
            msg += f'(step {max(n)}) {type(exception).__name__}: {exception}'
            if r'\x00\x00\x00\x00' in fr'{exception}':
                msg += f', try increasing the CHECK_PAUSE value ({CHECK_PAUSE}).'
            else:
                msg += f', check {Path(self.runlog.name).name} for details' if self.runlog else ''
        finally:
            # Kill possible remaining processes
            self.print2log('')
            for run in self.runs:
                run.kill()
            self.print2log(f'\n=====  {msg.replace("INFO","")}  =====')
            self.current_run = None
            self.update.progress()   # Reset progress time
            self.update.plot()
            try:
                if success:
                    conv_msg = self.convert_and_merge()
            except SystemError as error:
                self.update.message(f'{error}')
            finally:
                self.close()
                self.update.status(value=msg+conv_msg, newline=True)
        return success, msg + ' ' + conv_msg


    #--------------------------------------------------------------------------------
    def convert_and_merge(self):                                         # Simulation
    #--------------------------------------------------------------------------------
        ior = [run for run in self.runs if run.is_iorsim]
        # Post-process only for IORSim runs
        if not ior:
            return ''
        for func in self.output.post_process:
            sleep(0.05)  # Need a short break to make the GUI progressbar responsive
            success = getattr(self.output, func)(cancel=ior[0].stop_if_canceled)
            self.print2log(f'\n=====  {self.output.msg}  =====')
            #self.update.status(value=msg, newline=True)
            if not success:
                return self.output.msg
        return ''


    #--------------------------------------------------------------------------------
    def info_header(self, long_lognames=False):                          # Simulation
    #--------------------------------------------------------------------------------
        width = '10s'
        logfiles = [run.logname for run in self.runs]+[log.name for log in (self.runlog,) if log]
        case = Path(self.root).name
        s  = '\n'
        s += f'    {"Case":{width}}: {case}\n'
        mode = len(self.runs)<2 and self.runs[0].name or self.mode
        s += f'    {"Mode":{width}}: {mode.capitalize()}\n'
        s += f'    {"Days":{width}}: {self.end_time}' 
        if self.restart.days > 0:
            days = timedelta(days=self.restart.days)
            s += f' (restart after {days.days} days, at {(self.start + days).date()})'
            s += self.skiprest and ' (SKIPREST)' or ''
        s += '\n'
        if self.IOR_inp:
            ior = self.IOR_inp.get('INTEGRATION')
            s += f'    {"Timestep":{width}}: {ior.dtmin}{(f" - {ior.dtmax}" if ior.dtmin!=ior.dtmax else "")} days\n'
        s += (self.schedule and self.schedule.sch_file) and f'    {"Schedule":{width}}: start={self.schedule.start.date()}, days={self.schedule.end}{(self.schedule.skip_empty and ", skip empty entries" or "")}\n' or ''
        rundir = str(Path.cwd())
        s += f'    {"Run-dir":{width}}: {rundir}\n'
        casedir = str(Path(self.root).parent).replace(rundir, '<Run-dir>')
        s += f'    {"Case-dir":{width}}: {casedir}\n'
        if long_lognames:
            indent = f'\n    {"    ":{width}}: '
            s += f'    {"Log-files":{width}}: {indent.join([str(Path(file).resolve()) for file in logfiles])}\n'
        else:
            s += f'    {"Log-files":{width}}: {", ".join([Path(file).name for file in logfiles])} (in Case-dir)\n'
        s += f'    {"Version":{width}}: {__version__}\n'
        s += f'    {"Started":{width}}: ' + str(datetime.now()).split('.')[0] + '\n'
        s += '\n'
        return s

    #--------------------------------------------------------------------------------
    def cancel(self):                                                    # Simulation
    #--------------------------------------------------------------------------------
        for run in self.runs:
            if isinstance(run, Runner):
                run.cancel()

    # #--------------------------------------------------------------------------------
    # def compare_restart(self, ecl_keys=[], ior_keys=[], limit=None):     # Simulation
    # #--------------------------------------------------------------------------------
    #     ecl_unrst = UNRST_file(f'{self.root}_ECLIPSE.UNRST')
    #     ior = self.ior or Iorsim(root=self.root)   
    #     try:
    #         if self.runlog:
    #             self.runlog = open(self.runlog.name, 'a')
    #         #with open(self.runlog.name, 'a') as self.runlog:   
    #         self.print2log(f'\n Comparing {ecl_unrst.name()} and {ior.funrst.name()}:')
    #         n = 0
    #         for a,b in zip(ecl_unrst.data(*ecl_keys), ior.funrst.data(*ior_keys)):
    #             self.print2log(f'  ECL: {print_dict(a)}')
    #             self.print2log(f'  IOR: {print_dict(b)}')
    #             self.print2log('')
    #             n += 1
    #             if limit and n > limit:
    #                 break
    #     finally:
    #         self.runlog and self.runlog.close()


#====================================================================================
class Output:                                                                # Output
#====================================================================================
    #--------------------------------------------------------------------------------
    def __init__(self, root, convert=True, merge=True, del_convert=True, del_merge=True,
                 progress=None, status=None, **kwargs):
    #--------------------------------------------------------------------------------
        self.root = Path(root).with_suffix('').resolve()
        self.ior_funrst = FUNRST_file(str(root)+IOR_RESTART_NAMETAG)
        self.ior_unrst = UNRST_file(self.ior_funrst.path) #, end=IOR_RESTART_ENDKEY)
        self.slb_unrst = UNRST_file(root)
        self.slb_unrst_backup = UNRST_file(str(root)+SLB_BACKUP_NAMETAG)
        # The merged file ends with SATNUM (IORSim UNRST) instead of ENDSOL (Eclipse UNRST)
        self.merge_unrst = UNRST_file(f'{root}_MERGED.UNRST', end=self.ior_unrst.end)
        # if not convert:
        #     merge = False
        self.post_process = [k for k,v in {'convert':convert, 'merge':merge}.items() if v]
        self.merge_OK = self.root.with_name(MERGE_OK_FILE)
        self.del_convert = del_convert
        self.del_merge = del_merge
        self.progress = progress or self.progress_
        self.status = status or self.status_
        self.msg = ''
        self.prog = Progress(format='40#')
        self.starttime = None
        self.report_dates = None # Used in check()

    #--------------------------------------------------------------------------------
    def already_merged(self):                                                # Output
    #--------------------------------------------------------------------------------
        if self.merge_OK.is_file():
            merge_time = self.merge_OK.stat().st_ctime
            run_time = self.slb_unrst.creation_time()
            if merge_time > run_time:
                return True
        return False

    #--------------------------------------------------------------------------------
    def progress_(self, value=None, head=None):                              # Output
    #--------------------------------------------------------------------------------
        if value is None:
            self.prog.reset_time()
        elif value < 0:
            self.prog.reset(N=abs(value))
        else:
            self.prog.print(value, head=head)

    #--------------------------------------------------------------------------------
    def status_(self, value=None, **kwargs):                                 # Output
    #--------------------------------------------------------------------------------
        value = value.replace('INFO','').strip()
        print('\r   '+value+(80-len(value))*' ', end='\n' if kwargs.get('newline') else '')

    #--------------------------------------------------------------------------------
    def restore_unrst_backup(self):                                          # Output
    #--------------------------------------------------------------------------------
        if self.slb_unrst_backup.exists():
            self.slb_unrst_backup.replace(self.slb_unrst.path)
            return True
        return False
            
    #--------------------------------------------------------------------------------
    def process_time(self):                                                  # Output
    #--------------------------------------------------------------------------------
        return str(datetime.now()-self.starttime).split('.')[0]
    
    #--------------------------------------------------------------------------------
    def message(self, msg, newline=True):                                    # Output
    #--------------------------------------------------------------------------------
        self.msg = msg
        self.status(value=self.msg, newline=newline)

    #--------------------------------------------------------------------------------
    def check(self, unrst:UNRST_file):                                        # Output
    #--------------------------------------------------------------------------------
        if self.report_dates is None:
            sim = INIT_file(self.root).simulator()
            input_file = {'ecl':DATA_file, 'ix':IX_input}[sim](self.root)
            self.report_dates = input_file.report_dates()
        passed = list(unrst.dates()) == self.report_dates
        if not passed:
            raise SystemError(f'ERROR Output-file {unrst} did not pass the check.'
                                + ' The report dates does not match the input file.')

    #--------------------------------------------------------------------------------
    def convert(self, delete=None, check=False, **kwargs):                   # Output
    #--------------------------------------------------------------------------------
        if delete is None:
            delete = self.del_convert
        if self.ior_unrst.is_file():
            self.message('INFO Convert already complete!')
            return True
        error_msg = 'ERROR Unable to convert IORSim output: '
        if not self.ior_funrst.is_file():
            self.message(error_msg + f'{self.ior_funrst} is missing')
            return False
        self.starttime = datetime.now()
        try:
            # Convert file
            self.status(value='Converting restart file...')
            unrst = self.ior_funrst.as_unrst(
                        rename=((b'TEMP    ', b'TEMP_IOR'),),
                        progress=lambda n: self.progress(value=n, head='Convert'),
                        **kwargs)
            self.ior_unrst.path = unrst.path
            if check:
                self.check(self.ior_unrst)
            if delete:
                silentdelete(self.ior_funrst.path)
        except (SystemError, KeyboardInterrupt, struct_error) as error:
            # Convert failed or cancelled, delete converted file
            silentdelete(self.ior_unrst.path)
            msg = str(error)
            if isinstance(error, KeyboardInterrupt) or 'run stopped' in msg.lower():
                self.message('Convert cancelled')
                return False 
            raise SystemError(error_msg + f'{error}') from error
        self.message(f'Convert complete, process-time was {self.process_time()}')
        return True


    #--------------------------------------------------------------------------------
    def merge(self, cancel=lambda:None, check=False, delete=None, 
              force=False):                                                  # Output
    #--------------------------------------------------------------------------------
        # Merge Eclipse and IORSim restart files
        if delete is None:
            delete = self.del_merge
        if force:
            silentdelete(self.merge_OK)
        self.msg = ''
        self.progress()   # Reset progress
        self.status(value='Merging Eclipse and IORSim restart files...')
        if self.merge_OK.exists():
            self.message('Merge already complete!')
            return True
        error_msg = 'ERROR Unable to merge restart files: '
        unrst_files = (self.slb_unrst, self.ior_unrst)
        missing = [file.name for file in unrst_files if not file.is_file()]
        if missing:
            self.message(error_msg + f'{", ".join(missing)} is missing')
            return False
        try:
            self.starttime = datetime.now()
            # Find the common first step-index to use for both files/sections
            #begin, fileind = max((next(file.steps()), i) for i,file in enumerate(unrst_files))
            # Get the number of sections from the file with the highest initial step-index
            #num_sec = unrst_files[fileind].count_sections()# - begin
            num_sec = [file.count_sections() for file in unrst_files]
            if len(set(num_sec)) > 1:
                self.message(f'WARNING Merging files have different lengths: {num_sec}')
            self.progress(value=-max(num_sec))
            # Get end-keyword of the IORSim-file
            ior_end_block = next(self.ior_unrst.tail_blocks(), None)
            # Currently, the UNRST file from IORSim use wrong payload sizes
            # which cause tail_blocks() to fail. We need to apply a fix before merging 
            if not ior_end_block:
                self.status(value=f'Fixing errors in {self.ior_unrst}...')
                num_fix = self.ior_unrst.fix_errors()
                ior_end_block = next(self.ior_unrst.tail_blocks())
            self.merge_unrst.end = ior_end = ior_end_block.key()
            # Define the sections in the restart file where the stitching is done
            slb_data = self.slb_unrst.section_data2(start=('SEQNUM'  , 'startpos'), end=('ENDSOL', 'endpos'))
            ior_data = self.ior_unrst.section_data2(start=('DOUBHEAD', 'endpos')  , end=(ior_end,  'endpos'),
                                                    rename=((b'TEMP    ',b'TEMP_IOR'),))
            # Create merged UNRST file
            merged_file = self.merge_unrst.merge(slb_data, ior_data,
                                                  progress=lambda n: self.progress(value=n, head='Merge'),
                                                  cancel=cancel)
            self.merge_unrst.assert_no_duplicates(raise_error=False)
            if check:
                self.check(self.merge_unrst)
            if merged_file.is_file():
                # Backup the original Eclipse UNRST-file
                self.slb_unrst.replace(self.slb_unrst_backup.path)
                # Rename the merged UNRST to the original root.UNRST
                merged_file.replace(self.slb_unrst.path)
            else:
                self.message(error_msg)
                return False
        except (Exception, KeyboardInterrupt) as error:
            # Merge failed or cancelled, delete merged file
            silentdelete(self.merge_unrst.path)
            if isinstance(error, KeyboardInterrupt):
                return False, 'Merge cancelled'
            if DEBUG:
                trace_print_exc()
            raise SystemError(f'{error_msg}: {error}') from error
            #raise SystemError(f'{error_msg}: {exc_info()[1]}')
        if delete:
            silentdelete(self.slb_unrst_backup.path, self.ior_unrst.path)
        # Create file to avoid re-merging of merged UNRST-files 
        self.merge_OK.touch()
        self.message(f'Merge complete, process-time was {self.process_time()}')
        return True


#====================================================================================
class Tandem_output:                                                  # Tandem_output
#====================================================================================
    #--------------------------------------------------------------------------------
    def __init__(self, root, log=None):                               # Tandem_output
    #--------------------------------------------------------------------------------
        self.root = Path(root)
        self.unified_host = UNRST_file(root)
        self.unified_ior = UNRST_file(str(root)+IOR_RESTART_NAMETAG)
        self.root_dir = self.root.parent
        self.log = None
        if callable(log):
            self.log = log
        self.nchecks = 0
        cl = [c['Cl'] for _, c in IORSim_input(self.root).solutions().items()]
        self.cl_limit = (min(cl), max(cl))

    #--------------------------------------------------------------------------------
    def _ready_for_append(self, single_unrst:UNRST_file,
                          unified_unrst:UNRST_file):                  # Tandem_output
    #--------------------------------------------------------------------------------
        """
        Check that single unrst has newer data than unified unrst
        """
        if (d1:=next(single_unrst.days())) > (d2:=next(unified_unrst.days(tail=True), -1)):
            return True
        if self.log:
            self.log(f'***** Skipped append of {d1}, timestep {d2} is already present *****')
        return False

    #--------------------------------------------------------------------------------
    def _log_append(self, single_unrst:UNRST_file,
                    unified_unrst:UNRST_file):                        # Tandem_output
    #--------------------------------------------------------------------------------
        if self.log:
            self.log(f'Append {single_unrst.path.relative_to(self.root_dir)}'
                    f' --> {unified_unrst.path.relative_to(self.root_dir)}')

    #--------------------------------------------------------------------------------
    def append_ior(self, unrst:UNRST_file):                           # Tandem_output
    #--------------------------------------------------------------------------------
        self._log_append(unrst, self.unified_ior)
        # Fix possible errors in IORSim unrst-file
        start = datetime.now()
        nerror = unrst.fix_errors()
        if nerror > 0 and self.log:
            self.log(f'Fixed {nerror} errors in {unrst} (in {(datetime.now()-start).microseconds/1e6:.2f} sec)')
        # Check that single unrst has newer data than unified unrst
        if self._ready_for_append(unrst, self.unified_ior):
            self.unified_ior.append_bytes(unrst.binarydata())

    #--------------------------------------------------------------------------------
    def append_host(self, unrst:UNRST_file):                          # Tandem_output
    #--------------------------------------------------------------------------------
        self._log_append(unrst, self.unified_host)
        # Check that single unrst has newer data than unified unrst
        if self._ready_for_append(unrst, self.unified_host):
            unrst.blocks_to_file(self.unified_host, keys=('FLO*', 'FLR*', 'B?', 'R?'),
                                 invert=True, append=True)

    #--------------------------------------------------------------------------------
    def check_chlorine(self, unrst:UNRST_file):                       # Tandem_output
    #--------------------------------------------------------------------------------
        def log(*arg):
            self.log(5*' ', *arg)

        var = 'CL'
        self.nchecks += 1
        limit = self.cl_limit
        data = next(unrst.cellarray(var), None)
        percents = (0.01, 0.1, 1, 5, 10, 50)
        if data:
            values = getattr(data, var.upper())
            ncells = prod(values.shape)
            high = [limit[1], values.max()]
            low = [limit[0], values.min()]
            col = [(' ', ' High ', ' Low ')]
            head = ('Limit', 'Value')
            row_val = (high, low)
            col.extend([(f' {name} ', *[f' {v:.4f} ' for v in val]) for name, val in zip(head, zip(*row_val))])
            over, under = [], []
            for i in percents:
                over.append( npsum(values > (1 + 0.01*i) * limit[1]) )
                under.append( npsum(values < (1 - 0.01*i) * limit[0]) )
                col.append([f' {i} % '] + [f' {val} ({100*val/ncells:.0f}) ' for val in (over[-1], under[-1])])

            # Write table of check-data to log
            log()
            log('Number of cells with chlorine values outside limit:')
            width = [max(map(len, c)) for c in col]
            line = ([w*'-' for w in width], '+', '>')
            nrows = len(col[0]) - 1
            header, *rows = zip(zip(*col), repeat('|'), ('^', *nrows*'>'))
            for row, delim, just in (line, header, line, *rows, line):
                log(delim + delim.join(f'{r:{just}{w}}' for r,w in zip(row, width)) + delim)

            # Write check-data to file
            mode = 'w'
            if self.nchecks > 1:
                mode = 'a'
            with open(f'{self.root}_{var}_limits.dat', mode, encoding='utf8') as out:
                if mode == 'w':
                    out.write('#' + ' '.join(f'{txt:11}' for txt in ('Days',) + head + tuple(f'>{p}%' for p in percents)) + '\n')
                days = next(unrst.days())
                for dbl, itg in ((high,over), (low,under)):
                    out.write(' '.join(f'{d:10.5f}' for d in [days]+dbl) + ' '.join(f'{i:10d}' for i in itg) + '\n')




#############################################################################
#                                                                           #
#                    Various utility functions                              #
#                                                                           #
#############################################################################



#--------------------------------------------------------------------------------
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
    parser.add_argument('-only_iorsim',    help="Run only IORSim", action='store_true')
    parser.add_argument('-only_host',      help="Run only host simulator", action='store_true')
    parser.add_argument('-intersect',      default=0, help="1: Use Intersect instead of Eclipse, 2: Use Intersect and convert from Eclipse case", type=int)
    parser.add_argument('-v',              default=DEFAULT_LOG_LEVEL, help='Verbosity level, higher number increase verbosity, default is 3', type=int)
    if SCHEDULE_SKIP_EMPTY:
        parser.add_argument('-not_skip_empty', help='Do not skip empty schedule-file entries', action='store_true')
    else:
        parser.add_argument('-skip_empty', help='Skip schedule-file entries with no statements', action='store_true')
    parser.add_argument('-keep_files',     help='Interface-files are not deleted after completion', action='store_true')
    parser.add_argument('-to_screen',      help='Print program log to screen', action='store_true')
    # parser.add_argument('-only_convert',   help='Only convert+merge and exit', action='store_true')
    # parser.add_argument('-only_merge',     help='Only merge and exit', action='store_true')
    parser.add_argument('-keep_merge',     help='Keep the separate UNRST-files (IX/ECL and IORSim) after successful merge', action='store_true')
    parser.add_argument('-keep_convert',   help='Keep the FUNRST-file created by IORSim after successful convert', action='store_true')
    parser.add_argument('-ecl_alive',      help=f'Keep Eclipse alive at least {ECL_ALIVE_LIMIT} seconds', action='store_true')
    parser.add_argument('-ior_alive',      help='Keep IORSim alive', action='store_true')
    parser.add_argument('-check_input',    help='Check IORSim input file keywords', action='store_true', dest='check_input_kw')
    parser.add_argument('-logtag',          help='Add this tag to the log-files', type=int)
    args = vars(parser.parse_args())
    if SCHEDULE_SKIP_EMPTY:
        args['skip_empty'] = not args['not_skip_empty']
    # Look for case in case_dir if root is not a file
    args['root'] = Path(args['root']).with_suffix('')
    # Read iorexe from settings if argument is missing
    if not args['iorexe']:
        if iorsim:=File(settings_file).line_matching('iorsim'):
            args['iorexe'] = iorsim.split()[-1]
        else:
            raise SystemError('IORSim executable is missing')
    return args

#--------------------------------------------------------------------------------
def progress_without_end(text='', length=20):
#--------------------------------------------------------------------------------
    def progress(n):
        pos = n%length
        rest = length-pos-1
        print('\r' + text + ' [' + '-'*pos + '#' + '-'*rest + ']', end='', flush=True)
    return progress

@print_error
#--------------------------------------------------------------------------------
def runsim(root=None, time=None, iorexe=None, eclexe='eclrun', to_screen=False,
           check_unrst=True, check_rft=True, keep_files=False, 
           convert=True, del_convert=True, merge=True, del_merge=True,
           ecl_alive=False, ior_alive=False, only_host=False, only_iorsim=False, intersect=0,
           check_input=False, verbose=DEFAULT_LOG_LEVEL, logtag=None, skip_empty=SCHEDULE_SKIP_EMPTY,
           tandem=False, del_tandem=True, **kwargs):
#--------------------------------------------------------------------------------
    """
    Run IORSim with Eclipse/Intersect in forward- or backward-mode. 
    Also possible to run only Eclipse/Intersect or only IORSim (requires ECL/IX output)

    Args:
        root (Path or str, mandatory): _Path to input file without extension_
             
        time (int):
            Number of timesteps to run. Only valid for backward runs
            
        iorexe (str):
            Path to the IORSim executable

        eclexe (str): 'eclrun'
            Path to 'eclrun'. Should be in the search-path of the OS
            
        to_screen (bool): False
            If True, output is written to screen/terminal and not redirected to log-files
            
        check_unrst (bool): True
        
        check_rft (bool, optional): _description_. Defaults to True.
        keep_files (bool, optional): _description_. Defaults to False.
        only_convert (bool, optional): _description_. Defaults to False.
        only_merge (bool, optional): _description_. Defaults to False.
        convert (bool, optional): _description_. Defaults to True.
        merge (bool, optional): _description_. Defaults to True.
        delete (bool, optional): _description_. Defaults to True.
        ecl_alive (bool, optional): _description_. Defaults to False.
        ior_alive (bool, optional): _description_. Defaults to False.
        only_host (bool, optional): _description_. Defaults to False.
        only_iorsim (bool, optional): _description_. Defaults to False.
        intersect (int, optional): _description_. Defaults to 0.
        check_input (bool, optional): _description_. Defaults to False.
        verbose (_type_, optional): _description_. Defaults to DEFAULT_LOG_LEVEL.
        logtag (_type_, optional): _description_. Defaults to None.
        skip_empty (_type_, optional): _description_. Defaults to SCHEDULE_SKIP_EMPTY.

    Returns:
        _type_: _description_
    """
    #----------------------------------------
    def status(value=None, **x):
    #----------------------------------------
        if not to_screen and value:
            value = value.replace('INFO','').strip()
            print('\r   '+value+(80-len(value))*' ', end=x.get('newline') and '\n' or '')

    #----------------------------------------
    def message(text=None, **x):
    #----------------------------------------
        if text:
            print(f'\n\n     {text}\n')

    prog = Progress(format='40#')

    #----------------------------------------
    def progress(run=None, value=None, head=None, min=None, n0=None):
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
            prog.print(value, head=head, text=run and f'({run.name})' or '')
            
    # Check if we only run eclipse or iorsim
    mode, runs = None, ['eclipse', 'iorsim']
    if only_host or only_iorsim:
        mode = 'forward'
        runs = ['eclipse'] if only_host else ['iorsim']

    if intersect and runs[0] == 'eclipse':
        runs[0] = 'intersect'

    if intersect > 1 and IX_input.need_convert(root):
        conv_prog = progress_without_end(text='   Convert Eclipse case to Intersect', length=20)
        success, log = IX_input.from_eclipse(root, progress=conv_prog, freq=10)
        print('\r' + ' '*80, end='')
        if not success:
            raise SystemError(f'ERROR Unable to create Intersect input, check the log {log}')
            
    if tandem:
        mode = 'tandem'
    
    sim = Simulation(root=root, time=time, iorexe=iorexe, eclexe=eclexe,
                     check_unrst=check_unrst, check_rft=check_rft, keep_files=keep_files,
                     progress=progress, status=status, message=message, to_screen=to_screen,
                     convert=convert, merge=merge, del_merge=del_merge, del_convert=del_convert,
                     ecl_keep_alive=ecl_alive, ior_keep_alive=ior_alive,
                     run_names=runs, mode=mode, check_input_kw=check_input, verbose=verbose,
                     logtag=logtag, skip_empty=skip_empty, del_tandem=del_tandem, **kwargs)
    sim.prepare()
    if not sim.ready():
        return

    # if only_convert or only_merge:
    #     sim.convert_and_merge()
    #     return

    if not to_screen:
        print(sim.info_header(long_lognames=running_jupyter()))
    result, msg = sim.run()
    print()
    return sim

@print_error
#--------------------------------------------------------------------------------
def runsim_with_plot(plot=None, run=None, update=1):
#--------------------------------------------------------------------------------
    if plot is None:
        plot = {}
    if run is None:
        run = {}
    unsmry = UNSMRY_file(run.get('root'))
    plot['only_new'] = True
    live_plot = LivePlot(func=unsmry.plot, **plot)
    unsmry.delete()
    thread = Thread(target=runsim, kwargs=run)
    thread.start()
    live_plot.loop(thread=thread, wait=update)
    #live_plot.loop(wait=update)
    #while thread.is_alive():
    #    pass
    #live_plot.stop()
    #print('!!!!!!!!END!!!!!')


@print_error
#--------------------------------------------------------------------------------
def main(settings_file=None):
#--------------------------------------------------------------------------------
    from os import _exit as os_exit
    args = parse_input(settings_file=settings_file)
    runsim(root=args['root'], time=args['days'], check_unrst=(not args['no_unrst_check']), 
           check_rft=(not args['no_rft_check']), to_screen=args['to_screen'], eclexe=args['eclexe'], 
           iorexe=args['iorexe'], del_merge=(not args['keep_merge']), del_convert=(not args['keep_convert']), 
           keep_files=args['keep_files'], only_convert=args['only_convert'], only_merge=args['only_merge'], 
           ecl_alive=args['ecl_alive'] and ECL_ALIVE_LIMIT, 
           ior_alive=args['ior_alive'] and IOR_ALIVE_LIMIT, 
           only_host=args['only_host'], only_iorsim=args['only_iorsim'], 
           check_input=args['check_input_kw'], verbose=args['v'], logtag=args['logtag'], 
           skip_empty=args['skip_empty'])
    os_exit(0)


######################################################################################

if __name__ == '__main__':

    main()
