#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#import atexit
from collections import namedtuple
import os
from pathlib import Path
import sys
from argparse import ArgumentParser
#from shutil import which
from datetime import datetime, timedelta
from time import sleep
#from types import GeneratorType
from psutil import NoSuchProcess
import shutil
import traceback
from numpy import ceil

from IORlib.utils import flatten_list, matches, number_of_blocks, safeopen, Progress, check_endtag, warn_empty_file, silentdelete, delete_files_matching, file_contains
from IORlib.runner import runner
from IORlib.ECL import check_blocks, get_schedule_time_and_filepos, get_tsteps, unfmt_file, fmt_file, Section, input_days_and_steps as ECL_input_days_and_steps



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
        super().__init__(name='Eclipse', case=root, exe=exe, cmd=[exe, 'eclipse', root], **kwargs)                        
        self.unrst = Path(root+'.UNRST')
        self.unsmry = unfmt_file(root+'.UNSMRY')
        self.rft = Path(root+'.RFT')
        self.inputfile = Path(root+'.DATA')

    #--------------------------------------------------------------------------------
    def time_and_step(self):                                                # eclipse
    #--------------------------------------------------------------------------------
        #for block in self.unsmry.blocks(only_new=True):
        t, n = 0, 0
        for block in self.unsmry.tail_blocks():
            if block.key()=='PARAMS':
                t = block.data()[0]
            if block.key()=='MINISTEP':
                n = block.data()[0]
                break
        #print('{}: time = {}, step = {}'.format(self.name, t, n))
        return int(t), int(n)        

    #--------------------------------------------------------------------------------
    def delete_output_files(self):                                          # eclipse
    #--------------------------------------------------------------------------------
        silentdelete( [self.unrst, self.rft] +
                      [str(self.case)+ext for ext in ('.SMSPEC','.UNSMRY','.RTELOG','.RTEMSG','_ECLIPSE.UNRST')] )
    

    #--------------------------------------------------------------------------------
    def check_input(self):                                                  # eclipse
    #--------------------------------------------------------------------------------
        super().check_input()
        msg = 'WARNING Unable to start ' + self.name + ': '
        ### check root.DATA exists
        inp_file = str(self.case)+'.DATA'
        if not Path(inp_file).is_file():
            raise SystemError(msg + 'missing input file ' + inp_file)

    # #--------------------------------------------------------------------------------
    # def start(self):                                                        # eclipse
    # #--------------------------------------------------------------------------------
    #     # check input
    #     self.check_input()
    #     # delete output from previous runs
    #     #self.delete_output_files()
    #     # start run
    #     super().start()    


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
        if file_contains(str(self.case)+'.DATA', text='READDATA', comment='--'):
            raise SystemError('WARNING The current case cannot be used in forward-mode: '+
                              'Eclipse input contains the READDATA keyword.')



#====================================================================================
class ecl_backward(eclipse):                                           # ecl_backward
#====================================================================================
    #--------------------------------------------------------------------------------
    def __init__(self, check_unrst=True, check_rft=True, rft_size=True, **kwargs):
    #--------------------------------------------------------------------------------
        super().__init__(ext_iface='I{:04d}', ext_OK='OK', **kwargs)
        self.init_tsteps = kwargs.get('init_tsteps') or 1
        #print(self.init_tsteps)
        self.check_unrst = check_unrst
        self.check_rft = check_rft
        self.rft_size = rft_size
        self.unrst_check = check_blocks(self.unrst, start='SEQNUM', end='ENDSOL', var='nwell')
        self.rft_check = check_blocks(self.rft, start='TIME', end='CONNXT')

    #--------------------------------------------------------------------------------
    def check_input(self):                                             # ecl_backward
    #--------------------------------------------------------------------------------
        super().check_input()
        ### Check root.DATA exists and that READDATA keyword is NOT present
        if not file_contains(str(self.case)+'.DATA', text='READDATA', comment='--'):
            raise SystemError('WARNING The current case cannot be used in backward-mode: '+
                              'Eclipse input is missing the READDATA keyword.')

    #--------------------------------------------------------------------------------
    def start(self):                                                   # ecl_backward
    #--------------------------------------------------------------------------------
        # Start Eclipse in backward mode
        self.n = self.init_tsteps
        if self.echo:
            print('  Starting Eclipse...', end='', flush=True)
        self.interface_file('all').delete()
        # Need to create all interface files in advance to avoid Eclipse termination
        #[self.interface_file(i).create_empty() for i in range(1, self.N+1)] 
        [self.interface_file(i).create_empty() for i in range(self.n, self.N+self.n)] 
        self.OK_file().delete()
        super().start()  # eclipse.start()
        self.wait_for( self.unrst.exists, error=self.unrst.name+' not created')
        self.wait_for( self.rft.exists, error=self.rft.name+' not created')
        self.check_UNRST_file(nblocks=self.init_tsteps)
        self.nwell = self.unrst_check.var('nwell')
        nwell_max = (self.init_tsteps+1)*self.nwell
        rft_wells = self.check_RFT_file(nwell_max=nwell_max, nwell_min=self.nwell, limit=200)
        # If some wells are missing in the RFT-file we need to 
        # do the full RFT-check, not only the file-size check
        if rft_wells != nwell_max:
            self.rft_size = False
            info = 'Turned on full RFT-check (default is file-size check)' 
            self._print(info)
        #print(f'self.nwell: {self.nwell}, nwell_max: {(self.init_tsteps+1)*self.nwell}, rft_wells: {rft_wells}')
        self.suspend(check=True)
        if self.echo:
            print('\r  Eclipse started, log file is ' + self.get_logfile(), flush=True)
            print('  ' + self.timer.info) if self.timer else None
        if self.init_tsteps > 1:
            self.rft_size = None
        # only check RFT-file by size if all wells are initially written to the RFT-file 
        if self.rft_size and rft_wells == 2*self.nwell:
            # get size of RFT file
            self.rft_size = int(0.5*self.rft.stat().st_size)
            if 2*self.rft_size != self.rft.stat().st_size:
                self.print2log('\nWARNING! Initial size of RFT size not even!\n')

    #--------------------------------------------------------------------------------
    def run_one_step(self, satnum_file):                            # ecl_backward
    #--------------------------------------------------------------------------------
        self.rft_start_size = self.rft.stat().st_size
        ### run Eclipse
        #self.n = n + self.init_tsteps - 1
        #print('ecl, n: ',self.n)
        self.interface_file(self.n).copy(satnum_file, delete=True)
        self.OK_file().create_empty()
        self.resume(check=True)
        self.wait_for( self.OK_file().is_deleted, error=self.OK_file().name()+' not deleted')
        if self.check_unrst:
            self.check_UNRST_file()
        if self.check_rft:
            if self.rft_size:
                self.wait_for( self.check_RFT_size )
            else:
                self.check_RFT_file(nwell_max=self.nwell, nwell_min=1, limit=200)
        #sleep(1)
        self.suspend(check=True)
        self.n += 1

    #--------------------------------------------------------------------------------
    def check_UNRST_file(self, nblocks=1):                                             # eclipse
    #--------------------------------------------------------------------------------
        self.wait_for( self.unrst_check.blocks_complete, nblocks=nblocks, log=self.unrst_check.info,
                       error=self.unrst_check.file.name()+' not complete' )
        
    #--------------------------------------------------------------------------------
    def check_RFT_file(self, nwell_max=0, nwell_min=0, limit=10000):        # eclipse
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
                #self._print('', tag='')
                raise SystemError('ERROR No TIME blocks found in the RFT-file. Check for closed wells in the Eclipse log.')
        return nblocks

    #--------------------------------------------------------------------------------
    def check_RFT_size(self):                                               # eclipse
    #--------------------------------------------------------------------------------
        diff = self.rft.stat().st_size-self.rft_start_size
        if diff==self.rft_size:
            #if self.rft_check.file.tail_block_is('CONNXT'):
            return True
            #else:
            #    return False
        else:
            return False



#====================================================================================
class iorsim(runner):                                                        # iorsim
#====================================================================================
    #--------------------------------------------------------------------------------
    def __init__(self, root=None, exe='IORSimX', args='', **kwargs):
    #--------------------------------------------------------------------------------
        #print('iorsim.__init__: ',root, exe, args, kwargs)
        if '.exe' not in exe and sys.platform == 'win32':
            exe += '.exe'
        # IORSim only accepts root relative to the current directory
        abs_root = str(root)
        cwd = Path().cwd()
        if str(cwd) in str(root):
            root = Path(root).relative_to(cwd)
        cmd = [exe, '-root_name='+str(root)] + args.split()
        super().__init__(name='IORSim', case=root, exe=exe, cmd=cmd, **kwargs)
        self.trcconc = None
        self.funrst = Path(abs_root+'_IORSim_PLOT.FUNRST')
        self.inputfile = Path(abs_root+'.trcinp')


    #--------------------------------------------------------------------------------
    def check_input(self):                                                   # iorsim
    #--------------------------------------------------------------------------------
        super().check_input()

        msg = 'WARNING Unable to start IORSim: '
        ### check root.trcinp exists
        inp_file = str(self.case)+'.trcinp'
        if not Path(inp_file).is_file():
            raise SystemError(msg + 'missing input file ' + inp_file)

        ### check that Eclipse UNRST and RFT files exists
        for ext in ('.UNRST','.RFT'):
            fname = str(self.case)+ext
            if not Path(fname).is_file():
                raise SystemError(msg + 'missing Eclipse file ' + str(Path(fname).name))


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
        delete_files_matching(case+'*.trcconc')
        delete_files_matching(case+'*.trcprd')
        silentdelete(self.funrst)



#====================================================================================
class ior_forward(forward_mixin, iorsim):                               # ior_forward
#====================================================================================
    pass

#====================================================================================
class ior_backward(iorsim):                                            # ior_backward
#====================================================================================
    #--------------------------------------------------------------------------------
    #def __init__(self, dt=1, **kwargs):
    def __init__(self, **kwargs):
    #--------------------------------------------------------------------------------
        # Call iorsim.__init__()
        super().__init__(args='-readdata', ext_iface='IORSimI{:04d}', ext_OK='IORSimOK', **kwargs)
        self.satnum = Path('satnum.dat')   # Output-file from IORSim, read by Eclipse as an interface-file
        self.endtag = '-- IORSimX done.'
        self.satnum_check = check_endtag(file=self.satnum, endtag=self.endtag)  # Check if satnum-file is flushed


    #--------------------------------------------------------------------------------
    def delete_output_files(self):                                     # ior_backward
    #--------------------------------------------------------------------------------
        super().delete_output_files()
        silentdelete(self.satnum)

    #--------------------------------------------------------------------------------
    def start(self):                                                   # ior_backward
    #--------------------------------------------------------------------------------
        # Start IORSim backward run
        self.n = 1
        if self.echo:
            print('\n  Starting IORSim...', end='', flush=True)
        self.interface_file('all').delete()
        self.interface_file(self.n).create_empty()
        self.OK_file().create_empty()
        super().start() # iorsim.start()   
        self.wait_for( self.OK_file().is_deleted, error=self.OK_file().name()+' not deleted')
        self.suspend(check=True)
        if self.echo:
            print('\r  IORSim started, log file is ' + self.get_logfile(), flush=True)
            print('  ' + self.timer.info) if self.timer else None

    #--------------------------------------------------------------------------------
    def run_one_step(self):                                         # ior_backward
    #--------------------------------------------------------------------------------
        ### run IORSim
        #self.n = n
        self.n += 1
        #print('ior, n: ',self.n)
        self.interface_file(self.n).create_empty()
        self.OK_file().create_empty()
        self.resume(check=True)
        self.wait_for( self.OK_file().is_deleted, error=self.OK_file().name()+' not deleted')
        self.wait_for( self.satnum_check.find_endtag, error=self.satnum_check.file().name+' has no endtag')
        warn_empty_file(self.satnum, comment='--')
        self.suspend(check=True)

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
    def __init__(self, case, ext='.SCH', comment='--', end='/', tag='TSTEP'):
    #--------------------------------------------------------------------------------
        self.file = Path(case).with_suffix(ext)
        self.ifacefile = None
        self.comment = comment
        self.tag = tag
        self.end = end
        self.count = 0
        self.length = 0
        self.exists = self.file.is_file()
        if self.exists:
            self._schedule = self.read()
            self.length = len(self._schedule)

    #--------------------------------------------------------------------------------
    def read(self):                                                        # schedule     
    #--------------------------------------------------------------------------------
        ''' 
        Returns a list with tsteps at even positions followed by a list of schedule
        statements at odd positions:
        schedule = [10.0 ['WELSPECS', "P1"], 34.0, ['END']]
        '''
        if self.file.is_file():
            with open(self.file) as f:
                lines = f.readlines()
            # Remove whitespace
            lines = [line.strip() for line in lines if len(line)>0]
            # Remove empty lines
            lines = [line for line in lines if len(line)>0]
            # Remove commented lines
            lines = [line for line in lines if not line.startswith(self.comment)]
            dt, pos = get_schedule_time_and_filepos(lines)
            #print(dt, pos)
            kw = flatten_list([ [ lines[pos[n][0]:pos[n][1]] or '' ] for n in range(len(pos)) ])
            schedule = flatten_list([[sum(t), kw[n]] for n,t in enumerate(dt)])
            #print(schedule[:4])
            #print(schedule[-4:])
        return schedule

    #--------------------------------------------------------------------------------
    def set_interface_file(self, filename, filepos=4):                            # schedule
    #--------------------------------------------------------------------------------
        self.ifacefile = filename
        self.filepos = filepos

    #--------------------------------------------------------------------------------
    def next_actions(self):                                                        # schedule
    #--------------------------------------------------------------------------------
        return self.actions(inc=True)

    #--------------------------------------------------------------------------------
    def _get(self, n, inc=False):                                               # schedule
    #--------------------------------------------------------------------------------
        if self.count<self.length and self.count%2==n:
            val = self._schedule[self.count]
            if inc:
                self.count += 1
            #print(f'inc:{inc}, val:{val}')
            return val
        return False

    #--------------------------------------------------------------------------------
    def _set(self, n, val):                                               # schedule
    #--------------------------------------------------------------------------------
        if self.count<self.length and self.count%2==n:
            self._schedule[self.count] = val
            return self._schedule[self.count]
        return False

    #--------------------------------------------------------------------------------
    def tstep(self, **kwargs):                                               # schedule
    #--------------------------------------------------------------------------------
        return self._get(0, **kwargs)

    #--------------------------------------------------------------------------------
    def set_tstep(self, val):                                               # schedule
    #--------------------------------------------------------------------------------
        return self._set(0, val)

    #--------------------------------------------------------------------------------
    def actions(self, **kwargs):                                               # schedule
    #--------------------------------------------------------------------------------
        return '\n'.join(self._get(1, **kwargs) or '')

    #--------------------------------------------------------------------------------
    def append(self, text):                               # schedule
    #--------------------------------------------------------------------------------
        if not text:
            return
        with open(self.ifacefile, 'r') as f:
            lines = f.readlines()
        #n = safeindex(lines, self.endtag)
        n = len(lines) - self.filepos
        #print('append_to_satnum:',n, text)
        if n:
            lines.insert(n, text+'\n')
            with open(self.ifacefile, 'w') as f:
                f.write(''.join(lines))

    #--------------------------------------------------------------------------------
    def update(self):                                               # schedule
    #--------------------------------------------------------------------------------
        # Append next keywords to outfile if tstep==0 
        if self.tstep()==0:
            self.count += 1
            self.append(self.actions(inc=True))
            #print('actions:',self.actions(), 'count:', self.count, 'length:',self.length)
            return
        # Update schedule tstep by reading TSTEP from the interface-file
        tstep = get_tsteps(self.ifacefile)[0]
        #old_tstep = self.tstep()
        self.set_tstep( max(self.tstep()-int(tstep), 0) )
        #print(f'tstep: {tstep}, old_tstep: {old_tstep}, new_tstep: {self.tstep()}, count: {self.count}')

    #--------------------------------------------------------------------------------
    def check(self):                                               # schedule
    #--------------------------------------------------------------------------------
        # just a check...
        with open(self.ifacefile, 'r') as f:
            lines = f.readlines()
        print(self.ifacefile, ': ', lines[-10:])


#====================================================================================
class simulation:
#====================================================================================
    #--------------------------------------------------------------------------------
    def __init__(self, mode=None, root=None, pause=0, init_tstep=1, runs=[], to_screen=False, convert=True,
                 status=lambda **x:None, progress=lambda **x:None, plot=lambda **x:None, **kwargs):
    #--------------------------------------------------------------------------------
        #print('mode',mode,'root',root,'pause',pause,'runs',runs,'to_screen',to_screen,'convert',convert,
        #      'status',status,'progress',progress,'plot',plot,kwargs)
        #print(pause)
        self.name = 'ior2ecl'
        self.root = root
        self.update = namedtuple('update',['status','progress','plot'])(status, progress, plot)
        self.pause = pause
        self.init_tstep = init_tstep # backward mode: initial Eclipse TSTEP after READDATA
        self.convert = convert
        self.runlog = None
        if root and not to_screen:
            self.runlog = safeopen(Path(root).parent/(self.name+'.log'), 'w')
            #atexit.register(self.runlog.close)
        self.print2log = lambda txt: print(txt, file=self.runlog, flush=True)
        self.current_run = None
        self.runs = runs
        self.run_sim = None
        self.ior = self.ecl = None
        self.T = 0
        self.mode = mode
        self.schedule = None
        if root:
            kwargs.update({'root':str(root), 'runlog':self.runlog})
            self.prepare_mode(**kwargs)


    #-----------------------------------------------------------------------
    def prepare_mode(self, iorexe=None, eclexe=None, time=0, time_ecl=None, dt_ecl=None, **kwargs):
    #-----------------------------------------------------------------------
        if not self.mode:
            self.mode = self.mode_from_case()
        # Backward simulation
        if self.mode=='backward':
            #self.schedule = Schedule(self.root)
            if not dt_ecl:
                #dt_ecl = dtecl(self.root)
                dt_ecl = ior_input(var='dtecl', root=self.root)
            sum_tstep, len_tstep, tsteps = ECL_input_days_and_steps(self.root)
            if not time_ecl:
                time_ecl = sum_tstep
            # Need to calculate number of steps based on duration of 
            # simulation, initial TSTEP from .schedule-file, and TSTEP 
            # preceding READDATA in .DATA-file
            #dt_init = int( self.schedule.tstep() )
            dt_init = int( self.init_tstep )
            self.T = int(time)+int(time_ecl)+dt_init
            N = int(ceil((time+dt_init)/dt_ecl))
            kwargs.update({'N':N, 'T':self.T, 'init_tsteps':len_tstep})
            self.run_sim = self.backward
            self.runs = [ecl_backward(exe=eclexe, **kwargs), ior_backward(exe=iorexe, **kwargs)]
            self.ecl, self.ior = self.runs
            # Add satnum-file to schedule, filepos is position of TSTEP 
            self.schedule = Schedule(self.root)            
            self.schedule.set_interface_file(self.ior.satnum, filepos=4)
        # Forward simulation
        if self.mode=='forward':
            self.T = time
            kwargs.update({'T':self.T})
            self.run_sim = self.forward
            if not self.runs:
                self.runs = ('eclipse','iorsim')
            for name in self.runs:
                if name=='eclipse':
                    self.ecl = ecl_forward(exe=eclexe, **kwargs)
                if name=='iorsim':
                    self.ior = ior_forward(exe=iorexe, **kwargs)
            #print(self.T, self.runs)
            self.runs = [run for run in (self.ecl, self.ior) if run]

    #-----------------------------------------------------------------------
    def set_time(self, time):
    #-----------------------------------------------------------------------
        self.T = time
        for run in self.runs:
            run.set_time(time)

    #-----------------------------------------------------------------------
    def mode_from_case(self):
    #-----------------------------------------------------------------------
        data = str(self.root)+'.DATA'
        if Path(data).is_file():
            if file_contains(data, text='READDATA', comment='--'):
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


    #-----------------------------------------------------------------------
    def forward(self): 
    #-----------------------------------------------------------------------
        #print('Forward run')
        run_time = timedelta()
        ret = ''
        for run in self.runs:
            self.current_run = run.name.lower()
            run.check_input()
            run.delete_output_files()
            self.update.status(value='Starting '+run.name)
            self.update.progress(value=-run.T)
            run.start()
            self.update.status(value=run.name+' running')
            run.init_control_func(update=self.update) 
            run.wait_for_process_to_finish(pause=0.2, loop_func=run.control_func)
            t = run.time_and_step()[0]
            if t < run.T:
                raise SystemError('ERROR ' + run.name + ' stopped unexpectedly, check the log')
            run_time += run.run_time()
            ret = run.complete_msg(run_time=run_time)
        return ret


    #-----------------------------------------------------------------------
    def backward(self): 
    #-----------------------------------------------------------------------
        self.update.progress(value=-self.runs[0].T)
        for run in self.runs:
            run.check_input()
            run.delete_output_files()
            self.update.status(value='Starting ' + run.name + '...')
            run.start()
        ecl, ior = self.runs
        #ior.satnum.write_text(self.schedule.next_tstep())
        ior.satnum.write_text(f'TSTEP\n{self.init_tstep} /') 

        # Start timestep loop
        #for n in range(self.N_start, ecl.N+self.N_start):
        for n in range(ecl.N):
            self.print2log('\nLoop step {}'.format(n))
            ecl.run_one_step(ior.satnum)
            # Need a short stop after Eclipse has finished, otherwise IORSim sometimes stops 
            sleep(self.pause)
            # Run IORSim to prepare satnum input for the next Eclipse run
            ior.run_one_step() #n+ecl.init_tsteps)
            if self.schedule.exists:
                self.schedule.update()
                #self.schedule.check()
            ior.t = ior.time_and_step()[0]
            self.update.progress(value=ior.t)
            self.update.status(run=ior, mode='backward')
            self.update.plot()
            #print('ecl',ecl.t, ecl.T, ecl.n, ecl.N)
            #print('ior',ior.t, ior.T, ior.n, ior.N)
            if ior.t > ior.T:
                #print('t>T')
                raise SystemError(ior.complete_msg())
        # Timestep loop finished
        ecl.quit()
        ior.quit()
        return ecl.complete_msg()

    #-----------------------------------------------------------------------
    def info_header(self):
    #-----------------------------------------------------------------------
        logfiles = [run.log.name for run in self.runs]+[log.name for log in (self.runlog,) if log]
        case = Path(self.root).name
        print()
        print('   {:10s}: {}'.format('Case', case))
        print('   {:10s}: {}'.format('Mode', self.mode.capitalize())) 
        print('   {:10s}: {}'.format('Days', self.T), end='')
        if self.mode=='forward':
            print(' (update TSTEP in '+case+'.DATA to change number of days)')
        else:
            print()
        print('   {:10s}: {}'.format('Folder', Path(self.root).parent))
        print('   {:10s}: {}'.format('Log-files', ', '.join([Path(file).name for file in logfiles])))
        print()

    #-----------------------------------------------------------------------
    def run(self):
    #-----------------------------------------------------------------------
        msg = ''
        success = False
        try:
            msg = self.run_sim()
            success = True
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
            [run.kill_and_clean() for run in self.runs]
            self.print2log('\n======  ' + msg + '  ======')
            self.current_run = None
            self.update.progress(value=0)   # Reset progress time
            self.update.plot()
            self.update.status(value=msg)
            if self.convert and success:
                sleep(0.05)  # Need a short break here to make the GUI progressbar responsive
                complete, conv_msg = self.convert_restart_file(case=self.root)
                self.print2log('\n===== '+conv_msg+' ======')
                if not complete:
                    self.update.status(value=conv_msg)
                    msg += '\n'+conv_msg
            self.runs = []
            if self.runlog:
                self.runlog.close()
            return success, msg


    #-----------------------------------------------------------------------
    def convert_restart_file(self, case=None, fast=True):
    #-----------------------------------------------------------------------
        msg = ''
        complete = False
        ecl = self.ecl or eclipse(root=case)   
        ior = self.ior or iorsim(root=case)   
        if not ior.funrst.is_file():
            return complete, '' #str(ior.funrst) + ' does not exist'
        self.update.status(value='Converting restart file...')
        # Convert from formatted to unformatted restart file
        start = datetime.now()
        try:
            infile = fmt_file(ior.funrst)
            if fast:
                convert = infile.fast_convert
            else:
                N = number_of_blocks(file=ior.funrst, blockstart='SEQNUM')
                self.update.progress(value=-(N-1))
                convert = infile.convert 
            ior_unrst = convert(rename_duplicate=True, rename_key=('TEMP','TEMP_IOR'),
                                progress=lambda n: self.update.progress(value=n), 
                                cancel=ior.stop_if_canceled)
        except SystemError as e:
            msg = str(e)
            if 'run stopped' in msg.lower():
                msg = 'Convert cancelled'
            return complete, msg
        except KeyboardInterrupt:
            return complete, 'Convert cancelled'

        # Merge Eclipse and IORSim restart files
        if ecl.unrst.is_file() and ior_unrst.is_file():
            backup = Path(str(ecl.case)+'_ECLIPSE.UNRST')
            if backup.is_file():
                # This is a pure IORSim run and backup already exists; restore backup
                shutil.copy(backup, ecl.unrst)
            else:
                # No backup exists; create backup copy
                shutil.copy(ecl.unrst, backup)
        else:
            self.update.status(value='Convert finished, unable to merge Eclipse and IORSim files')
            return
        # Define the sections in the restart files where they are stitched together
        ecl_sec = Section(ecl.unrst, start_before='SEQNUM', end_before='SEQNUM', skip_sections=(0,))
        ior_sec = Section(ior_unrst, start_after='DOUBHEAD', end_before='SEQNUM')
        self.update.status(value='Merging Eclipse and IORSim restart files...')
        merged_file = Path(str(ior.case)+'_MERGED.UNRST')
        merged_file = unfmt_file(merged_file).create(ecl_sec, ior_sec)
        # Rename merged UNRST-file to original restart file
        if not merged_file:
            raise SystemError('WARNING Unable to merge {} and {}'.format(ecl.unrst, ior_unrst))
        if merged_file.is_file():
            merged_file.replace(ecl.unrst)
        self.update.status(value='Simulation complete, restart file ready')
        msg = 'Convert and merge of restart file completed, process-time was '+str(datetime.now()-start).split('.')[0]
        complete = True    
        return complete, msg
        #self.progress(N)



#############################################################################


#-----------------------------------------------------------------------
def ior_input(var=None, root=None):
#-----------------------------------------------------------------------
#
#  Get dtecl from IORSim input file .trcinp
#  Assumed format:    
#    
#  *INTEGRATION
#  # tstart  tstop
#    0.0  1.e99
#  # dtmin dtmax 
#    0.0  1.e99
#  # dtecl dteclmax 
#    5      20 
#  # metnum
#    0
#
    file=f'{root}.trcinp'
    pos = var_group = None
    if var == 'dtecl':
        var_group = '\*INTEGRATION'
        pos = 4
    # Find position after var_group name 
    end = [m.span()[1] for m in matches(file=file, pattern=fr'{var_group}\s*\n+')]
    num = '\d+\.?e?E?\d*'
    # Find uncommented lines with two numbers
    val = [float(s.decode()) for m in matches(file=file, pattern=fr'\n+\s*{num}\s*{num}', pos=end[0]) for s in m.group(0).split()]
    # Find commented lines with variable names
    #if name:
    #    name = [s.decode() for m in matches(file=file, pattern=fr'(?<=#)\s*\D+\s*\D*', pos=end[0]) for s in m.group(0).split()]
    if val[pos]==0:
        raise SystemError('WARNING IORSim timestep (dtecl) is zero')
    return val[pos]


# #-----------------------------------------------------------------------
# def dtecl(root, ext='.trcinp'):                             # 
# #-----------------------------------------------------------------------
# #
# #  Get dtecl from IORSim input file .trcinp
# #  Assumed format:    
# #    
# #  *INTEGRATION
# #  # tstart  tstop
# #    0.0  1.e99
# #  # dtmin dtmax 
# #    0.0  1.e99
# #  # dtecl dteclmax 
# #    5      20 
# #  # metnum
# #    0
# #
#     read = False
#     dt = ()
#     with open(str(root)+ext) as f:
#         for line in f:
#             line = line.lstrip()
#             if line.startswith('#'):
#                 continue
#             if line.startswith('*INTEGRATION'):
#                 read = True
#                 continue
#             if read:
#                 dt += tuple(line.split())
#                 if len(dt) > 5:
#                     break
#     val = int(float(dt[4]))
#     if val==0:
#         raise SystemError('WARNING IORSim timestep (dtecl) is zero')
#     return val


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
    print('\n   Missing IORSim executable: '+str(iorexe)+'\n')
    raise SystemExit

#--------------------------------------------------------------------------------
def case_from_casedir(case_dir, root):
#--------------------------------------------------------------------------------
    # Find case in casedir if given DATA-file is missing
    case_dir = Path(case_dir)
    if case_dir.is_dir() and (case_dir/root/(root+'.DATA')).is_file():
        return case_dir/root/root
    print('\n   '+root+'.DATA'+' not found in '+str(case_dir/root)+'\n')
    raise SystemExit

#--------------------------------------------------------------------------------
def parse_input(case_dir=None, settings_file=None):
#--------------------------------------------------------------------------------
    description = 'Script for running IORSim and Eclipse in backward and forward mode'
    parser = ArgumentParser(description=description)
    parser.add_argument('root',            help='Eclipse case name without .DATA')
    parser.add_argument('days',            help='Time interval of the simulation, if 0 only convert is performed', type=int)
    parser.add_argument('-eclexe',         default='eclrun', help="Name of excecutable, default is 'eclrun'")
    parser.add_argument('-iorexe',         help="Name of IORSim executable, default is 'IORSimX'"                  )
    parser.add_argument('-no_unrst_check', help='Backward mode: do not check flushed UNRST-file', action='store_true')
    parser.add_argument('-no_rft_check',   help='Backward mode: do not check flushed RFT-file', action='store_true')
    parser.add_argument('-full_rft_check', help='Backward mode: Full check of RFT-file, default is to only check size', action='store_true')
    parser.add_argument('-pause',          default=0.5, help='Backward mode: pause between Eclipse and IORSim runs', type=float)
    parser.add_argument('-init_tstep',     default=1.0, help='Backward mode: initial Eclipse TSTEP', type=float)
    parser.add_argument('-v',              default=3, help='Verbosity level, higher number increase verbosity, default is 3', type=int)
    parser.add_argument('-keep_files',     help='Interface-files are not deleted after completion', action='store_true')
    parser.add_argument('-to_screen',      help='Print program log to screen', action='store_true')
    parser.add_argument('-only_convert',   help='Only convert and exit', action='store_true')
    args = vars(parser.parse_args())
    # Look for case in case_dir if root is not a file
    if case_dir and not Path(args['root']+'.DATA').is_file():
        args['root'] = case_from_casedir(case_dir, args['root'])
    # Read iorexe from settings if argument is missing
    if settings_file and not args['iorexe']:
        args['iorexe'] = iorexe_from_settings(settings_file, args['iorexe'])
    return args


#--------------------------------------------------------------------------------
def runsim(root=None, time=None, iorexe=None, eclexe='eclrun', to_screen=False, pause=0.5, 
           init_tstep=1.0, check_unrst=True, check_rft=True, rft_size=True, keep_files=False, only_convert=False):
#--------------------------------------------------------------------------------
    prog = Progress(format='40#')
    #----------------------------------------
    def progress(run=None, value=None):
    #----------------------------------------
        if value and value<0:
            prog.reset(N=abs(value))
            return
        if run:
            value = run.t
        prog.print(value)

    #----------------------------------------
    def status(value=None, **x):
    #----------------------------------------
        not to_screen and value and print('\r   '+value+50*' ', end='')

    sim = simulation(root=root, time=time, pause=pause, init_tstep=init_tstep, iorexe=iorexe, eclexe=eclexe, 
                     check_unrst=check_unrst, check_rft=check_rft, rft_size=rft_size, 
                     keep_files=keep_files, progress=progress, status=status, to_screen=to_screen)

    if only_convert:
        complete, conv_msg = sim.convert_restart_file(case=sim.root)
        return

    if sim.mode=='forward':
        sim.set_time(ECL_input_days_and_steps(root)[0])
    sim.info_header()
    result, msg = sim.run()
    not to_screen and print('\r   '+msg.replace('INFO','').strip()+'              \n')


#--------------------------------------------------------------------------------
def main(case_dir='GUI/cases', settings_file='GUI/settings.txt'):
#--------------------------------------------------------------------------------
    args = parse_input(case_dir=case_dir, settings_file=settings_file)
    runsim(root=args['root'], time=args['days'], check_unrst=(not args['no_unrst_check']), check_rft=(not args['no_rft_check']), rft_size=(not args['full_rft_check']), 
           to_screen=args['to_screen'], pause=args['pause'], init_tstep=args['init_tstep'], eclexe=args['eclexe'], iorexe=args['iorexe'],
           keep_files=args['keep_files'], only_convert=args['only_convert'])
    os._exit(0)


######################################################################################

if __name__ == '__main__':

    main()
