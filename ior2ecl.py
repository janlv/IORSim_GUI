#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import atexit
from pathlib import Path
import sys
from argparse import ArgumentParser
from shutil import which
from datetime import datetime, timedelta
from time import sleep
from types import GeneratorType
from psutil import NoSuchProcess
import shutil
import traceback

from IORlib.utils import safeopen, Progress, check_endtag, warn_empty_file, loop_until_2, silentdelete, assert_python_version, exit_without_atexit, delete_files_matching, file_contains
from IORlib.runner import runner
from IORlib.ECL import check_blocks, unfmt_file, fmt_file, Section


#--------------------------------------------------------------------------------
def main():
#--------------------------------------------------------------------------------
    try:
        assert sys.version_info >= (3,6)
    except AssertionError:
        raise SystemExit('\nThe minimal python version for this script is 3.6, you are running {}.{}\n'.
                         format(*sys.version_info))
        #assert_python_version(major=3, minor=6)
    
    try:
        print()
        args = parse_input()
        sim = ior2ecl(root=args['root'], dt=args['dt'], nsteps=args['n'], eclrun=args['eclrun'], iorsim=args['iorsim'],
                      iorargs=args['iorargs'], v=args['v'], timer=args['timer'], keep_files=args['keep_files'],
                      to_screen=args['to_screen']) 
        sim.check_input()
        sim.init_eclipse_run()
        sim.init_iorsim_run()
        
        print()
        sim.start_eclipse()
        sim.start_iorsim()

        print()
        for t in range(1, sim.nsteps+1):
            if sim.progress:
                sim.progress.print(sim.t) 
            sim.run_one_step()
        print()
            
        sim.terminate_runs()
        sim.close_logfiles()
        exit_without_atexit()  # do not call functions registered with atexit 
        
    except KeyboardInterrupt:
        print()
        print()
        print('  Interrupted by user, aborting....')
        print()
    except SystemError as e:
        raise SystemExit('\n'+str(e)+'\nExiting...')
        

#--------------------------------------------------------------------------------
def parse_input():
#--------------------------------------------------------------------------------
    parser = ArgumentParser()
    parser.add_argument('-root',       help='Eclipse case name without .DATA', required=True)
    parser.add_argument('-dt',         help='Timestep length', type=float, required=True)
    parser.add_argument('-n',          help='Number of timesteps', type=int, required=True)
    parser.add_argument('-eclrun',     help="Name of excecutable, default is 'eclrun'", default='eclrun')
    parser.add_argument('-iorsim',     help="Name of IORSim executable, default is 'IORSimX'", default='IORSimX')
    parser.add_argument('-iorargs',    help='Additional arguments passed to IORSim, should be quoted', default='')
    parser.add_argument('-v',          help='Verbosity level, higher number increase verbosity, default is 3', type=int, default=3)
    parser.add_argument('-timer',      help='Record execution times, default is FALSE', action='store_true')
    parser.add_argument('-keep_files', help='Interface-files are not deleted after completion', action='store_true')
    parser.add_argument('-to_screen',  help='Logging output is sent to the terminal', action='store_true')
    args = vars(parser.parse_args())
    return args



# #--------------------------------------------------------------------------------
# def wait_for(runner, func, *args, log=False, assert_running=True, error=None, raise_error=True,
#              limit=100000, sleep_sec=0.01, kill_func=None, kill_msg=None, **kwargs):
# #--------------------------------------------------------------------------------
#     runner._print('calling wait_for( ' + func.__qualname__ + ' )...', v=3, end='')
#     if assert_running:
#         assert_running = runner.assert_running
#     try:
#         n = loop_until(func, *args, **kwargs, limit=limit, assert_running=assert_running,
#                        error=error, sleep_sec=sleep_sec, kill_func=kill_func, kill_msg=kill_msg)
#         runner._print(' {:d} loops'.format(n), v=3, tag='')
#         if callable(log):
#             runner._print(log())
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
    def init_control_func(self, status=lambda *x:None, progress=lambda *x:None, plot=lambda:None, 
                              pause=0.01, count=5, **kwargs):
    #--------------------------------------------------------------------------------
        self.status = status
        self.progress = progress
        self.plot = plot
        self.loop_count = 0
        self.pause = pause
        self.count = count

    #--------------------------------------------------------------------------------
    def control_func(self):
    #--------------------------------------------------------------------------------
        self.stop_if_canceled()
        self.loop_count += 1
        #print(self.loop_count)
        if self.loop_count == self.count:
            self.loop_count = 0
            self.t = self.stop_if_limit_reached(limit='time')
            #print(self.t, self.T, self.n, self.N)
            self.progress(self.t)
            self.status('{}/{} days'.format(self.t, self.T))
            self.plot()


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
    def __init__(self, check_unrst=True, check_rft=True, **kwargs):
    #--------------------------------------------------------------------------------
        super().__init__(ext_iface='I{:04d}', ext_OK='OK', **kwargs)
        self.check_unrst = check_unrst
        self.check_rft = check_rft
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
        if self.echo:
            print('  Starting Eclipse...', end='', flush=True)
        self.interface_file('all').delete()
        # Need to create all interface files in advance to avoid Eclipse termination
        [self.interface_file(i).create_empty() for i in range(1, self.N+1)] 
        self.OK_file().delete()
        super().start()  # eclipse.start()
        self.wait_for( self.unrst.exists, error=self.unrst.name+' not created')
        self.wait_for( self.rft.exists, error=self.rft.name+' not created')
        self.check_UNRST_file()
        self.nwell = self.unrst_check.var('nwell')
        rft_wells = self.check_RFT_file(nwell_max=2*self.nwell, nwell_min=self.nwell, limit=100)
        self.suspend()
        if self.echo:
            print('\r  Eclipse started, log file is ' + self.get_logfile(), flush=True)
            print('  ' + self.timer.info) if self.timer else None
        self.rft_size = None
        # only check RFT-file by size if all wells are initially written to the RFT-file 
        if rft_wells == 2*self.nwell:
            # get size of RFT file
            self.rft_size = int(0.5*self.rft.stat().st_size)
            if 2*self.rft_size != self.rft.stat().st_size:
                self.print2log('\nWARNING! Initial size of RFT size not even!\n')

    #--------------------------------------------------------------------------------
    def run_one_step(self, n, satnum_file):                            # ecl_backward
    #--------------------------------------------------------------------------------
        self.rft_start_size = self.rft.stat().st_size
        ### run Eclipse
        self.n = n
        self.interface_file(n).copy(satnum_file, delete=True)
        self.OK_file().create_empty()
        self.resume(check=True)
        self.wait_for( self.OK_file().is_deleted, error=self.OK_file().name()+' not deleted')
        if self.check_unrst:
            self.check_UNRST_file()
        if self.check_rft:
            if self.rft_size:
                self.wait_for( self.check_RFT_size )
            else:
                self.check_RFT_file(nwell_max=self.nwell, nwell_min=1, limit=100)
        self.suspend()

    #--------------------------------------------------------------------------------
    def check_UNRST_file(self):                                             # eclipse
    #--------------------------------------------------------------------------------
        self.wait_for( self.unrst_check.blocks_complete, nblocks=1, log=self.unrst_check.info,
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
                raise SystemError('ERROR Check of ' + self.rft_check.file.name() + ' failed! No new TIME-blocks written to file')
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

    # #--------------------------------------------------------------------------------
    # def status_message(self):
    # #--------------------------------------------------------------------------------
    #     message = ''
    #     if self.check_unrst:
    #         unrst = self.unrst_check.start_values()
    #         message += ':  {} : {}'.format(unrst['start'].rstrip(), unrst['values'][0])
    #     if self.check_rft:
    #         rft = self.rft_check.start_values()
    #         message += ', {} : {:.2f}'.format(rft['start'].rstrip(), rft['values'][0]) 
    #     return message


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
        abs_root = root
        cwd = Path().cwd()
        if str(cwd) in root:
            root = Path(root).relative_to(cwd)
        cmd = [exe, '-root_name='+str(root)] + args.split()
        super().__init__(name='IORSim', case=root, exe=exe, cmd=cmd, **kwargs)
        self.trcconc = None
        self.funrst = Path(abs_root+'_IORSim_PLOT.FUNRST')


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
    def __init__(self, dt=1, **kwargs):
    #--------------------------------------------------------------------------------
        # Call iorsim.__init__()
        super().__init__(args='-readdata', ext_iface='IORSimI{:04d}', ext_OK='IORSimOK', **kwargs)
        self.dt = dt   # Timestep in the first satnum-file prepared for the first Eclipse step (do we need this?)
        self.satnum = Path('satnum.dat')   # Output-file from IORSim, read by Eclipse as an interface-file
        self.satnum_check = check_endtag(file=self.satnum, endtag='-- IORSimX done.')  # Check if satnum-file is flushed

    #--------------------------------------------------------------------------------
    def delete_output_files(self):                                     # ior_backward
    #--------------------------------------------------------------------------------
        super().delete_output_files()
        silentdelete(self.satnum)

    #--------------------------------------------------------------------------------
    def start(self):                                                   # ior_backward
    #--------------------------------------------------------------------------------
        # Start IORSim backward run
        if self.echo:
            print('\n  Starting IORSim...', end='', flush=True)
        self.interface_file('all').delete()
        self.interface_file(1).create_empty()
        self.OK_file().create_empty()
        super().start() # iorsim.start()   
        self.wait_for( self.OK_file().is_deleted, error=self.OK_file().name()+' not deleted')
        self.suspend()
        self.satnum.write_text('\nTSTEP\n' + str(self.dt) + '  / \n')
        if self.echo:
            print('\r  IORSim started, log file is ' + self.get_logfile(), flush=True)
            print('  ' + self.timer.info) if self.timer else None
    
    #--------------------------------------------------------------------------------
    def run_one_step(self, n):                                         # ior_backward
    #--------------------------------------------------------------------------------
        ### run IORSim
        self.n = n
        self.interface_file(n).create_empty()
        self.OK_file().create_empty()
        self.resume(check=True)
        self.wait_for( self.OK_file().is_deleted, error=self.OK_file().name()+' not deleted')
        self.wait_for( self.satnum_check.find_endtag, error=self.satnum_check.file().name+' has no endtag')
        warn_empty_file(self.satnum, comment='--')
        self.suspend()


    # #--------------------------------------------------------------------------------
    # def set_satnum_tstep(self, tstep):                                 # ior_backward
    # #--------------------------------------------------------------------------------
    #     data = []
    #     with open(self.satnum) as file:
    #         for line in file:
    #             if line.strip().startswith('TSTEP'):
    #                 next(file)
    #                 line = 'TSTEP\n' + str(tstep) + '  / \n'
    #                 print(line)
    #             data.append(line)
    #     with open(self.satnum, 'w') as file:
    #         file.writelines(data)

    #--------------------------------------------------------------------------------
    def quit(self):                                                    # ior_backward
    #--------------------------------------------------------------------------------
        self.interface_file(self.n+1).append('Quit')
        self.OK_file().create_empty()
        super().quit()
        

#====================================================================================
class simulation:
#====================================================================================
    #--------------------------------------------------------------------------------
    def __init__(self, mode, root=None, pause=0.5, runs=None, iorexe=None, eclexe=None, to_screen=False, echo=False, 
                 status=lambda *x:None, progress=lambda *x:None, plot=lambda:None, **kwargs):
    #--------------------------------------------------------------------------------
        self.name = 'ior2ecl'
        # Functions 
        self.status = status
        self.progress = progress
        self.plot = plot
        #
        self.pause = pause
        #if echo:
        #    print('  Welcome to ' + self.name + '!' )
        #    print('  This script runs IORSim together with Eclipse')
        self.runlog = None
        if not to_screen:
            self.runlog = safeopen(Path(root).parent/(self.name+'.log'), 'w')
            if echo:
                print('  Log-file is ' + self.runlog.name )
            atexit.register(self.runlog.close)
        self.print2log = lambda txt: print(txt, file=self.runlog, flush=True)
        self.current_run = None
        self.runs = []
        self.ior = self.ecl = None
        kwargs['root'] = root
        kwargs['runlog'] = self.runlog
        if mode=='backward':
            self.run_sim = self.backward
            self.runs = [ecl_backward(exe=eclexe, **kwargs), ior_backward(exe=iorexe, **kwargs)]
            self.ecl, self.ior = self.runs
        if mode=='forward':
            self.run_sim = self.forward
            for name in runs:
                if name=='eclipse':
                    self.ecl = ecl_forward(exe=eclexe, **kwargs)
                if name=='iorsim':
                    self.ior = ior_forward(exe=iorexe, **kwargs)
            self.runs = [run for run in (self.ecl, self.ior) if run]


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
        for run in self.runs:
            self.current_run = run.name.lower()
            run.check_input()
            run.delete_output_files()
            self.status('Starting ' + run.name)
            self.progress(0)
            run.start()
            self.status(run.name + ' running')
            run.init_control_func(status=self.status, progress=self.progress, plot=self.plot)
            run.wait_for_process_to_finish_2(pause=0.2, loop_func=run.control_func)
            t = run.time_and_step()[0]
            if t < run.T:
                raise SystemError('ERROR ' + run.name + ' stopped unexpectedly, check the log')
            run_time += run.run_time()
        return run.complete_msg(run_time=run_time)


    #-----------------------------------------------------------------------
    def backward(self): 
    #-----------------------------------------------------------------------
        #print('Backward run')
        for run in self.runs:
            run.check_input()
            run.delete_output_files()
            self.status('Starting ' + run.name + '...')
            run.start()
        ecl, ior = self.runs
        # Start timestep loop
        for n in range(1, ecl.N+1):
            self.print2log('\nReport step {}'.format(n))
            ecl.run_one_step(n, ior.satnum)
            # Need a short stop after Eclipse has finished, otherwise IORSim sometimes stops 
            sleep(self.pause)
            # Run IORSim to prepare satnum input for the next Eclipse run
            ior.run_one_step(n+1)
            t = ior.time_and_step()[0]
            #print(n,t,', ecl: ',ecl.n,ecl.N,ecl.t,ecl.T,', ior: ',ior.n,ior.N,ior.t,ior.T)
            self.progress(t)
            self.status('{}/{} days'.format(t, ecl.T))
            self.plot()
            if t > ior.T:
                raise SystemError(ecl.complete_msg())
        # Timestep loop finished
        ecl.quit()
        ior.quit()
        return ecl.complete_msg()


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
        except:  # Catch all other exceptions 
            msg = traceback.format_exc()
        finally:
            # Kill possible remaining processes
            [run.kill_and_clean() for run in self.runs]
            self.print2log('\n======  ' + msg + '  ======')
            self.runlog.close()
            self.current_run = None
            self.progress(0)   # Reset progress time
            self.plot()
            self.status(msg)
            if success:
                N = min([run.time_and_step()[1] for run in self.runs])  # Steps for the convert progress 
                sleep(0.05)  # Need a short break to make sure the progressbar is responsive
                self.convert_restart_file(N=N, case=self.runs[0].case)
            return success, msg


    #-----------------------------------------------------------------------
    def convert_restart_file(self, N=0, case=None):
    #-----------------------------------------------------------------------
        if not self.ior:
            return
        ecl = self.ecl or eclipse(root=case)   
        ior = self.ior or iorsim(root=case)    
        self.progress(-N)
        self.status('Converting restart file...')
        # Convert from formatted to unformatted restart file
        #print(ior.funrst)
        ior_unrst = fmt_file(ior.funrst).convert(rename_duplicate=True, rename_key=('TEMP','TEMP_IOR'),
                                                 progress=self.progress) 
        # Merge unformatted Eclipse and IORSim files
        #print(ecl.unrst)
        if ecl.unrst.is_file() and ior_unrst.is_file():
            backup = Path(str(ecl.case)+'_ECLIPSE.UNRST')
            if backup.is_file():
                # This is a pure IORSim run and backup already exists; restore backup
                shutil.copy(backup, ecl.unrst)
            else:
                # No backup exists; create backup copy
                shutil.copy(ecl.unrst, backup)
        else:
            self.status('Convert finished, unable to merge Eclipse and IORSim files')
            return
        # Define sections in the restart files where merging 
        ecl_sec = Section(ecl.unrst, start_before='SEQNUM', end_before='SEQNUM', skip_sections=(0,))
        ior_sec = Section(ior_unrst, start_after='DOUBHEAD', end_before='SEQNUM')
        self.status('Merging Eclipse and IORSim restart files...')
        #ior = Path(ior_unrst)
        merged_file = Path(str(ior.case)+'_MERGED.UNRST')
        merged_file = unfmt_file(merged_file).create(ecl_sec, ior_sec)
        #print(merged)
        # Rename merged UNRST-file to original restart file
        if not merged_file:
            raise SystemError('WARNING Unable to merge {} and {}'.format(ecl.unrst, ior_unrst))
        if merged_file.is_file():
            merged_file.replace(ecl.unrst)
        self.status('Simulation complete, restart file ready')
        self.progress(N)


        
######################################################################################

if __name__ == '__main__':
    
    main()

    
