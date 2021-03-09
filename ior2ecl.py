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

from IORlib.utils import safeopen, Progress, check_endtag, warn_empty_file, loop_until_2, silentdelete, assert_python_version, exit_without_atexit, delete_files_matching, file_contains
from IORlib.runner import runner
from IORlib.ECL import check_blocks, unfmt_file #, unfmt_file


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
    def __init__(self, **kwargs):
    #--------------------------------------------------------------------------------
        #print('eclipse.__init__',root,N,kwargs)
        root = kwargs.pop('root', None)
        root = str(root)        
        exe = kwargs.pop('exe', None) or 'eclrun' # Default executable
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
                      [self.case+'.'+ext for ext in ('SMSPEC','UNSMRY','RTELOG','RTEMSG')] )
    
    #--------------------------------------------------------------------------------
    def check_input(self):                                                  # eclipse
    #--------------------------------------------------------------------------------
        super().check_input()
        msg = 'WARNING Unable to start ' + self.name + ': '
        ### check root.DATA exists
        inp_file = str(self.case)+'.DATA'
        if not Path(inp_file).is_file():
            raise SystemError(msg + 'missing input file ' + inp_file)

    #--------------------------------------------------------------------------------
    def start(self):                                                        # eclipse
    #--------------------------------------------------------------------------------
        # check input
        self.check_input()
        # delete output from previous runs
        #self.delete_output_files()
        # start run
        super().start()    


#====================================================================================
class forward_mixin:
#====================================================================================
    #--------------------------------------------------------------------------------
    def init_control_functions(self, status_func=lambda *x:None, update_func=lambda *x:None, pause=0.01, count=5):
    #--------------------------------------------------------------------------------
        self.status_func = status_func
        self.update_func = update_func
        self.loop_count = 0
        self.pause = pause
        self.count = count

    #--------------------------------------------------------------------------------
    def loop_func(self):
    #--------------------------------------------------------------------------------
        #self.assert_running_and_stop_if_canceled()
        self.stop_if_canceled()
        self.loop_count += 1
        #print(self.loop_count)
        if self.loop_count == self.count:
            self.loop_count = 0
            self.t = self.stop_if_limit_reached(limit='time')
            self.update_func(self.t)
            self.status_func('{}/{} days'.format(self.t, self.T))


    #--------------------------------------------------------------------------------
    def execute(self, **kwargs):
    #--------------------------------------------------------------------------------
        self.delete_output_files()
        self.init_control_functions(**kwargs)
        self.status_func('Starting ' + self.name)
        self.update_func(0)
        self.start()
        self.status_func(self.name + ' running')
        self.wait_for_process_to_finish_2(pause=0.2, loop_func=self.loop_func)
        #days = 10
        t = self.time_and_step()[0]
        #print(t, self.T)
        if t < self.T:
            #if 10<run.N:
            raise SystemError('ERROR ' + self.name + ' stopped unexpectedly, check the log')



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
        if file_contains(self.case + '.DATA', text='READDATA', comment='--'):
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
        if not file_contains(self.case + '.DATA', text='READDATA', comment='--'):
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
        #[self.interface_file(i).create_empty() for i in range(1, self.nsteps+1)] 
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
    def __init__(self, args='', **kwargs):
    #--------------------------------------------------------------------------------
        #print('iorsim.__init__: ',kwargs)
        root = kwargs.pop('root', None)
        exe = kwargs.pop('exe', None) or 'IORSimX' # Default executable
        if '.exe' not in exe and sys.platform == 'win32':
            exe += '.exe'
        #print(exe)
        # IORSim only accepts root relative to the current directory
        root = Path(root)
        cwd = Path().cwd()
        if str(cwd) in str(root):
            root = root.relative_to(cwd)
        #print(root)
        cmd = [exe, '-root_name='+str(root)] + args.split()
        #print(kwargs)
        super().__init__(name='IORSim', case=root, exe=exe, cmd=cmd, **kwargs)
        self.trcconc = None

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
        delete_files_matching(case+'*.FUNRST')

    #--------------------------------------------------------------------------------
    def start(self):                                                         # iorsim
    #--------------------------------------------------------------------------------
        # check input
        self.check_input()
        # delete output from previous runs
        #self.delete_output_files()
        # start run
        super().start()    



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
        #self.delete_output_files()
        #self.check_input()
        # start run
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
        #self.interface_file(self.n+2).append('Quit')
        self.interface_file(self.n+1).append('Quit')
        self.OK_file().create_empty()
        super().quit()
        #self.wait_for_process_to_finish_2()
        

#====================================================================================
class simulation:
#====================================================================================
    # #--------------------------------------------------------------------------------
    # def __init__(self, root=None, dt=None, nsteps=None, eclrun='eclrun', iorsim='IORSimX',
    #              iorargs='', v=3, timer=False, keep_files=False, to_screen=False, echo=False,
    #              check_unrst=True, check_rft=True, readdata=True, step_unit='days'):
    # #--------------------------------------------------------------------------------
    #--------------------------------------------------------------------------------
    def __init__(self, iorsim=None, eclrun=None, to_screen=False, **kwargs):
    #--------------------------------------------------------------------------------
        name = 'ior2ecl'
        #self.root = root
        #self.n = 0
        self.kwargs = kwargs
        echo = kwargs.get('echo') or False
        root = kwargs.get('root') or False
        self.N = kwargs.get('N') or 0
        #if echo:
        #    print('  Welcome to ' + self.name + '!' )
        #    print('  This script runs IORSim together with Eclipse')
        ### runlog
        self.runlog = None
        if not to_screen:
            self.runlog = safeopen(Path(root).parent/(name+'.log'), 'w')
            if echo:
                print('  Log-file is ' + self.runlog.name )
            atexit.register(self.runlog.close)
        self.print2log = lambda txt: print(txt, file=self.runlog, flush=True)
        ### progress    
        self.progress = False
        if not to_screen and echo:
            self.progress = Progress(N=self.N)
        #self.dt = dt
        #self.nsteps = nsteps
        #self.readdata = readdata
        #self.v = v
        #self.timer = timer
        #self.keep_files = keep_files
        self.runs = []
        #self.starttime = datetime.now()
        self.run = {'forward' :{'eclipse':ecl_forward,  'iorsim':ior_forward},
                    'backward':{'eclipse':ecl_backward, 'iorsim':ior_backward}}
        self.exe = {'eclipse':eclrun, 'iorsim':iorsim}
        self.current_run = None

    #-----------------------------------------------------------------------
    def cancel(self):
    #-----------------------------------------------------------------------
        for run in self.runs:
            run.canceled = True
            #print(run.name + ' canceled!')

    # #-----------------------------------------------------------------------
    # def init_forward_run(self, name):
    # #-----------------------------------------------------------------------
    #     if name=='eclipse':
    #         return ecl_forward(self.root, exe=self.exe[name], runlog=self.runlog, **self.kwargs)
    #     if name=='iorsim':
    #         return ior_forward(self.root, exe=self.exe[name], runlog=self.runlog, **self.kwargs)

    #-----------------------------------------------------------------------
    #def forward(self, run_names, **kwargs):
    #-----------------------------------------------------------------------


    #-----------------------------------------------------------------------
    def run_forward(self, run_names, **kwargs):
    #-----------------------------------------------------------------------
        msg = ''
        self.runs = []
        result = False
        run = None
        run_time = timedelta()
        #start = datetime.now()
        try:
            for name in run_names:
                run = self.run['forward'][name](exe=self.exe[name], runlog=self.runlog, **self.kwargs)
                self.runs.append(run)
                self.current_run = name
                run.execute(**kwargs)
                run_time += run.run_time()
            msg = run.complete_msg(run_time=run_time)
            result = True
        except (SystemError, ProcessLookupError, NoSuchProcess) as e:
            msg = str(e)
            #print('except:',msg)
            if ('simulation complete') in msg.lower():
                result = True
            else:
                result = False
        finally:
            # Set number of steps for convert_FUNRST progress 
            self.N = min([run.time_and_step()[1] for run in self.runs])  
            #print(self.N)  
            # kill possible remaining processes
            for run in self.runs:
                #print(run.name,': ',run.n)
                #self.N = run.n
                run.kill_and_clean()
            self.print2log('\n======  ' + msg + '  ======')
            self.runlog.close()
            self.current_run = None
            return result, msg
                        
    #-----------------------------------------------------------------------
    def run_backward(self, pause=0.5, status_func=lambda *x:None, update_func=lambda *x:None):
    #-----------------------------------------------------------------------
        msg = ''
        self.runs = []
        start = datetime.now()
        try:
            for name in ('eclipse', 'iorsim'):
                run = self.run['backward'][name](exe=self.exe[name], runlog=self.runlog, **self.kwargs)
                self.runs.append(run)
                status_func('Starting ' + run.name + '...')
                run.delete_output_files()
                run.start()
            ecl, ior = self.runs
            # Start timestep loop
            for n in range(1, ecl.N+1):
                self.print2log('\nReport step {}'.format(n))
                ecl.run_one_step(n, ior.satnum)
                # Need a short stop after Eclipse has finished, otherwise IORSim sometimes stops 
                sleep(pause)
                # Run IORSim to prepare satnum input for the next Eclipse run
                ior.run_one_step(n+1)
                t = ior.time_and_step()[0]
                print(n,t,', ecl: ',ecl.n,ecl.N,ecl.t,ecl.T,', ior: ',ior.n,ior.N,ior.t,ior.T)
                update_func(t)
                status_func('{}/{} days'.format(t, ecl.T))
                if t > ior.T:
                    #raise SystemError(self.complete_msg(start))
                    raise SystemError(ecl.complete_msg())
            # Timestep loop finished
            ecl.quit()
            ior.quit()
            #msg = self.complete_msg(start)
            msg = ecl.complete_msg()
            result = True
        except (SystemError, NoSuchProcess) as e:
            msg = str(e)
            if msg.startswith('INFO Simulation complete'):
                result = True
            else:
                result = False
        finally:
            self.N = min([r.n for r in self.runs])    
            for run in self.runs:
                run.kill_and_clean()
            self.print2log('\n======  ' + msg + ' ======')
            self.runlog.close() 
            return result, msg
                        

    # #--------------------------------------------------------------------------------
    # def complete_msg(self, start):
    # #--------------------------------------------------------------------------------
    #     return 'INFO Simulation complete, run-time was '+ str(datetime.now()-start).split('.')[0]


    # #--------------------------------------------------------------------------------
    # def loop_all(self):
    # #--------------------------------------------------------------------------------
    #     for t in range(1, self.nsteps+1):
    #         #self.print2log('\nReport step {}'.format(t))
    #         if self.progress:
    #             self.progress.update(self.n) 
    #         self.run_one_step()

    # #--------------------------------------------------------------------------------
    # def run_one_step(self, pause=0.5):
    # #--------------------------------------------------------------------------------
    #     self.n += 1
    #     self.print2log('\nReport step {}'.format(self.n))
    #     # Run Eclipse 
    #     self.ecl.run_one_step(self.n, self.ior.satnum)
    #     # Need a short stop after Eclipse has finished, otherwise 
    #     # IORSim sometimes stops 
    #     sleep(pause)
    #     # Run IORSim to prepare satnum input for the next Eclipse run
    #     self.ior.run_one_step(self.n+1)
    #     # Return days from IORSim output
    #     days = self.ior.days()
    #     self.t = days
    #     return days

    # #--------------------------------------------------------------------------------
    # def status_message(self, n):
    # #--------------------------------------------------------------------------------
    #     return 'Step {}/{} '.format(n, self.nsteps) + self.ecl.status_message()



#====================================================================================
class ior2ecl:
#====================================================================================
    #--------------------------------------------------------------------------------
    def __init__(self, root=None, dt=None, nsteps=None, eclrun='eclrun', iorsim='IORSimX',
                 iorargs='', v=3, timer=False, keep_files=False, to_screen=True, quiet=False,
                 check_unrst=True, check_rft=True, readdata=True, step_unit='days'):
    #--------------------------------------------------------------------------------
        self.n = 0
        self.t = 0
        self.step_unit = step_unit
        self.name = 'ior2ecl'
        self.starttime = datetime.now()
        if not quiet:
            print('  Welcome to ' + self.name + '!' )
            print('  This script runs IORSim with back-coupling to Eclipse')
        self.quiet = quiet
        self.root = root
        self.dt = dt
        self.nsteps = nsteps
        self.eclrun = eclrun
        self.iorsim = iorsim
        self.iorargs = iorargs
        self.readdata = readdata
        self.v = v
        self.timer = timer
        self.keep_files = keep_files
        self.casedir = Path(self.root).parent
        self.ecl = None
        self.unrst = None
        self.unrst_check = None
        self.check_unrst = check_unrst
        self.rft = None
        self.rft_check = None
        self.check_rft = check_rft
        self.ior = None
        self.satnum = None
        self.satnum_check = None
        self.print2log = None
        self.runlog = None
        self.is_killed = False
        ### runlog
        if not to_screen:
            self.runlog = safeopen(self.casedir/(self.name+'.log'), 'w')
            if not quiet:
                print('  Log-file is ' + self.runlog.name )
            atexit.register(self.runlog.close)
        self.print2log = lambda txt: print(txt, file=self.runlog, flush=True)
        ### progress    
        self.progress = False
        if not to_screen and not quiet:
            self.progress = Progress(N=self.nsteps)
        # output files for reading days
        self.ecl_days = 0
        self.ior_days = 0
        self.trcconc = None
        self.unsmry = unfmt_file(self.root+'.UNSMRY')

    #--------------------------------------------------------------------------------
    def wait_for(self, runner, func, *args, limit=100000, error='ERROR wait_for()', pause=0.01, 
                raise_error=True, log=None, **kwargs):
    #--------------------------------------------------------------------------------
        def loop_func(n):
            runner.assert_running()
            if self.is_killed:
                steps = str(self.t)
                if self.step_unit != 'days':
                    steps = str(self.n)
                raise SystemError('INFO Run stopped after ' + steps + ' ' + self.step_unit)    
            if n > limit:
                if raise_error:
                    raise SystemError(error)
                else:
                    runner._print(' limit exceeded', v=3, tag='')            
                    return -1

        runner._print('calling wait_for( {}, limit={} )...'.format(func.__qualname__, limit), v=3, end='')
        n = loop_until_2(func, *args, **kwargs, pause=pause, loop_func=loop_func)
        if n<0:
            return False    
        runner._print(str(n) + ' loops', v=3, tag='')
        if callable(log):
            runner._print(log())
        return True


    #--------------------------------------------------------------------------------
    def _runner(self, name=None, case=None, cmd=None, ext_iface=None, ext_OK=None):
    #--------------------------------------------------------------------------------
        return runner(name=name, case=case, cmd=cmd, ext_iface=ext_iface, ext_OK=ext_OK,
                      verbose=self.v, timer=self.timer, runlog=self.runlog)
            
            
    #--------------------------------------------------------------------------------
    def init_eclipse_run(self):
    #--------------------------------------------------------------------------------
        ### init Eclipse
        self.ecl = self._runner(name = 'Eclipse',
                                case = self.root,
                                cmd  = [self.eclrun, 'eclipse', self.root],
                                ext_iface = 'I{:04d}',
                                ext_OK = 'OK')
        self.unrst = Path(self.root+'.UNRST')
        self.unrst_check = check_blocks(self.unrst, start='SEQNUM', end='ENDSOL', var='nwell')
        self.rft = Path(self.root+'.RFT')
        self.rft_check = check_blocks(self.rft, start='TIME', end='CONNXT')
        # delete output from previous runs
        self.delete_eclipse_files()
        # check input
        self.check_eclipse_input()
    
        
    #--------------------------------------------------------------------------------
    def init_iorsim_run(self):
    #--------------------------------------------------------------------------------
        ### init IORSim
        # IORSim only accepts root relative to the current directory
        root = Path(self.root)
        cwd = Path().cwd()
        if str(cwd) in self.root:
            root = root.relative_to(cwd)
        cmd = [self.iorsim, '-root_name={}'.format(root)]
        if self.readdata:
            cmd.append('-readdata')
        self.ior = self._runner(name = 'IORSim',
                                case = self.root,
                                cmd = cmd + list(self.iorargs.split()),
                                ext_iface = 'IORSimI{:04d}',
                                ext_OK = 'IORSimOK')
        self.satnum = Path('satnum.dat')
        self.satnum_check = check_endtag(file=self.satnum, endtag='-- IORSimX done.')
        # delete output from previous runs
        self.delete_iorsim_files()
        # check input
        self.check_iorsim_input()

    #--------------------------------------------------------------------------------
    def init_run(self, run):
    #--------------------------------------------------------------------------------
        run = run.strip().lower()
        if run=='eclipse':
            self.init_eclipse_run()
            return self.ecl
        if run=='iorsim':
            self.init_iorsim_run()
            return self.ior
        raise SystemError("WARNING ior2ecl.init_run() expects 'eclipse' or 'iorsim' but got " + run)


    # #--------------------------------------------------------------------------------
    # def init_runs(self, run='all'):
    # #--------------------------------------------------------------------------------
    #     self.init_eclipse_run()
    #     self.init_iorsim_run()
        
    #--------------------------------------------------------------------------------
    def check_UNRST_file(self):
    #--------------------------------------------------------------------------------
        self.wait_for( self.ecl, self.unrst_check.blocks_complete, nblocks=1, log=self.unrst_check.info,
                       error=self.unrst_check.file.name()+' not complete' )
        
    #--------------------------------------------------------------------------------
    def check_RFT_file(self, nwell_max=0, nwell_min=0, limit=10000):
    #--------------------------------------------------------------------------------
        ###
        ###  cannot always require nblocks=2*nwell in the initial RFT-check. In some situations
        ###  all wells may not be ready after the TSTEP in the DATA-file. The RFT-check
        ###  starts to look for 2*nwell blocks. If the check fails, the check is repeated
        ###  with nblocks-1, and so on until nblocks==nwell.
        ###
        for nblocks in range(nwell_max, nwell_min-1, -1):
            passed = self.wait_for( self.ecl, self.rft_check.blocks_complete, nblocks=nblocks, log=self.rft_check.info,
                                    error=self.rft_check.file.name()+' not complete', limit=limit, raise_error=False)
            if passed:
                break
            if nblocks==nwell_min:
                raise SystemError('ERROR Check of ' + self.rft_check.file.name() + ' failed! No new TIME-blocks written to file')
        return nblocks

    #--------------------------------------------------------------------------------
    def delete_eclipse_files(self):
    #--------------------------------------------------------------------------------
        silentdelete( [self.unrst, self.rft] +
                      [self.root+'.'+ext for ext in ('SMSPEC','UNSMRY','RTELOG','RTEMSG')] )
    
    #--------------------------------------------------------------------------------
    def delete_iorsim_files(self):
    #--------------------------------------------------------------------------------
        silentdelete(self.satnum)
        delete_files_matching(self.root+'*.trcconc')
        delete_files_matching(self.root+'*.trcprd')
        delete_files_matching(self.root+'*.FUNRST')


    #--------------------------------------------------------------------------------
    def start_eclipse(self): 
    #--------------------------------------------------------------------------------
        # start Eclipse
        if not self.quiet:
            print('  Starting Eclipse...', end='', flush=True)
        ecl = self.ecl
        ecl.interface_file('all').delete()
        # Need to create all interface files in advance to avoid Eclipse termination
        [ecl.interface_file(i).create_empty() for i in range(1, self.nsteps+1)] 
        ecl.OK_file().delete()
        ecl.start()
        self.wait_for( ecl, self.unrst.exists, error=self.unrst.name+' not created')
        self.wait_for( ecl, self.rft.exists, error=self.rft.name+' not created')
        self.check_UNRST_file()
        self.nwell = self.unrst_check.var('nwell')
        rft_wells = self.check_RFT_file(nwell_max=2*self.nwell, nwell_min=self.nwell, limit=100)
        ecl.suspend()
        if not self.quiet:
            print('\r  Eclipse started, log file is ' + ecl.get_logfile(), flush=True)
            print('  ' + ecl.timer.info) if ecl.timer else None
        self.rft_size = None
        # only check RFT-file by size if all wells are initially written to the RFT-file 
        if rft_wells == 2*self.nwell:
            # get size of RFT file
            self.rft_size = int(0.5*self.rft.stat().st_size)
            if 2*self.rft_size != self.rft.stat().st_size:
                self.print2log('\nWARNING! Initial size of RFT size not even!\n')

                
    #--------------------------------------------------------------------------------
    def start_iorsim(self): 
    #--------------------------------------------------------------------------------
        # start IORSim
        if not self.quiet:
            print('\n  Starting IORSim...', end='', flush=True)
        ior = self.ior
        ior.interface_file('all').delete()
        ior.interface_file(1).create_empty()
        #silentdelete(self.satnum)
        ior.OK_file().create_empty()
        #delete_files_matching(self.root+'*.trcconc')
        #delete_files_matching(self.root+'*.trcprd')
        #self.delete_iorsim_files()
        ior.start()    
        self.wait_for( ior, ior.OK_file().is_deleted, error=ior.OK_file().name()+' not deleted')
        ior.suspend()
        self.satnum.write_text('\nTSTEP\n' + str(self.dt) + '  / \n')
        if not self.quiet:
            print('\r  IORSim started, log file is ' + ior.get_logfile(), flush=True)
            print('  ' + ior.timer.info) if ior.timer else None
    
    
    # #--------------------------------------------------------------------------------
    # def start_runs(self):
    # #--------------------------------------------------------------------------------
    #     self.start_eclipse()
    #     self.start_iorsim()
        
        
    #--------------------------------------------------------------------------------
    def loop_all(self):
    #--------------------------------------------------------------------------------
        for t in range(1, self.nsteps+1):
            #self.print2log('\nReport step {}'.format(t))
            if self.progress:
                self.progress.update(self.n) 
            self.run_one_step()

    # #--------------------------------------------------------------------------------
    # def RFT_is_closed(self):
    # #--------------------------------------------------------------------------------
    #     #open_files = [Path(f.path).name for p in self.ecl.children+[self.ecl.parent] for f in p.open_files()]
    #     open_files = [f for p in self.ecl.children+[self.ecl.parent] for f in p.open_files()]
    #     for f in open_files:
    #         if Path(f.path).name == self.rft.name:
    #             print(f)
        
    #--------------------------------------------------------------------------------
    def check_RFT_size(self):
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
    def run_one_step(self, pause=0.5):
    #--------------------------------------------------------------------------------
        self.n += 1
        self.print2log('\nReport step {}'.format(self.n))

        self.rft_start_size = self.rft.stat().st_size
        ### run Eclipse
        ecl = self.ecl
        ecl.interface_file(self.n).copy(self.satnum, delete=True)
        ecl.OK_file().create_empty()
        ecl.resume(check=True)
        self.wait_for( ecl, ecl.OK_file().is_deleted, error=ecl.OK_file().name()+' not deleted')
        if self.check_unrst:
            self.check_UNRST_file()
        if self.check_rft:
            if self.rft_size:
                self.wait_for( ecl, self.check_RFT_size)
            else:
                self.check_RFT_file(nwell_max=self.nwell, nwell_min=1, limit=100)
        ecl.suspend()
        sleep(pause)

        ### run IORSim
        ior = self.ior
        ior.interface_file(self.n+1).create_empty()
        ior.OK_file().create_empty()
        ior.resume(check=True)
        self.wait_for( ior, ior.OK_file().is_deleted, error=ior.OK_file().name()+' not deleted')
        self.wait_for( ior, self.satnum_check.find_endtag, error=self.satnum_check.file().name+' has no endtag')
        warn_empty_file(self.satnum, comment='--')
        ior.suspend()

        ### Get days from IORSim output
        days = self.iorsim_days()
        self.t = days
        return days
        # root = Path(self.root) 
        # for outfile in root.parent.glob(root.stem+'*.trcconc'):
        #     with open(outfile) as out:
        #         for line in out:
        #             pass
        #     break
        # days = line.strip().split()[0]
        # self.t = days
        # return int(days)

    #--------------------------------------------------------------------------------
    def days(self, run):
    #--------------------------------------------------------------------------------
        run = run.strip().lower()
        if run=='eclipse':
            return self.eclipse_days()
        if run=='iorsim':
            return self.iorsim_days()

    #--------------------------------------------------------------------------------
    def eclipse_days(self):
    #--------------------------------------------------------------------------------
        for block in self.unsmry.blocks(only_new=True):
            if block.key()=='PARAMS':
                self.ecl_days = block.data()[0]
        return int(self.ecl_days)        


    #--------------------------------------------------------------------------------
    def iorsim_days(self):
    #--------------------------------------------------------------------------------
        # Output file for reading days
        if not self.trcconc:
            root = Path(self.root)
            for outfile in root.parent.glob(root.stem+'*.trcconc'):
                if outfile.is_file():
                    self.trcconc = outfile
                break
        ### Get days from IORSim output
        with open(self.trcconc) as out:
           for line in out:
               pass
        line = line.strip()
        #print('line: |'+line+'|')
        if line:
            self.ior_days = float(line.split()[0])
        #print('iorsim_days: '+str(self.ior_days))
        return int(self.ior_days)


    #--------------------------------------------------------------------------------
    def set_satnum_tstep(self, tstep):
    #--------------------------------------------------------------------------------
        data = []
        with open(self.satnum) as file:
            for line in file:
                if line.strip().startswith('TSTEP'):
                    next(file)
                    line = 'TSTEP\n' + str(tstep) + '  / \n'
                    #print(line)
                data.append(line)
        with open(self.satnum, 'w') as file:
            file.writelines(data)


    #--------------------------------------------------------------------------------
    def status_message(self, n):
    #--------------------------------------------------------------------------------
        message = 'Step {}/{} '.format(n, self.nsteps)
        if self.check_unrst:
            unrst = self.unrst_check.start_values()
            message += ':  {} : {}'.format(unrst['start'].rstrip(), unrst['values'][0])
        if self.check_rft:
            rft = self.rft_check.start_values()
            message += ', {} : {:.2f}'.format(rft['start'].rstrip(), rft['values'][0]) 
        return message


    #--------------------------------------------------------------------------------
    def terminate(self, run):
    #--------------------------------------------------------------------------------
        self.print2log('\nTerminating ' + run.name)
        run.resume()
        #print(run.name + ' resumed')
        try:
            run.wait_for_process_to_finish_2(pause=0.01, limit=1000, error='RUNNING')
        except SystemError as e:
            if str(e).startswith('RUNNING'):
                self.print2log(run.name + ' did not finish properly and was killed')
                run.kill()
            else:   
                raise e
        self.clean_up(run)        


    #--------------------------------------------------------------------------------
    def terminate_eclipse(self):
    #--------------------------------------------------------------------------------
        self.terminate(self.ecl)
        
    #--------------------------------------------------------------------------------
    def terminate_iorsim(self):
    #--------------------------------------------------------------------------------
        self.ior.interface_file(self.n+2).append('Quit')
        self.ior.OK_file().create_empty()
        self.terminate(self.ior)
        
    #--------------------------------------------------------------------------------
    def terminate_runs(self):
    #--------------------------------------------------------------------------------
        self.terminate_eclipse()
        self.terminate_iorsim()
        
    #--------------------------------------------------------------------------------
    def clean_up(self, run):
    #--------------------------------------------------------------------------------
        ### cleaning up
        if not self.keep_files:
            run.interface_file('all').delete()
        #if not self.quiet:
        #    print()
        
    #--------------------------------------------------------------------------------
    def runs(self):
    #--------------------------------------------------------------------------------
        return (run for run in (self.ecl, self.ior) if run)


    #--------------------------------------------------------------------------------
    def kill_and_clean(self, runs):
    #--------------------------------------------------------------------------------
        if not isinstance(runs, (tuple, list, GeneratorType)):
            runs = (runs,)
        for run in runs:
            run.kill()
            run.log.close()
            self.clean_up(run)


    #--------------------------------------------------------------------------------
    def check_eclipse_input(self):
    #--------------------------------------------------------------------------------
        ### check if executables exist on the system
        exe = self.eclrun
        if which(exe) is None:
            raise SystemError('WARNING Executable not found: ' + exe)

        ### check root.DATA exists and if READDATA keyword is present or not
        data = self.root + '.DATA'
        if Path(data).is_file():
            DATA_with_readdata = file_contains(data, text='READDATA', comment='--')
            if DATA_with_readdata and not self.readdata:
                raise SystemError('WARNING The current case cannot be used in forward-mode: '+
                                  'Eclipse input contains the READDATA keyword.')
            if not DATA_with_readdata and self.readdata:
                # READDATA is missing
                raise SystemError('WARNING The current case cannot be used in backward-mode: '+
                                  'Eclipse input is missing the READDATA keyword.')

            
    #--------------------------------------------------------------------------------
    def check_iorsim_input(self):
    #--------------------------------------------------------------------------------
        def raise_error(msg):
               raise SystemError('WARNING Unable to start IORSim: ' + msg )

        if '.exe' not in self.iorsim and sys.platform == 'win32':
            self.iorsim += '.exe'

        ### check if executables exist on the system
        exe = self.iorsim
        if which(exe) is None:
            raise_error('Missing executable ' + exe)

        ### check root.trcinp exists
        inp_file = self.root + '.trcinp'
        if not Path(inp_file).is_file():
            raise_error('Missing input file ' + inp_file)

        ### check that Eclipse UNRST and RFT files exists
        for ext in ('.UNRST','.RFT'):
            fname = self.root+ext
            if not Path(fname).is_file():
                raise_error('Missing Eclipse file ' + str(Path(fname).name))


        
    #--------------------------------------------------------------------------------
    def close_logfiles(self):
    #--------------------------------------------------------------------------------
        if self.runlog:
            self.runlog.close()
            self.runlog = None
        for run in self.runs():
            run.log.close()

        
        
######################################################################################

if __name__ == '__main__':
    
    main()

    
