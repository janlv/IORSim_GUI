#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import atexit
from pathlib import Path
import sys
#from __future__ import print_function
from argparse import ArgumentParser
from shutil import which
from datetime import datetime
from time import sleep

from IORlib.utils import safeopen, Progress, check_endtag, warn_empty_file, loop_until, silentdelete, assert_python_version, exit_without_atexit, delete_files_matching, file_contains
from IORlib.runner import runner
from IORlib.ECL import check_blocks


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
        sim.init_runs()
        
        print()
        sim.start_runs()

        print()
        for t in range(1, sim.nsteps+1):
            if sim.progress:
                sim.progress.print(t) 
            sim.run_one_step(t)
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



#--------------------------------------------------------------------------------
def wait_for(runner, func, *args, log=False, assert_running=True, error=None, raise_error=True,
             limit=100000, sleep_sec=0.01, kill_func=None, **kwargs):
#--------------------------------------------------------------------------------
    runner._print('calling wait_for( ' + func.__qualname__ + ' )...', v=3, end='')
    if assert_running:
        assert_running = runner.assert_running
    try:
        n = loop_until(func, *args, **kwargs, limit=limit, assert_running=assert_running,
                       error=error, sleep_sec=sleep_sec, kill_func=kill_func)
        runner._print(' {:d} loops'.format(n), v=3, tag='')
        if callable(log):
            runner._print(log())
    except SystemError as e:
        if raise_error or str(e).startswith('assert_running'):
            raise e
        else:
            n = str(e).split('>')[-1].split()[0]
            runner._print(n + '  loops', v=3, tag='')            
            return False
    else:
        return True


#====================================================================================
class ior2ecl:
#====================================================================================
    #--------------------------------------------------------------------------------
    def __init__(self, root=None, dt=None, nsteps=None, eclrun='eclrun', iorsim='IORSimX',
                 iorargs='', v=3, timer=False, keep_files=False, to_screen=True, quiet=False,
                 check_unrst=True, check_rft=True, readdata=True):
    #--------------------------------------------------------------------------------
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

        
    #--------------------------------------------------------------------------------
    def init_runs(self):
    #--------------------------------------------------------------------------------
        self.init_eclipse_run()
        self.init_iorsim_run()

        
    #--------------------------------------------------------------------------------
    def check_UNRST_file(self, kill_func=None):
    #--------------------------------------------------------------------------------
        wait_for( self.ecl, self.unrst_check.blocks_complete, nblocks=1, log=self.unrst_check.info,
                  error=self.unrst_check.file.name()+' not complete', kill_func=kill_func)
        
    #--------------------------------------------------------------------------------
    def check_RFT_file(self, nwell_max=0, nwell_min=0, limit=10000, kill_func=None):
    #--------------------------------------------------------------------------------
        ###
        ###  cannot always require nblocks=2*nwell in the initial RFT-check. In some situations
        ###  all wells may not be ready after the TSTEP in the DATA-file. The RFT-check
        ###  starts to look for 2*nwell blocks. If the check fails, the check is repeated
        ###  with nblocks-1, and so on until nblocks==nwell.
        ###
        for nblocks in range(nwell_max, nwell_min-1, -1):
            passed = wait_for( self.ecl, self.rft_check.blocks_complete, nblocks=nblocks, log=self.rft_check.info,
                               error=self.rft_check.file.name()+' not complete', limit=limit, raise_error=False,
                               kill_func=kill_func)
            if passed:
                break
            if nblocks==nwell_min:
                raise SystemError('Check of ' + self.rft_check.file.name() + ' failed! No new TIME-blocks written to file')
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
    def start_eclipse(self, kill_func=None):
    #--------------------------------------------------------------------------------
        ### start Eclipse
        if not self.quiet:
            print('  Starting Eclipse...', end='', flush=True)
        ecl = self.ecl
        ecl.interface_file('all').delete()
        [ecl.interface_file(i).create_empty() for i in range(1, self.nsteps+1)] # create all now to avoid Eclipse termination
        ecl.OK_file().delete()
        self.delete_eclipse_files()
        ecl.start()
        wait_for( ecl, self.unrst.exists, error=self.unrst.name+' not created', kill_func=kill_func )
        wait_for( ecl, self.rft.exists, error=self.rft.name+' not created', kill_func=kill_func )
        self.check_UNRST_file(kill_func=kill_func )
        self.nwell = self.unrst_check.var('nwell')
        rft_wells = self.check_RFT_file(nwell_max=2*self.nwell, nwell_min=self.nwell, limit=100,
                                        kill_func=kill_func )
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
                print('WARNING! Initial size of RFT size not even!')

                
                
    #--------------------------------------------------------------------------------
    def start_iorsim(self, kill_func=None):
    #--------------------------------------------------------------------------------
        ### start IORSim
        if not self.quiet:
            print('\n  Starting IORSim...', end='', flush=True)
        ior = self.ior
        ior.interface_file('all').delete()
        ior.interface_file(1).create_empty()
        #silentdelete(self.satnum)
        ior.OK_file().create_empty()
        #delete_files_matching(self.root+'*.trcconc')
        #delete_files_matching(self.root+'*.trcprd')
        self.delete_iorsim_files()
        ior.start()    
        wait_for( ior, ior.OK_file().is_deleted, error=ior.OK_file().name()+' not deleted', kill_func=kill_func )
        ior.suspend()
        self.satnum.write_text('\nTSTEP\n' + str(self.dt) + '  / \n')
        if not self.quiet:
            print('\r  IORSim started, log file is ' + ior.get_logfile(), flush=True)
            print('  ' + ior.timer.info) if ior.timer else None
    
    
    #--------------------------------------------------------------------------------
    def start_runs(self, kill_func=None):
    #--------------------------------------------------------------------------------
        self.start_eclipse(kill_func=kill_func)
        self.start_iorsim(kill_func=kill_func)
        
        
    #--------------------------------------------------------------------------------
    def loop_all(self):
    #--------------------------------------------------------------------------------
        for t in range(1, self.nsteps+1):
            #self.print2log('\nReport step {}'.format(t))
            if self.progress:
                self.progress.update(t) 
            self.run_one_step(t)

    #--------------------------------------------------------------------------------
    def RFT_is_closed(self):
    #--------------------------------------------------------------------------------
        #open_files = [Path(f.path).name for p in self.ecl.children+[self.ecl.parent] for f in p.open_files()]
        open_files = [f for p in self.ecl.children+[self.ecl.parent] for f in p.open_files()]
        for f in open_files:
            if Path(f.path).name == self.rft.name:
                print(f)

        
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
    def run_one_step(self, t, kill_func=None):
    #--------------------------------------------------------------------------------
        self.print2log('\nReport step {}'.format(t))

        self.rft_start_size = self.rft.stat().st_size
        ### run Eclipse
        ecl = self.ecl
        ecl.interface_file(t).copy(self.satnum, delete=True)
        ecl.OK_file().create_empty()
        ecl.resume(check=True)
        wait_for( ecl, ecl.OK_file().is_deleted, error=ecl.OK_file().name()+' not deleted', kill_func=kill_func )
        if self.check_unrst:
            self.check_UNRST_file(kill_func=kill_func)
        if self.check_rft:
            if self.rft_size:
                wait_for( ecl, self.check_RFT_size, kill_func=kill_func )
            else:
                self.check_RFT_file(nwell_max=self.nwell, nwell_min=1, limit=100, kill_func=kill_func )
        ecl.suspend()
        sleep(0.5)

        ### run IORSim
        ior = self.ior
        ior.interface_file(t+1).create_empty()
        ior.OK_file().create_empty()
        ior.resume(check=True)
        wait_for( ior, ior.OK_file().is_deleted, error=ior.OK_file().name()+' not deleted', kill_func=kill_func )
        wait_for( ior, self.satnum_check.find_endtag, error=self.satnum_check.file().name+' has no endtag', kill_func=kill_func )
        warn_empty_file(self.satnum, comment='--')
        ior.suspend()


    #--------------------------------------------------------------------------------
    def status_message(self, t, unrst=True, rft=True):
    #--------------------------------------------------------------------------------
        message = 'Step {}/{} '.format(t, self.nsteps)
        if unrst:
            unrst = self.unrst_check.start_values()
            message += ':  {} : {}'.format(unrst['start'].rstrip(), unrst['values'][0])
        if rft:
            rft = self.rft_check.start_values()
            message += ', {} : {:.2f}'.format(rft['start'].rstrip(), rft['values'][0]) 
        return message

    
    #--------------------------------------------------------------------------------
    def terminate_eclipse_run(self):
    #--------------------------------------------------------------------------------
        self.print2log('\nTerminating Eclipse')
        ecl = self.ecl
        self.ecl.resume()
        self.ecl.wait_for_process_to_quit()
        self.clean_up(self.ecl)        

        
    #--------------------------------------------------------------------------------
    def terminate_iorsim_run(self):
    #--------------------------------------------------------------------------------
        self.print2log('\nTerminating IORSim')
        self.ior.interface_file(self.nsteps+2).append('Quit')
        self.ior.OK_file().create_empty()
        self.ior.resume()
        self.ior.wait_for_process_to_quit()
        self.clean_up(self.ior)

        
    #--------------------------------------------------------------------------------
    def terminate_runs(self):
    #--------------------------------------------------------------------------------
        self.terminate_eclipse_run()
        self.terminate_iorsim_run()

        
    #--------------------------------------------------------------------------------
    def clean_up(self, run):
    #--------------------------------------------------------------------------------
        ### cleaning up
        if not self.keep_files:
            run.interface_file('all').delete()
        if not self.quiet:
            print()
        

    #--------------------------------------------------------------------------------
    def kill_run(self, run):
    #--------------------------------------------------------------------------------
        self.print2log('\nKilling processes')
        run.kill()
        self.clean_up(run)

    #--------------------------------------------------------------------------------
    def kill_eclipse(self):
    #--------------------------------------------------------------------------------
        self.kill_run(self.ecl)

    #--------------------------------------------------------------------------------
    def kill_iorsim(self):
    #--------------------------------------------------------------------------------
        self.kill_run(self.ior)

    #--------------------------------------------------------------------------------
    def kill_runs(self):
    #--------------------------------------------------------------------------------
        self.kill_run(self.ecl)
        self.kill_run(self.ior)

        
    #--------------------------------------------------------------------------------
    def check_eclipse_input(self):
    #--------------------------------------------------------------------------------
        ### check if executables exist on the system
        exe = self.eclrun
        if which(exe) is None:
            raise SystemError('Executable not found: ' + exe)

        ### check root.DATA exists and if READDATA keyword is present or not
        data = self.root + '.DATA'
        if Path(data).is_file():
            DATA_with_readdata = file_contains(data, text='READDATA', comment='--')
            if DATA_with_readdata and not self.readdata:
                raise SystemError('The current case cannot be used in forward-mode: '+
                                  'the Eclipse DATA-file contains the READDATA keyword.')
            if not DATA_with_readdata and self.readdata:
                # READDATA is missing
                raise SystemError('The current case cannot be used in backward-mode: '+
                                  'the Eclipse DATA-file is missing the READDATA keyword.')

            
    #--------------------------------------------------------------------------------
    def check_iorsim_input(self):
    #--------------------------------------------------------------------------------
        if '.exe' not in self.iorsim and sys.platform == 'win32':
            self.iorsim += '.exe'

        ### check if executables exist on the system
        exe = self.iorsim
        if which(exe) is None:
            raise SystemError('Executable not found: ' + exe)

        ### check root.trcinp exists
        inp_file = self.root + '.trcinp'
        if not Path(inp_file).is_file():
            raise_error('IORSim input file, ' + inp_file + ', is missing!')


    #--------------------------------------------------------------------------------
    def check_input(self):
    #--------------------------------------------------------------------------------
        self.check_eclipse_input()
        self.check_iorsim_input()
            
        
    #--------------------------------------------------------------------------------
    def close_logfiles(self):
    #--------------------------------------------------------------------------------
        if self.runlog:
            self.runlog.close()
        self.ecl.log.close()
        self.ior.log.close()


        
# #====================================================================================
# class run_forward:
# #====================================================================================
#     def __init__(name, case, cmd):
#         self.run = runner(name=name, case=case, cmd=cmd)
    
        
        
######################################################################################

if __name__ == '__main__':
    
    main()

    
