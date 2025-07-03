# src/iorsimlib/__init__.py
from iorsim.version import __version__

#from .utils import utils
from .ior2ecl import (Output, IORSim_input, IORSim_output, Simulation, runsim, runsim_with_plot, main, 
                      ECL_ALIVE_LIMIT, IOR_ALIVE_LIMIT, SCHEDULE_SKIP_EMPTY, DEFAULT_LOG_LEVEL, 
                      LOG_LEVEL_MAX, LOG_LEVEL_MIN)

__all__ = ['Output', 'IORSim_input', 'IORSim_output', 'Simulation', 'runsim', 'runsim_with_plot', 'main',
           'ECL_ALIVE_LIMIT', 'IOR_ALIVE_LIMIT', 'SCHEDULE_SKIP_EMPTY', 'DEFAULT_LOG_LEVEL',
           'LOG_LEVEL_MAX', 'LOG_LEVEL_MIN',
           '__version__']
