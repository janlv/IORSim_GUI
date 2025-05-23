from collections import defaultdict, namedtuple
#from datetime import datetime
from itertools import product, islice
from math import inf
from pathlib import Path
from dataclasses import dataclass
#from time import perf_counter
from time import perf_counter
from typing import Iterable, NamedTuple
from numpy import (array as nparray, abs as npabs, asarray, copyto, count_nonzero, cumsum, 
    empty, float64, int32, ndarray, sum as npsum, diff as npdiff, any as npany,
    zeros, ones, stack, concatenate, moveaxis, prod, where, add as npadd, repeat as nprepeat,
    arange, ones_like, bincount, min as npmin, max as npmax)
from numba import njit, prange, boolean
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from .ECL import UNRST_file, RFT_file, INIT_file, unfmt_block
from .utils import batched, flatten, roll_xyz, neighbour_connections


ENDSOL = unfmt_block.from_data('ENDSOL', [], 'mess')

#---------------------------------------------------------------------------------
def unrstfile(root):
#---------------------------------------------------------------------------------
    """
    Returns the path to the output file in the given root directory.
    """
    path = Path(root).with_name(Path(root).stem + '_tracer.UNRST')
    return UNRST_file(path)

#---------------------------------------------------------------------------------
def little_endian(arr):
#---------------------------------------------------------------------------------
    """
    Convert a numpy array to little-endian format.
    """
    #start = perf_counter()
    # If array is already in native byte order, return it unchanged
    if arr.dtype.isnative:
        return arr
    # Otherwise swap bytes and reinterpret the data in one step
    native_arr = arr.dtype.newbyteorder('=')
    return arr.byteswap().view(native_arr)
    #print(f'byteswap: {perf_counter() - start:.3e} s')
    #return arr_be


# m = c * Vp, m = mass, c = conc, Vp = vol_phase
# Implicit integration (without chemical reactions):
#         c^n * Vp^n + dt * Qin^n * Cin^n+1
# c^n+1 = ---------------------------------      (Eq.1)
#             Vp^n+1 + dt * Qout^n+1
# Volume balance:
#   Vp^n+1 - Vp^n = dt * (Qin^n - Qout^n)
# Forced volume balance:
#   Vp^n+1 + dt * Qout^n =                       (Eq.2)
# (Eq.2) in (Eq.1) yields
#         c^n * Vp^n + dt * Qin^n * Cin^n+1
# c^n+1 = ---------------------------------      (Eq.3)
#                Vp^n + dt * Qin^n

@njit(parallel=True)
#--------------------------------------------------------------------------------
def implicit_numba(sorted_blocks, nb_conn, rate_in, rate_out, dt, sat, sat_old,
                        rporv, rporv_old, conc, inj_conc, force_vol_balance):
#--------------------------------------------------------------------------------
    # Equations are taken from: https://github.com/ahiorth/GeoChem/blob/main/doc/report.pdf
    num_concs = conc.shape[-1]
    nb = sorted_blocks.shape[0]
    tot_mass_res = zeros(num_concs)
    tot_mass_inj = zeros(num_concs)
    tot_mass_prod = zeros(num_concs)
    # mass = empty_like(conc)
    mass = zeros(num_concs)
    for idx in prange(nb):
        i, j, k = sorted_blocks[idx]
        vol_old = sat_old[i, j, k] * rporv_old[i, j, k]
        vol_new = sat[i, j, k] * rporv[i, j, k]
        # mass[i, j, k, :] = conc[i, j, k, :] * vol_old + dt * inj_conc[i, j, k, :] * rate_in[i, j, k, 6]
        # Nominator of eq. 95 (page 29) where C^n+1_inflow only includes well-injection 
        mass = conc[i, j, k, :] * vol_old + dt * inj_conc[i, j, k, :] * rate_in[i, j, k, 6]
        if force_vol_balance:
            # Denominator of eq. 101 (page 29)
            denom = vol_old + dt * npsum(rate_in[i, j, k, :])
        else:
            # Denominator of eq. 95 (page 29)
            denom = vol_new + dt * npsum(rate_out[i, j, k, :])
        # Inflow mass from blocks
        inflow = zeros(num_concs)
        for m in range(6):
            r = rate_in[i, j, k, m]
            if r > 0.0:
                ii, jj, kk = nb_conn[i, j, k, m]
                inflow += r * conc[ii, jj, kk, :]
        # Oppdater mass og conc
        # mass[i, j, k, :] += dt * inflow
        # conc[i, j, k, :] = mass[i, j, k, :] / denom
        mass += dt * inflow
        conc[i, j, k, :] = mass / denom
        # Sum mass for all blocks
        tot_mass_res += conc[i, j, k, :] * vol_new
        tot_mass_inj += dt * rate_in[i, j, k, 6] * inj_conc[i, j, k, :]
        tot_mass_prod += dt * rate_out[i, j, k, 6] * conc[i, j, k, :]
    return conc, (tot_mass_res, tot_mass_inj, tot_mass_prod)

@njit(parallel=True)
#--------------------------------------------------------------------------------
def implicit_numba_nnc(sorted_blocks, nb_conn, rate_in, rate_out, nnc_rate_in,
                       nnc_rate_out, nnc_indices, nnc_pos, nnc_blocks, dt, 
                       sat, sat_old, rporv, rporv_old, conc, inj_conc, 
                       force_vol_balance):
#--------------------------------------------------------------------------------
    # Equations are taken from: https://github.com/ahiorth/GeoChem/blob/main/doc/report.pdf
    num_concs = conc.shape[-1]
    nb = sorted_blocks.shape[0]
    tot_mass_res = zeros(num_concs)
    tot_mass_inj = zeros(num_concs)
    tot_mass_prod = zeros(num_concs)
    #mass = empty_like(conc)
    #mass = zeros(num_concs)
    for idx in prange(nb):
        i, j, k = sorted_blocks[idx]
        vol_old = sat_old[i, j, k] * rporv_old[i, j, k]
        vol_new = sat[i, j, k] * rporv[i, j, k]
        #mass[i, j, k, :] = conc[i, j, k, :] * vol_old + dt * inj_conc[i, j, k, :] * rate_in[i, j, k, 6]
        #mass = conc[i, j, k, :] * vol_old + dt * inj_conc[i, j, k, :] * rate_in[i, j, k, 6]
        # Inflow from injection wells
        C_inflow = inj_conc[i, j, k, :] * rate_in[i, j, k, 6]
        ncc_ind = nnc_indices[:, i, j, k]
        # Entries in nnc_blocks and nnc_rate where: 
        #  [i,j,k] is nnc1
        nnc_pos1 = nnc_pos[0, ncc_ind[0, 0]: ncc_ind[0, 1]]
        #  [i,j,k] is nnc2
        nnc_pos2 = nnc_pos[1, ncc_ind[1, 0]: ncc_ind[1, 1]]
        if force_vol_balance:
            # Sum inflow rates into [i,j,k], including NNC
            sum_rate_in = 0
            for m in range(7): # 6 neighbours + well-injection
                sum_rate_in += rate_in[i, j, k, m]
            for m in nnc_pos1: # nnc1 <-- nnc2
                sum_rate_in += nnc_rate_in[m]
            for m in nnc_pos2: # nnc2 <-- nnc1 == nnc1 --> nnc2
                sum_rate_in += nnc_rate_out[m]
            # Denominator of eq. 101 (page 29) in the report.pdf
            denom = vol_old + dt * sum_rate_in
        else:
            # Sum outflow rates from [i,j,k], including NNC
            sum_rate_out = 0
            for m in range(7): # 6 neighbours + well-injection
                sum_rate_out += rate_out[i, j, k, m]
            for m in nnc_pos1: # nnc1 --> nnc2
                sum_rate_out += nnc_rate_out[m]
            for m in nnc_pos2: # nnc2 --> nnc1 == nnc1 <-- nnc2
                sum_rate_out += nnc_rate_in[m]
            # Denominator of eq. 95 (page 29) in the report.pdf
            denom = vol_new + dt * sum_rate_out
        #inflow = zeros(num_concs)
        # Inflow from blocks
        for m in range(6):
            r = rate_in[i, j, k, m]
            if r > 0.0:
                ii, jj, kk = nb_conn[i, j, k, m]
                C_inflow += r * conc[ii, jj, kk, :]
                # for nn in prange(idx, nb):
                #     xi, xj, xk = sorted_blocks[nn]
                #     if ii == xi and jj == xj and kk == xk:
                #         print('WARNING1: Block not visited:', idx, nn, (i,j,k), (ii, jj, kk))
        # Inflow from NNC
        for m in nnc_pos1: # nnc1 <-- nnc2
            r = nnc_rate_in[m]
            if r > 0.0:
                ii, jj, kk = nnc_blocks[1, m]
                C_inflow += r * conc[ii, jj, kk, :]
        for m in nnc_pos2: # nnc2 <-- nnc1 == nnc1 --> nnc2
            r = nnc_rate_out[m]
            if r > 0.0:
                ii, jj, kk = nnc_blocks[0, m]
                C_inflow += r * conc[ii, jj, kk, :]
        # Update mass and conc
        # mass[i, j, k, :] += dt * inflow
        # conc[i, j, k, :] = mass[i, j, k, :] / denom
        #mass += dt * inflow
        #conc[i, j, k, :] = mass / denom
        # Equation 95 (page 29) in the report.pdf
        conc[i, j, k, :] = ( conc[i, j, k, :] * vol_old + dt * C_inflow ) / denom       
        # Sum mass for all blocks
        tot_mass_res += conc[i, j, k, :] * vol_new
        tot_mass_inj += dt * rate_in[i, j, k, 6] * inj_conc[i, j, k, :]
        tot_mass_prod += dt * rate_out[i, j, k, 6] * conc[i, j, k, :]
    return conc, (tot_mass_res, tot_mass_inj, tot_mass_prod)

@dataclass
#====================================================================================
class Tracer:                                                          # Tracer
#====================================================================================
    step: int = -1
    time: float = 0.0
    dt: float = 0.0
    names: tuple = None
    conc: ndarray = None
    tot_mass: ndarray = None
    #data: tuple = None
    cycles: int = -1
    header_pos: tuple = None
    host_unrst: UNRST_file = None
    #prod_mass: dict = None

    #--------------------------------------------------------------------------------
    def __post_init__(self):
    #--------------------------------------------------------------------------------
        self.conc = dict(zip(self.names, moveaxis(self.conc, -1, 0)))
        # tot_mass is a tuple of 3 arrays: (mass_res, mass_inj, mass_prod)
        # where each array is of shape (num_concs,)
        self.tot_mass = dict(zip(self.names, zip(*self.tot_mass)))

    #--------------------------------------------------------------------------------
    def __str__(self):
    #--------------------------------------------------------------------------------
        txt = f'step: {self.step: 4d}, time: {self.time:9.3f} days, dt: {self.dt:6.3f} days'
        if self.cycles > -1:
            txt += f', cycles: {self.cycles}'
        return txt

    #--------------------------------------------------------------------------------
    def save_mass(self, echo=True, filename='tracer_mass.txt'):
    #--------------------------------------------------------------------------------    
        filename = self.host_unrst.with_name(filename)
        mode = 'a'
        txt = ''
        if self.step == 1:
            if echo:
                print(f'Writing mass to {filename}')
            mode = 'w'
            # Write header
            txt = '# step       time           dt        '
            for name in self.names:
                txt += f'{name+'_res':20} {name+'_inj':20} {name+'_prod':20}'
            txt += '\n'
        # Write data
        txt += f'{self.step:6d} {self.time:13.6e} {self.dt:13.6e} '
        for name in self.names:
            txt += ' '.join(f'{v:20.13e}' for v in self.tot_mass[name])
        with open(filename, mode, encoding='utf8') as out:
            out.write(txt + '\n')
    
    #--------------------------------------------------------------------------------
    def limits(self):
    #--------------------------------------------------------------------------------
        min_max = (f'{name}: ({npmin(data):.3f}, {npmax(data):.3f})' for name, data in self.conc.items())
        return ', '.join(min_max)

    #--------------------------------------------------------------------------------
    def to_unrst(self):                                          # Tracer
    #--------------------------------------------------------------------------------
        # Write copy header from host UNRST-file and append tracer data
        unrst = unrstfile(self.host_unrst.path)
        mode = 'ab'
        if not unrst.exists() or unrst.end_step() > self.step:
            mode = 'wb'
        #target = self.host_unrst.with_name(self.host_unrst.stem + '_tracer.UNRST')
        with open(unrst.path, mode) as out:
            with self.host_unrst.mmap() as hostfile:
                if mode == 'wb':
                    print(f'Creating new UNRST-file: {unrst}')
                    # Also copy initial header from the host UNRST file
                    out.write(hostfile[self.header_pos[0][1]])
                    out.write(ENDSOL.as_bytes())
                # Copy header from the host UNRST file
                out.write(hostfile[self.header_pos[self.step][1]])
            # Write tracer data
            for name, data in self.conc.items():
                block = unfmt_block.from_data(name, data.flatten(order='F'), 'float')
                out.write(block.as_bytes())
            out.write(ENDSOL.as_bytes())
        return unrst

#Rate = namedtuple('Rate', 'time rate_in rate_out')
#====================================================================================
class Rate(NamedTuple):
#====================================================================================
    time: float
    rate_in: ndarray
    rate_out: ndarray

#Well = namedtuple('Well', 'time name pos rate')
#====================================================================================
class Well(NamedTuple):
#====================================================================================
    time: float
    name: str
    pos: ndarray
    rate: ndarray


#====================================================================================
class Flow():                                                                  # Flow
#====================================================================================
    """
    The Flow Class provides functionality for handling flow-related data and computations 
    in reservoir simulations. It integrates data from UNRST and RFT files, manages 
    reservoir and well rates, and supports tracer computations.
    Attributes:
        unrst (UNRST_file): Object for handling UNRST file data.
        dim (tuple): Dimensions of the reservoir grid.
        rft (RFT_file): Object for handling RFT file data.
        convert (Convert or bool): Conversion object for surface to reservoir rates.
        phases (tuple): Phases involved in the simulation (e.g., 'wat', 'oil', 'gas').
        dt_accuracy (float): Allowed difference in UNRST and RFT time steps.
        res_block (bool): Indicates if block rates are in reservoir conditions.
        res_well (bool): Indicates if well rates are in reservoir conditions.
        block_keys (list): Keys for block (UNRST) rates.
        well_keys (list): Keys for well (RFT) rates.
        nnc (NNC or bool): Object for handling non-neighboring connections (NNC).
        sort (Sort): Object for sorting blocks based on flow.
        seqnum (int): Sequence number for the current time step.
        time (float): Current simulation time.
        _neigh_connects (None or array): Cached neighbor connections.
    Methods:
        __init__(root, phase='wat', res_block=False, res_well=False):
            Initializes the Flow object with the given root directory and parameters.
        sat_rporv(sync=True):
            Yields saturation and pore volume data for each time step.
        neighbour_connections():
            Returns neighbor connections for the reservoir grid.
        tracer_explicit(name, init, inlet, force_vol_balance=False):
            Performs explicit tracer computations for the reservoir.
        tracer_implicit(name, init, inlet, force_vol_balance=False):
            Performs implicit tracer computations for the reservoir.
        rates():
            Yields rates for interblock and well flows.
        data():
            Yields combined flow data including saturation, pore volume, and rates.
        interblock():
            Yields interblock flow rates from the UNRST file.
        wells(zerobase=True):
            Yields well flow rates from the RFT file.
        _wellrate_from_tubing(tubflows, wellplt, connxt):
            Computes well rates from tubing flow data.
        check_time_sync(*times):
            Checks if the provided times are synchronized within the allowed accuracy.
    """

    #--------------------------------------------------------------------------------
    def __init__(self, root, phase='wat', res_block=False, res_well=False):    # Flow
    #--------------------------------------------------------------------------------
        self.unrst = UNRST_file(root)
        self.dim = self.unrst.dim()
        self.rft = RFT_file(root)
        self.convert = False
        self.phases = (phase,)
        self.dt_accuracy = 1e-3  # Allowed difference in UNRST and RFT time steps
        self.res_block = res_block
        self.res_well = res_well
        if res_block is False or res_well is False:
            # Need to convert from surface rates to reservoir rates
            self.convert = Convert(self.unrst, phase)
            if phase != 'wat':
                # Need both oil and gas rates to convert to reservoir rates
                other, = set(('oil', 'gas')) - set(phase)
                self.phases = (phase, other)
        # Define keys for block (UNRST) and well (RFT) rates
        RO = 'R' if res_block else 'O'
        self.block_keys = [f'FL{RO}{p.upper()}{ijk}+' for p,ijk in product(self.phases, 'IJK')]
        #self.unrst.check_missing_keys(*self.block_keys)
        TUB_RAT = 'TUBL' if res_well else 'RAT'
        #print(self.phases)
        #self.well_keys = [f'CON{owg}{TUB_RAT}' for owg in 'OWG']
        PH = [ph[0].upper() for ph in self.phases]
        self.well_keys = [f'CON{ph}{TUB_RAT}' for ph in PH]
        #print(self.well_keys)
        self.rft.check_missing_keys(*self.well_keys)
        nnc = NNC(root, self.phases)
        self.nnc = nnc if nnc.exists else False
        self.sort = Sort(self.dim, nnc=self.nnc)
        self.seqnum = 0
        self.time = 0
        self._neigh_connects = None

    #--------------------------------------------------------------------------------
    def sat_rporv(self, sync=True, min_sat=None):                                       # Flow
    #--------------------------------------------------------------------------------
        Data = namedtuple('Data', 'time dt rporv rporv_old sat sat_old')
        blockdata = self.unrst.blockdata('DOUBHEAD', 0, 'S'+self.phases[0].upper(), 'RPORV')
        # Initial data
        if sync:
            time_old, *data = next(blockdata, None)
            sat_old, rporv_old = self.unrst.reshape_dim(*data)
        else:
            time_old, sat_old, rporv_old = -1, -1, -1
        for time, *data in blockdata:
            sat, rporv = self.unrst.reshape_dim(*data)
            if min_sat is not None:
                too_low = sat < min_sat
                if nonzero := count_nonzero(too_low):
                    print(f'WARNING: {nonzero} negative {self.phases[0]} saturations')
                    copyto(sat, min_sat, where=too_low)
            yield Data(time, time - time_old, rporv, rporv_old, sat, sat_old)
            time_old, sat_old, rporv_old = time, sat, rporv

    #--------------------------------------------------------------------------------
    def neighbour_connections(self):                                           # Flow
    #--------------------------------------------------------------------------------
        if self._neigh_connects is None:
            self._neigh_connects = neighbour_connections(self.dim)
        return self._neigh_connects


    #--------------------------------------------------------------------------------
    def tracer_explicit(self, name, init, inlet, force_vol_balance=False):      # Flow
    #--------------------------------------------------------------------------------
        # m = c * Vp, m = mass, c = conc, Vp = vol_phase
        # Explicit integration (without chemical reactions):
        #   m^n+1 = c^n * Vp^n + dt * (Qin^n * Cin^n - Qout^n * C^n)
        #   m^n+1 = c^n * (Vp^n - dt * Qout^n) + dt * Qin^n * Cin^n   (Eq.1)
        # Volume balance:
        #   Vp^n+1 - Vp^n = dt * (Qin^n - Qout^n)
        # Forced volume balance:
        #   Vp^n+1 - dt * Qin^n = Vp^n - dt * Qout^n                  (Eq.2)
        # (Eq.2) in (Eq.1) yields
        #   m^n+1 = c^n * (Vp^n+1 - dt * Qin^n) + dt * Qin^n * Cin^n  (Eq.3)
        #   c^n+1 = m^n+1 / Vp^n+1

        # Blocks:
        #   Conc. for the 3 positive (ijk+1) and the 3 negative (ijk-1) block connections are
        #   obtained using roll along x-, y-, and z-axis. Negative roll brings ijk+1 conc. to
        #   the ijk-block, while positive roll brings ijk-1 conc. to the ijk-block.
        # Well:
        #   Injection conc. is connection 7

        #Tracer = namedtuple('Tracer', 'time dt conc prod_mass cfl')
        conc = stack(init, axis=-1)
        inj_conc = stack([i * ones(self.dim) for i in inlet], axis=-1)
        rates = self.rates()
        # if self.nnc_key:
        #     nnc_rates = self.nnc_rates()
        sat_rporv = self.sat_rporv()
        for rate, data in zip(rates, sat_rporv):
            self.check_time_sync(rate.time, data.time)
            vol_phase_old = data.sat_old * data.rporv_old
            vol_phase = data.sat * data.rporv
            if force_vol_balance:
                # First part of (Eq.3)
                mass = conc * (vol_phase - rate.dt * npsum(rate.rate_in, axis=-1))[..., None]
            else:
                # First part of (Eq.1)
                mass = conc * (vol_phase_old - rate.dt * npsum(rate.rate_out, axis=-1))[..., None]
            # Use 'as_scalar' = True in roll_xyz to move all conc. values
            conc_inflow = concatenate((roll_xyz(conc, -1, True), roll_xyz(conc, 1, True), inj_conc[..., None]), axis=-1)
            # Second part of (Eq.1) or (Eq.3)
            mass += rate.dt * npsum((rate.rate_in[..., None, :] * conc_inflow), axis=-1)
            # if self.nnc_key:
            if self.nnc:
                #nnc = next(nnc_rates, None)
                nnc_rate = self.nnc.get_rate(self.seqnum)
                # rate_out is flow into nnc2 from nnc1, get conc from nnc1
                npadd.at(mass, self.nnc.ijk[1], rate.dt * nnc_rate.rate_out[:, None] * conc[self.nnc.ijk[0]])
                #mass[self.nnc_ijk[1]] += rate.dt * nnc.rate_out[:, None] * conc[self.nnc_ijk[0]]
                # rate_in is flow into nnc1 from nnc2, get conc from nnc2
                npadd.at(mass, self.nnc.ijk[0], rate.dt * nnc_rate.rate_in[:, None] * conc[self.nnc.ijk[1]])
                #mass[self.nnc_ijk[0]] += rate.dt * nnc.rate_in[:, None] * conc[self.nnc_ijk[1]]
            conc = mass / vol_phase[..., None]
            prod_mass = rate.dt * rate.rate_out[..., 6][..., None] * conc
            #cfl = rate.dt * npsum(rate.rate_in, axis=-1) / vol_phase
            output = (zip(name, moveaxis(var, -1, 0)) for var in (conc, prod_mass))
            yield Tracer(rate.time, rate.dt, *[dict(out) for out in output])#, cfl)
            vol_phase_old = vol_phase

    #--------------------------------------------------------------------------------
    def tracer_implicit(self, names, init, inlet, force_vol_balance=False):                        # Flow
    #--------------------------------------------------------------------------------
        # File-positions of the sections in the host UNRST-file. Used to obtain the 
        # SEQNUM - STARTSOL header from the host UNRST-file when writing tracer data 
        # to the tracer UNRST-file.
        # Add header position to the tracer data for writing to the UNRST-file
        header_pos = tuple(self.unrst.section_slices(('SEQNUM', 'startpos'), ('STARTSOL', 'endpos')))
        conc = stack(init, axis=-1)
        rates = self.rates()
        sat_rporv = self.sat_rporv()
        inj_conc = stack([i * ones(self.dim) for i in inlet], axis=-1)
        nb_conn = self.neighbour_connections()
        for n, (rate, data) in enumerate(zip(rates, sat_rporv)):
            step = n + 1
            self.check_time_sync(rate.time, data.time)
            self.check_step_sync(step)
            nnc_rate = self.nnc.get_rate(self.seqnum) if self.nnc else None
            sorted_blocks = self.sort.topological(rate.rate_out[..., :6], nnc_rate, check=False, weighted=False)
            if nnc_rate:
                conc, tot_mass = implicit_numba_nnc(
                    sorted_blocks, nb_conn, little_endian(rate.rate_in), little_endian(rate.rate_out),
                    little_endian(nnc_rate.rate_in), little_endian(nnc_rate.rate_out), self.nnc.indices,
                    self.nnc.pos, self.nnc.nnc, float64(rate.dt), little_endian(data.sat), 
                    little_endian(data.sat_old), little_endian(data.rporv), little_endian(data.rporv_old), 
                    conc, inj_conc, force_vol_balance)
            else:
                conc, tot_mass = implicit_numba(
                    sorted_blocks, nb_conn, rate.rate_in, rate.rate_out, float64(rate.dt), data.sat,
                    data.sat_old, data.rporv, data.rporv_old, conc, inj_conc, force_vol_balance)
            
            #prod_mass = rate.dt * rate.rate_out[..., 6][..., None] * conc
            yield Tracer(step, rate.time, rate.dt, names, conc, tot_mass,
                         cycles=self.sort.cycle_id, header_pos=header_pos,
                         host_unrst=self.unrst)


    # #--------------------------------------------------------------------------------
    # def timesynced_zip(self, *iterables):                                      # Flow
    # #--------------------------------------------------------------------------------
    #     for it in zip(*iterables):
    #         self.check_time_sync(*[i.time for i in it])
    #         yield it

    #--------------------------------------------------------------------------------
    def rates(self):                                                           # Flow
    #--------------------------------------------------------------------------------
        Rates = namedtuple('Rates', 'time dt rate_in rate_out')
        block_flow = self.interblock()
        well_flow = self.wells()
        # Initial time
        # Need to skip the inital block of the UNRST-file to sync with the RFT-file
        time = next(block_flow).time
        # Loop over interblock flows and well flows
        for block, wells in zip(block_flow, well_flow):
            # Loop over active wells
            well_rate = {io:zeros(self.dim) for io in ('in', 'out')}
            for well in wells:
                self.check_time_sync(well.time, block.time)
                #kind = 'in' if well.rate[0] < 0 else 'out'
                kind = 'in' if npany(well.rate < 0) else 'out'
                well_rate[kind][well.pos] += npabs(well.rate)
            dt = block.time - time
            # Last index is connection number. Block rates are the 6 first connections
            # (one for each face), and well rates are connection 7.
            rate_in = concatenate((block.rate_in, well_rate['in'][..., None]), axis=-1)
            rate_out = concatenate((block.rate_out, well_rate['out'][..., None]), axis=-1)
            yield Rates(block.time, dt, rate_in, rate_out)
            time = block.time


    #--------------------------------------------------------------------------------
    def data(self):                                                           # Flow
    #--------------------------------------------------------------------------------
        Flowdata = namedtuple('Flowdata', 'time dt rporv rporv_old sat sat_old rate_in rate_out')  # bw bw_old')
        for rate, data in zip(self.rates(), self.sat_rporv()):
            self.check_time_sync(rate.time, data.time)
            rate_in = npsum(rate.rate_in, axis=-1)
            rate_out = npsum(rate.rate_out, axis=-1)
            yield Flowdata(data.time, data.dt, data.rporv, data.rporv_old,
                           data.sat, data.sat_old, rate_in, rate_out)

    #--------------------------------------------------------------------------------
    def combined_rate(self, block:Rate, well:Rate) -> Rate:                             # Flow
    #--------------------------------------------------------------------------------
        # Combine well and block rates
        self.check_time_sync(well.time, block.time)
        rate_in = concatenate((block.rate_in, well.rate_in[..., None]), axis=-1)
        rate_out = concatenate((block.rate_out, well.rate_out[..., None]), axis=-1)
        return Rate(well.time, rate_in=rate_in, rate_out=rate_out)

    #--------------------------------------------------------------------------------
    def wellrate(self, wells: Iterable[Well]) -> Rate:                          # Flow
    #--------------------------------------------------------------------------------
        # wells: iterable of Well namedtuples
        #self.check_time_sync(*[well.time for well in wells])
        rate_in = zeros(self.dim)
        rate_out = zeros(self.dim)
        days = []
        for well in wells:
            days.append(well.time)
            if npany(well.rate < 0):
                # Injection well
                if npany(well.rate > 0):
                    raise ValueError(f'Injection well {well.name} has positive rates: {well.rate}')
                rate_in[well.pos] += npabs(well.rate)
            else:
                # Production well
                rate_out[well.pos] += well.rate
        self.check_time_sync(*days)
        return Rate(time=days[0], rate_in=rate_in, rate_out=rate_out)

    #--------------------------------------------------------------------------------
    def blockrate(self, days, rate:ndarray) -> Rate:                           # Flow
    #--------------------------------------------------------------------------------
        #Blockrate = namedtuple('Blockrate', 'time rate_in rate_out')
        rate_in = -rate * (rate < 0)
        rate_out = rate * (rate > 0)
        rates_in = concatenate((rate_in, roll_xyz(rate_out)), axis=-1)
        rates_out = concatenate((rate_out, roll_xyz(rate_in)), axis=-1)
        return Rate(time=days, rate_in=rates_in, rate_out=rates_out)

    #--------------------------------------------------------------------------------
    def blockrate_from_file(self):                           # Flow
    #--------------------------------------------------------------------------------
        # Read flow rates from UNRST-file
        blockdata = self.unrst.blockdata('SEQNUM', 'DOUBHEAD', 0, *self.block_keys)
        for self.seqnum, self.time, *data in blockdata:
            data = batched(self.unrst.reshape_dim(*data), 3)
            rates = {ph:stack(next(data), axis=-1) for ph in self.phases}
            if not self.res_block:
                rates = self.convert.surf_to_res(rates, self.seqnum)
            else:
                rates = rates[self.phases[0]]
            yield self.blockrate(self.time, rates)

    #--------------------------------------------------------------------------------
    def interblock(self):                           # Flow
    #--------------------------------------------------------------------------------
        # Read flow rates from UNRST-file
        Blockflow = namedtuple('Blockflow', 'time rate_in rate_out')
        phase = self.phases[0]
        blockdata = self.unrst.blockdata('SEQNUM', 'DOUBHEAD', 0, *self.block_keys)
        for self.seqnum, self.time, *data in blockdata:
            data = batched(self.unrst.reshape_dim(*data), 3)
            rates = {ph:stack(next(data), axis=-1) for ph in self.phases}
            if not self.res_block:
                rates = self.convert.surf_to_res(rates, self.seqnum)
            else:
                rates = rates[phase]
            inflow = rates < 0
            outflow = rates > 0
            rate_in = -rates * inflow
            rate_out = rates * outflow
            rates_in = concatenate((rate_in, roll_xyz(rate_out)), axis=-1)
            rates_out = concatenate((rate_out, roll_xyz(rate_in)), axis=-1)
            yield Blockflow(self.time, rates_in, rates_out)


    #--------------------------------------------------------------------------------
    def wells(self):                                             # Flow
    #--------------------------------------------------------------------------------
        #Well = namedtuple('Well', 'time name pos rate')
        #keys = ('TIME', 'CONIPOS', 'CONJPOS', 'CONKPOS', 'WELLPLT', 'WELLETC', 'CONNXT')
        keys = ('TIME', 'CONIPOS', 'CONJPOS', 'CONKPOS', 'WELLETC')
        rates = []
        current_time = next(self.rft.read('time'))
        rftdata = self.rft.blockdata(*keys, *self.well_keys, singleton=True)
        #for time, i, j, k, wellplt, welletc, connxt, *wellrat in rftdata:
        for days, i, j, k, welletc, *wellrat in rftdata:
            days = days[0]
            wellname = welletc[1]
            if days > current_time:
                yield rates
                current_time = days
                rates = []
            if self.res_well:
                raise ValueError('Reservoir well rates not supported')
                # wellrat = self._wellrate_from_tubing(wellrat, wellplt, connxt)
            wellrat = dict(zip(self.phases, wellrat))
            # Subtract 1 to convert from 1-based to 0-based indexing
            pos = (i - 1, j - 1, k - 1)
            if not self.res_well:
                # Convert surface rates to reservoir rates
                wellrat = self.convert.surf_to_res(wellrat, self.seqnum, pos=pos)
            rates.append(Well(days, wellname, pos, wellrat))
        # yield last rates because other yield is only called when time changes
        yield rates

    # #--------------------------------------------------------------------------------
    # def _wellrate_from_tubing(self, tubflows, wellplt, connxt):                # Flow
    # #--------------------------------------------------------------------------------
    #     # Seems that this only works for reservoir rates, and the accuracy is not very good
    #     wellplt = nparray(wellplt)
    #     phase_rates = wellplt[5]*(wellplt[:3]/sum(wellplt[:3]))
    #     connxt = atleast_1d(connxt)
    #     rates = []
    #     for i, tub in enumerate(tubflows):
    #         wellrate = phase_rates[i]
    #         tub = atleast_1d(tub)
    #         injector = wellrate < 0
    #         idx = list(argsort(connxt))
    #         miss = missing_elements(connxt) or [len(connxt)]
    #         is_neigh = ones(len(idx) + len(miss), dtype=np_bool)
    #         for m in miss:
    #             idx.insert(m, (m if m<len(connxt) else 0) if injector else m-1)
    #             is_neigh[m] = 0
    #         #is_neigh[list(miss)] = 0
    #         if injector:
    #             # Injector
    #             diff = zeros(len(tub))
    #             for i in range(len(tub)):
    #                 if is_neigh[i+1]:
    #                     diff[idx[i+1]] = tub[i] - tub[idx[i+1]]
    #             diff[idx[0]] = wellrate - tub[idx[0]]
    #         else:
    #             # Producer
    #             diff = tub - is_neigh[1:]*tub[idx[1:]]
    #         rates.append(diff)
    #     return rates

    #--------------------------------------------------------------------------------
    def check_time_sync(self, *times):                                        # Flow
    #--------------------------------------------------------------------------------
        """
        Validates synchronization of time values.

        Parameters:
        *times: Variable length argument list of time values to check for synchronization.

        Raises:
        ValueError: If time differences exceed the allowed accuracy.
        """
        # if len(times) < 2:
        #     times = list(times) + [self.time]
        if npany(npabs(npdiff(times)) > self.dt_accuracy):
            raise ValueError(f'Time error in Flow: {times}. Expected time differences to be within {self.dt_accuracy}.')

    #--------------------------------------------------------------------------------
    def check_step_sync(self, *steps):                                        # Flow
    #--------------------------------------------------------------------------------
        """
        Validates synchronization of step values.

        Parameters:
        *steps: Variable length argument list of step numbers to validate.

        Raises:
        ValueError: If steps are not synchronized.
        """
        if set(steps) != set([self.seqnum]):
            raise ValueError(f"Step mismatch detected in Flow: expected step {self.seqnum}, but got {steps}")

#================================================================================
class NNC():                                          # NNC
#================================================================================
    """
    Non-neighbouring connections (NNC) are used to connect blocks that are not
    among the 6 neighbouring blocks. The blocks are given as two lists of block-numbers
    (1-based) in the INIT-file, NNC1 and NNC2. The function init.non_neigh_conn() 
    is used to set up the 0-based block indice arrays nnc (2, N, 3) and ijk (2, 3, N). 
    
    The corresponding NNC flow-rates are obtained by the function get_rate(). This 
    function returns rate_in and rate_out, and reads new FLRpppN+ entries from the 
    UNRST-file only if the seqnum is different from the previous one. Positive flow 
    rates (rate_out) means flow from nnc1 to ncc2, and negative flow rates (rate_in) 
    means flow from nnc2 to nnc1. 
    
    Note that a given nnc block may have several connections to other nnc blocks, and it 
    may appear both as a nnc1- and a nnc2-block. Hence, to access all out-rates from a block,
    we need to get all matching nnc1 blocks with positive rates, and all matching nnc2 
    blocks with negative rates. 
    """
    
    nnc_rate = namedtuple('nnc_rate', 'rate_in rate_out')

    #----------------------------------------------------------------------------
    def __init__(self, root, phases):                 # Convert
    #----------------------------------------------------------------------------
        # Non-neighbouring connections
        # NB! Seems that only reservoir NNC rates are available in UNRST,
        # so only one phase is used
        self.unrst = UNRST_file(root)
        self.key = None
        self.exists = False
        self.seqnum = None
        self.blockdata = None
        self.rate_in = None
        self.rate_out = None
        init = INIT_file(root)
        self.nnc = init.non_neigh_conn()
        if self.nnc[0]:
            self.exists = True
            self.nnc = asarray(self.nnc, dtype=int32)
            #print(self.nnc)
            self.ijk = moveaxis(self.nnc, 1, 2)
            dim = self.unrst.dim()
            tmp_map = [Map(nnc, dim) for nnc in self.nnc]
            self.indices = stack([m.keys for m in tmp_map])
            self.pos = stack([m.values for m in tmp_map])
            self.key = f'FLR{phases[0].upper()}N+'
            self.start()

    #----------------------------------------------------------------------------
    def start(self):                                         # NNC
    #----------------------------------------------------------------------------
        self.seqnum = -1
        self.blockdata = self.unrst.blockdata('SEQNUM', self.key, singleton=True)

    #----------------------------------------------------------------------------
    def read_rate(self, step=1):                                         # NNC
    #----------------------------------------------------------------------------
        # Read NNC rates from UNRST-file at reservoir conditions, i.e. FLRpppN+ where
        # ppp is OIL, WAT, or GAS
        # Using islice here to be able to skip steps, typically the initial step
        seqnum, rate = next(islice(self.blockdata, step-1, None))
        self.seqnum = seqnum[0]
        #rate = nparray(data)
        self.rate_in = -rate * (rate < 0)
        self.rate_out = rate * (rate > 0)

    #----------------------------------------------------------------------------
    def get_rate(self, seqnum):                                         # NNC
    #----------------------------------------------------------------------------
        if seqnum < self.seqnum:
            # Start from beginning
            self.start()
        if seqnum > self.seqnum:
            # Step forward
            self.read_rate(seqnum - self.seqnum)
        if seqnum != self.seqnum:
            raise ValueError(f'SEQNUM mismatch in NNC: {seqnum} != {self.seqnum}')
        return self.nnc_rate(self.rate_in, self.rate_out)

    #----------------------------------------------------------------------------
    def out_connections(self, rate:nnc_rate):               # NNC
    #----------------------------------------------------------------------------
        out_conn = defaultdict(list)
        for n1, n2, rin, rout in zip(*self.nnc, rate.rate_in, rate.rate_out):
            if rout > 0:
                out_conn[tuple(n1)].append(tuple(n2))
            if rin > 0:
                out_conn[tuple(n2)].append(tuple(n1))
        return out_conn
    

#================================================================================
class Convert():                                          # Convert
#================================================================================
    #----------------------------------------------------------------------------
    def __init__(self, unrst:UNRST_file, phase):                 # Convert
    #----------------------------------------------------------------------------
        self.unrst = unrst
        self.phase = phase
        self.pvt = None
        self.seqnum = None
        self.keys = ('BW',)
        if phase != 'wat':
            self.keys = ('RS', 'RV', 'BO', 'BG')
        self.blockdata = None #self.unrst.blockdata('SEQNUM', *self.keys)
        self.start()

    #----------------------------------------------------------------------------
    def start(self):                                         # Convert
    #----------------------------------------------------------------------------
        self.seqnum = -1
        self.blockdata = self.unrst.blockdata('SEQNUM', *self.keys)

    #----------------------------------------------------------------------------
    def read_pvt(self, step=1):                                         # Convert
    #----------------------------------------------------------------------------
        # Using islice here to be able to skip steps, typically the initial step
        self.seqnum, *data = next(islice(self.blockdata, step-1, None))
        self.pvt = dict(zip(self.keys, self.unrst.reshape_dim(*data)))
        return self.pvt

    #----------------------------------------------------------------------------
    def surf_to_res(self, rate:dict, seqnum, pos=None):               # Convert
    #----------------------------------------------------------------------------
        # From IORSim code
        # rate.oil   = Bo * (     STC_rate.oil - Rv*STC_rate.gas )/( 1.0 - Rs*Rv );
        # rate.gas   = Bg * ( -Rs*STC_rate.oil +    STC_rate.gas )/( 1.0 - Rs*Rv );
        # rate.water = Bw *       STC_rate.water;

        # Only read new PVT data if seqnum has changed.
        # Why? PVT-data and rate-data are read from the same UNRST-file, but 
        # in separate functions. This is to make sure PVT-data and rate-data
        # are always in sync
        if seqnum < self.seqnum:
            # Start from beginning
            self.start()
        if seqnum > self.seqnum:
            # Step forward
            self.read_pvt(seqnum - self.seqnum)
        if seqnum != self.seqnum:
            raise ValueError(f'SEQNUM mismatch in Convert: {seqnum} != {self.seqnum}')
        pvt = self.pvt
        if rate[self.phase].ndim > 3:
            # Add extra dimension for broadcasting
            pvt = {k:v[..., None] for k,v in self.pvt.items()}
        if pos is not None:
            pvt = {k:v[pos] for k,v in self.pvt.items()}
        if self.phase in ('oil', 'gas'):
            denom = 1 - pvt['RS'] * pvt['RV']
            if npany(denom == 0):
                raise ValueError("Denominator is zero, which will cause a division by zero error.")
            if self.phase == 'oil':
                resrate = pvt['BO'] * (              rate['oil'] - pvt['RV'] * rate['gas']) / denom
            else:
                resrate = pvt['BG'] * ( -pvt['RS'] * rate['oil'] +             rate['gas']) / denom
        else:
            resrate = pvt['BW'] * rate['wat']
        return resrate


#================================================================================
class Sort:
#================================================================================
    
    #----------------------------------------------------------------------------
    def __init__(self, shape, nnc=None):
    #----------------------------------------------------------------------------
        """
        in_degrees:      3D-array shape (X, Y, Z)
        out_connections: 5D-array shape (X, Y, Z, 6, 3), hvor:
                         out_connections[x,y,z,d] = (nx,ny,nz) eller (-1,-1,-1)
                         Retninger d=0..5 tilsvarer (+x,+y,+z,-x,-y,-z)
        """
        self.shape = shape
        self.N = prod(shape)
        self.D = 6
        self.connections = neighbour_connections(shape).reshape(self.N * self.D, 3)
        self.nnc_flat_conn = None
        if nnc:
            self.nnc_flat_conn = nparray([self.to_flat(n) for n in nnc.nnc], dtype=int32)
            #print(self.nnc_flat_conn.shape)
        self.u = nprepeat(arange(self.N, dtype=int32), self.D)               # (N*6,)
        self.adj = None
        self.cycle_id = None

    #----------------------------------------------------------------------------
    def topological(self, rate_out, nnc_rate=None, check=False, echo=False,
                    weighted=False):
    #----------------------------------------------------------------------------
        """
        Breaks cycles, then manually runs Kahn's algorithm
        on our CSR matrix to determine a flow order.
        """
        self.adj = self.adjacency_matrix(rate_out, nnc_rate, weighted=weighted)
        if self.has_cycle():
            if weighted:
                self.remove_cycles_weighted(echo=echo)
            else:
                self.remove_cycles(echo=echo)            

        # Create indptr and indices for kahn_numba
        indptr  = self.adj.indptr    # length N+1
        indices = self.adj.indices   # length = number of edges

        # Init in-degree as a 1D-array

        if weighted:
            in_deg = self.adj.getnnz(axis=0).astype(int32)
        else:
            in_deg = nparray(self.adj.sum(axis=0), dtype=int32).ravel()  # shape (N,)
        
        # Run Kahn's algoritm
        order, idx = kahn_numba(indptr, indices, in_deg)
        
        # Check for possible cycles
        if idx < self.N:
            print(f'Warning: Cycles remain — topological sorting incomplete: {idx} < {self.N}')
            order = order[:idx]

        if check:
            self.check_order(order)
        return self.to_coords(order)

    #----------------------------------------------------------------------------
    def check_order(self, order):                                          # Sort
    #----------------------------------------------------------------------------
        pos = empty(self.N, dtype=int32)
        for idx, node in enumerate(order):
            pos[node] = idx
        u_indices = self.adj.indptr[:-1]
        v_indices = self.adj.indices
        errors = check_order_numba(u_indices, v_indices, pos, self.N)
        if errors.size > 0:
            print('ERRORS in topological sorting: source calculated after target')
            u_c = self.to_coords(errors[:, 0])
            v_c = self.to_coords(errors[:, 1])
            for u, v in zip(u_c, v_c):
                print(f' {u} -> {v}  ')
        else:
            print('Topological sorting is valid!')

    #----------------------------------------------------------------------------
    def adjacency_matrix(self, rate_out, nnc_rate, weighted=False):        # Sort
    #----------------------------------------------------------------------------
        # --- vanlige naboer -----------------------------------
        # flat ut rate_out og finn gyldige nabo-indekser
        # u is source, v is target: u -> v
        if weighted:
            flat_rate_out = rate_out.reshape(self.N * self.D)
            valid_nb = flat_rate_out > 0     
        else:
            valid_nb = rate_out.reshape(self.N * self.D) > 0        # bool-vektor, lengde N*6
        u_nb = self.u[valid_nb]                                # kilde indekser
        v_nb = self.to_flat(self.connections[valid_nb])        # flate destinasjons indekser
        if weighted:
            data_nb = flat_rate_out[valid_nb]
        else:
            data_nb = ones_like(u_nb)

        # --- non-neighboring connections ----------------------
        if nnc_rate:
            #print('NNC-rates found!')
            # rate_out er flow inn i nnc2 fra nnc1, legg til kobling nnc1 -> nnc2
            valid_nnc1 = nnc_rate.rate_out > 0
            u_nnc1 = self.nnc_flat_conn[0][valid_nnc1]
            v_nnc1 = self.nnc_flat_conn[1][valid_nnc1]
            if weighted:
                data_nnc1 = nnc_rate.rate_out[valid_nnc1]
            else:
                data_nnc1 = ones_like(u_nnc1)
            # rate_in er flow inn i nnc1 fra nnc2, legg til kobling nnc2 -> nnc1
            valid_nnc2 = nnc_rate.rate_in > 0
            u_nnc2 = self.nnc_flat_conn[1][valid_nnc2]
            v_nnc2 = self.nnc_flat_conn[0][valid_nnc2]
            if weighted:
                data_nnc2 = nnc_rate.rate_in[valid_nnc2]
            else:
                data_nnc2 = ones_like(u_nnc2)
            # --- samle alt og bygg én sparse matrise --------------
            u_nb = concatenate([u_nb, u_nnc1, u_nnc2])
            v_nb = concatenate([v_nb, v_nnc1, v_nnc2])
            data_nb = concatenate([data_nb, data_nnc1, data_nnc2])

        # print('Adjacency matrix:', self.N, 'x', self.N, 'with', len(u_nb), 'edges')
        # bygger en N×N matrise der A[u[i], v[i]] = data[i]
        return csr_matrix((data_nb, (u_nb, v_nb)), shape=(self.N, self.N), dtype=int32)

    #----------------------------------------------------------------------------
    def has_cycle(self):
    #----------------------------------------------------------------------------
        """
        Check for cycles in the adjacency matrix using Numba-compatible operations.
        """
        indptr = self.adj.indptr
        indices = self.adj.indices
        return has_cycle_numba(indptr, indices, self.N)

    #----------------------------------------------------------------------------
    def remove_cycles(self, echo=False):
    #----------------------------------------------------------------------------
        """Breaks all cycles by removing edges from a LIL representation."""
        # 1) Gjør om til LIL for trygg mutasjon
        A_lil = self.adj.tolil()

        self.cycle_id = 0
        while True:
            # 2) Finn SCC på CSR (gjør om midlertidig)
            A_csr = A_lil.tocsr()
            n_comp, labels = connected_components(csgraph=A_csr, directed=True, connection='strong', return_labels=True)
            counts = bincount(labels, minlength=n_comp)
            cyc_comps = where(counts > 1)[0]
            if cyc_comps.size == 0:
                # Ingen sykluser igjen
                if echo:
                    print(f"Completed cycle breaking, removed {self.cycle_id} cycles.")
                break

            comp_id = int(cyc_comps[0])
            self.cycle_id += 1

            # 3) Hent nodene i denne komponenten
            comp_nodes = where(labels == comp_id)[0]

            # 4) Fjern én kant i LIL‐matrisen
            removed = False
            for u in comp_nodes:
                # nabo‐flater fra LIL kan hentes raskt:
                for idx, v in enumerate(A_lil.rows[u]):
                    if labels[v] == comp_id:
                        if echo:
                            uc, vc = self.to_coords([u, v])
                            print(f"[Cycle {self.cycle_id}] Removing edge {uc} -> {vc} " +
                                f"in component {comp_id} (size {counts[comp_id]})")
                        # fjern elementet
                        del A_lil.rows[u][idx]
                        del A_lil.data[u][idx]
                        removed = True
                        break
                if removed:
                    break

            if not removed:
                # trygghetsjekk – skal ikke skje
                print(f"Warning: No edge to remove in component {comp_id}.")
                break

        # 5) Legg CSR‐versjonen tilbake i self.adj
        self.adj = A_lil.tocsr()
    
    #----------------------------------------------------------------------------
    def remove_cycles_weighted(self, echo=False):
    #----------------------------------------------------------------------------    
        """Bryter alle sykluser ved å fjerne minst-vektede kanter i hver SCC."""
        A_lil = self.adj.tolil()
        self.cycle_id = 0

        while True:
            # 1) Finn SCC på CSR
            A_csr = A_lil.tocsr()
            n_comp, labels = connected_components(csgraph=A_csr, directed=True,
                connection='strong', return_labels=True
            )
            counts = bincount(labels, minlength=n_comp)
            cyc_comps = where(counts > 1)[0]
            if cyc_comps.size == 0:
                if echo:
                    print(f"Ferdig med syklusbryting, fjernet {self.cycle_id} kanter.")
                break

            comp_id = int(cyc_comps[0])
            comp_nodes = where(labels == comp_id)[0]
            self.cycle_id += 1

            # 2) Finn den kantpar (u, idx) med minste vekt i denne komponenten
            min_w = inf
            min_u = None
            min_idx = None
            for u in comp_nodes:
                for idx, v in enumerate(A_lil.rows[u]):
                    if labels[v] == comp_id:
                        w = A_lil.data[u][idx]  # vekta på kanten u->v
                        if w < min_w:
                            min_w, min_u, min_idx = w, u, idx

            # 3) Fjern akkurat den minste kanten
            if min_u is not None:
                v = A_lil.rows[min_u][min_idx]
                if echo:
                    uc, vc = self.to_coords([min_u, v])
                    print(f"[Cycle {self.cycle_id}] Fjerner kant {uc} -> {vc} (vekt={min_w})")
                del A_lil.rows[min_u][min_idx]
                del A_lil.data[min_u][min_idx]
            else:
                # Bør aldri skje så lenge det finnes sykler
                print(f"Warning: Ingen kant funnet i komponent {comp_id}.")
                break

        # 4) Sett resultatet tilbake i self.adj
        self.adj = A_lil.tocsr()


    #--------------------------------------------------------------------------------
    def to_coords(self, flat, as_list=False):
    #--------------------------------------------------------------------------------
        """
        Converts a 1D array of flat indices `flat` to (x, y, z) coordinates.
        flat: array of shape (N,)
        shape: tuple of grid dimensions (X, Y, Z)
        Returns: list of tuples with coordinates of shape (N, 3)
        """

        # if not isinstance(flat, ndarray):
        #     flat = nparray(flat, dtype=int32)
        flat = asarray(flat, dtype=int32)
        #flat = asarray(flat, dtype=int32)
        #x, y, z = to_coords_parallel(flat, nparray(self.shape, dtype=int32))
        YZ = self.shape[1] * self.shape[2]
        x = flat // YZ
        rem = flat % YZ
        y = rem // self.shape[2]
        z = rem % self.shape[2]
        if as_list:
            return list(zip(x.tolist(), y.tolist(), z.tolist()))
        return stack((x, y, z), axis=1)

    #--------------------------------------------------------------------------------
    def to_flat(self, coords, as_list=False):
    #--------------------------------------------------------------------------------
        """Convert 3D coordinates to flat index."""
        # if not isinstance(coords, ndarray):
        #     coords = nparray(coords, dtype=int32)
        coords = asarray(coords, dtype=int32)
        x, y, z = coords.T
        flat = x * (self.shape[1] * self.shape[2]) + y * self.shape[2] + z      # (M,)
        if as_list:
            return flat.tolist()
        return flat
    

#============================================================================
class Map:
#============================================================================

    #------------------------------------------------------------------------
    def __init__(self, src, dim):
    #------------------------------------------------------------------------
        tmp = defaultdict(list)
        for idx, s in enumerate(src):
            tmp[tuple(s)].append(idx)
        csum = cumsum([0] + [len(v) for v in tmp.values()])
        idx = stack((csum[:-1], csum[1:]), axis=-1)
        self.keys = zeros((*dim, 2), dtype=int32)
        self.keys[tuple(zip(*tmp.keys()))] = idx
        self.values = nparray(list(flatten(tmp.values())))

    #------------------------------------------------------------------------
    def __getitem__(self, key):
    #------------------------------------------------------------------------
        i, j, k = key
        a, b = self.keys[i, j, k]
        return self.values[a:b]


#============================================================================
# Numba-routines
#============================================================================

@njit(parallel=True)
#----------------------------------------------------------------------------
def check_order_numba(u_indices, v_indices, pos, N):
#----------------------------------------------------------------------------
    # Preallocate a fixed-size array for errors (worst case: all edges are errors)
    max_edges = v_indices.shape[0]
    errors = empty((max_edges, 2), dtype=int32)
    error_count = 0
    for u in prange(N):
        start = u_indices[u]
        end = u_indices[u + 1]
        for idx in range(start, end):
            v = v_indices[idx]
            if pos[u] >= pos[v]:
                errors[error_count, 0] = u
                #errors[error_count, 1] = pos[u]
                errors[error_count, 1] = v
                #errors[error_count, 3] = pos[v]
                error_count += 1
    return errors[:error_count]

@njit
#----------------------------------------------------------------------------
def kahn_numba(indptr, indices, in_deg):
#----------------------------------------------------------------------------
    N = in_deg.shape[0]
    queue = empty(N, int32)
    order = empty(N, int32)
    head = 0
    tail = 0
    idx  = 0

    # init kø
    for u in range(N):
        if in_deg[u] == 0:
            queue[tail] = u
            tail += 1

    # kjør
    while head < tail:
        u = queue[head]
        head += 1
        order[idx] = u
        idx += 1
        for ptr in range(indptr[u], indptr[u+1]):
            v = indices[ptr]
            in_deg[v] -= 1
            if in_deg[v] == 0:
                queue[tail] = v
                tail += 1

    return order[:idx], idx


@njit
#--------------------------------------------------------------------------------
def has_cycle_numba(indptr, indices, N):
#--------------------------------------------------------------------------------
    """
    Check for cycles in a graph represented by a CSR adjacency matrix.

    Parameters:
    indptr (array): CSR matrix indptr array.
    indices (array): CSR matrix indices array.
    N (int): Number of nodes in the graph.

    Returns:
    bool: True if a cycle is detected, False otherwise.
    """
    visited = empty(N, dtype=boolean)
    rec_stack = empty(N, dtype=boolean)

    visited[:] = False
    rec_stack[:] = False

    for node in range(N):
        if not visited[node]:
            if has_cycle_dfs(node, indptr, indices, visited, rec_stack):
                return True
    return False


@njit
#--------------------------------------------------------------------------------
def has_cycle_dfs(node, indptr, indices, visited, rec_stack):
#--------------------------------------------------------------------------------
    """
    Perform a DFS to detect cycles in the graph.

    Parameters:
    node (int): Current node being visited.
    indptr (array): CSR matrix indptr array.
    indices (array): CSR matrix indices array.
    visited (array): Array to track visited nodes.
    rec_stack (array): Array to track recursion stack.

    Returns:
    bool: True if a cycle is detected, False otherwise.
    """
    visited[node] = True
    rec_stack[node] = True

    for ptr in range(indptr[node], indptr[node + 1]):
        neighbor = indices[ptr]
        if not visited[neighbor]:
            if has_cycle_dfs(neighbor, indptr, indices, visited, rec_stack):
                return True
        elif rec_stack[neighbor]:
            return True

    rec_stack[node] = False
    return False


# @njit(parallel=True)
# #--------------------------------------------------------------------------------
# def to_coords_parallel(flat, shape):
# #--------------------------------------------------------------------------------
#     n = flat.shape[0]
#     x = empty(n, dtype=np.int32)
#     y = empty(n, dtype=np.int32)
#     z = empty(n, dtype=np.int32)
#     stride_yz = shape[1] * shape[2]
#     for i in prange(n):
#         idx = flat[i]
#         xi = idx // stride_yz
#         rem = idx - xi * stride_yz
#         yi = rem // shape[2]
#         zi = rem - yi * shape[2]
#         x[i] = xi
#         y[i] = yi
#         z[i] = zi
#     return x, y, z


# @njit(parallel=True)
# #--------------------------------------------------------------------------------
# def to_flat_parallel(coords, shape):
# #--------------------------------------------------------------------------------
#     # coords er en (n,3)-array av int32
#     n = coords.shape[0]
#     flat = np.empty(n, dtype=np.int64)
#     stride_yz = shape[1] * shape[2]
#     stride_z  = shape[2]
#     for i in prange(n):
#         xi = coords[i, 0]
#         yi = coords[i, 1]
#         zi = coords[i, 2]
#         flat[i] = xi * stride_yz + yi * stride_z + zi
#     return flat

# HOW TO EDIT PROPERTIES IN INTERSECT USING PYTHON:
    # MODEL_DEFINITION

    #------------------------------------------------------------------------------
    # This script modifies the fraction of CO2 dissolved in water after every
    # timestep. This will cause IX to abort with the error-message:
    #
    #     WARNING  Validation of BoxPropertyEdit['Edit_1_0']:
    #              Box restriction for Grid edit CoarseGrid on flat grid CoarseGrid: ni=20 nj=1 nk=1 cannot be performed.
    #     ERROR    Cell property edit for 'AQUEOUS_COMPONENT_MOLE_FRACTION["CO2"]' is not valid after the simulation has started.
    #
    #------------------------------------------------------------------------------

    #CustomControl "ScriptTest" {
    #    ExecutionPosition=END_TIMESTEP
    #
    #    InitialScript=@{
    #from itertools import product, chain
    #from math import prod
    #}@
    #
    #    Script=@{
    #t = FieldManagement().CurrentTime.get().days()
    #name = 'Edit_' + str(t).replace('.', '_')
    #
    #grid = StructuredInfo('CoarseGrid')
    #start = grid.FirstCellId.get()
    #dim = [getattr(grid, f'NumberCellsIn{x}').get() for x in ('IJK')]
    #limits = list(chain(*zip((1,1,1), dim)))
    #
    #box_edit = GridMgr().RegionFamilyMgr().create_box_property_edit(name)
    #box_edit.Grid.set(["CoarseGrid"])
    #attrs = [a+b for a,b in product('IJK','12')]
    #for attr, lim in zip(attrs, limits):
    #    print(attr, lim)
    #    getattr(box_edit, attr).set([lim]) 
    #box_edit.Property.set(['AQUEOUS_COMPONENT_MOLE_FRACTION["CO2"]'])
    #N = prod(dim)
    #expr = '[' + ' '.join('0.02' for _ in range(N)) + ']'
    #print(expr)
    #box_edit.Expression.set([expr])
    #}@
    #
    #}
