from collections import defaultdict, namedtuple, deque
from itertools import chain, product, islice
from math import inf
from operator import ne
from numpy import (array as nparray, abs as npabs, empty, int64, ndarray,
                   sum as npsum, diff as npdiff, any as npany, zeros, ones, stack, concatenate,
                   moveaxis, prod, where, argwhere, atleast_1d, argsort, bool_ as np_bool,
                   add as npadd, repeat as nprepeat, arange, ones_like, bincount)
from numba import njit
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from .ECL import UNRST_file, RFT_file, INIT_file
from .utils import batched, roll_xyz, missing_elements, neighbour_connections



@njit
def implicit_numba(sorted_blocks, nb_conn, rate, data, conc, inj_conc, force_vol_balance=False):
    """
    Perform implicit integration for tracer concentration updates.

    Parameters:
        sorted_blocks (list): List of sorted block indices (x, y, z).
        nb_conn (ndarray): Neighbor connections for each block.
        rate (object): Object containing rate_in, rate_out, and dt.
        vol_phase (ndarray): Current phase volume for each block.
        vol_phase_old (ndarray): Previous phase volume for each block.
        conc (ndarray): Concentration array with shape (nx, ny, nz, 3).
        mass (ndarray): Mass array with shape (nx, ny, nz, 3).
        force_vol_balance (bool): Whether to enforce volume balance.

    Returns:
        ndarray: Updated concentration array.
    """
    vol_phase_old = data.sat_old * data.rporv_old
    vol_phase = data.sat * data.rporv
    mass = conc * vol_phase_old[..., None] + rate.dt * inj_conc * rate.rate_in[..., 6][..., None]
    for b in sorted_blocks:
        bx, by, bz = b
        if force_vol_balance:
            # Denominator of (Eq.3)
            denom = (
                vol_phase_old[bx, by, bz]
                + rate.dt * npsum(rate.rate_in[bx, by, bz])
            )
        else:
            # Denominator of (Eq.1)
            denom = (
                vol_phase[bx, by, bz]
                + rate.dt * npsum(rate.rate_out[bx, by, bz])
            )

        valid_conn = rate.rate_in[bx, by, bz, :6] > 0
        inflow_mass = zeros(conc.shape[-1])  # Initialize for 3 species
        if npany(valid_conn):
            for i in range(valid_conn.shape[0]):
                if valid_conn[i]:
                    nx, ny, nz = nb_conn[bx, by, bz, i]
                    inflow_mass += (
                        rate.rate_in[bx, by, bz, i] * conc[nx, ny, nz, :]
                    )
            mass[bx, by, bz, :] += rate.dt * inflow_mass

        # Update concentration
        conc[bx, by, bz, :] = mass[bx, by, bz, :] / denom

    return conc


#====================================================================================
class Flow():                                                                  # Flow
#====================================================================================

    #--------------------------------------------------------------------------------
    def __init__(self, root, phase='wat', res_block=False, res_well=False):    # Flow
    #--------------------------------------------------------------------------------
        self.unrst = UNRST_file(root)
        self.dim = self.unrst.dim()
        self.rft = RFT_file(root)
        self.convert = False
        self.phases = (phase,)
        self.dt_accuracy = 1e-4  # Allowed difference in UNRST and RFT time steps
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
        self.unrst.check_missing_keys(*self.block_keys)
        TUB_RAT = 'TUBL' if res_well else 'RAT'
        self.well_keys = [f'CON{owg}{TUB_RAT}' for owg in 'OWG']
        self.rft.check_missing_keys(*self.well_keys)
        self.nnc = NNC(root, self.phases)
        self.sort = Sort(self.dim)
        self.seqnum = 0
        self.time = 0
        self._neigh_connects = None

    #--------------------------------------------------------------------------------
    def sat_rporv(self, sync=True):                                       # Flow
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

        Tracer = namedtuple('Tracer', 'time dt conc prod_mass cfl')
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
            if self.nnc.exists:
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
            cfl = rate.dt * npsum(rate.rate_in, axis=-1) / vol_phase
            output = (zip(name, moveaxis(var, -1, 0)) for var in (conc, prod_mass))
            yield Tracer(rate.time, rate.dt, *[dict(out) for out in output], cfl)
            vol_phase_old = vol_phase

    #--------------------------------------------------------------------------------
    def sort_blocks_by_flow(self, rate):                            # Flow
    #--------------------------------------------------------------------------------
        in_degrees = npsum(where(rate.rate_in[..., :6] > 0, 1, 0), axis=-1)
        out_conn = where(rate.rate_out[...,:6, None] > 0, self.neighbour_connections(), -1)
        if self.nnc.exists:
            nnc_rate = self.nnc.get_rate(self.seqnum)
            # rate_out is flow into nnc2 from nnc1
            npadd.at(in_degrees, self.nnc.ijk[1], nnc_rate.rate_out > 0)
            # rate_in is flow into nnc1 from nnc2
            npadd.at(in_degrees, self.nnc.ijk[0], nnc_rate.rate_in > 0)
            nnc_out_conn = self.nnc.out_connections(nnc_rate)
        # Start with blocks with no in-degrees (inlets)
        queue = deque([tuple(ind) for ind in argwhere(in_degrees == 0).tolist()])
        
        sorted_blocks = []
        while queue:
            node = queue.popleft()
            sorted_blocks.append(node)
            connections = (tuple(nb) for nb in out_conn[node].tolist() if nb[0] >= 0)
            if self.nnc.exists:
                connections = chain(connections, nnc_out_conn.get(node, []))
            for link in connections:
                in_degrees[link] -= 1
                if in_degrees[link] == 0:
                    queue.append(link)
        num_blocks = prod(self.dim)
        if len(sorted_blocks) != num_blocks:
            raise RuntimeError(f'ERROR: Not all blocks are sorted: {len(sorted_blocks)} < {num_blocks}')
        return sorted_blocks


    #--------------------------------------------------------------------------------
    def tracer_implicit(self, name, init, inlet, force_vol_balance=False):                        # Flow
    #--------------------------------------------------------------------------------
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
        Tracer = namedtuple('Tracer', 'time dt conc prod_mass')
        #dim = self.unrst.dim()
        conc = stack(init, axis=-1)
        rates = self.rates()
        sat_rporv = self.sat_rporv()
        inj_conc = stack([i * ones(self.dim) for i in inlet], axis=-1)
        nb_conn = self.neighbour_connections()
        for rate, data in zip(rates, sat_rporv):
            self.check_time_sync(rate.time, data.time)

            sorted_blocks = self.sort.topological(rate.rate_out[..., :6])
            conc = implicit_numba(sorted_blocks, nb_conn, rate, data, conc, inj_conc, force_vol_balance)

            # vol_phase_old = data.sat_old * data.rporv_old
            # vol_phase = data.sat * data.rporv
            # # Initialize mass with c^n * Vp^n and mass from injectors
            # # inflow_conc[0] : inflow rate, inflow_conc[1:4] : index of inflow connection blocks
            # # mass = conc * vol_phase_old[..., None] + rate.dt * inj_conc * rate.rate_in[..., 6][..., None]
            # rate_in = rate.rate_in[..., :6]
            # for b in self.sort.topological(rate.rate_out[..., :6]):
            #     if force_vol_balance:
            #         # Denominator of (Eq.3)
            #         denom = vol_phase_old[b] + rate.dt * npsum(rate.rate_in[b])
            #     else:
            #         # Denominator of (Eq.1)
            #         denom = vol_phase[b] + rate.dt * npsum(rate.rate_out[b])
            #     # Nominator of (Eq.1) or (Eq.3)
            #     # Filter and compute inflow contributions in a vectorized manner
            #     valid_conn = rate_in[b] > 0
            #     if npany(valid_conn):
            #         #inflow_conc = rate_in[b][valid_conn, None] * conc[tuple(zip(*nb_conn[b][valid_conn]))]
            #         inflow_conc = rate_in[b][valid_conn, None] * conc[*nb_conn[b][valid_conn].T]
            #         mass[*b, :] += rate.dt * npsum(inflow_conc, axis=0)
            #     conc[*b, :] = mass[*b, :] / denom

            prod_mass = rate.dt * rate.rate_out[..., 6][..., None] * conc
            #cfl = rate.dt * npsum(rate.rate_in, axis=-1) / vol_phase
            output = (zip(name, moveaxis(var, -1, 0)) for var in (conc, prod_mass))
            yield Tracer(rate.time, rate.dt, *[dict(out) for out in output])
            #vol_phase_old = vol_phase


    #--------------------------------------------------------------------------------
    def rates(self):                                                           # Flow
    #--------------------------------------------------------------------------------
        Rates = namedtuple('Rates', 'time dt rate_in rate_out')
        block_flow = self.interblock()
        well_flow = self.wells()
        #dim = self.unrst.dim()
        # nnc = None
        # if self.nnc_key:
        #     nnc_rates = self.nnc_rates()
        #     nnc = next(nnc_rates, None)
        # Initial time
        # Need to skip the inital block of the UNRST-file to sync with the RFT-file
        time = next(block_flow).time
        # Loop over interblock flows and well flows
        for block, wells in zip(block_flow, well_flow):
            # Loop over active wells
            well_rate = {io:zeros(self.dim) for io in ('in', 'out')}
            for well in wells:
                self.check_time_sync(well.time, block.time)
                kind = 'in' if well.rate[0] < 0 else 'out'
                well_rate[kind][well.pos] += npabs(well.rate)
            dt = block.time - time
            # Last index is connection number. Block rates are the 6 first connections
            # (one for each face), and well rates are connection 7.
            rate_in = concatenate((block.rate_in, well_rate['in'][..., None]), axis=-1)
            rate_out = concatenate((block.rate_out, well_rate['out'][..., None]), axis=-1)
            # if nnc:
            #     # NNC rates are connection 8 (nnc1 in/out) and 9 (nnc2 in/out)
            #     nnc = next(nnc_rates, None)
            #     rate_in = concatenate((rate_in, nnc.rate_in), axis=-1)
            #     rate_out = concatenate((rate_out, nnc.rate_out), axis=-1)
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


    # #--------------------------------------------------------------------------------
    # def nnc_rates(self):                                                  # Flow
    # #--------------------------------------------------------------------------------
    #     NNC = namedtuple('NNC', 'rate_in rate_out')
    #     dim2 = (*self.unrst.dim(), 2)
    #     for seqnum, rate in self.unrst.blockdata('SEQNUM', self.nnc_key, singleton=True):
    #         if seqnum[0] != self.seqnum:
    #             raise ValueError(f'SEQNUM mismatch for NNC-rates: {seqnum[0]}, {self.seqnum}')
    #         rate = nparray(rate)
    #         rate_in = -rate * (rate < 0)
    #         rate_out = rate * (rate >= 0)
    #         nnc_in = zeros(dim2)
    #         # Into nnc1 (from nnc2)
    #         nnc_in[self.nnc_ijk[0]][..., 0] = rate_in
    #         # Into nnc2 (from nnc1)
    #         nnc_in[self.nnc_ijk[1]][..., 1] = rate_out
    #         nnc_out = zeros(dim2)
    #         # From nnc1 (into nnc2)
    #         nnc_out[self.nnc_ijk[0]][..., 0] = rate_out
    #         # From nnc2 (into nnc1)
    #         nnc_out[self.nnc_ijk[1]][..., 1] = rate_in
    #         yield NNC(nnc_in, nnc_out)

    # #--------------------------------------------------------------------------------
    # def nnc_rates(self, sync=True):                                     # Flow
    # #--------------------------------------------------------------------------------
    #     # Read NNC rates from UNRST-file at reservoir conditions, i.e. FLRpppN+ where 
    #     # ppp is OIL, WAT, or GAS
    #     NNC = namedtuple('NNC', 'rate_in rate_out')
    #     blockdata = self.unrst.blockdata('SEQNUM', self.nnc_key, singleton=True)
    #     if sync:
    #         # Skip the initial block to sync with the RFT-file
    #         next(blockdata, None)
    #     for seqnum, rate in blockdata:
    #         if sync and seqnum[0] != self.seqnum:
    #             raise ValueError(f'SEQNUM mismatch for NNC-rates: {seqnum[0]}, {self.seqnum}')
    #         rate = nparray(rate)
    #         rate_in = -rate * (rate < 0)
    #         rate_out = rate * (rate > 0)
    #         yield NNC(rate_in, rate_out)


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
    def wells(self, zerobase=True):                                             # Flow
    #--------------------------------------------------------------------------------
        Well = namedtuple('Well', 'time name pos rate')
        keys = ('TIME', 'CONIPOS', 'CONJPOS', 'CONKPOS', 'WELLPLT', 'WELLETC', 'CONNXT')
        rates = []
        current_time = next(self.rft.read('time'))
        rftdata = self.rft.blockdata(*keys, *self.well_keys, singleton=True)
        for time, i, j, k, wellplt, welletc, connxt, *wellrat in rftdata:
            time = time[0]
            wellname = welletc[1]
            if time > current_time:
                yield rates
                current_time = time
                rates = []
            if self.res_well:
                wellrat = self._wellrate_from_tubing(wellrat, wellplt, connxt)
            wellrat = {ph:nparray(rat) for ph,rat in zip(('oil', 'wat', 'gas'), wellrat)}
            pos = (i, j, k)
            if zerobase:
                # Use zero-based position indices
                pos = tuple([x-1 for x in p] for p in pos)
            if not self.res_well:
                # Convert surface rates to reservoir rates
                wellrat = self.convert.surf_to_res(wellrat, self.seqnum, pos=pos)
            rates.append(Well(time, wellname, pos, wellrat))


    #--------------------------------------------------------------------------------
    def _wellrate_from_tubing(self, tubflows, wellplt, connxt):                # Flow
    #--------------------------------------------------------------------------------
        # Seems that this only works for reservoir rates, and the accuracy is not very good
        wellplt = nparray(wellplt)
        phase_rates = wellplt[5]*(wellplt[:3]/sum(wellplt[:3]))
        connxt = atleast_1d(connxt)
        rates = []
        for i, tub in enumerate(tubflows):
            wellrate = phase_rates[i]
            tub = atleast_1d(tub)
            injector = wellrate < 0
            idx = list(argsort(connxt))
            miss = missing_elements(connxt) or [len(connxt)]
            is_neigh = ones(len(idx) + len(miss), dtype=np_bool)
            for m in miss:
                idx.insert(m, (m if m<len(connxt) else 0) if injector else m-1)
                is_neigh[m] = 0
            #is_neigh[list(miss)] = 0
            if injector:
                # Injector
                diff = zeros(len(tub))
                for i in range(len(tub)):
                    if is_neigh[i+1]:
                        diff[idx[i+1]] = tub[i] - tub[idx[i+1]]
                diff[idx[0]] = wellrate - tub[idx[0]]
            else:
                # Producer
                diff = tub - is_neigh[1:]*tub[idx[1:]]
            rates.append(diff)
        return rates

    #--------------------------------------------------------------------------------
    def check_time_sync(self, *times):             # Flow
    #--------------------------------------------------------------------------------
        times = list(times)
        if len(times) < 2:
            times.append(self.time)
        if npany(npabs(npdiff(times)) > self.dt_accuracy):
            raise ValueError(f'Time difference in Rates-class: {times}')

#================================================================================
class NNC():                                          # NNC
#================================================================================
    nnc_rate = namedtuple('nnc_rate', 'rate_in rate_out')

    #----------------------------------------------------------------------------
    def __init__(self, root, phases):                 # Convert
    #----------------------------------------------------------------------------
        # Non-neighbouring connections
        # NB! Seems that only reservoir NNC is available 
        # in UNRST, so only one phase is used
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
            self.ijk = [tuple(zip(*n))for n in self.nnc]
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
        seqnum, data = next(islice(self.blockdata, step-1, None))
        self.seqnum = seqnum[0]
        rate = nparray(data)
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
        if pos:
            pvt = {k:v[tuple(pos)] for k,v in self.pvt.items()}
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
    def __init__(self, shape):
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
        self.u = nprepeat(arange(self.N), self.D)               # (N*6,)
        self.adj = None

    #----------------------------------------------------------------------------
    def topological(self, rate_out):
    #----------------------------------------------------------------------------
        """
        Breaks cycles, then manually runs Kahn's algorithm
        on our CSR matrix to determine a flow order.
        """
        self.adj = self.adjacency_matrix(rate_out)
        if self.has_cycle():
            self.remove_cycles()

        # Create indptr and indices for kahn_numba
        indptr  = self.adj.indptr    # length N+1
        indices = self.adj.indices   # length = number of edges

        # Init in-degree as a 1D-array
        in_deg = nparray(self.adj.sum(axis=0)).ravel()  # shape (N,)

        # Run Kahn's algoritm
        order, idx = kahn_numba(indptr, indices, in_deg)
        
        # Check for possible cycles
        if idx < self.N:
            print("Warning: Cycles remain — topological sorting incomplete.")
            order = order[:idx]

        return self.to_coords(order, as_list=True)

    #--------------------------------------------------------------------------------
    def to_coords(self, flat, as_list=False):
    #--------------------------------------------------------------------------------
        """
        Converts a 1D array of flat indices `flat` to (x, y, z) coordinates.
        flat: array of shape (N,)
        shape: tuple of grid dimensions (X, Y, Z)
        Returns: list of tuples with coordinates of shape (N, 3)
        """

        if not isinstance(flat, ndarray):
            flat = nparray(flat)
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
        if not isinstance(coords, ndarray):
            coords = nparray(coords)
        x, y, z = coords.T
        flat = x * (self.shape[1] * self.shape[2]) + y * self.shape[2] + z      # (M,)
        if as_list:
            return flat.tolist()
        return flat

    #----------------------------------------------------------------------------
    def adjacency_matrix(self, rate_out):
    #----------------------------------------------------------------------------
        valid = rate_out.reshape(self.N * self.D) > 0          # (N*6,)
        u_valid = self.u[valid]
        # Beregn destinasjons-ID v
        v = self.to_flat(self.connections[valid])          # (M,)
        # Bygg sparse adjacency-matrise
        data = ones_like(u_valid, dtype=int)
        return csr_matrix((data, (u_valid, v)), shape=(self.N, self.N), dtype=int)


    # #----------------------------------------------------------------------------
    # def active_connections(self, pos, adj=None):
    # #----------------------------------------------------------------------------
    #     adj = self.adj if adj is None else adj
    #     return self.to_coords(adj[*self.to_flat(pos)].indices, as_list=True)

    #----------------------------------------------------------------------------
    def active_connections(self, adj=None):
    #----------------------------------------------------------------------------
        """
        Returns a defaultdict mapping each grid cell (x, y, z) to a list of its active connections.

        Parameters:
        adj (csr_matrix, optional): The adjacency matrix representing connections. If None, uses self.adj.
        """
        adj = self.adj if adj is None else adj
        conn = defaultdict(list)
        coords = (self.to_coords(axis, as_list=True) for axis in adj.nonzero())
        for u_coord, v_coord in zip(*coords):
            conn[u_coord].append(v_coord)
        return conn

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
    def remove_cycles(self):
    #----------------------------------------------------------------------------
        """Breaks all cycles by removing edges from a LIL representation."""
        # 1) Gjør om til LIL for trygg mutasjon
        A_lil = self.adj.tolil()

        cycle_id = 0
        while True:
            # 2) Finn SCC på CSR (gjør om midlertidig)
            A_csr = A_lil.tocsr()
            n_comp, labels = connected_components(csgraph=A_csr, directed=True, connection='strong', return_labels=True)
            counts = bincount(labels, minlength=n_comp)
            cyc_comps = where(counts > 1)[0]
            if cyc_comps.size == 0:
                # Ingen sykluser igjen
                print(f"Completed cycle breaking, removed {cycle_id} cycles.")
                break

            comp_id = int(cyc_comps[0])
            cycle_id += 1

            # 3) Hent nodene i denne komponenten
            comp_nodes = where(labels == comp_id)[0]

            # 4) Fjern én kant i LIL‐matrisen
            removed = False
            for u in comp_nodes:
                # nabo‐flater fra LIL kan hentes raskt:
                for idx, v in enumerate(A_lil.rows[u]):
                    if labels[v] == comp_id:
                        print(f"[Cycle {cycle_id}] Removing edge {u}->{v} " +
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
    

#============================================================================
# Numba-routines
#============================================================================

@njit
#----------------------------------------------------------------------------
def kahn_numba(indptr, indices, in_deg):
#----------------------------------------------------------------------------
    N = in_deg.shape[0]
    queue = empty(N, int64)
    order = empty(N, int64)
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
    visited = empty(N, dtype=np_bool)
    rec_stack = empty(N, dtype=np_bool)

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

@njit
#--------------------------------------------------------------------------------
def to_coords(flat, shape):
#--------------------------------------------------------------------------------
    YZ = shape[1] * shape[2]
    x = flat // YZ
    rem = flat % YZ
    y = rem // shape[2]
    z = rem % shape[2]
    return x, y, z

@njit
#--------------------------------------------------------------------------------
def to_flat(coords, shape):
#--------------------------------------------------------------------------------
    """ Convert 3D coordinates to flat index. """
    x, y, z = coords.T
    return x * (shape[1] * shape[2]) + y * shape[2] + z

