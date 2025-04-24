from collections import namedtuple, deque
from itertools import product, islice
from numpy import (array as nparray, abs as npabs, sum as npsum, diff as npdiff, any as npany,
                   zeros, ones, stack, concatenate, moveaxis, prod, where, argwhere, atleast_1d,
                   argsort, bool_ as np_bool)
from .ECL import UNRST_file, RFT_file, INIT_file
from .utils import batched, roll_xyz, missing_elements, neighbour_connections

#====================================================================================
class Flow():                                                                  # Flow
#====================================================================================

    #--------------------------------------------------------------------------------
    def __init__(self, root, phase='wat', res_block=False, res_well=False):    # Flow
    #--------------------------------------------------------------------------------
        self.unrst = UNRST_file(root)
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
        # Non-neighbouring connections
        # NB! Seems that only reservoir NNC is available 
        # in UNRST, so only one phase is used
        init = INIT_file(self.unrst.path)
        self.nnc_ijk = [tuple(zip(*nnc))for nnc in init.non_neigh_conn()]
        self.nnc_key = ''
        if self.nnc_ijk[0]:
            self.nnc_key = f'FLR{self.phases[0].upper()}N+'
        self.seqnum = 0
        self.time = 0
        self._neigh_connects = None
        self._index_array = None

    #--------------------------------------------------------------------------------
    def sat_rporv(self):                                 # Flow
    #--------------------------------------------------------------------------------
        Data = namedtuple('Data', 'time dt rporv rporv_old sat sat_old')
        blockdata = self.unrst.blockdata('DOUBHEAD', 0, 'S'+self.phases[0].upper(), 'RPORV')
        # Initial data
        time_old, *data = next(blockdata, None)
        sat_old, rporv_old = self.unrst.reshape_dim(*data)
        for time, *data in blockdata:
            sat, rporv = self.unrst.reshape_dim(*data)
            yield Data(time, time - time_old, rporv, rporv_old, sat, sat_old)
            time_old, sat_old, rporv_old = time, sat, rporv

    #--------------------------------------------------------------------------------
    def neighbour_connections(self):                                                # Flow
    #--------------------------------------------------------------------------------
        if self._neigh_connects is None:
            self._neigh_connects = neighbour_connections(self.unrst.dim())
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
        # NNC:
        #   Conc. for flow nnc2 -> nnc1 is connection 8, while conc. for
        #   flow nnc1 -> nnc2 is connection 9

        Tracer = namedtuple('Tracer', 'time dt conc prod_mass cfl')
        conc = stack(init, axis=-1)
        dim = self.unrst.dim()
        inj_conc = stack([i * ones(dim) for i in inlet], axis=-1)
        rates = self.rates()
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
            if self.nnc_key:
                conc_nnc = zeros((*conc.shape[:3], 2))
                # NNC inflow[0] is flow into nnc1 from nnc2, get conc from nnc2
                conc_nnc[self.nnc_ijk[0]][..., 0] = conc[self.nnc_ijk[1]]
                # NNC inflow[1] is flow into nnc2 from nnc1, get conc from nnc1
                conc_nnc[self.nnc_ijk[1]][..., 1] = conc[self.nnc_ijk[0]]
                # The NNC conc. is connection 8 and 9
                conc_inflow = concatenate((conc_inflow, conc_nnc), axis=-1)
            # Second part of (Eq.1) or (Eq.3)
            mass += rate.dt * npsum((rate.rate_in[..., None, :] * conc_inflow), axis=-1)
            conc = mass / vol_phase[..., None]
            prod_mass = rate.dt * rate.rate_out[..., 6][..., None] * conc
            cfl = rate.dt * npsum(rate.rate_in, axis=-1) / vol_phase
            output = (zip(name, moveaxis(var, -1, 0)) for var in (conc, prod_mass))
            yield Tracer(rate.time, rate.dt, *[dict(out) for out in output], cfl)
            vol_phase_old = vol_phase


    #--------------------------------------------------------------------------------
    def sort_blocks_by_flow(self, rate):                                 # Flow
    #--------------------------------------------------------------------------------
        in_degrees = npsum(where(rate.rate_in[..., :6] > 0, 1, 0), axis=-1)
        out_conn = where(rate.rate_out[...,:6, None] > 0, self.neighbour_connections(), -1)
        # Start with blocks with no in-degrees (inlets)
        queue = deque([tuple(ind) for ind in argwhere(in_degrees == 0)])
        
        sorted_blocks = []
        while queue:
            node = queue.popleft()
            sorted_blocks.append(node)
            connections = [tuple(nb) for nb in out_conn[node] if nb[0] >= 0]
            for link in connections:
                in_degrees[link] -= 1
                if in_degrees[link] == 0:
                    queue.append(link)
        num_blocks = prod(self.unrst.dim())
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
        Tracer = namedtuple('Tracer', 'time dt conc prod_mass cfl')
        dim = self.unrst.dim()
        conc = stack(init, axis=-1)
        rates = self.rates()
        sat_rporv = self.sat_rporv()
        inj_conc = stack([i * ones(dim) for i in inlet], axis=-1)
        for rate, data in zip(rates, sat_rporv):
            self.check_time_sync(rate.time, data.time)
            vol_phase_old = data.sat_old * data.rporv_old
            vol_phase = data.sat * data.rporv
            # Initialize mass with c^n * Vp^n and mass from injectors
            mass = conc * vol_phase_old[..., None] + rate.dt * inj_conc * rate.rate_in[..., 6][..., None]
            # inflow_conc[0] : inflow rate, inflow_conc[1:4] : index of inflow connection blocks
            rate_in = rate.rate_in[..., :6]
            nb_conn = self.neighbour_connections()
            # if self.nnc_key:
            #     # Skip well rates at index 6
            #     #rate_in = rate.rate_in[..., (0, 1, 2, 3, 4, 5, 7, 8)]
            #     rate_in = concatenate((rate_in, rate.rate_in[..., 7:]), axis=-1)
            #     connect_nnc = zeros((*conc.shape[:3], 3, 2), dtype=int)
            #     connect_nnc[self.nnc_ijk[0]][..., 0] = self.nnc_ijk[1]
            #     connect_nnc[self.nnc_ijk[1]][..., 1] = self.nnc_ijk[0]
            #     connections = concatenate((connections, connect_nnc), axis=-1)
            #inflow_conn = concatenate((rate.rate_in[..., :6, None], self.neighbour_connections()), axis=-1, dtype=object)
            #inflow_conn = concatenate((rate_in[..., None], nb_conn), axis=-1, dtype=object)
            #print(inflow_conn.shape)
            for b in self.sort_blocks_by_flow(rate):
                if force_vol_balance:
                    # Denominator of (Eq.3)
                    denom = vol_phase_old[b] + rate.dt * npsum(rate.rate_in[b])
                else:
                    # Denominator of (Eq.1)
                    denom = vol_phase[b] + rate.dt * npsum(rate.rate_out[b])
                # Nominator of (Eq.1) or (Eq.3)
                # Filter and compute inflow contributions in a vectorized manner
                valid_conn = rate_in[b] > 0
                if npany(valid_conn):
                    inflow_conc = rate_in[b][valid_conn, None] * conc[tuple(zip(*nb_conn[b][valid_conn]))]
                    mass[*b, :] += rate.dt * npsum(inflow_conc, axis=0)
                conc[*b, :] = mass[*b, :] / denom
            prod_mass = rate.dt * rate.rate_out[..., 6][..., None] * conc
            cfl = rate.dt * npsum(rate.rate_in, axis=-1) / vol_phase
            output = (zip(name, moveaxis(var, -1, 0)) for var in (conc, prod_mass))
            yield Tracer(rate.time, rate.dt, *[dict(out) for out in output], cfl)
            vol_phase_old = vol_phase


    #--------------------------------------------------------------------------------
    def rates(self):                                                    # Flow
    #--------------------------------------------------------------------------------
        Rates = namedtuple('Rates', 'time dt rate_in rate_out')
        block_flow = self.interblock()
        well_flow = self.wells()
        dim = self.unrst.dim()
        nnc = None
        if self.nnc_key:
            nnc_rates = self.nnc_rates()
            nnc = next(nnc_rates, None)
        # Initial time
        time = next(block_flow).time
        # Loop over interblock flows and well flows
        for block, wells in zip(block_flow, well_flow):
            # Loop over active wells
            well_rate = {io:zeros(dim) for io in ('in', 'out')}
            for well in wells:
                self.check_time_sync(well.time, block.time)
                kind = 'in' if well.rate[0] < 0 else 'out'
                well_rate[kind][well.pos] += npabs(well.rate)
            dt = block.time - time
            # Last index is connection number. Block rates are the 6 first connections 
            # (one for each face), and well rates are connection 7.
            rate_in = concatenate((block.rate_in, well_rate['in'][..., None]), axis=-1)
            rate_out = concatenate((block.rate_out, well_rate['out'][..., None]), axis=-1)
            if nnc:
                # NNC rates are connection 8 (nnc1 in/out) and 9 (nnc2 in/out)
                nnc = next(nnc_rates, None)
                rate_in = concatenate((rate_in, nnc.rate_in), axis=-1)
                rate_out = concatenate((rate_out, nnc.rate_out), axis=-1)
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
    def nnc_rates(self):                                                  # Flow
    #--------------------------------------------------------------------------------
        NNC = namedtuple('NNC', 'rate_in rate_out')
        dim2 = (*self.unrst.dim(), 2)
        for seqnum, rate in self.unrst.blockdata('SEQNUM', self.nnc_key, singleton=True):
            if seqnum[0] != self.seqnum:
                raise ValueError(f'SEQNUM mismatch for NNC-rates: {seqnum[0]}, {self.seqnum}')
            rate = nparray(rate)
            rate_in = -rate * (rate < 0)
            rate_out = rate * (rate >= 0)
            nnc_in = zeros(dim2)
            # Into nnc1 (from nnc2)
            nnc_in[self.nnc_ijk[0]][..., 0] = rate_in
            # Into nnc2 (from nnc1)
            nnc_in[self.nnc_ijk[1]][..., 1] = rate_out
            nnc_out = zeros(dim2)
            # From nnc1 (into nnc2)
            nnc_out[self.nnc_ijk[0]][..., 0] = rate_out
            # From nnc2 (into nnc1)
            nnc_out[self.nnc_ijk[1]][..., 1] = rate_in
            yield NNC(nnc_in, nnc_out)


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

