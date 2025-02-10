from itertools import repeat, islice
from time import sleep
from threading import Thread
from numpy import zeros, array as nparray, mean as npmean
from numpy.linalg import norm as npnorm
from IORlib.ECL import EGRID_file, IX_input, UNRST_file, RFT_file
from IORlib.utils import any_cell_in_box, flatten
from pyvista import PolyData, Label, Arrow, Sphere, Cylinder
from pyvistaqt import BackgroundPlotter


#====================================================================================
class Plotter():                                                            # Plotter
#====================================================================================

    #--------------------------------------------------------------------------------
    def __init__(self, root, wells=True, only_active=True, scale=(1, -1, -5), edges=True,
                 size=None, culling=None, title=None):    # Plotter
    #--------------------------------------------------------------------------------
        self.root = root
        self.egrid = EGRID_file(root)
        self.show_edges = edges
        self.culling = culling
        #self.var = None #var
        #self.limit = limit
        self.plotter = BackgroundPlotter(window_size=size, allow_quit_keypress=True, title=title, multi_samples=4,
                                         line_smoothing=True, point_smoothing=True)
        self.wells = wells
        self.only_active = only_active
        self.scale = scale
        self.ijk = None
        #self.dim = None
        #self.limit = None
        self.dim = None
        self.tube_opacity = None
        self.tube_size = None
        self.tube_height = None
        self.tube_radius = None
        self.sphere_radius = None
        self.well_grid_opacity = None
        self.res_grid_opacity = None
        self.label_fontsize = 12
        self.arrow_scale = None
        # self.start = None
        # self.stop = None
        # self.step = 1
        # self.rft = None
        # self.unrst = None
        self.grid_box = None
        # self.box_mask = None
        # self.well_mask = None
        # self.res_mask = None
        # self.wells = None
        #self.active_wells = None
        self.grid = None
        self.labels = []
        self.tubes = {}
        self.datestring = None
        #self.tube_grid = {}
        # self.well_actor = {}
        #self.well_grid = None
        self.grid_mask = []
        #self.well_off = {}
        self.set_tube_values()
        self.set_grid_values()

    #--------------------------------------------------------------------------------
    def set_tube_values(self, opacity=0.75, size=10, height=700, radius=35, sphere=45, 
                        fontsize=10, arrow_scale=400):
    #--------------------------------------------------------------------------------
        self.tube_opacity = opacity
        self.tube_size = size
        self.tube_height = height
        self.tube_radius = radius
        self.sphere_radius = sphere
        self.tube_fontsize = fontsize
        self.arrow_scale = arrow_scale

    #--------------------------------------------------------------------------------
    def set_grid_values(self, well_opacity=0.75, res_opacity=1):            # Plotter
    #--------------------------------------------------------------------------------
        self.well_grid_opacity = well_opacity
        self.res_grid_opacity = res_opacity

    #--------------------------------------------------------------------------------
    def make_grid_and_masks(self, ijk=None):                                # Plotter
    #--------------------------------------------------------------------------------
        self.dim = self.egrid.nijk()
        self.ijk = ijk or [(0,d) for d in self.dim]
        self.grid_box = [slice(*dir) for dir in self.ijk]
        # Mask out the box defined by the ijk-tuple 
        box_mask = zeros(self.dim, dtype=bool)
        box_mask[*self.grid_box] = True
        # Create grid
        self.grid = self.egrid.grid(*self.ijk, self.scale)
        # Create well and reservoir masks
        well_mask = zeros(self.dim, dtype=bool)
        if self.wells:
            wellpos = self.add_wells()
            well_mask[*zip(*wellpos)] = True
            well_mask *= box_mask
        # Exclude well-cells from the reservoir grid
        res_mask = well_mask == False
        res_mask *= box_mask
        return res_mask, well_mask

    #--------------------------------------------------------------------------------
    def add_wells(self, tube_opacity=0.75):                                  # Plotter
    #--------------------------------------------------------------------------------
        #self.only_active = only_active
        inp = IX_input(self.root)
        wname, wtype = zip(*inp.wells())
        wpos = inp.wellpos(*wname)
        if self.only_active:
            wells_inside = [n for n,cells in enumerate(wpos) if any_cell_in_box(cells, self.ijk)]
            wname, wtype, wpos = zip(*((wname[n], wtype[n], wpos[n]) for n in wells_inside))
        wellpos = list(flatten(wpos))
        # Create tubes and add to plotter
        for name, typ, pos in zip(wname, wtype, wpos):
            tube_grid = self.add_well_tube(name, typ, pos)
            act = self.plotter.add_mesh(tube_grid, opacity=tube_opacity)
            #act.mapper.dataset = act.mapper.dataset.flip_z()
            #act.mapper.dataset = act.mapper.dataset.flip_y()
            self.tubes[(name, typ)] = act
        return wellpos


    #--------------------------------------------------------------------------------
    def active_wells_iter(self):
    #--------------------------------------------------------------------------------
        # Get active wells
        rft = RFT_file(self.root)
        if self.only_active:
            active_wells = rft.active_wells()
        else:
            inp = IX_input(self.root)
            all_wells = inp.wellnames()
            active_wells = repeat(all_wells)
        return active_wells


    #--------------------------------------------------------------------------------
    def add_well_tube(self, name, typ, pos, fontsize=12, arrow_scale=400, sphere_radius=45, 
                      tube_height=700, tube_radius=35):                                # Plotter
    #--------------------------------------------------------------------------------
        #print(name, typ, pos)
        cell_centers = npmean(list(self.egrid.cell_corners(pos)), axis=1)*nparray(self.scale)
        tgrid = PolyData()
        tip = cell_centers[0].copy()
        tip[2] = self.grid.bounds[-1] + tube_height
        self.labels.append(Label(name, position=tip + (0, 0, 0.5*arrow_scale), size=fontsize))
        dir = -1 if 'INJ' in typ else 1
        tgrid += Arrow(tip - (0, 0, 0.5*arrow_scale*dir), (0, 0, dir), scale=arrow_scale)
        tip[2] -= 0.6*arrow_scale
        for center in cell_centers:
            tgrid += Sphere(center=center, radius=sphere_radius)
            A = center - tip
            length = npnorm(A)
            # Center of cylinder is the midpoint between the center two cells
            tgrid += Cylinder(center=npmean((tip, center), axis=0), direction=A/length, radius=tube_radius, height=length)
            # tip is the start of the new tube (or end of the current tube)            
            tip = center
        return tgrid

    #--------------------------------------------------------------------------------
    def add_grid_from_mask(self, mask, varname, scalar, limit, opacity):                                  # Plotter
    #--------------------------------------------------------------------------------
        grid = self.grid.extract_cells(mask[*self.grid_box].flatten())
        grid[varname] = scalar[mask].flatten()
        #grid['opacity'] = opacity[mask].flatten()
        grid.set_active_scalars(varname)
        act = self.plotter.add_mesh(grid, scalars=varname, lighting=False, show_edges=self.show_edges, 
                                    cmap='jet', clim=limit, opacity=opacity, culling=self.culling)
        #act.mapper.dataset = act.mapper.dataset.flip_z()
        #act.mapper.dataset = act.mapper.dataset.flip_y()
        self.grid_mask.append((grid, mask))

    #--------------------------------------------------------------------------------
    def plot(self, varname, ijk=None, startdate=None, start=None, stop=None, step=1, limit=None):                                  # Plotter
    #--------------------------------------------------------------------------------
        """
        startdate : (year, month, day)
        """
        #self.var = var
        #self.ijk = ijk or [(0,d) for d in self.dim]
        unrst = UNRST_file(self.root)
        if startdate:
            start = unrst.section(date=startdate)
        celldata = unrst.cellarray(varname, start=start, stop=stop, step=step)
        active_wells = islice(self.active_wells_iter(), start, stop, step)
        # Get grid-data for the first plot
        data = next(celldata)
        res_mask, well_mask = self.make_grid_and_masks(ijk)
        scalar = getattr(data, varname)
        #opacity = ones(scalar.shape)
        self.add_grid_from_mask(res_mask, varname, scalar, limit, opacity=self.res_grid_opacity)
        if self.wells:
            # Add grid of well cells
            self.add_grid_from_mask(well_mask, varname, scalar, limit, opacity=self.well_grid_opacity)
            # Get well-data for the first plot
            time, act_wells = next(active_wells)
            self.update_tubes(act_wells)
            # Add labels
            for label in self.labels:
                self.plotter.add_actor(label)
        # Plot features
        #self.plotter.view_yz()
        self.plotter.view_xy()
        self.plotter.show_axes()
        self.update_datestring(data.date)

        def plot():
            for data, (time, act_wells) in zip(celldata, active_wells):
                if self.wells:
                    self.update_tubes(act_wells)
                scalar = getattr(data, varname)
                self.update_scalar(varname, scalar)
                if limit is None:
                    self.update_range(scalar)
                self.update_datestring(data.date)
                sleep(0.25)

        thread = Thread(target=plot)
        thread.start()

    #--------------------------------------------------------------------------------
    def update_range(self, scalar):                                  # Plotter
    #--------------------------------------------------------------------------------
        #print([(i, grid) for i,(grid,_) in enumerate(self.grid_mask)])
        min_max = ((scalar[mask].min(), scalar[mask].max()) for _,mask in self.grid_mask)
        _min, _max = zip(*min_max)
        if self.plotter.mapper:
            self.plotter.mapper.scalar_range = (min(_min), max(_max))
        #self.plotter.update_scalar_bar_range((min(_min), max(_max)))
        

    #--------------------------------------------------------------------------------
    def update_tubes(self, active_wells):                                  # Plotter
    #--------------------------------------------------------------------------------
        color = {'WATER_INJECTOR':'blue', 'GAS_INJECTOR':'green', 'PRODUCER':'red'}
        for (name,typ), tube in self.tubes.items():
            if name in active_wells:
                tube.prop.color = color[typ]
            else:
                tube.prop.color = 'white'

    #--------------------------------------------------------------------------------
    def update_scalar(self, varname, scalar):                                  # Plotter
    #--------------------------------------------------------------------------------
        for grid, mask in self.grid_mask:
            grid[varname] = scalar[mask].flatten()

    #--------------------------------------------------------------------------------
    def update_datestring(self, date, size=10):                                  # Plotter
    #--------------------------------------------------------------------------------
        if self.datestring:
            self.datestring.SetVisibility(False)
        self.datestring = self.plotter.add_text(str(date).split()[0], font_size=size)
