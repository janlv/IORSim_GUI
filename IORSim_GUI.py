#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# importing libraries 
import os
#from PySide6.QtWidgets import QStatusBar, QDialog, QTextEdit, QWidget, QMainWindow, QApplication, QLabel, QPushButton, QGridLayout, QVBoxLayout, QHBoxLayout, QLineEdit, QPlainTextEdit, QDialogButtonBox, QCheckBox, QToolBar, QProgressBar, QGroupBox, QComboBox, QFrame, QFileDialog, QMessageBox
#from PySide6.QtGui import  QAction, QActionGroup, QColor, QColorConstants, QFont, QIcon, QSyntaxHighlighter, QTextCharFormat, QTextCursor 
#from PySide6.QtCore import QObject, Signal, Slot, QRunnable, QThreadPool, Qt, QRegularExpression
#from PySide2.QtWidgets import QDialogButtonBox, QStatusBar, QDialog, QWidget, QMainWindow, QApplication, QLabel, QPushButton, QGridLayout, QVBoxLayout, QHBoxLayout, QLineEdit, QPlainTextEdit, QDialogButtonBox, QCheckBox, QAction, QActionGroup, QToolBar, QProgressBar, QGroupBox, QComboBox, QFrame, QFileDialog, QMessageBox
#from PySide2.QtGui import QFont, QIcon, QSyntaxHighlighter, QTextCharFormat, QTextCursor, QColor 
#from PySide2.QtCore import QObject, Signal, Slot, QRunnable, QThreadPool, Qt, QRegExp
from PyQt5.QtWidgets import QStatusBar, QDialog, QWidget, QMainWindow, QApplication, QLabel, QPushButton, QGridLayout, QVBoxLayout, QHBoxLayout, QLineEdit, QPlainTextEdit, QDialogButtonBox, QCheckBox, QAction, QActionGroup, QToolBar, QProgressBar, QGroupBox, QComboBox, QFrame, QFileDialog, QMessageBox
from PyQt5.QtGui import QColor, QFont, QIcon, QSyntaxHighlighter, QTextCharFormat, QTextCursor 
from PyQt5.QtCore import QObject, pyqtSignal as Signal, pyqtSlot as Slot, QRunnable, QThreadPool, Qt, QRegExp
import sys, traceback
from time import sleep
from pathlib import Path
from matplotlib.colors import to_rgb as colors_to_rgb
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from numpy import genfromtxt, asarray 
from collections import namedtuple
import shutil
import warnings
import copy

from ior2ecl import simulation, main as ior2ecl_main
from IORlib.utils import Progress, flat_list, get_keyword, get_substrings, is_file_ignore_suffix_case, return_matching_string, delete_all, file_contains, upper_and_lower
from IORlib.ECL import get_tsteps, unfmt_file
import GUI_icons

gui_dir = Path('GUI')
case_dir = gui_dir/'cases'
input_file = gui_dir/'input.txt'
settings_file = gui_dir/'settings.txt'

default_font = 'default'
default_size = 10
default_weight = 50

blue   = QColor(31,119,180)  #1f77b4 
orange = QColor(255,127,14)  #ff7f0e  
green  = QColor(44,160,44)   #2ca02c
red    = QColor(214,39,40)   #d62728
violet = QColor(148,103,189) #9467bd
brown  = QColor(140,86,75)   #8c564b
pink   = QColor(227,119,194) #e377c2
gray   = QColor(127,127,127) #7f7f7f
yellow = QColor(188,189,34)  #bcbd22
turq   = QColor(23,190,207)  #17becf

#-----------------------------------------------------------------------
def open_file_dialog(win, text, filetype):
#-----------------------------------------------------------------------
    options = QFileDialog.Options() | QFileDialog.DontUseNativeDialog
    fileName, _ = QFileDialog.getOpenFileName(win, text, "", filetype, options=options)
    return fileName


#-----------------------------------------------------------------------
def create_action(win, text=None, shortcut=None, tip=None, func=None, icon=None, **kwargs):
#-----------------------------------------------------------------------
    act = QAction(text, win, **kwargs)
    if shortcut:
        act.setShortcut(shortcut)
    if tip:
        act.setStatusTip(tip)
    if icon:
        #act.setIcon(QIcon(str(icon_path/Path(icon))))
        act.setIcon(QIcon(':'+icon)) #QIcon(str(icon_path/Path(icon))))
    act.triggered.connect(func)
    #act.changed.connect(func)
    return act


#-----------------------------------------------------------------------
def get_species_iorsim(root):
#-----------------------------------------------------------------------
    file = f'{root}.trcinp'
    species = get_keyword(file, keyword='\*solution')
    if species:
        species = species[0][1::2]
    else:
        # Read old input format
        species = flat_list(get_keyword(file, keyword='\*SPECIES'))
    return species
    # species = []
    # with open(str(root)+'.trcinp') as f:
    #     for line in f:
    #         if line.startswith('*SPECIES'):
    #             (kw, specie) = line.split()
    #             species.append(specie)
    # return species
                

#-----------------------------------------------------------------------
def get_wells_iorsim(root):
#-----------------------------------------------------------------------
    file = f'{root}.trcinp'
    out_wells = flat_list(get_keyword(file, keyword='\*PRODUCER'))
    in_wells = flat_list(get_keyword(file, keyword='\*INJECTOR'))
    if not out_wells or not in_wells:
        # Read old input format
        out_wells = get_keyword(file, keyword='\*OUTPUT')[0][1:]
        w = get_keyword(file, keyword='\*WELLSPECIES')[0]
        in_wells = w[1:1+int(w[0])]
    return out_wells, in_wells
    # def get_wells(num, wells, line):
    #     if num is not None:
    #         if num==0:
    #             # first line is number of wells
    #             num = int(line.split()[0])
    #         else:
    #             num -= 1
    #             wells.append(line.split()[0])
    #             if num==0:
    #                 num = None
    #     return num, wells
                    
    # out_well = []
    # in_well = []
    # if root:
    #     n_out = n_in = None
    #     if not Path(str(root)+'.trcinp').is_file():
    #         raise SystemError('trcinp-file is missing')
    #     with open(str(root)+'.trcinp') as f:
    #         for line in f:
    #             if line.lstrip().startswith('#') or line.isspace():
    #                 continue
    #             if line.lstrip().startswith('*OUTPUT'):
    #                 n_out = 0
    #                 continue
    #             if line.lstrip().startswith('*WELLSPECIES'):
    #                 n_in = 0
    #                 continue
    #             n_out, out_well = get_wells(n_out, out_well, line)
    #             n_in, in_well = get_wells(n_in, in_well, line)

    # #print(out_well, in_well)
    # return out_well, in_well
    
#-----------------------------------------------------------------------
def get_eclipse_well_yaxis_fluid(root):
#-----------------------------------------------------------------------
    fil = str(root)+'.DATA'
    if not Path(fil).is_file():
        raise SystemError(fil + ' is missing!')
    summary = well = False
    vars = []
    wells = []
    with open(fil) as f:
        for line in f:
            if line.lstrip().startswith('--') or line.isspace():
                continue
            if line.lstrip().upper().startswith('SUMMARY'):
                summary = True
                continue
            if line.lstrip().upper().startswith('SCHEDULE'):
                break
            if summary:
                kw = line.strip()
                #print(kw)
                if kw[0] in ('F','R'):
                    vars.append(kw)
                    #print('add: '+kw)
                if kw[0] == 'W':
                    vars.append(kw)
                    well = True
                    continue
                if well:
                    if '/' in kw:
                        kw = kw[:-1].strip()
                        well = False
                    if ',' in kw:
                        kw = kw.split(',')
                    elif ' ' in kw:
                        kw = kw.split()
                    elif len(kw)>0:
                        kw = (kw,)
                    else:
                        kw = []
                    for k in kw:
                        wells.append(k.strip())                  
    vars = list(set(vars))
    #print('vars',vars)
    if len(vars)==0:
        raise SystemError('No variables in SUMMARY section.'+
                          '\n\nEclipse plotting disabled.')
    wells = list(set(wells))
    wells = [w.replace("'","") for w in wells]
    #print(wells)
    F = {'O':'Oil', 'W':'Water', 'G':'Gas'}
    P = {'P':'prod', 'R':'rate'}
    fluids = list(set([F[v[1]] for v in vars if v[1] in F.keys()]))
    yaxis = list(set([P[v[-1]] for v in vars if v[-1] in P.keys()]))
    if any([v[0]=='F' and v[-1] in ('P','R') for v in vars]):
        wells.insert(0, 'Field')
    #fluids.insert(-1, 'Temp')
    return wells, yaxis, fluids 
            
    
#-----------------------------------------------------------------------
def show_message(window, kind, text='', extra='', detail=None):
#-----------------------------------------------------------------------
    kind = kind.lower()
    if kind=='info':
        title = 'Information'
        icon = QMessageBox.Information
    elif kind=='question':
        title = 'Question'
        icon = QMessageBox.Information
    elif kind=='warning':
        title = 'Warning'
        icon = QMessageBox.Warning
    elif kind=='error':
        title = 'Error'
        icon = QMessageBox.Critical
    else:
        raise SystemError(f'Unrecognized kind-option in show_message(): {kind}')
    msg = QMessageBox(window)
    msg.setWindowTitle(title)
    msg.setIcon(icon)
    msg.setText(text)
    msg.setInformativeText(extra)
    if detail:
        msg.setdetailedText(detail)
    msg.setStandardButtons(QMessageBox.Ok)  # | QMessageBox.Cancel)
    #msg.exec_()
    msg.show()


#-----------------------------------------------------------------------
def delete_all_widgets_in_layout(layout):
#-----------------------------------------------------------------------
    '''
   Deletes all widgets in a given layout, and all its
   nested layout recursively

    '''
    for i in reversed(range(layout.count())):
        #print(i, type(layout))
        widget = layout.itemAt(i).widget()
        if widget:
            #print(type(widget))
            layout.removeWidget(widget)
            widget.deleteLater()
        else:
            layout2 = layout.itemAt(i).layout()
            if layout2:
                #print(type(layout2))
                delete_all_widgets_in_layout(layout2)

                
#-----------------------------------------------------------------------
def to_rgb(color):
#-----------------------------------------------------------------------
    #return 'rgb{}'.format( tuple(asarray(asarray(matplotlib.colors.to_rgb(color))*255, dtype='int')) )
    return 'rgb{}'.format( tuple(asarray(asarray(colors_to_rgb(color))*255, dtype='int')) )
    

#-----------------------------------------------------------------------
def set_checkbox(box, value, block_signal=True):
#-----------------------------------------------------------------------
    box.blockSignals(block_signal)
    box.setChecked(value)
    #box.setEnabled(value)
    box.blockSignals(not block_signal)


#-----------------------------------------------------------------------
def str_to_bool(s):
#-----------------------------------------------------------------------
    if s.lower() == 'true':
        return True
    elif s.lower() == 'false':
        return False
    else:
        return s
        #raise ValueError("Valid case-insensitive strings in str_to_bool() are 'true','false','1','0', got " + s)

#===========================================================================
class VLine(QFrame):
#===========================================================================
    def __init__(self):
        super(VLine, self).__init__()
        self.setFrameShape(self.VLine|self.Sunken)

#===========================================================================
class HLine(QFrame):
#===========================================================================
    def __init__(self):
        super(HLine, self).__init__()
        self.setFrameShape(self.HLine|self.Sunken)


#===========================================================================
class worker_signals(QObject):                                              
#===========================================================================
    finished = Signal()
    result = Signal(int)
    error = Signal(tuple)
    #progress = Signal(int)
    progress = Signal(tuple)
    plot = Signal()
    #show_message = Signal(tuple)
    show_message = Signal(str)
    status_message = Signal(str)
    stop = Signal()
    
#===========================================================================
class base_worker(QRunnable):                                              
#===========================================================================
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs
        self.signals = worker_signals()
        # Signal alias
        self.status_message = self.signals.status_message.emit
        self.show_message = self.signals.show_message.emit
        self.update_progress = self.signals.progress.emit
        self.update_plot = self.signals.plot.emit

    @Slot()
    #-----------------------------------------------------------------------
    def run(self):
    #-----------------------------------------------------------------------
        try:
            result = self.runnable()
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit()


#===========================================================================
class sim_worker(base_worker):                                              
#===========================================================================
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sim = None
        self.signals.stop.connect(self.stop_sim)
        self.days_box = kwargs.get('days_box') or None

    #-----------------------------------------------------------------------
    def current_run(self):
    #-----------------------------------------------------------------------
        if self.sim:
            return self.sim.current_run 

    #-----------------------------------------------------------------------
    def stop_sim(self):
    #-----------------------------------------------------------------------
        if self.sim:
            self.sim.cancel()


    @Slot()
    #-----------------------------------------------------------------------
    def runnable(self):
    #-----------------------------------------------------------------------
        self.progress_min = None
        #------------------------------------
        def progress(run=None, value=None, min=None, n0=None):
        #------------------------------------
            if min is not None:
                self.progress_min = min
            if value == 0:
                self.progress_min = None
            if run and value is None:
                value = run.t
            self.update_progress((int(value), min, n0))
        #------------------------------------
        def status(run=None, value=None, mode=None, **x):
        #------------------------------------
            if run and not value:
                count = f'{int(run.t)}'
                if self.progress_min:
                    count = f'({int(self.progress_min)} + {int(run.t-self.progress_min)})'
                value = f'{run.name}   {count} / {run.T} days'
                if mode == 'forward':
                    value = run.name + ' ' + value
            self.status_message(value)
        #------------------------------------
        def plot(run=None, value=None):
        #------------------------------------
            self.update_plot()
        #------------------------------------
        def message(text=None, **kwargs):
        #------------------------------------
            text and self.show_message(text)

        self.sim = simulation(status=status, progress=progress, plot=plot, message=message, **self.kwargs)
        self.days_box.setText(str(int(self.sim.get_time())))
        result, msg = self.sim.run()
        self.show_message(msg)
        return result


#===========================================================================
class Mpl_canvas(FigureCanvasQTAgg):                                              
#===========================================================================
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        #self.axes = fig.add_subplot(subplot)
        super(Mpl_canvas, self).__init__(self.fig)


                
#===========================================================================
class User_input(QDialog):                                              
#===========================================================================
    #-----------------------------------------------------------------------
    def __init__(self, parent=None, title=None, head=None, label=None, text=None, delete_src=False):
    #-----------------------------------------------------------------------
        #super(User_input, self).__init__(*args, **kwargs)
        super(User_input, self).__init__(parent)
        self.setWindowTitle(title)
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.func = None
        intro = QLabel()
        intro.setText(head)
        intro.setWordWrap(True)
        self.layout.addWidget(intro)
        ### input
        self.inp_layout = QHBoxLayout()
        self.layout.addLayout(self.inp_layout)
        self.lbl = QLabel(label)
        self.lbl.setStyleSheet('padding: 20px')
        self.var = QLineEdit(self)
        self.var.setFixedWidth(250)
        self.var.setText(text)
        self.inp_layout.addWidget(self.lbl)
        self.inp_layout.addWidget(self.var)
        ### buttons
        self.btn_layout = QHBoxLayout()
        self.layout.addLayout(self.btn_layout)
        buttons = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        self.button_box = QDialogButtonBox(buttons)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        self.btn_layout.addWidget(self.button_box)

    #-----------------------------------------------------------------------
    def set_func(self, func):
    #-----------------------------------------------------------------------
        self.func = func
        
    #-----------------------------------------------------------------------
    def accept(self):
    #-----------------------------------------------------------------------
        if self.func:
            self.func()
        super().accept()

        
#===========================================================================
class Settings(QDialog):                                              
#===========================================================================
    #-----------------------------------------------------------------------
    def __init__(self, parent=None, file=None):
    #-----------------------------------------------------------------------
        super(Settings, self).__init__(parent)
        self.setWindowTitle('Settings')
        self.setMinimumSize(400,300)
        self.get = {}
        self.set = {}
        self.required = []
        self.default = {'eclrun'      : 'eclrun', 
                        'unrst'       : True, 
                        'rft'         : True, 
                        'dt'          : '1', 
                        'convert'     : True,
                        'del_convert' : True,
                        'merge'       : True,
                        'del_merge'   : True,
                        'stop_child'  : True,
                        'pause'       : '0.5'}
        self.abs_path = False
        self.initUI()
        self.file = Path(file) 
        self.load()
        
        
    #-----------------------------------------------------------------------
    def initUI(self):                                             # settings
    #-----------------------------------------------------------------------
        grid = QGridLayout()
        self.setLayout(grid)
        grid.setColumnStretch(0,15) 
        grid.setColumnStretch(1,70) 
        grid.setColumnStretch(2,15) 

        tool_tip = {'iorsim'      : 'Path to the IORSim executable',
                    'iorarg'      : 'Additional arguments passed to IORSim',
                    'eclrun'      : "Eclipse command, default is 'eclrun'",
                    'unrst'       : 'Check that the UNRST-file is properly flushed before suspending Eclipse',
                    'rft'         : 'Check that the RFT-file is properly flushed before suspending Eclipse',
                    'dt'          : 'Eclipse starts before IORSim and the first Eclipse TSTEP is given here. Subsequent TSTEPs are provided by IORSim via the interface-file.',
                    'convert'     : 'Convert IORSim formatted output to unformatted format (readable by ResInsight), and also add output from the Eclipse run',
                    'del_convert' : 'Delete the original formatted IORSim output if the convert is successful',
                    'merge'       : 'Merge the unformatted output from Eclipse and IORSim into one file',
                    'del_merge'   : 'Delete the separate output-files from Elipse and IORSim if the merge is successful',
                    'stop_child'  : 'Stop both Eclipse parent and child process to increase stability (~5% performance drop)',
                    'pause'       : 'A short break between Eclipse and IORSim runs might be necessary to improve the stability of backward runs'}

        ### IORSim executable
        n = 0
        var, text = 'iorsim', 'IORSim program'
        # label, setting, button = self.new_line()
        widget = self.new_line(var=var, text=text, required=True, open_func=self.open_ior_prog)
        grid.addWidget(widget[0] , n, 0)
        # layout given as (row, col, rowspan, colspan)
        grid.addWidget(widget[1] , n, 1)
        grid.addWidget(widget[2] , n, 2)
        widget[1].setToolTip(tool_tip[var])
        self.iorsim = widget[1]
            
        # ### IORSim args
        # show = False
        # var, text = 'iorarg', 'IORSim arguments'
        # widget = self.new_line(var=var, text=text, required=False)
        # #for w in widget[1:]:
        # widget[1].setToolTip(tool_tip[var])
        # self.iorarg = widget[1]
        # if show:
        #     n += 1
        #     self.layout.addWidget(widget[0] , n, 0)
        #     self.layout.addWidget(widget[1] , n, 1)
        
        ### Eclipse executable
        n += 1
        var, text = 'eclrun', 'Eclipse program'
        widget = self.new_line(var=var, text=text, required=True, open_func=self.open_ecl_prog)
        grid.addWidget(widget[0] , n, 0)
        grid.addWidget(widget[1] , n, 1)
        grid.addWidget(widget[2] , n, 2)
        #for w in widget[1:]:
        widget[1].setToolTip(tool_tip[var])
        self.eclrun = widget[1]
        
        ### Space
        n += 1
        grid.addWidget(QLabel(), n, 0)

        ### Output options
        n += 1
        label = QLabel()
        label.setText('Output options')
        grid.addWidget(label, n, 0)

        ### Convert
        n += 1
        layout = QGridLayout()
        grid.addLayout(layout, n, 1)
        var, text = 'convert', 'Convert to unformatted output'
        self.convert = self.new_box(var=var, text=text)
        self.convert.setToolTip(tool_tip[var])
        layout.addWidget(self.convert, 0, 0)
        var, text = 'del_convert', 'Delete original after convert'
        self.del_convert = self.new_box(var=var, text=text)
        self.del_convert.setToolTip(tool_tip[var])
        layout.addWidget(self.del_convert, 0, 1)

        ### Merge
        n += 1
        var, text = 'merge', 'Merge Eclipse and IORSim output'
        self.merge = self.new_box(var=var, text=text)
        self.merge.setToolTip(tool_tip[var])
        layout.addWidget(self.merge, 1, 0)
        var, text = 'del_merge', 'Delete originals after merge'
        self.del_merge = self.new_box(var=var, text=text)
        self.del_merge.setToolTip(tool_tip[var])
        layout.addWidget(self.del_merge, 1, 1)

        ### Space
        n += 1
        grid.addWidget(QLabel(), n, 0)

        ### Backward options
        n += 1
        label = QLabel()
        label.setText('Backward options')
        grid.addWidget(label, n, 0)

        # initial timestep
        n += 1
        layout = QGridLayout()
        grid.addLayout(layout, n, 1) # (layout, row, col, rowspan, colspan)
        var, text = 'dt', 'Initial timestep passed to Eclipse '
        lbl_dt, self.dt = self.new_line(var=var, text=text, required=True)
        self.dt.setFixedWidth(40)
        self.dt.setToolTip(tool_tip[var])
        lbl_dt.setToolTip(tool_tip[var])
        layout.addWidget(lbl_dt, 0, 0)
        box = QHBoxLayout()
        layout.addLayout(box, 0, 1)
        box.addWidget(self.dt)
        txt = QLabel()
        txt.setText('TSTEP')
        box.addWidget(txt)

        # n += 1
        # pause between runs
        var, text = 'pause', 'Pause before IORSim resumes '
        lbl_pause, self.pause = self.new_line(var=var, text=text)
        # self.pause.setToolTip(tool_tip[var])
        # lbl_pause.setToolTip(tool_tip[var])
        # self.pause.setFixedWidth(40)
        # layout.addWidget(lbl_pause, 1, 0)
        # box = QHBoxLayout()
        # layout.addLayout(box, 1, 1)
        # box.addWidget(self.pause)
        # sec = QLabel()
        # sec.setText('seconds')
        # box.addWidget(sec)

        n += 1
        # unrst file check
        lbl = QLabel()
        lbl.setText('Eclipse outfile checks ')
        layout.addWidget(lbl, 2, 0)    
        var, text = 'unrst', 'UNRST-file'
        self.unrst = self.new_box(var=var, text=text)
        self.unrst.setToolTip(tool_tip[var])
        box = QHBoxLayout()
        layout.addLayout(box, 2, 1)
        box.addWidget(self.unrst)
        var, text = 'rft', 'RFT-file'
        self.rft = self.new_box(var=var, text=text)
        self.rft.setToolTip(tool_tip[var])
        box.addWidget(self.rft)

        n += 1
        # stop child process
        lbl = QLabel()
        lbl.setText('Suspend Eclipse child-process')
        layout.addWidget(lbl, 3, 0)    
        var, text = 'stop_child', ''
        self.stop_child = self.new_box(var=var, text=text)
        [a.setToolTip(tool_tip[var]) for a in (lbl, self.stop_child)]
        self.stop_child.setEnabled(False)
        layout.addWidget(self.stop_child, 3, 1)

        # Space
        n += 1
        grid.addWidget(QLabel(), n, 0)

        ### OK / Cancel buttons
        n += 1
        #yes_no = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        yes_no = QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        self.yes_no_btns = QDialogButtonBox(yes_no)
        grid.addWidget(self.yes_no_btns, n, 0, 1, 3)
        self.yes_no_btns.accepted.connect(self.on_OK_click)
        self.yes_no_btns.rejected.connect(self.reject)
        
    #-----------------------------------------------------------------------
    def new_box(self, var=None, text='', required=False):
    #-----------------------------------------------------------------------
        if required:
            self.required.append(var)
        box = QCheckBox(text)
        self.get[var] = box.isChecked
        self.set[var] = box.setChecked
        self.set_default(var)
        return box

    #-----------------------------------------------------------------------
    def new_line(self, var=None, text='', required=False, open_func=None):
    #-----------------------------------------------------------------------
        if required:
            self.required.append(var)
        label = QLabel()
        label.setText(text)
        line = QLineEdit()
        #line.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding )
        #line.setMinimumSize(100,25)
        self.get[var] = line.text 
        self.set[var] = line.setText
        self.set_default(var)
        button = False
        if open_func:
            #button = QDialogButtonBox(QDialogButtonBox.Open)
            button = QDialogButtonBox(QDialogButtonBox.StandardButton.Open)
            button.clicked.connect(open_func)
        if button:
            return label, line, button
        else:
            return label, line

        
    #-----------------------------------------------------------------------
    def set_default(self, var):
    #-----------------------------------------------------------------------
        if var in self.default:
            self.set[var](self.default[var])
        
    #-----------------------------------------------------------------------
    def open_ior_prog(self):
    #-----------------------------------------------------------------------
        fname = open_file_dialog(self, 'Locate IORSim program', 'All Files (*)')
        if fname:
            if not self.abs_path:
                try:
                    fname = str(Path(fname).relative_to(Path.cwd()))
                except ValueError:
                    pass
            self.set['iorsim'](fname)
            
    #-----------------------------------------------------------------------
    def open_ecl_prog(self):
    #-----------------------------------------------------------------------
        fname = open_file_dialog(self, 'Locate eclrun program', 'All Files (*)')
        if fname:
            self.set['eclrun'](fname)

    #-----------------------------------------------------------------------
    def on_OK_click(self):
    #-----------------------------------------------------------------------
        if self.save():
            self.done(1)
        
    #-----------------------------------------------------------------------
    def save(self):                                      # settings
    #-----------------------------------------------------------------------
        self.file.touch(exist_ok=True)
        with open(self.file, 'w') as f:
            f.write('# This is a settings-file for ior2ecl_GUI.py, do not edit.\n')
            for var,val in self.get.items():
                if var in self.required and len(val())==0:
                    show_message(self, 'error', text=var+' cannot be empty!')
                    return False
                line = '{} {}'.format(var,val())
                f.write(line+'\n')
                #print(line)
        return True
    
    #-----------------------------------------------------------------------
    def load(self):                                      # settings
    #-----------------------------------------------------------------------
        if self.file.is_file():
            with open(self.file) as f:
                for line in f:
                    if line.lstrip().startswith('#'):
                        continue
                    try:
                        (var, val) = line.split()
                    except ValueError:
                        var = line.rstrip()
                        val = ''
                    else:
                        self.set[var.strip()]( str_to_bool(val.strip()) )

                        
        
#===========================================================================
class main_window(QMainWindow):                                    # main_window 
#===========================================================================

    #-----------------------------------------------------------------------
    def __init__(self, *args, **kwargs):                       # main_window
    #-----------------------------------------------------------------------
        super(main_window, self).__init__(*args, **kwargs)
        self.setWindowTitle('IORSim') 
        self.setGeometry(300, 100, 1200, 800)
        self.setMinimumHeight(600)
        self.setMinimumWidth(800)
        #self.setContentsMargins(2,2,2,2)
        self.setWindowIcon(QIcon(':program_icon'))
        self.font = QFont().defaultFamily()
        self.menu_fontsize = 7
        self.plot_lines = None
        self.data = {}
        self.plot_ref_data = {}
        self.ecl_boxes = {}
        self.ior_boxes = {}
        self.current_view = None
        self.days = None
        self.log_file = None
        self.unsmry = None
        self.worker = None
        self.convert = None
        self.max_3_checked = []
        self.plot_prop = {}
        self.view = False
        self.plot_ref = None
        self.progress = None
        self.vscroll = {}
        gui_dir.mkdir(exist_ok=True)
        self.settings = Settings(self, file=str(settings_file))
        self.casedir = case_dir 
        self.input_file = input_file #gui_dir/'input.txt'
        #self.input = {'root':None, 'ecl_days':None, 'dtecl':None, 'days':None, 'step':None, 'species':[], 'mode':None} #, 'case':None}
        self.input = {'root':None, 'ecl_days':None, 'days':None, 'step':None, 'species':[], 'mode':None}
        self.input_to_save = ['root','days','mode']
        self.load_input()
        self.initUI()
        self.set_input_field()
        self.threadpool = QThreadPool()
        self.show()

    #-----------------------------------------------------------------------
    def initUI(self):                                          # main_window
    #-----------------------------------------------------------------------
        self.create_actions()
        self.create_menus()
        self.create_toolbar()
        self.create_statusbar()
        self.create_central_widget()
        
    #-----------------------------------------------------------------------
    def create_actions(self):                                  # main_window
    #-----------------------------------------------------------------------
        ### actions
        self.set_act = create_action(self, text='&Settings', icon='gear', shortcut='Ctrl+S',
                                     tip='Edit settings', func=self.settings.open)
        self.start_act = create_action(self, text=None, icon='start', shortcut='Ctrl+R',
                                       tip='Run simulation', func=self.run_sim)
        self.stop_act = create_action(self, text=None, icon='stop', shortcut='Ctrl+E',
                                      tip='Stop simulation', func=self.killsim)
#        self.help_act = create_action(self, text='Keyboard shortcuts',  shortcut='',
#                                      tip='Display help', func=self.help_win.open_win)
        self.exit_act = create_action(self, text='&Exit', icon='control-power', shortcut='Ctrl+Q',
                                      tip='Exit application', func=self.quit)
        self.add_case_act = create_action(self, text='Add case...', icon='document--plus',
                                          func=self.add_case_from_file)
        self.dupl_case_act = create_action(self, text='Duplicate current case...', icon='document-copy',
                                           func=self.duplicate_current_case)
        self.rename_case_act = create_action(self, text='Rename current case...', icon='document-rename',
                                             func=self.rename_current_case)
        self.clear_case_act = create_action(self, text='Clear current case', icon='document',
                                            func=self.clear_current_case)
        self.delete_case_act = create_action(self, text='Delete current case', icon='document--minus',
                                             func=self.delete_current_case)
        self.plot_act = create_action(self, text='Plot', icon='guide', func=self.view_plot, checkable=True)
        self.plot_act.setChecked(True)
        self.ecl_inp_act = create_action(self, text='Eclipse input file', icon='document-e',
                                         func=self.view_eclipse_input, checkable=True)
        self.ior_inp_act = create_action(self, text='IORSim input file', icon='document-i',
                                         func=self.view_iorsim_input, checkable=True)
        self.chem_inp_act = create_action(self, text='IORSim geochem file', icon='document-g',
                                          func=self.view_geochem_input, checkable=True)
        self.schedule_file_act = create_action(self, text='IORSim schedule file', icon='document-i',
                                          func=self.view_schedule_file, checkable=True)
        self.ecl_log_act = create_action(self, text='Eclipse log file', icon='terminal',
                                         func=self.view_eclipse_log, checkable=True)
        self.ior_log_act = create_action(self, text='IORSim log file', icon='terminal',
                                         func=self.view_iorsim_log, checkable=True)
        self.py_log_act = create_action(self, text='Program log file', icon='terminal',
                                        func=self.view_program_log, checkable=True)
        # fwd = create_action(self, text='Forward', icon='',
        #                     func=self.mode_forward, checkable=True)
        # back = create_action(self, text='Backward', icon='',
        #                      func=self.mode_backward, checkable=True)
        # ecl = create_action(self, text='Eclipse', icon='',
        #                     func=self.mode_eclipse, checkable=True)
        # ior = create_action(self, text='IORSim', icon='',
        #                     func=self.mode_iorsim, checkable=True)
        # self.mode_act = {'forward':fwd, 'backward':back, 'eclipse':ecl, 'iorsim':ior}
        #self.show_manual_act = create_action(self, text='View user manual', icon='document', func=self.on_show_manual)
                
        
    #-----------------------------------------------------------------------
    def create_menus(self):                                          # main_window
    #-----------------------------------------------------------------------
        ### menu
        menu = self.menuBar()
        #self.setStyleSheet('QMainWindow::menuBar { padding: 10px; }')
        file_menu = menu.addMenu('&File')
        file_menu.addAction(self.add_case_act)
        file_menu.addAction(self.dupl_case_act) 
        file_menu.addAction(self.rename_case_act)
        file_menu.addAction(self.clear_case_act)
        file_menu.addAction(self.delete_case_act)
        file_menu.addSeparator()
        file_menu.addAction(self.exit_act)
        # mode_menu = menu.addMenu('&Mode')
        # self.mode_ag = QActionGroup(self)
        # for act in self.mode_act.values():
        #     mode_menu.addAction(act)
        #     self.mode_ag.addAction(act)
        edit_menu = menu.addMenu('&Edit')
        self.view_ag = QActionGroup(self)
        for act in (self.ecl_inp_act, self.ior_inp_act, self.chem_inp_act, self.schedule_file_act):
            edit_menu.addAction(act)
            self.view_ag.addAction(act)
        edit_menu.addSeparator()
        edit_menu.addAction(self.set_act)
        view_menu = menu.addMenu('&View')
        self.view_ag.addAction(self.plot_act)
        view_menu.addAction(self.plot_act)
        view_menu.addSeparator()
        for act in (self.ecl_log_act, self.ior_log_act, self.py_log_act):
            view_menu.addAction(act)
            self.view_ag.addAction(act)
        
        #help_menu = menu.addMenu('&Help')
        #help_menu.addAction(self.help_act)
        
        
    #-----------------------------------------------------------------------
    def create_toolbar(self):                                  # main_window
    #-----------------------------------------------------------------------
        ### toolbar
        self.toolbar = QToolBar('Toolbar')
        self.toolbar.setStyleSheet('QToolBar{spacing:15px; padding:5px;}')
        self.addToolBar(self.toolbar)
        #self.toolbar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toolbar.setStyleSheet('QToolButton { padding: 0px 0px 0px 0px}')
        self.create_toolbar_widgets()

    #-----------------------------------------------------------------------
    def create_toolbar_widgets(self):                           # main_window
    #-----------------------------------------------------------------------
        ### simulation controls
        widgets = {'run'    : QComboBox(),
                   'case'    : QComboBox(),
                   'days'    : QLineEdit(),
                   'compare' : QComboBox()} 
        tips = ('Set running mode',
                'Choose a case, or add a new from the Case-menu',
                'Set total time interval',
                'Compare current case to a previous case')
        #ql = QLabel()
        #ql.setText('')
        #ql.setFixedWidth(70)
        #ql.setStatusTip('Set simulation mode from the Mode menu')
        #self.toolbar.addWidget(ql)
        #self.mode_label = ql
        #self.toolbar.addSeparator()
        for i,(text,wid) in enumerate(list(widgets.items())[:3]):
            ql = QLabel()
            ql.setText(text.capitalize())
            ql.setStatusTip(tips[i])
            wid.setStatusTip(tips[i])
            self.toolbar.addWidget(ql)
            self.toolbar.addWidget(wid)
            self.toolbar.addSeparator()
        # Add start and stop buttons
        self.toolbar.addAction(self.start_act)
        self.toolbar.addAction(self.stop_act)
        self.toolbar.addSeparator()
        # last widget
        text, wid = list(widgets.items())[-1]
        i = len(widgets)-1
        ql = QLabel()
        ql.setText(text.capitalize())
        ql.setStatusTip(tips[i])
        wid.setStatusTip(tips[i])
        self.toolbar.addWidget(ql)
        self.toolbar.addWidget(wid)
        # mode
        self.modes = ['forward','backward','eclipse','iorsim']
        mode_names = ['Forward','Backward','Eclipse','IORSim']
        self.mode_cb = widgets['run']
        #self.mode_cb.setStyleSheet('QComboBox {min-width: 100px;}')
        self.mode_cb.addItems(mode_names)
        #self.mode_cb.setPlaceholderText('Choose mode')
        self.mode_cb.setCurrentIndex(-1)
        self.mode_cb.currentIndexChanged[int].connect(self.on_mode_select)
        # case
        self.case_cb = widgets['case']
        self.case_cb.setStyleSheet('QComboBox {min-width: 200px;}')
        self.case_cb.currentIndexChanged[int].connect(self.on_case_select)
        # steps
        self.days_box = widgets['days']
        self.days_box.setFixedWidth(80)
        self.days_box.setObjectName('days')
        self.days_box.textChanged[str].connect(self.on_input_change)
        # reference
        self.ref_case = widgets['compare']
        self.ref_case.setObjectName('compare')
        self.ref_case.setStyleSheet('QComboBox {min-width: 200px;}')
        self.ref_case.currentIndexChanged[int].connect(self.on_compare_select)
        self.ref_case.setProperty('lastitem',0)    

    #-----------------------------------------------------------------------
    def create_statusbar(self):                                          # main_window
    #-----------------------------------------------------------------------
        ### statusbar
        self.remaining_time = QLabel()
        self.remaining_time.setStyleSheet('QLabel {margin-top:10px; margin-bottom: 10px;}')
        self.update_remaining_time()
        self.progressbar = QProgressBar()
        self.progressbar.setStyleSheet('QProgressBar {max-width: 300px; max-height: 15px; padding: 0px;}\nQProgressBar::chunk {background-color: #2ca02c;}')
        self.progressbar.setFormat('')
        self.reset_progressbar()
        statusbar = QStatusBar()
        statusbar.setStyleSheet("QStatusBar { border-top: 1px solid lightgrey; }\nQStatusBar::item { border:None; };"); 
        self.messages = QLabel()
        statusbar.addPermanentWidget(self.messages)
        statusbar.addPermanentWidget(self.progressbar, stretch=0)
        statusbar.addPermanentWidget(self.remaining_time, stretch=0)
        self.setStatusBar(statusbar)


    #-----------------------------------------------------------------------
    def create_central_widget(self):                                          # main_window
    #-----------------------------------------------------------------------
        ### central widget
        # layout given as (row, col, rowspan, colspan)
        # self.position = {'input'       : (0, 0),
        #                  'ior_menu': (1, 0),
        #                  'ecl_menu': (2, 0),
        #                  'plot'        : (0, 1, 3, 1)}
        self.position = {'ior_menu' : (0, 0),
                         'ecl_menu' : (1, 0),
                         'plot'     : (0, 1, 2, 1)}
        self.layout = QGridLayout()
        self.layout.setContentsMargins(10,10,10,10)
        self.layout.setSpacing(10)
        self.layout.setColumnStretch(0,25)
        self.layout.setColumnStretch(1,75)
        self.layout.setRowStretch(0,50)
        self.layout.setRowStretch(1,50)
        #self.layout.setRowStretch(2,35)
        widget = QWidget()
        #widget.setStyleSheet('QWidget {border: 1px solid black}')
        widget.setLayout(self.layout)
        self.setCentralWidget(widget)
        # iorplot menu
        self.create_ior_menu()
        # eclplot menu
        self.create_ecl_menu()
        # plot
        self.create_plot_field()
        self.create_editor_field()
        #print(self.layout.columnCount(), self.layout.rowCount())
        #margins = self.contentsMargins()
        #margins.setBottom(0)    
        #self.setContentsMargins(0,0,0,0)        


    #-----------------------------------------------------------------------
    def set_input_field(self):
    #-----------------------------------------------------------------------
        ### set values from input-file or default
        days = 100
        ### case
        self.cases = self.read_case_dir()
        self.case = self.input.get('root')
        self.create_caselist(choose=self.case)
        ### number of steps
        #if self.input['nsteps']:
        #    nsteps = self.input['nsteps']
        #self.days_box.setText(str(nsteps))
        self.days_box.setText(str(self.input.get('days') or days))
        
        
    #-----------------------------------------------------------------------
    def save_input(self):                                   # main_window
    #-----------------------------------------------------------------------
        self.input_file.touch(exist_ok=True)
        with open(self.input_file, 'w') as f:
            f.write('# This is an input-file for ior2ecl_GUI.py, do not edit.\n')
            #for var,val in self.input.items():
            for var in self.input_to_save:
                line = '{} {}'.format(var,self.input.get(var) or '')
                f.write(line+'\n')
                #print('saved input: '+line)
        return True
    
    #-----------------------------------------------------------------------
    def load_input(self):                                   # main_window
    #-----------------------------------------------------------------------
        if self.input_file.is_file():
            with open(self.input_file) as f:
                for line in f:
                    if line.lstrip().startswith('#'):
                        continue
                    try:
                        (var, val) = line.split()
                    except ValueError:
                        var = line.rstrip()
                        val = None
                    finally:
                        try:
                            v = int(val) 
                        except (TypeError,ValueError):
                            v = val
                        finally:
                            self.input[var] = v
        #for var,val in self.input.items():
        #    print('{} = {} ({})'.format(var,val,type(val)))
                            
    #-----------------------------------------------------------------------
    def set_variables_from_casefiles(self):                # main_window
    #-----------------------------------------------------------------------
        inp = self.input
        #inp['dtecl'] = inp['ecl_days'] = inp['species'] = None
        inp['ecl_days'] = inp['species'] = None
        if inp['root']:
            #inp['dtecl']   = ior_input(var='dtecl', root=inp['root'])
            #inp['ecl_days'] = ECL_input_days_and_steps(inp['root'])[0]
            inp['ecl_days'] = int(sum(get_tsteps(inp['root']+'.DATA')))
            inp['species'] = get_species_iorsim(inp['root'])


    #-----------------------------------------------------------------------
    def set_plot_properties(self):                # main_window
    #-----------------------------------------------------------------------
        #colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

        species = enumerate(self.input['species'] or [])
        prop = {}
        prop['color'] = None
        prop['line'] = None
        prop['alpha'] = None
        species = self.input['species']
        if species:
            prop['color'] = {specie:colors[i%len(colors)] for i,specie in enumerate(species)}
            prop['line'] = {specie:'-' for i,specie in enumerate(species)}
            prop['alpha'] = {specie:1.0 for i,specie in enumerate(species)}
            for var in ('Temp','Temp_ecl'):
                prop['color'][var] = '#000000' 
                prop['line'][var] = '--' 
                prop['alpha'][var] = 0.5
            var = 'Oil'
            prop['color'][var] = '#d62728' # red
            prop['line'][var] = '-' 
            prop['alpha'][var] = 1.0
            var = 'Water'
            prop['color'][var] = '#1f77b4' # blue
            prop['line'][var] = '-' 
            prop['alpha'][var] = 1.0
            var = 'Gas'
            prop['color'][var] = '#2ca02c' # green
            prop['line'][var] = '-' 
            prop['alpha'][var] = 1.0
        self.plot_prop = prop

            
    #-----------------------------------------------------------------------
    def create_caselist(self, remove=None, insert=None, choose=None):
    #-----------------------------------------------------------------------
        if remove:
            self.cases.pop(self.case_nr(remove))
        if insert:
            self.cases.insert(0, insert)
            self.cases = sorted([str(case) for case in self.cases])
        # case combobox
        self.case_cb.blockSignals(True)
        self.case_cb.clear()
        items = [Path(f).stem for f in self.cases]
        self.case_cb.addItems(items)
        self.case_cb.setCurrentIndex(-1)
        self.case_cb.blockSignals(False)
        # ref case combobox
        self.ref_case.blockSignals(True)
        self.ref_case.clear()
        self.ref_case.addItems(['None']+items)
        self.ref_case.setCurrentIndex(0)
        self.ref_case.blockSignals(False)
        if choose:
            self.case_cb.setCurrentIndex(self.case_nr(choose))

    #-----------------------------------------------------------------------
    def missing_case_error(self, tag=''):
    #-----------------------------------------------------------------------
        show_message(self, 'warning', text='No case selected!\nAdd a case from the File-menu')

    #-----------------------------------------------------------------------
    def missing_file_error(self, tag=''):
    #-----------------------------------------------------------------------
        show_message(self, 'warning', text=f'{tag}-file is missing')

        
    #-----------------------------------------------------------------------
    def add_case(self, case, rename=False, choose_new=True):   # main_window
    #-----------------------------------------------------------------------
        self.case = self.copy_case(case, rename=rename)
        if not self.case:
            return None
        #choose = None
        #if choose_new:
        #    choose = self.case
        #self.create_caselist(insert=self.case, choose=choose)

                
    #-----------------------------------------------------------------------
    def copy_case(self, case, rename=False, choose_new=True): 
    #-----------------------------------------------------------------------
        case = Path(case)
        from_root = case.parent/case.stem
        to_root = self.casedir/case.stem.upper()/case.stem.upper()
        if rename:
            name = Path(rename).stem
            to_root = self.casedir/name/name
        if to_root.parent.is_dir():
            head = f'A case named {to_root.stem} already exists, please choose another name'
            rename = User_input(self, title='Choose new case name', head=head, label='New case name', text=str(to_root.stem))
            def func():
                newname = Path(rename.var.text().upper()).stem
                self.copy_case(case, rename=newname)
            rename.set_func(func)
            rename.open()
            return None
        try:
            to_root.parent.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            show_message(self, 'warning', text=f'A case named {to_root.name} already exists, case not added.')
            return None
        self.copy_case_files(from_root, to_root) 
        self.case = str(to_root)
        choose = None
        if choose_new:
            choose = self.case
        self.create_caselist(insert=self.case, choose=choose)
        return self.case

    
    #-----------------------------------------------------------------------
    def copy_case_files(self, from_root, to_root):             # main_window
    #-----------------------------------------------------------------------
        #print('COPY: {} -> {}'.format(from_root, to_root))
        for ext in ('.DATA','.trcinp','.geocheminp','.sch','.SCH'):
            #from_fil = str(from_root)+ext
            #if Path(from_fil).is_file():
            from_fil = from_root.with_suffix(ext)
            if from_fil.is_file():
                #to_fil = str(to_root)+ext
                #shutil.copy(from_fil, to_fil)
                shutil.copy(from_fil, to_root.with_suffix(ext))
        for ext in upper_and_lower(('INC','DAT','GRDECL','EGRID','VFP')):
            for fil in Path(from_root).parent.glob('*.'+ext):
                shutil.copy(str(fil), str(Path(to_root).parent/fil.name))
        #ext = '.schedule'
        #from_fil = from_root.with_suffix(ext)
        #if from_fil.is_file():
        #    shutil.copy(from_fil, to_root.with_suffix(ext))
        #else:
        #    schedule = Schedule(to_root)

        
    #-----------------------------------------------------------------------
    def read_case_dir(self):
    #-----------------------------------------------------------------------
        cases = []
        if self.casedir.is_dir():
            for d in Path(self.casedir).glob('*'):
                if d.is_dir():
                    cases.append(str(d/d.name))
        return cases

    #-----------------------------------------------------------------------
    def case_nr(self, case):
    #-----------------------------------------------------------------------
        if not case:
            return -1
        nr = -1
        try:
            cases = [Path(case) for case in self.cases]
            nr = cases.index(Path(case))
            return nr
        except ValueError:
            show_message(self, 'warning', text="Case '{}' not found!".format(case))
            return nr
            
    #-----------------------------------------------------------------------
    def update_menu_boxes(self, data, block_signal=True):       # main_window
    #-----------------------------------------------------------------------
        if data=='ecl':
            if not self.ecl_boxes:
                return
            #yaxis = ('prod','rate')
            boxlist = self.ecl_boxes
        if data=='ior':
            if not self.ior_boxes:
                return
            #yaxis = ('prod','conc')
            boxlist = self.ior_boxes 
        if not boxlist['yaxis'] or not boxlist['well']:
            return
        for box in self.max_3_checked:
            #print('Remove:' + box.objectName())
            set_checkbox(box, False, block_signal=block_signal)
        self.max_3_checked = []
        well = list(boxlist['well'].keys())[0] 
        for box in ([val for val in boxlist['yaxis'].values()] + [boxlist['well'][well],]):
            #print('Add: '+box.objectName())
            set_checkbox(box, True, block_signal=block_signal)
            self.max_3_checked.append(box)

        
    #-----------------------------------------------------------------------
    #def set_mode(self, mode, box=False, tip=None, days=None, func=None, run=None):  # main_window
    def set_mode(self, mode, box=False, tip=None, days=None, run=None):  # main_window
    #-----------------------------------------------------------------------
        if not self.case:
            self.sender().setChecked(False)
            self.missing_case_error(tag='set_mode: ')
            return False
        self.mode = self.input['mode'] = mode
        if days:
            self.days_box.setText(str(days))
        if tip:
            self.days_box.setStatusTip(tip)
        self.days_box.setEnabled(box)
        #self.run_func = func
        if not isinstance(run, tuple):
            run = (run,)
        self.run = run
        fh = open(str(Path(self.case).parent/'mode.gui'), 'w')
        fh.write(mode+'\n')
        #print('mode write: '+mode)
        fh.close()


    #-----------------------------------------------------------------------
    def on_mode_select(self, nr):                               # main_window
    #-----------------------------------------------------------------------
        self.reset_progress_and_message()
        if nr<0:
            return
        self.max_days = None
        mode = self.modes[nr]
        fwd_tip = 'Edit TSTEP in Eclipse input to change the total time interval'
        back_tip = 'Set total time interval'
        if mode=='forward':
            # need to run Eclipse before IORSim
            self.set_mode(mode, days=self.input['ecl_days'], tip=fwd_tip, box=False, run=('eclipse','iorsim'))
        elif mode=='backward':
            self.set_mode(mode, box=True, tip=back_tip)
        elif mode=='eclipse':
            self.set_mode(mode, days=self.input['ecl_days'], box=False, tip=fwd_tip, run=mode)
            self.update_menu_boxes('ecl')
            self.create_plot()
        elif mode=='iorsim':
            days = [1,]
            if self.read_ecl_data():
                days = self.data['ecl'].get('days') or [1,]
            self.max_days = int(days[-1])
            self.set_mode(mode, days=self.max_days, box=True, tip='Set total time interval, maximun is '+str(self.max_days), run=mode)
            self.update_menu_boxes('ior')
            self.create_plot()
        else:
            raise SystemError('ERROR Uknown mode: ' + mode)

            
    #-----------------------------------------------------------------------
    def on_input_change(self, text):                           # main_window
    #-----------------------------------------------------------------------
        name = self.sender().objectName()
        #var = {'dt':'Timestep', 'nsteps':'Number of steps'}
        #var = {'nsteps':'Number of steps'}
        var = {'days':'Number of days'}
        if not text:
            self.input[name] = 0
            return
        try:
            val = int(text)
        except:
            show_message(self, 'error', text=var[name]+' must be an integer!')
        else:
            self.input[name] = val
            
    #-----------------------------------------------------------------------
    def on_case_select(self, nr):                              # main_window
    #-----------------------------------------------------------------------
        self.reset_progress_and_message()
        if self.cases:
            #self.sender().blockSignals(True)
            #self.sender().setCurrentIndex(nr)
            #self.sender().blockSignals(False)
            self.input['root'] = self.case = str(self.cases[nr])
            # set simulation mode based on READDATA keyword in .DATA-file
            mode = 'forward'
            #print(self.case, self.input['root'])
            try:
                if file_contains(self.case+'.DATA', text='READDATA', comment='--'):
                    mode = 'backward'
                    self.days_box.setEnabled(False)
                    #print('file_contains: '+mode)
            except FileNotFoundError as e:
                show_message(self, 'error', text='The Eclipse DATA-file is missing for this case')
            # set mode from file
            mode_file = Path(self.case).parent/'mode.gui'
            if mode_file.is_file():
                mode = open(mode_file).readline().strip()
            # on_mode_select() is called in prepare_case() and 
            # dont need to be triggered here    
            self.mode_cb.blockSignals(True)
            self.mode_cb.setCurrentIndex(self.modes.index(mode))
            self.mode_cb.blockSignals(False)
            self.prepare_case(self.input['root'])
                
    #-----------------------------------------------------------------------
    def on_compare_select(self, nr):                              # main_window
    #-----------------------------------------------------------------------
        #self.reset_progress_and_message()
        self.update_message()
        if nr>0:
            ior = ecl = False
            case = str(self.cases[nr-1])
            data = copy.deepcopy(self.data)
            #print('BEFORE')
            #print_dict(self.data)
            # Eclipse
            if self.read_ecl_data(case=case, reinit=True):
                self.plot_ref_data['ecl'] = copy.deepcopy(self.data['ecl'])
                ecl = True
            self.unsmry = None
            # IOR
            if self.read_ior_data(case=case):
                self.plot_ref_data['ior'] = copy.deepcopy(self.data['ior'])
                ior = True
            # copy original data back
            self.data = data
            #print(case, ecl, ior, self.mode)
            if (self.mode=='eclipse' and not ecl) or (self.mode=='iorsim' and not ior) or (not ecl or not ior):
                self.ref_case.setCurrentIndex(self.ref_case.property('lastitem'))
                self.update_message('Cannot compare {} against {}'.format(Path(case).name, Path(self.case).name))
                #self.ref_case.setCurrentIndex(0)
                return
            self.plot_ref = case
            #print('AFTER')
            #print_dict(self.data)
        else:
            self.plot_ref = None
            self.plot_ref_data = {}
            self.ref_plot_lines = {}
            #print(self.plot_ref)
        self.ref_case.setProperty('lastitem',nr)    
        self.create_plot()
        #self.plot_ref = None
        
        
    #-----------------------------------------------------------------------
    def add_case_from_file(self):                   # main_window
    #-----------------------------------------------------------------------
        self.reset_progress_and_message()
        case = open_file_dialog(self, 'Locate Eclipse DATA-file', 'DATA files (*.DATA)')
        if case:
            root = Path(case.split('.DATA')[0])
            self.add_case(root)

            
    #-----------------------------------------------------------------------
    def clear_current_case(self):                              # main_window
    #-----------------------------------------------------------------------
        self.reset_progress_and_message()
        try:
            if not self.case:
                self.missing_case_error(tag='clear_case: ')
                return False
            case = Path(self.case)
            for fil in case.parent.glob('*UNRST'):
                fil.unlink()
            clean_dir = case.parent/'CLEAN'
            clean_dir.mkdir(exist_ok=True)
            # copy case-files to the CLEAN-folder
            self.copy_case_files(case, clean_dir/case.stem)
            # delete all files in case-folder
            for fil in case.parent.glob('*'):
                if fil.is_file():
                    fil.unlink()
            # copy case-files back from CLEAN-folder
            self.copy_case_files(clean_dir/case.stem, case)
            # delete all sub-folders and their files 
            for d in case.parent.iterdir():
                if d.is_dir():
                    delete_all(d)
            self.data = {}
            self.unsmry = None # read again
            self.create_plot()
        except PermissionError as e:
            show_message(self, 'error', text='Unable to clear case, '+str(e))
            return False
        #self.prepare_case(self.case)

    #-----------------------------------------------------------------------
    def delete_case(self, case):                              # main_window
    #-----------------------------------------------------------------------
        #print('Deleting ' + str(case))
        delete_all(Path(case).parent)
        # remove case from caselist
        self.create_caselist(remove=str(case))


    #-----------------------------------------------------------------------
    def delete_current_case(self):                              # main_window
    #-----------------------------------------------------------------------
        self.reset_progress_and_message()
        #print(self.case)
        if not self.case:
            self.missing_case_error(tag='delete: ')
            #print('return')
            return False
        self.delete_case(self.case)
        self.input['root'] = self.case = None
        self.max_3_checked = []
        if self.current_view.objectName()=='editor':
            self.view_file(None)
        self.prepare_case(None)


        
    #-----------------------------------------------------------------------
    def duplicate_current_case(self):                              # main_window
    #-----------------------------------------------------------------------
        self.reset_progress_and_message()
        if not self.case:
            self.missing_case_error(tag='duplicate: ')
            return False
        new_name = User_input(self, title='Duplicate current case', label='Name of duplicate case', text=Path(self.case).name)
        def func():
            from_case = self.case
            name = Path(new_name.var.text().upper()).stem
            to_case = self.casedir/name/name
            #print(str(from_case)+' -> '+str(to_case))
            self.add_case(from_case, rename=to_case)
        new_name.set_func(func)
        new_name.open()
        
    #-----------------------------------------------------------------------
    def rename_current_case(self):                              # main_window
    #-----------------------------------------------------------------------
        self.reset_progress_and_message()
        if not self.case:
            self.missing_case_error(tag='rename: ')
            return False
        rename = User_input(self, title='Rename current case', label='New case name', text=Path(self.case).name)
        def func():
            oldname = Path(self.case).stem
            newname = rename.var.text().upper()
            casedir = Path(self.casedir)
            newdir = (casedir/oldname).rename(casedir/newname)
            #print(str(casedir/oldname)+' => '+str(newdir))
            for x in newdir.iterdir():
                if x.is_file() and oldname in str(x):
                    new = str(x.name).replace(oldname,newname)
                    #print(str(x)+' -> '+str(newdir/new))
                    x.rename(newdir/new)
            newroot = casedir/newname/newname
            self.create_caselist(remove=self.case, insert=newroot, choose=newroot)
            ## set self.case to the new name
            #self.add_case(self.case, rename=newname) #, delete_src=True)
            #self.delete_case(old_case)
        rename.set_func(func)
        rename.open()
        

    #-----------------------------------------------------------------------
    def prepare_case(self, root):
    #-----------------------------------------------------------------------
        self.reset_progress_and_message()
        self.plot_lines = {}
        self.ref_plot_lines = {}
        try:
            self.ref_case.setCurrentIndex(0)
            self.out_wells, self.in_wells = get_wells_iorsim(root)
            self.set_variables_from_casefiles()
            #print('prepare_case: '+str(ind))
            if root:
                self.on_mode_select(self.mode_cb.currentIndex())
            self.set_plot_properties()
            self.update_ior_menu()
        except SystemError as e:
            self.show_message_text(str(e))
        self.data = {}
        self.unsmry = None  # Signals to re-read Eclipse data
        self.read_ior_data()
        self.read_ecl_data()
        self.update_ecl_menu()
        self.create_plot()
        if self.view_ag.checkedAction():
           self.view_ag.checkedAction().trigger()
        #self.update_view_area()
        # enable/disable geochem edit action
        if self.input['root'] and Path(self.input['root']+'.geocheminp').is_file():
            self.chem_inp_act.setEnabled(True)
        else:
            self.chem_inp_act.setEnabled(False)
        # enable/disable schedule edit action
        if self.input['root'] and is_file_ignore_suffix_case( self.input['root']+'.SCH' ):
            self.schedule_file_act.setEnabled(True)
        else:
            self.schedule_file_act.setEnabled(False)

            
    #-----------------------------------------------------------------------
    def create_ecl_menu(self):                                # main_window
    #-----------------------------------------------------------------------
        self.ecl_menu = QGroupBox()
        self.ecl_menu.setTitle('ECLIPSE plot options')
        self.layout.addWidget(self.ecl_menu, *self.position['ecl_menu']) # 
        self.ecl_menu_layout = QHBoxLayout()
        self.ecl_menu.setLayout(self.ecl_menu_layout)
        self.ecl_menu_col = []
        for i in range(2):
            c = QVBoxLayout()
            c.setAlignment(Qt.AlignTop)
            self.ecl_menu_col.append(c)
            self.ecl_menu_layout.addLayout(c)
        
    #-----------------------------------------------------------------------
    def on_ecl_var_click(self):
    #-----------------------------------------------------------------------
        #print('on_ecl_var_click')
        name = self.sender().objectName()
        is_checked = self.sender().isChecked()
        self.update_plot_line(name, is_checked)
        if self.plot_ref_data:
            #print('ref_data')
            self.update_plot_line(name, is_checked, lines=self.ref_plot_lines, set_data=False)
        self.canvas.draw()

    #-----------------------------------------------------------------------
    def create_ior_menu(self):                                # main_window
    #-----------------------------------------------------------------------
        self.ior_menu_group = QGroupBox()
        self.ior_menu_group.setTitle('IORSim plot options')
        self.layout.addWidget(self.ior_menu_group, *self.position['ior_menu']) # 
        self.ior_menu_layout = QHBoxLayout()
        self.ior_menu_group.setLayout(self.ior_menu_layout)
        self.ior_menu_col = []
        for i in range(2):
            c = QVBoxLayout()
            c.setAlignment(Qt.AlignTop)
            self.ior_menu_col.append(c)
            self.ior_menu_layout.addLayout(c)

    #-----------------------------------------------------------------------
    def new_box_with_line_layout(self, name, boxname=None, func=None, linestyle='solid', color=None):
    #-----------------------------------------------------------------------
        if not boxname:
            boxname=name
        box = self.new_checkbox(name=boxname, toggle=True, func=func, pad_left=15) 
        #box.setStyleSheet('padding-left: 15px ')
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        if not color and self.plot_prop:
            color = to_rgb(self.plot_prop['color'][name])
        if color:
            line.setStyleSheet('border: 3px '+linestyle+' '+color)
        label = QLabel(name)
        #font = QFont()
        label.setFont(QFont(self.font, self.menu_fontsize))
        layout = QHBoxLayout()
        layout.addWidget(box,1)
        layout.addWidget(line,1)
        layout.addWidget(label,3)
        return layout, box

    #-----------------------------------------------------------------------
    def new_checkbox(self, text='', name='', func=None, toggle=False, pad_left=10, size=15):
    #-----------------------------------------------------------------------
        box = QCheckBox(text)
        box.setObjectName(name)
        box.setFont(QFont(self.font, self.menu_fontsize))
        #box.setStyleSheet('padding-left: 10px;')
        box.setStyleSheet('QCheckBox { padding-left: '+str(pad_left)+'px; }\nQCheckBox::indicator { width: '+str(size)+'px; height: '+str(size)+'px;};')
        if toggle:
            box.toggle()
        if func: #is None:
            #func = self.update_plot
            box.stateChanged.connect(func)
        return box
        
    #-----------------------------------------------------------------------
    def update_ior_menu(self, checked=True):                   # main_window
    #-----------------------------------------------------------------------
        #print('update_ior_menu')
        delete_all_widgets_in_layout(self.ior_menu_layout)
        if not self.input['root']:
            return False
        col = self.ior_menu_col
        self.ior_boxes = {}
        # add conc / prod boxes
        lbl = QLabel()
        lbl.setText('Y-axis')
        lbl.setStyleSheet('padding-left: 10px')
        #lbl.setFont(QFont(self.font, self.menu_fontsize))
        col[0].addWidget(lbl)
        self.ior_boxes['yaxis'] = {}
        for text in ('Prod', 'Conc'):
            box = self.new_checkbox(text=text+'.', name='yaxis '+text.lower()+' ior', func=self.on_ior_menu_click)
            col[0].addWidget(box, alignment=Qt.AlignTop)
            self.ior_boxes['yaxis'][text.lower()] = box
        # add well boxes
        lbl = QLabel()
        lbl.setText('Wells')
        #lbl.setFont(QFont(self.font, self.menu_fontsize))
        lbl.setStyleSheet('padding-top: 10px; padding-left: 10px')
        col[0].addWidget(lbl)
        self.ior_boxes['well'] = {}
        for i,well in enumerate(self.out_wells or []):
            box = self.new_checkbox(text=well, name='well '+well+' ior', func=self.on_ior_menu_click)
            col[0].addWidget(box, alignment=Qt.AlignTop)
            self.ior_boxes['well'][well] = box
        # add specie boxes
        lbl = QLabel()
        lbl.setText('Variables')
        #lbl.setFont(QFont(self.font, self.menu_fontsize))
        lbl.setStyleSheet('padding-left: 15px')
        col[1].addWidget(lbl)
        self.ior_boxes['var'] = {}
        for i,specie in enumerate(self.input['species'] or []):
            layout, box = self.new_box_with_line_layout(specie, func=self.on_specie_click)
            self.ior_boxes['var'][specie] = box
            col[1].addLayout(layout)
        # add temperature box
        layout, box = self.new_box_with_line_layout('Temp', linestyle='dotted', color='#707070', func=self.on_specie_click)
        self.ior_boxes['var']['Temp'] = box
        col[1].addLayout(layout)
        # set default checked boxes and add them to the checked list
        #self.max_3_checked = []
        if checked:
            self.update_menu_boxes('ior')
            # for box in [self.ior_boxes['yaxis']['conc']]+[b for i,b in enumerate(self.ior_boxes['well'].values()) if i<2]:
            #     set_checkbox(box, True, block_signal=False)
            #     self.max_3_checked.append(box)
            

    #-----------------------------------------------------------------------
    def init_ecl_data(self, case=None):
    #-----------------------------------------------------------------------
        #  WOPR    - well oil rate,
        #  WWPR    - well water rate
        #  WTPCHEA - well temp (Temp_ecl)
        #  WOPT    - well oil prod
        #  WWCT    - well water cut (prod)
        #  WWIR    - well water injection rate
        #  WWIT    - well water injection prod
        #  FOPT    - field oil prod total
        #  FWIT    - field water injection total
        #  FWCT    - field water cut total (prod)
        #  ROIP    - Reservoir oil in place

        #print('init_ecl_data: start')
        varlist = ['WOPR','WWPR','WTPCHEA','WOPT','WWIR','WWIT',
                   'FOPR','FOPT','FGPR','FGPT','FWPR','FWPT'] #,'FWIT','FWIR'] 
        #print('{} : {}'.format(type(self.input['root']),self.input['root']))
        datafile = self.input['root']
        if case:
            datafile = case
        smspec = Path(datafile+'.SMSPEC')
        unsmry = Path(datafile+'.UNSMRY')
        #print(unsmry, smspec)
        if not smspec.is_file() or not unsmry.is_file():
            return False
        #print('init_ecl_data: inside')
        self.unsmry = unfmt_file(unsmry)
        smspec = unfmt_file(smspec)

        ### read variable specifications
        ecl_data = namedtuple('ecl_data','time fluid wells yaxis units indx', defaults=(None,))
        varnames = measure = None
        #try:
        for block in smspec.blocks():
            if block.key() == 'KEYWORDS':
                varnames = get_substrings(block.data()[0], 8)
            elif block.key() == 'WGNAMES':
                ecl_data.wells = [s for s in get_substrings(block.data()[0], 8)]
            elif block.key() == 'MEASRMNT':
                data = block.data()[0].lower()
                width = len(data)/max(len(varnames), 1)
                measure = get_substrings(data, width)
            elif block.key() == 'UNITS':
                ecl_data.units = get_substrings(block.data()[0], 8)                
        #except (SystemError,TypeError) as e:
        #    print(e)
        #    self.unsmry = None # so that we call this function again
        #    return
        #if not varnames:
        if any([not v for v in (varnames, ecl_data.wells, measure, ecl_data.units)]):
            self.unsmry = None # so that we call this function again
            print('return in ecl_init_data()')
            return
        ecl_data.time = varnames.index('TIME')
        fluid_type = {'O':'Oil', 'W':'Water', 'G':'Gas'}
        ecl_data.fluid = {var:('Temp_ecl' if ecl_data.units[i]=='DEG C' else fluid_type.get(var[1])) for i,var in enumerate(varnames)}
        yaxis_type = ['prod','rate']
        ecl_data.yaxis = {var:return_matching_string(yaxis_type, measure[i]) for i,var in enumerate(varnames)}
        ecl_data.yaxis['WTPCHEA'] = 'rate'
        ecl_data.indx = {var:[] for var in varlist}

        ### prepare data dict 
        ecl = {}
        ecl['days'] = []
        for w in [wn for wn in  set(ecl_data.wells) if not ':+:' in wn]:
            ecl[w] = {}
            for y in set(yaxis_type):
                ecl[w][y] = {}
                for f in list(fluid_type.values())+['Temp_ecl']:
                    #print(w,y,f)
                    ecl[w][y][f] = []
                    
        for var in varlist:
            match = False
            for i,name in enumerate(varnames):
                well = ecl_data.wells[i]
                if var==name and not ':+:' in well:
                    match = True
                    ecl_data.indx[var].append(i)
                    y = ecl_data.yaxis[var]
                    f = ecl_data.fluid[var]
                    ecl[well][y][f+' var'] = [var]
                    if 'Temp' in f:
                        ecl[well]['prod'][f] = ecl[well]['rate'][f] 
            if not match:
                del ecl_data.indx[var]
                #print('WARNING! Variable {} not found in {}'.format(var, self.unsmry.name()))
        self.ecl_data = ecl_data
        self.data['ecl'] = ecl
        return True

        
    #-----------------------------------------------------------------------
    def read_ecl_data(self, case=None, reinit=False):
    #-----------------------------------------------------------------------
        #print('read_ecl_data')
        datafile = self.input['root']
        if case:
            datafile = case
        if not datafile:
            self.data['ecl'] = {}
            return False
        ### read data
        if reinit or not self.unsmry:
            if not self.init_ecl_data(case=case):
                return False
        for block in self.unsmry.blocks(only_new=True):
            if block.key()=='PARAMS':
                data = block.data()
                time = data[self.ecl_data.time]
                if time==0.0:
                    continue
                self.data['ecl']['days'].append( time )
                for var,index in self.ecl_data.indx.items():
                    yaxis = self.ecl_data.yaxis[var]
                    fluid = self.ecl_data.fluid[var]
                    wells = self.ecl_data.wells
                    for i in index:
                        self.data['ecl'][wells[i]][yaxis][fluid].append(data[i])
        return True

                        
    #-----------------------------------------------------------------------
    def update_ecl_menu(self, case=None):                       # main_window
    #-----------------------------------------------------------------------
        root = self.input['root']
        if case:
            root = case
        # delete checkboxes before creating new
        delete_all_widgets_in_layout(self.ecl_menu_layout)
        if not root: # or not self.ecl_yaxis:
            return False
        try:
            wells, yaxis, fluids = get_eclipse_well_yaxis_fluid(root)
        except SystemError as e:
            lbl = QLabel()
            lbl.setText(str(e))
            self.ecl_menu_layout.addWidget(lbl)
            return
        #print(wells, yaxis, fluids)
        col = self.ecl_menu_col
        self.ecl_boxes = {}
        # prod/rate
        lbl = QLabel()
        lbl.setText('Y-axis')
        lbl.setStyleSheet('padding-left: 10px')
        col[0].addWidget(lbl)
        self.ecl_boxes['yaxis'] = {}
        for i,name in enumerate(yaxis): 
            box = self.new_checkbox(text=name.capitalize(), name='yaxis '+name+' ecl',
                                    func=self.on_ecl_plot_click)
            self.ecl_boxes['yaxis'][name] = box
            col[0].addWidget(box, alignment=Qt.AlignTop)
        # wells
        lbl = QLabel()
        lbl.setText('Wells')
        lbl.setStyleSheet('padding-top: 10px; padding-left: 10px')
        col[0].addWidget(lbl)
        self.ecl_boxes['well'] = {}
        for i,well in enumerate(wells):
            box = self.new_checkbox(text=well, name='well '+well+' ecl', func=self.on_ecl_plot_click)
            if well=='Field':
                box.setObjectName('well FIELD ecl')
            col[0].addWidget(box, alignment=Qt.AlignTop)
            self.ecl_boxes['well'][well] = box
        # variables
        lbl = QLabel()
        lbl.setText('Variables')
        lbl.setStyleSheet('padding-left: 15px')
        col[1].addWidget(lbl)
        self.ecl_boxes['var'] = {}
        for var in fluids: 
            layout, box = self.new_box_with_line_layout(var, func=self.on_ecl_var_click)
            self.ecl_boxes['var'][var] = box
            col[1].addLayout(layout)
        layout, box = self.new_box_with_line_layout('Temp', boxname='Temp_ecl', linestyle='dotted',
                                                    color='#707070', func=self.on_ecl_var_click)
        self.ecl_boxes['var']['Temp_ecl'] = box
        self.ecl_menu_col[1].addLayout(layout)

        
    #-----------------------------------------------------------------------
    def on_ecl_plot_click(self):
    #-----------------------------------------------------------------------
        self.update_checked_list(self.sender())
        #if self.plot_ref_data:
        #    self.plot_ref = True
        self.create_plot()

                    
    #-----------------------------------------------------------------------
    def create_plot_field(self):                                # main_window
    #-----------------------------------------------------------------------
        self.plot_group = QGroupBox()
        self.current_view = self.plot_group
        self.plot_group.setTitle('Plot')
        self.plot_group.setObjectName('plot')
        layout = QVBoxLayout()
        self.plot_group.setLayout(layout)
        self.layout.addWidget(self.plot_group, *self.position['plot'])
        self.canvas = Mpl_canvas(self)
        toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(toolbar)
        layout.addWidget(self.canvas)
        self.checked_boxes = {}
        self.plotted_lines = {}

        
    #-----------------------------------------------------------------------
    def create_editor_field(self):                                # main_window
    #-----------------------------------------------------------------------
        width = 60
        height = 25
        def new_button(text=None, icon=None, func=None):
            btn = None
            if icon:
                btn = QPushButton(icon=QIcon(':'+icon))
            else:
                btn = QPushButton(text=text)
            #if icon:
            #    btn.setIcon(QIcon(':'+icon))
            #btn.setStyleSheet('QPushButton {max-width}')
            btn.setFixedWidth(width)
            btn.setFixedHeight(height)
            btn.clicked.connect(func)
            return btn

        layout = QVBoxLayout()
        buttons = QHBoxLayout()
        layout.addLayout(buttons)
        self.editor = QPlainTextEdit()
        #self.vscroll_pos = None
        layout.addWidget(self.editor)
        self.editor.textChanged.connect(self.activate_save)
        self.editor.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.editor_group = QGroupBox()
        self.editor_group.setObjectName('editor')
        self.editor_group.setLayout(layout)
        ### Refresh button
        self.refresh_btn = new_button(text='Refresh', func=self.editor_refresh)
        buttons.addWidget(self.refresh_btn)
        ### Save button
        self.save_btn = new_button(text='Save', func=self.save_text)
        buttons.addWidget(self.save_btn)
        ### Undo button
        self.undo_btn = new_button(text='Undo', func=self.editor_undo)
        buttons.addWidget(self.undo_btn)
        ### Redo button
        self.redo_btn = new_button(text='Redo', func=self.editor_redo)
        buttons.addWidget(self.redo_btn)
        ### End button
        self.end_btn = new_button(text='End', func=self.goto_end)
        buttons.addWidget(self.end_btn)
        ### Search field
        self.search_pos = []
        self.search_field = QLineEdit()
        self.search_field.setFixedHeight(height)
        self.search_field.setClearButtonEnabled(True) 
        self.search_field.setPlaceholderText('Search text')
        self.search_field.textChanged.connect(self.search_text)
        buttons.addWidget(self.search_field)
        ### Prev button
        self.prev_btn = new_button(text='Prev', func=self.search_prev)
        buttons.addWidget(self.prev_btn)
        ### Next button
        self.next_btn = new_button(text='Next', func=self.search_next)
        buttons.addWidget(self.next_btn)

    #-----------------------------------------------------------------------
    def goto_end(self):
    #-----------------------------------------------------------------------
        self.editor_refresh()
        self.editor.moveCursor(QTextCursor.End)
        
    #-----------------------------------------------------------------------
    def search_next(self):
    #-----------------------------------------------------------------------
        #print(self.search_string+' '+str(self.search_pos))
        start = 0
        if self.search_pos:
            start = self.search_pos[-1]
        self.search_text(self.search_field.text(), start=start)
        
    #-----------------------------------------------------------------------
    def search_prev(self):
    #-----------------------------------------------------------------------
        #print(self.search_string+' '+str(self.search_pos))
        if self.search_pos:
            string = self.search_field.text()
            if len(self.search_pos)==1:
                pos = self.search_pos[0]
            else:
                pos = self.search_pos.pop()
                if self.editor.textCursor().position() == pos:
                    pos = self.search_pos.pop()
            self.set_cursor(pos-len(string), string)
        
    #-----------------------------------------------------------------------
    def search_text(self, string, start=0, ignore_case=True):
    #-----------------------------------------------------------------------
        #print('search_text: '+string+' '+str(start))
        if start==0:
            self.search_pos = []          
        text = self.editor.toPlainText()
        if ignore_case:
            text = text.lower()
            string = string.lower()
        pos = text[start:].find(string)
        if pos < 0:
            return
        pos += start
        #print(pos)
        self.search_pos.append(pos+len(string))
        #self.search_string = string
        self.set_cursor(pos, string)
        
    #-----------------------------------------------------------------------
    def set_cursor(self, start, string, center=True):
    #-----------------------------------------------------------------------
        cursor = self.editor.textCursor()
        cursor.setPosition(start)
        cursor.setPosition(start+len(string), QTextCursor.KeepAnchor)   
        #print(cursor.blockNumber(), cursor.columnNumber())
        self.editor.setTextCursor(cursor)
        #self.editor.verticalScrollBar().setValue(self.editor.verticalScrollBar().value()+10)
        if center:
            cursor = self.editor.cursorRect()
            cursor_line = int(cursor.y()/cursor.height())
            page_height = self.editor.geometry().height()-self.editor.horizontalScrollBar().height()
            mid_line = int(0.5*page_height/cursor.height())
            shift = cursor_line-mid_line
            vbar = self.editor.verticalScrollBar()
            newpos = shift + vbar.value()
            if newpos>0:
                vbar.setValue(newpos)            
                #print(newpos)
        
    #-----------------------------------------------------------------------
    def save_text(self):
    #-----------------------------------------------------------------------
        #print('save_text')
        #self.editor.blockSignals(True)
        with open(self.editor.objectName(), 'w') as f:
            f.write(self.editor.toPlainText())
        #self.cursor_pos = self.editor.textCursor().position()
        #self.vscroll_pos = self.editor.verticalScrollBar().value()
        #print(self.vscroll_pos)
        self.save_btn.setEnabled(False)
        self.prepare_case(self.case)
        #self.editor.blockSignals(False)

    #-----------------------------------------------------------------------
    def activate_save(self):
    #-----------------------------------------------------------------------
        #print(self.sender())
        self.save_btn.setEnabled(True)
        
    #-----------------------------------------------------------------------
    def editor_undo(self):                               # main_window
    #-----------------------------------------------------------------------
        #self.vscroll_pos = self.editor.verticalScrollBar().value()
        self.editor.undo()
            
    #-----------------------------------------------------------------------
    def editor_redo(self):                               # main_window
    #-----------------------------------------------------------------------
        #self.vscroll_pos = self.editor.verticalScrollBar().value()
        self.editor.redo()
            
    #-----------------------------------------------------------------------
    def view_file(self, file, title=''):                                # main_window
    #-----------------------------------------------------------------------
        curr_file = self.editor.objectName()
        if curr_file:
            self.vscroll[curr_file] = self.editor.verticalScrollBar().value()
        self.reset_progress_and_message()
        # Clear search field
        self.search_field.setText('')
        # Avoid re-opening file after it is saved
        if str(file) == self.editor.objectName():
            return
        self.editor.setObjectName(str(file))
        #self.editor.setObjectName('')
        self.highlight = None
        self.search_field.setPlaceholderText('Search text')
        text = ''
        if file and Path(file).is_file():
            text = open(file).read()
            #self.editor.setObjectName(str(file))
        self.editor.setPlainText(text)
        vscroll = self.vscroll.get(str(file)) or 0
        self.editor.verticalScrollBar().setValue(vscroll)
        #self.editor_text = text
        self.editor_group.setTitle(title)
        self.current_view.setParent(None)
        self.layout.addWidget(self.editor_group, *self.position['plot'])
        self.current_view = self.editor_group
        #if self.vscroll_pos:
        #    self.editor.verticalScrollBar().setValue(self.vscroll_pos)
            #print(self.cursor_pos)
 
    #-----------------------------------------------------------------------
    def editor_refresh(self):
    #-----------------------------------------------------------------------
        file = self.editor.objectName()
        self.vscroll[file] = self.editor.verticalScrollBar().value()
        if file and Path(file).is_file():
            text = open(file).read()
            self.editor.setObjectName(str(file))
            self.editor.setPlainText(text)
        vscroll = self.vscroll.get(str(file)) or 0
        self.editor.verticalScrollBar().setValue(vscroll)

    #-----------------------------------------------------------------------
    def view_input_file(self, ext=None, title=None, comment='#', keywords=[]):                                # main_window
    #-----------------------------------------------------------------------
        # Avoid re-opening file after it is saved
        self.log_file = None
        if self.this_file_is_open_in_editor(ext):
            return
        if self.input['root']:
            fil = is_file_ignore_suffix_case( self.input['root']+'.'+ext )
            if fil and fil.is_file():
                self.view_file(fil, title=title+', '+str(fil.name))        
                self.highlight = Highlighter(self.editor.document(), comment=comment, keywords=keywords)
                self.save_btn.setEnabled(False)
                self.undo_btn.setEnabled(True)
                self.redo_btn.setEnabled(True)
            else:
                self.sender().setChecked(False)
                self.sender().parent().missing_file_error(tag=ext)
                return False
        
        else:
            self.sender().setChecked(False)
            self.sender().parent().missing_case_error(tag='input: ')
            return False

    #-----------------------------------------------------------------------
    def this_file_is_open_in_editor(self, ext):
    #-----------------------------------------------------------------------
        #print(self.input['root']+'.'+ext)
        if self.input['root'] and (self.input['root']+'.'+ext == self.editor.objectName()):
            return True
        return False


    #-----------------------------------------------------------------------
    def view_eclipse_input(self):                 # main_window
    #-----------------------------------------------------------------------
        ext='DATA'
        title='Eclipse input file'
        comment='--'
        # Avoid re-opening file after it is saved
        if self.this_file_is_open_in_editor(ext):
            return
        # Sections
        sections = [red, QFont.Bold, Qt.CaseInsensitive, '\\b','\\b','RUNSPEC','GRID','EDIT','PROPS' ,'REGIONS',
                    'SOLUTION','SUMMARY','SCHEDULE','OPTIMIZE']
        # Global keywords
        globals = [blue, QFont.Normal, Qt.CaseSensitive, '\\b','\\b','COLUMNS','DEBUG','DEBUG3','ECHO','END',
                    'ENDINC','ENDSKIP','SKIP','SKIP100','SKIP300','EXTRAPMS','FORMFEED','GETDATA',
                    'INCLUDE','MESSAGES','NOECHO','NOWARN','WARN']
        # Common keywords
        common = [green, QFont.Normal, Qt.CaseSensitive, r"\b",r'\b','TITLE','CART','DIMENS','FMTIN','FMTOUT',
                    'FMTOUT','UNIFOUT','UNIFIN','OIL','WATER','GAS','VAPOIL','DISGAS','FIELD','METRIC','LAB','START','WELLDIMS','REGDIMS','TRACERS',
                    'NSTACK','TABDIMS','NOSIM','GRIDFILE','DX','DY','DZ','PORO','BOX','PERMX','PERMY','PERMZ','TOPS',
                    'INIT','RPTGRID','PVCDO','PVTW','DENSITY','PVDG','ROCK','SPECROCK','SPECHEAT','TRACER','TRACERKP',
                    'TRDIFPAR','TRDIFIDE','SATNUM','FIPNUM','TRKPFPAR','TRKPFIDE','RPTSOL','RESTART','PRESSURE','SWAT',
                    'SGAS','RTEMPA','TBLKFA1','TBLKFIDE','TBLKFPAR','FOPR','FOPT','FGPR','FGPT','FWPR','FWPT','FWCT','FWIR',
                    'FWIT','FOIP','ROIP','WTPCHEA','WOPR','WWPR','WWIR','WBHP','WWCT','WOPT','WWIT','WTPRA1','WTPTA1','WTPCA1',
                    'WTIRA1','WTITA1','WTICA1','CTPRA1','CTIRA1','FOIP','ROIP','FPR','TCPU','TCPUTS','WNEWTON','ZIPEFF','STEPTYPE',
                    'NEWTON','NLINEARP','NLINEARS','MSUMLINS','MSUMNEWT','MSUMPROB','WTPRPAR','WTPRIDE','WTPCPAR','WTPCIDE','RUNSUM',
                    'SEPARATE','WELSPECS','COMPDAT','WRFTPLT','TSTEP','DATES','SKIPREST','WCONINJE','WCONPROD','WCONHIST','WTEMP','RPTSCHED',
                    'RPTRST','TUNING','READDATA', 'ROCKTABH','GRIDUNIT','NEWTRAN','MAPAXES','EQLDIMS','ROCKCOMP','TEMP',
                    'GRIDOPTS','VFPPDIMS','VFPIDIMS','AQUDIMS','SMRYDIMS','CPR','FAULTDIM','MEMORY','EQUALS','MINPV',
                    'COPY','MULTIPLY']
        self.view_input_file(ext=ext, title=title, comment=comment, keywords=[sections, globals, common])
        
    #-----------------------------------------------------------------------
    def view_iorsim_input(self):                                # main_window
    #-----------------------------------------------------------------------
        ext='trcinp'
        title='IORSim input file'
        comment='#'
        if self.this_file_is_open_in_editor(ext):
            return
        # Mandatory keywords
        mandatory = [blue, QFont.Bold, Qt.CaseInsensitive, '\\', '\\b', '*RESTART_WRITE','*RESTART_FILE','*GRIDPLOT_WRITE','*GRIDPLOT_FILE' ,
                     '*RESTART_READ','*INTEGRATION','*OUTPUT','*WELLPLOT_INTERVAL','*END']
        # Optional keywords
        optional = [green, QFont.Normal, Qt.CaseInsensitive, '\\', '\\b', '*TEMPERATURE', '*TRACER_LGR','*N_TRACER','*NAME','*K_WATER','*K_OIL','*K_GAS',
                    '*DW','*DO','*DG','*K_ADS','*C_INIT','*CONC_INJECTION','*REACTING_SYSTEM','*INTEGRATE_SPECIES',
                    '*MODELTYPE','*SPECIES','*MODELTEMPLATE','*TINIT','*MODELINSTANCE','*WELLSPECIES']
        self.view_input_file(ext=ext, title=title, comment=comment, keywords=[mandatory, optional])
        
    #-----------------------------------------------------------------------
    def view_geochem_input(self):                                # main_window
    #-----------------------------------------------------------------------
        self.view_input_file(ext='geocheminp', title='IORSim geochem file', comment='#')
        
    #-----------------------------------------------------------------------
    def view_schedule_file(self):                                # main_window
    #-----------------------------------------------------------------------
        self.view_input_file(ext='SCH', title='Schedule file for backward runs', comment='--')
        
    #-----------------------------------------------------------------------
    def view_log(self, logfile, title=None):                                # main_window
    #-----------------------------------------------------------------------
        if not self.case:
            self.sender().setChecked(False)
            self.sender().parent().missing_case_error(tag='log: ')
            return False
        self.log_file = Path(self.case).parent/logfile
        self.view_file(self.log_file, title=title)
        self.save_btn.setEnabled(False)
        self.undo_btn.setEnabled(False)
        self.redo_btn.setEnabled(False)
        
    #-----------------------------------------------------------------------
    def view_eclipse_log(self):                                # main_window
    #-----------------------------------------------------------------------
        self.view_log('eclipse.log', title='ECLIPSE logfile')
        
    #-----------------------------------------------------------------------
    def view_iorsim_log(self):                                # main_window
    #-----------------------------------------------------------------------
        self.view_log('iorsim.log', title='IORSim logfile')
    
    #-----------------------------------------------------------------------
    def view_program_log(self):                                # main_window
    #-----------------------------------------------------------------------
        self.view_log('ior2ecl.log', title='Program logfile')
    
    #-----------------------------------------------------------------------
    def update_log(self):                                # main_window
    #-----------------------------------------------------------------------
        if self.log_file:
            #print('update_log:',self.log_file)
            self.editor.setPlainText(open(self.log_file).read())
            #self.editor.setCenterOnScroll(True)
            #self.editor.ensureCursorVisible()
            self.editor.moveCursor(QTextCursor.End)
            self.save_btn.setEnabled(False)
    
    #-----------------------------------------------------------------------
    def view_plot(self):                                # main_window
    #-----------------------------------------------------------------------
        #if not self.case:
        #    #self.sender().setChecked(False)
        #    self.missing_case_error('plot: ')
        #    return False
        self.log_file = None
        self.editor.setObjectName('')
        if self.current_view:
            self.current_view.setParent(None)
        self.layout.addWidget(self.plot_group, *self.position['plot'])
        self.current_view = self.plot_group
        if not self.worker:# and self.case:
            self.read_ior_data()
            self.read_ecl_data()
            self.update_all_plot_lines()

            
    #-----------------------------------------------------------------------
    def update_checked_list(self, box): 
    #-----------------------------------------------------------------------
        #print('update_checked_list: '+self.sender().objectName())
        max_3 = self.max_3_checked
        if box.isChecked():
            max_3.append(box)
        else:
            max_3.remove(box)
        names = [b.objectName().split() for b in max_3]
        data = [n[2] for n in names]
        if len(set(data)) > 1:
            # both ecl and ior are checked
            for ie in ('ior','ecl'):
                if data.count(ie)>2:
                    # check if 2 yaxis or 2 wells are checked
                    for yw in ('yaxis','well'):
                        ind = [i for i,(a,b,c) in enumerate(names) if a==yw and c==ie]
                        if len(ind)>1:
                            box = max_3.pop(min(ind))
                            set_checkbox(box, False)
        else:
            # only ecl or ior is checked
            if len(max_3)>3:
                box = max_3.pop(0)
                set_checkbox(box, False)
            names = [b.objectName().split()[0] for b in max_3]    
            for key in ('yaxis','well'):
                if names.count(key)>2:
                    box = max_3.pop(names.index(key))
                    set_checkbox(box, False)
                
            
    #-----------------------------------------------------------------------
    def on_specie_click(self):
    #-----------------------------------------------------------------------
        #print('on_specie_click')
        specie = self.sender().objectName()
        is_checked = self.sender().isChecked()
        self.update_plot_line(specie, is_checked)
        if self.plot_ref_data:
            self.update_plot_line(self.sender().objectName(), is_checked, lines=self.ref_plot_lines, set_data=False)
        self.canvas.draw()

    #-----------------------------------------------------------------------
    def update_plot_line(self, name, is_checked, lines=None, set_data=True):
    #-----------------------------------------------------------------------
        #print('update_plot_line')
        if not lines:
            lines = self.plot_lines
        if not lines:
            self.create_plot_lines()
        if not name in lines:
            return
        for ax,line in lines[name].items():
            line.set_visible(is_checked)
            if is_checked and set_data:
                well, yaxis, var, data = line.get_label().split()
                if not self.data[data]:
                    return
                xdata = self.data[data]['days']
                ydata = self.data[data][well][yaxis][var]
                line.set_data(xdata, ydata)
            ax.relim(visible_only=True)
            ax.autoscale_view()
        if 'temp' in name.lower():
            for ax in lines[name].keys():
                ax.set_visible(is_checked)

            
    # #-----------------------------------------------------------------------
    # def update_plot_line_v2(self, name, is_checked):
    # #-----------------------------------------------------------------------
    #     if not name in self.plot_lines:
    #         # create new line
    #         line, = ax[ind].plot(xdata, ydata, color=self.plot_prop['color'][var],
    #                              linestyle=self.plot_prop['line'][var], alpha=self.plot_prop['alpha'][var],
    #                              lw=2, label=well+' '+yaxis+' '+var+' '+data)
    #         ax[ind].relim(visible_only=True)
    #         ax[ind].autoscale_view()
    #         line.set_visible(var_box[var].isChecked())

    #     #print('update_plot_line ' + name)
    #     for ax,line in self.plot_lines[name].items():
    #         line.set_visible(is_checked)
    #         if is_checked:
    #             well, yaxis, var, data = line.get_label().split()
    #             #print(var, varname)
    #             line.set_data(self.data[data]['days'], self.data[data][well][yaxis][var])
    #         ax.relim(visible_only=True)
    #         ax.autoscale_view()
    #     if name.lower() == 'temp':
    #         for ax in self.plot_lines[name].keys():
    #             ax.set_visible(is_checked)
            

    #-----------------------------------------------------------------------
    def update_axes_limits(self):
    #-----------------------------------------------------------------------
        #try:
        for ax in self.plot_axes:
            ax.relim(visible_only=True)
            ax.autoscale_view()
        #except ValueError:
        #    print('ValueError in update_axes_limits()')
            
    #-----------------------------------------------------------------------
    def on_ior_menu_click(self):
    #-----------------------------------------------------------------------
        #print('ior_menu: '+self.sender().objectName())
        self.update_checked_list(self.sender())
        #if self.plot_ref_data:
        #    self.plot_ref = True
        self.create_plot()

        
    #-----------------------------------------------------------------------
    def read_ior_data(self, case=None):
    #-----------------------------------------------------------------------
        #  Format:
        #     data['days']
        #     data[well]['conc'|'prod'][var]
        #
        #print('read_ior_data')
        self.data['ior'] = {}
        datafile = self.input['root']
        if case:
            datafile = case
        if not datafile:
            return False
        root = Path(datafile)
        files = list(root.parent.glob(root.name+'_W_*.trc*'))
        if len(files)<1 or not files[0].is_file():# or files[0].stat().st_size<210:
            # last check is to avoid UserWarning from genfromtxt about: Empty input file
            return False
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            data = genfromtxt(str(files[0]))
        if data.ndim < 2: # if ndim==1 only one line of data, abort....
            return False
        inp = self.input
        ior = {}
        for w in self.out_wells:
            ior[w] = {}
            ior[w]['conc'] = {}
            ior[w]['prod'] = {}
        for file in files:
            if not file.is_file():
                continue
            # read data
            well, yaxis = file.name.split('_W_')[-1].split('.trc')
            if yaxis=='prd':
                yaxis = 'prod'
            data = genfromtxt(str(file))
            try:
                ior['days'] = data[1:,0]
                for i,name in enumerate(inp['species']):
                    ior[well][yaxis][name] = data[1:,i+1]
                if 'conc' in yaxis:
                    ior[well]['conc']['Temp'] = data[1:,-1]
                    ior[well]['prod']['Temp'] = data[1:,-1]
            except (KeyError, IndexError, TypeError) as e:
                return False
        self.data['ior'] = ior
        return True

    
    #-----------------------------------------------------------------------
    def update_axes_names(self):                   # main_window
    #-----------------------------------------------------------------------                
    #
    #  Checkboxes are named 'yaxis prod ior', 'yaxis conc ior', 'yaxis rate ecl'
    #  'well P1 ior', 'well I1 ecl'
    #
        #print('update_axes_names')
        #print(self.max_3_checked)
        name0 = [b.objectName().split()[0] for b in self.max_3_checked]    
        y_idx = [i for i,x in enumerate(name0) if x=='yaxis']
        w_idx = [i for i,x in enumerate(name0) if x=='well']
        name1 = [b.objectName().split()[1] for b in self.max_3_checked]            
        name2 = [b.objectName().split()[2] for b in self.max_3_checked]            
        self.ioraxes_names = []
        for y in y_idx:
            for w in w_idx:
                if name2[y] == name2[w]:
                    self.ioraxes_names.append(name1[y]+' '+name1[w]+' '+name2[w])
    
    
    #-----------------------------------------------------------------------
    def create_plot(self):#, xlim=False):                         # main_window
    #-----------------------------------------------------------------------                
        #print('create_plot')
        self.plot_lines = {}
        self.ref_plot_lines = {}
        if self.plot_ref_data:
            self.plot_ref = True
        # clear figure 
        self.canvas.fig.clf()
        if not self.input['root']:
            self.canvas.draw()
            return False
        self.update_axes_names()
        #if len(self.canvas.fig.axes)>0:
        #    print('AXES NOT DELETED')
        # create new axes
        self.temp_axis = []
        self.plot_axes = []
        #self.plot_axes = {}
        nplot = len(self.ioraxes_names)
        #print(nplot)
        #print(self.ioraxes_names)
        src = {'ecl':'Eclipse', 'ior':'IORSim'}
        for i, ax_name in enumerate(self.ioraxes_names):
            yaxis, well, data = ax_name.split()
            ax = self.canvas.fig.add_subplot(nplot, 1, i+1)
            #self.plot_axes[ax_name] = ax
            self.plot_axes.append(ax)
            ax.set_label(ax_name)
            ax.set_xlabel('days')
            ax.title.set_text(src[data]+', well '+well)
            if yaxis=='conc':
                ax.set_ylabel('concentration [mol/L]')
            elif yaxis=='prod':
                ylabel = 'production '
                if data=='ior':
                    ylabel += '[mass/day]'
                elif data=='ecl':
                    ylabel += '[SM3]'
                ax.set_ylabel(ylabel)
            elif yaxis=='rate':
                ax.set_ylabel('rate [SM3/day]')                
            ax.autoscale_view()
            # add temperature axis
            axx = ax.twinx()
            self.plot_axes.append(axx)            
            #self.plot_axes[ax_name+' temp'] = axx
            if data=='ior':
                axx.set_visible(self.ior_boxes['var']['Temp'].isChecked())
            elif data=='ecl':
                axx.set_visible(self.ecl_boxes['var']['Temp_ecl'].isChecked())
            #axx.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            axx.ticklabel_format(axis='y', style='plain', useOffset=False) #scilimits=[-1,1])
            axx.set_label('Temp '+ax_name)
            axx.set_ylabel('Temperature [C]')
            axx.autoscale_view()
            #if self.worker:
            #    ax.set_xlim(*self.xlim)
            #    axx.set_xlim(*self.xlim)
        self.create_plot_lines()
        self.canvas.fig.subplots_adjust(top=.93, hspace=0.4, right=.87)
        self.canvas.draw()
        return True

    
    #-----------------------------------------------------------------------
    def create_plot_lines(self):
    #-----------------------------------------------------------------------
        #print('create_plot_lines')
        self.plot_lines = {}
        lines = {}
        self.ref_plot_lines = {}
        ref_lines = {}
        ax = self.plot_axes   # or self.canvas.fig.axes
        for i, ax_name in enumerate(self.ioraxes_names):
            yaxis, well, data = ax_name.split()
            if data=='ior':
                var_box = self.ior_boxes['var']
            elif data=='ecl':
                var_box = self.ecl_boxes['var']
            var_list = var_box.keys() 
            the_data = self.data.get(data)
            for var in var_list:
                if 'temp' in var.lower():
                    ind = i*2+1 # extra right axis for temperature
                else:
                    ind = i*2   # main axes
                line = None
                if the_data:
                    try:
                        xdata = the_data['days']
                        ydata = the_data[well][yaxis][var]
                        if len(xdata) != len(ydata):
                            continue
                    except KeyError as e:
                        continue
                    line, = ax[ind].plot(xdata, ydata, color=self.plot_prop['color'][var],
                                         linestyle=self.plot_prop['line'][var],
                                         alpha=self.plot_prop['alpha'][var],
                                         lw=2, label=well+' '+yaxis+' '+var+' '+data)
                if line:
                    if var not in lines:
                        lines[var] = {}
                    line.set_visible(var_box[var].isChecked())
                    lines[var][ax[ind]] = line
                # if we have a compare case
                ref_line = None
                if self.plot_ref:
                    refdata = self.plot_ref_data.get(data)
                    if refdata: # and var_box[var].isChecked():
                        try:
                            xdata = refdata['days']
                            ydata = refdata[well][yaxis][var]
                            if len(xdata) != len(ydata):
                                continue
                        except KeyError as e:
                            continue                        
                        ref_line, = ax[ind].plot(refdata['days'], refdata[well][yaxis][var],
                                                 color=self.plot_prop['color'][var],
                                                 linestyle=self.plot_prop['line'][var], alpha=0.4,
                                                 lw=2, label=well+' '+yaxis+' '+var+' '+data)
                if ref_line:
                    if var not in ref_lines:
                        ref_lines[var] = {}
                    ref_line.set_visible(var_box[var].isChecked())
                    ref_lines[var][ax[ind]] = ref_line
                ax[ind].relim(visible_only=True)
                ax[ind].autoscale_view()
            self.plot_lines = lines
            self.ref_plot_lines = ref_lines
        if self.plot_ref:
            self.plot_ref = None
            
    # #-----------------------------------------------------------------------
    # def plot_line(self, axis, var, data, axisname, var_box, lines):           # main_window
    # #-----------------------------------------------------------------------
    #     if not data:
    #         return
    #     try:
    #         y, w, d = axisname.split() # yaxis well data
    #         xdata = data['days']
    #         ydata = data[w][y][var]
    #         if len(xdata) != len(ydata):
    #             return
    #     except KeyError as e:
    #         return
    #     line, = axis.plot(xdata, ydata,
    #                       color=self.plot_prop['color'][var],
    #                       linestyle=self.plot_prop['line'][var],
    #                       alpha=self.plot_prop['alpha'][var],
    #                       lw=2,
    #                       label=w+' '+y+' '+var+' '+d)
    #     if var not in lines:
    #         lines[var] = {}
    #     line.set_visible(var_box[var].isChecked())
    #     lines[var][ax[ind]] = line

            
    #-----------------------------------------------------------------------
    def update_remaining_time(self, text='0:00:00'):           # main_window
    #-----------------------------------------------------------------------
        #self.remaining_time.setText('Remaining time:  ' + text)
        self.remaining_time.setText(text)
        self.remaining_time.repaint()


    #-----------------------------------------------------------------------
    def update_all_plot_lines(self): #, read_data=True):
    #-----------------------------------------------------------------------
        #print('update_all_plot_lines')
        # update plot-lines
        #if read_data:
        #    self.read_ior_data()
        #    self.read_ecl_data()
        #print('update_all_plot_lines: {}'.format(self.plot_lines))
        if not self.plot_lines:
            self.create_plot_lines()
        for name,ax_line in self.plot_lines.items():
            for ax,line in ax_line.items():
                #print(var, ax, line)
                well, yaxis, var, data = line.get_label().split()
                if not self.data.get(data):
                    return
                if data=='ior':
                    is_checked = self.ior_boxes['var'][name].isChecked()
                else: # data=='ecl'
                    is_checked = self.ecl_boxes['var'][name].isChecked()
                line.set_visible(is_checked)
                if is_checked:
                    try:
                        xdata = self.data[data]['days']
                        ydata = self.data[data][well][yaxis][var]
                        line.set_data(xdata, ydata)
                    except KeyError:
                        pass
                #try:
                #    ax.relim(visible_only=True)
                #    ax.autoscale_view()
                #except ValueError:
                #    print('ValueError in update_all_plot_lines()')
                #    #pass
        try:
            self.update_axes_limits()
            self.canvas.draw()
        except ValueError as e:
            pass
            #print('ValueError: '+str(e))

    #-----------------------------------------------------------------------
    def update_progressbar(self, t):
    #-----------------------------------------------------------------------
        if t>=0:
            #print(t, self.progressbar.value(), self.progressbar.maximum())
            self.progressbar.setValue(t)

    #-----------------------------------------------------------------------
    def reset_progress_and_message(self):
    #-----------------------------------------------------------------------
        if not self.worker:
            self.reset_progressbar()
            self.update_remaining_time()
            self.update_message()

    #-----------------------------------------------------------------------
    def reset_progressbar(self, N=None):
    #-----------------------------------------------------------------------
        #if N==0:
        #    N=-1
        if N:
            self.progressbar.setMaximum(N)
        #v = 0
        #if N<0:
        #    v = N
        #self.progressbar.setValue(1)
        self.progressbar.reset()
        #print(self.progressbar.minimum(), self.progressbar.maximum(), self.progressbar.value())

    #-----------------------------------------------------------------------
    #def update_progress(self, t=None, min=None):
    def update_progress(self, t_min_n0_tuple):
    #-----------------------------------------------------------------------
        t, min, n0 = t_min_n0_tuple
        if n0 is not None:
            self.progress and self.progress.reset_time(n=n0)
        if min is not None:
            self.progressbar.setMinimum(int(min))
            self.progress and self.progress.set_min(int(min))
        if t is not None:
            if t==0: 
                self.update_remaining_time()
                self.progressbar.setMinimum(0)
                if self.progress:
                    self.progress.set_min(0)
                return
            elif t<0:
                N = abs(int(t)) 
                self.reset_progressbar(N=N)
                self.progress = Progress(N=N)
                return  
            self.update_progressbar(t)
            self.update_remaining_time( self.progress.remaining_time(t) )
        #print(self.progressbar.minimum(), self.progressbar.maximum(), self.progressbar.value())
        
    #-----------------------------------------------------------------------
    def update_view_area(self):
    #-----------------------------------------------------------------------
        #A = datetime.now()
        if self.mode == 'backward':
            self.read_ior_data()
            self.read_ecl_data()
        #B = datetime.now()
        #print(B-A, ', ', end='')

        self.days = None
        #print('Mode: '+self.mode)
        #if self.mode in ('forward','eclipse','iorsim') and self.worker and self.worker.current:
        if self.mode in ('forward','eclipse','iorsim') and self.worker and self.worker.current_run():
            #print('Current: '+self.worker.current)
            #run = self.worker.current
            run = self.worker.current_run()
            if run in ('ecl','eclipse','eclrun'):
                self.read_ecl_data()
                run = 'ecl'
            if run in ('ior','iorsim'):
                self.read_ior_data()
                run = 'ior'
            if self.data.get(run):
                self.days = (self.data[run]).get('days')
                #print(self.days[-1])

        view = self.current_view.objectName()
        #print('view:', view)
        if view=='plot':
            self.update_all_plot_lines()
        elif view=='editor':
            self.update_log()
        #print(datetime.now()-B)

    #-----------------------------------------------------------------------
    def input_OK(self):
    #-----------------------------------------------------------------------
        i = self.input
        #if i['nsteps']==0:
        if self.case_cb.currentIndex() < 0:
            show_message(self, 'warning', text='You need to choose a case from the case drop-down list, or add a new case from File -> Add case.')
            return False
        if i['days']==0:
            show_message(self, 'warning', text='Total time interval is missing.')
            return False
        # if self.mode=='backward' and i['days']<i['dtecl']:
        #     show_message(self, 'warning', text='Total time interval of {} days is less than the IORSim timestep of {} days'.format(i['days'],i['dtecl']))
        #     return False
        if self.max_days and i['days']>self.max_days:
            show_message(self, 'warning', text='The Eclipse output read by IORSim currently sets a limit of ' + str(self.max_days) + 
                                               ' days on the time interval. Run Eclipse with a higher TSTEP to increase the maximun time interval.')
            return False
        #if i['dt']==0:
        if not self.settings.get['dt']() or int(self.settings.get['dt']())==0:
            show_message(self, 'warning', text='Timestep missing in settings')
            self.settings.open()
            return False
        if not i['root']:
            show_message(self, 'warning', text='No input-case selected')
            return False
        if not self.settings.get['iorsim']():
            show_message(self, 'warning', text='IORSim program missing in Settings')
            self.settings.open()
            return False
        # if self.mode != 'eclipse' and i['dtecl'] == 0:
        #     show_message(self, 'warning', text='IORSim timestep is zero')
        #     return False
        return True
    
        
    #-----------------------------------------------------------------------
    def set_toolbar_enabled(self, value):                                  # main_window
    #-----------------------------------------------------------------------
        self.start_act.setEnabled(value)
        self.case_cb.setEnabled(value)
        self.mode_cb.setEnabled(value)
        if self.mode=='backward':
            self.days_box.setEnabled(value)
        #self.sim_cb.setEnabled(value)

    #-----------------------------------------------------------------------
    def run_sim(self):                                    # main_window
    #-----------------------------------------------------------------------
        #print('run_mode:',self.mode, self.run)
        if not self.input_OK():
            return
        # Clear data
        self.data = {}
        self.unsmry = None
        # Clear messages and progress
        self.reset_progress_and_message()
        # Disable toolbar
        self.set_toolbar_enabled(False)
        i = self.input
        s = self.settings
        # backward mode
        if self.mode=='backward':
            kwargs = {'mode':'backward', 'check_unrst':s.get['unrst'](), 'check_rft':s.get['rft'](), 'rft_size':False}
        # forward mode
        elif self.mode in ('forward','eclipse','iorsim'):
            kwargs = {'mode':'forward', 'runs':self.run}
        # start simulation
        for opt in ('convert','del_convert','merge','del_merge'):
            kwargs[opt] = s.get[opt]()
        self.worker = sim_worker(root=i['root'], time=i['days'], iorexe=s.get['iorsim'](), eclexe=s.get['eclrun'](), 
                                 pause=float(s.get['pause']()), init_tstep=float(s.get['dt']()), 
                                 stop_children=s.get['stop_child'](), days_box=self.days_box, **kwargs)
        self.worker.signals.status_message.connect(self.update_message)
        self.worker.signals.show_message.connect(self.show_message_text)
        self.worker.signals.progress.connect(self.update_progress)
        self.worker.signals.plot.connect(self.update_view_area)
        self.worker.signals.finished.connect(self.run_finished)
        self.threadpool.start(self.worker)
        

    #-----------------------------------------------------------------------
    def show_message_text(self, text):
    #-----------------------------------------------------------------------
        text = text.lstrip()
        kind = 'error'
        if text.startswith(('ERROR','WARNING','INFO')):
            kind = text.split()[0]
            text = text.replace(kind,'').lstrip()
            kind = kind.lower()
        show_message(self, kind, text=text)
        return text
        
    #-----------------------------------------------------------------------
    def show_message(self, par):
    #-----------------------------------------------------------------------
        show_message(self, par[0], text=par[1])
        
    #-----------------------------------------------------------------------
    def run_finished(self):
    #-----------------------------------------------------------------------
        #print(datetime.now()-self.starttime)
        self.set_toolbar_enabled(True)
        #if self.worker.success:
        #    self.convert_FUNRST()   
        self.worker = None


    #-----------------------------------------------------------------------
    def killsim(self):                                          # main_window
    #-----------------------------------------------------------------------
        if self.worker:
            self.worker.signals.stop.emit()
            # If quit while running simulation, we wait for up to 5 seconds
            # to allow the processes to quit before we kill them
            n = 0
            while self.worker.sim and self.worker.sim.runs and n<500:
                n += 1
                #print(n)
                sleep(0.01)
            
        
    #-----------------------------------------------------------------------
    def quit(self):                                            # main_window
    #-----------------------------------------------------------------------
        self.killsim()
        #self.settings.close()
        #self.help_win.close()
        #self.write_casefile()
        #self.write_case_dir()
        self.save_input()
        QApplication.quit()
                

    #-----------------------------------------------------------------------
    def update_message(self, text='', limit=100):
    #-----------------------------------------------------------------------
        if text.startswith(('WARNING','ERROR','INFO')):
            text = text.replace(text.split()[0],'')
        #print('|'+text+'|')
        if len(text) < limit:            
            self.messages.setText(text)
        #self.statusBar().showMessage(text) 
        
### 
###  Adapted from code at:         
###  https://github.com/pyside/Examples/blob/master/examples/richtext/syntaxhighlighter.py
###
class Highlighter(QSyntaxHighlighter):
    def __init__(self, parent=None, comment='#', color=Qt.gray, keywords=[]):
#    def __init__(self, parent=None, comment='#', color=QColorConstants.Gray, keywords=[]):
        super(Highlighter, self).__init__(parent)

        self.highlightingRules = []
        #print(keywords)
        while keywords:
            kword = keywords.pop(0)
            keywordFormat = QTextCharFormat()
            keywordFormat.setForeground(kword.pop(0))
            keywordFormat.setFontWeight(kword.pop(0))
            case_sens = kword.pop(0)
            front = kword.pop(0)
            back = kword.pop(0)
            keywordPatterns = [front + kw + back for kw in kword]
            #keywordPatterns = kword
            #print(keywordPatterns)
            #keywordPatterns = ["\\b*TEMPERATURE\\b", "\\b*INTEGRATION\\b", "\\b*MODELTYPE\\b"]
            
            self.highlightingRules.extend( [(QRegExp(pattern, cs=case_sens), keywordFormat) for pattern in keywordPatterns] )
            #self.highlightingRules.extend( [(QRegularExpression(pattern, cs=case_sens), keywordFormat) for pattern in keywordPatterns] )
        singleLineCommentFormat = QTextCharFormat()
        singleLineCommentFormat.setForeground(color)
        self.highlightingRules.append((QRegExp(comment+'[^\n]*'), singleLineCommentFormat))
        #self.highlightingRules.append((QRegularExpression(comment+'[^\n]*'), singleLineCommentFormat))

    def highlightBlock(self, text):
        for pattern, format in self.highlightingRules:
            expression = QRegExp(pattern)
            #expression = QRegularExpression(pattern)
            index = expression.indexIn(text)
            while index >= 0:
                length = expression.matchedLength()
                self.setFormat(index, length, format)
                index = expression.indexIn(text, index + length)


                
###################################
#                                 #
#            M A I N              #
#                                 #
###################################

if __name__ == '__main__':
    # Need to set the locale under Linux to avoid datetime.strptime errors
    os.putenv("LC_ALL", "C")
    if len(sys.argv) > 1:
        print()
        print('   This is the terminal-version of IORSim_GUI')
        print('   Start IORSim_GUI without arguments to open the GUI')
        print()
        ior2ecl_main(case_dir=case_dir, settings_file=settings_file)
    else:
        app = QApplication(sys.argv) 
        #app.setFont(QFont(default_font, pointSize=default_size, weight=default_weight))
        window = main_window()
        app.exec_()
        os._exit(0)
        #exit_without_atexit()

    
