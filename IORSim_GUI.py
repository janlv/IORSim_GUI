#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
#print(sys.version_info)
from pathlib import Path
#-----------------------------------------------------------------------
def resource_path():
#-----------------------------------------------------------------------
    try:
        path = Path(sys._MEIPASS)
    except AttributeError:
        path = Path.cwd()
    return path

# Default settings
default_casedir = Path.cwd()/'IORSim_cases'
default_settings_file = Path.home()/'.iorsim_settings.dat'

# Update files
this_file = Path(sys.argv[0])

# Guide files
iorsim_guide = "file:///" + str(resource_path()).replace('\\','/') + "/guides/IORSim_2021_User_Guide.pdf"
script_guide = "file:///" + str(resource_path()).replace('\\','/') + "/guides/IORSim_GUI_guide.pdf"

# GitHub
github_url = "https://github.com/janlv/IORSim_GUI/"
latest_release = github_url +"releases/latest"

# External libraries
from PySide6.QtWidgets import QStatusBar, QDialog, QWidget, QMainWindow, QApplication, QLabel, QPushButton, QGridLayout, QVBoxLayout, QHBoxLayout, QLineEdit, QPlainTextEdit, QDialogButtonBox, QCheckBox, QToolBar, QProgressBar, QGroupBox, QComboBox, QFrame, QFileDialog, QMessageBox
from PySide6.QtGui import QPalette, QAction, QActionGroup, QColor, QFont, QIcon, QSyntaxHighlighter, QTextCharFormat, QTextCursor
from PySide6.QtCore import QDir, QCoreApplication, QSize, QUrl, QObject, Signal, Slot, QRunnable, QThreadPool, Qt, QRegularExpression, QRect, QPoint
from PySide6.QtWebEngineCore import QWebEnginePage, QWebEngineSettings
from PySide6.QtWebEngineWidgets import QWebEngineView
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.colors import to_rgb as colors_to_rgb
from matplotlib.figure import Figure
from numpy import genfromtxt, asarray

# Python libraries
from traceback import format_exc, print_exc, format_exc
from time import sleep
from collections import namedtuple
from shutil import copy as shutil_copy
import warnings
from copy import deepcopy 
from functools import partial
from requests import get as requests_get
# Suppress warnings about using verify=False in requests_get
from urllib3 import disable_warnings
disable_warnings()

# Local libraries
from ior2ecl import ECL_ALIVE_LIMIT, IOR_ALIVE_LIMIT, Iorsim, Simulation, main as ior2ecl_main, __version__, DEFAULT_LOG_LEVEL, LOG_LEVEL_MAX, LOG_LEVEL_MIN
from IORlib.utils import Progress, flat_list, get_keyword, get_substrings, is_file_ignore_suffix_case, read_file, replace_line, return_matching_string, delete_all, file_contains, write_file
from IORlib.ECL import Input_file as ECL_input, unfmt_file

QDir.addSearchPath('icons', resource_path()/'icons/')

# class WebEngineView(QWebEngineView):
#     def javaScriptConsoleMessage(self, level, msg, line, sourceID):
#         pass


#===========================================================================
class GUI_color(QColor):
#===========================================================================
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def as_hex(self):
        return '#'+hex(self.rgb()).split('0xff')[-1]

#===========================================================================
class color:
#===========================================================================
    black  = GUI_color(00,00,00)
    white  = GUI_color(255,255,255)
    blue   = GUI_color(31,119,180)  #1f77b4 
    orange = GUI_color(255,127,14)  #ff7f0e  
    green  = GUI_color(44,160,44)   #2ca02c
    red    = GUI_color(214,39,40)   #d62728
    violet = GUI_color(148,103,189) #9467bd
    brown  = GUI_color(140,86,75)   #8c564b
    pink   = GUI_color(227,119,194) #e377c2
    gray   = GUI_color(127,127,127) #7f7f7f
    yellow = GUI_color(188,189,34)  #bcbd22
    turq   = GUI_color(23,190,207)  #17becf
    as_tuple = (blue, orange, green, red, violet, brown, pink, gray, yellow, turq)


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
        #act.setIcon(QIcon(':'+icon))
        act.setIcon(QIcon(f'icons:{icon}'))
    act.triggered.connect(func)
    #act.changed.connect(func)
    return act


#-----------------------------------------------------------------------
def get_species_iorsim(root, raise_error=True):
#-----------------------------------------------------------------------
    file = f'{root}.trcinp'
    if not Path(file).is_file():
        if raise_error:
            raise SystemError(f'ERROR {Path(file).name} is missing')
        else:
            return []
    species = get_keyword(file, '\*solution', end='\*')
    #print(species)
    if species:
        species = species[0][1::2]
    else:
        # Read old input format
        species = flat_list(get_keyword(file, '\*SPECIES', end='\*'))
        species = [s for s in species if isinstance(s, str)]
    # Change pH to H 
    species = [s if s.lower() != 'ph' else 'H' for s in species]
    return species

#-----------------------------------------------------------------------
def get_tracers_iorsim(root, raise_error=True):
#-----------------------------------------------------------------------
    file = f'{root}.trcinp'
    if not Path(file).is_file():
        if raise_error:
            raise SystemError(f'ERROR {Path(file).name} is missing')
        else:
            return []
    tracers = flat_list(get_keyword(file, '\*NAME', end='\*'))
    if tracers:
        tracers = [t+f for t in tracers for f in ('_wat', '_oil', '_gas')]
    #print(tracers)
    return tracers
                
#-----------------------------------------------------------------------
def get_wells_iorsim(root):
#-----------------------------------------------------------------------
    file = f'{root}.trcinp'
    #print(file)
    if not Path(file).is_file():
        return [],[]
    in_wells, out_wells = [], []
    out_wells = flat_list(get_keyword(file, '\*PRODUCER', end='\*'))
    in_wells = flat_list(get_keyword(file, '\*INJECTOR', end='\*'))
    if not out_wells or not in_wells:
        # Read old input format
        ow = get_keyword(file, '\*OUTPUT', end='\*')
        if ow:
            out_wells = ow[0][1:]
        w = get_keyword(file, '\*WELLSPECIES', end='\*')
        if w and w[0]:
            #print(w)
            w = w[0]
            in_wells = w[1:1+int(w[0])]
    #print(out_wells, in_wells)
    return out_wells, in_wells


#-----------------------------------------------------------------------
def get_eclipse_well_yaxis_fluid(root):
#-----------------------------------------------------------------------
    fil = str(root)+'.DATA'
    if not Path(fil).is_file():
        raise SystemError(fil + ' is missing!')
    summary = well = False
    vars = []
    wells = []
    encoding = None
    try:
        with open(fil) as f:
            for line in f:
                l = f.readline()
    except UnicodeDecodeError:
        #encoding = 'ISO-8859-1'
        encoding = 'latin-1'
    with open(fil, encoding=encoding) as f:
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
def show_message(window, kind, text='', extra='', ok_text=None, wait=False, detail=None, button=None):
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
    #if width:
    #msg.setBaseSize(QSize(width, height))
    #msg.setStyleSheet('QLabel{min-width: '+ str(width) +'px;}')
    #msg.setMinimumWidth(width)
    msg.setIcon(icon)
    msg.setText(text)
    msg.setInformativeText(extra)
    if detail:
        msg.setdetailedText(detail)
    msg.setStandardButtons(QMessageBox.Ok)
    if ok_text:
        ok_btn = msg.button(QMessageBox.Ok)
        ok_btn.setText(ok_text)
    if button:
        btn_txt, btn_func = button
        def func():
            msg.done(1) # Better than msg.close()
            btn_func()
        btn = msg.addButton(btn_txt, QMessageBox.YesRole)
        btn.clicked.disconnect()
        btn.clicked.connect(func)
    if wait:
        msg.exec_()
    else:
        msg.show()
    return msg

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
    # return 'rgb{}'.format( tuple(asarray(asarray(colors_to_rgb(color))*255, dtype='int')) )
    return f'rgb{tuple(asarray(asarray(colors_to_rgb(color))*255, dtype="int"))}'
    

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
        self.print_exception = self.kwargs.get('print_exception')
        if self.print_exception is None:
            self.print_exception = True

    @Slot()
    #-----------------------------------------------------------------------
    def run(self):
    #-----------------------------------------------------------------------
        try:
            result = self.runnable()
        except:
            if self.print_exception:
                print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals and self.signals.error.emit((exctype, value, format_exc()))
        else:
            self.signals and self.signals.result.emit(result)
        finally:
            self.signals and self.signals.finished.emit()


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
                count = f'{run.t:.2f}'.rstrip('0').rstrip('.')
                if self.progress_min:
                    count = f'({self.progress_min:.0f} + {(run.t-self.progress_min):.2f}'.rstrip('0').rstrip('.')+')'
                value = f'{run.name}   {count} / {run.T:.0f} days'
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
            #text and self.show_message(text)
            self.show_message(text)

        result, msg = False, ''
        self.sim = Simulation(status=status, progress=progress, plot=plot, message=message, **self.kwargs)
        if self.sim.ready():
            self.days_box.setText(str(int(self.sim.get_time())))
            result, msg = self.sim.run()
        self.show_message(msg)
        return result

#===========================================================================
class download_worker(base_worker):
#===========================================================================
    #
    #  Download updated executable in update_dir as a separate process
    #
    #-----------------------------------------------------------------------
    def __init__(self, new_version, folder):
    #-----------------------------------------------------------------------
        super().__init__(print_exception=False)
        self.running = False         
        self.new_version = new_version
        folder = Path(folder)
        if this_file.suffix == '.py':
            # Download a zip archive 
            ext = '.zip'
            self.url = 'https://api.github.com/repos/janlv/IORSim_GUI/zipball/v' + new_version
        else:
            # Download the compiled executable
            ext = this_file.suffix
            self.url = github_url + 'releases/download/v' + new_version + '/' + this_file.name 
        self.savename = folder/(this_file.stem + '_v' + new_version + ext)

    #-----------------------------------------------------------------------
    def runnable(self):
    #-----------------------------------------------------------------------
        self.running = True                     # Set to False to abort download
        resp = requests_get(self.url, stream=True, verify=False)
        if not resp.status_code == 200:
            raise SystemError(f'{self.url} not found!')
        tot_size = int(resp.headers.get('content-length', 0)) or len(resp.content)
        block_size = 1024
        self.update_progress((-tot_size, None, None))
        self.status_message(f'Downloading version {self.new_version} of {this_file}')
        size = 0
        with open(self.savename, 'wb') as file:
            for data in resp.iter_content(block_size):
                if not self.running:
                    return
                size += len(data)
                #print(f'\r{size/tot_size:.2f}%', end='')
                self.update_progress((size, None, None))
                file.write(data)
        if tot_size != 0 and tot_size != size:
            msg = f'Size mismatch when downloading {self.filename}: got {size} bytes, expected {tot_size} bytes'
            raise SystemError(msg)

#===========================================================================
class Mpl_canvas(FigureCanvasQTAgg):                                              
#===========================================================================
    #-----------------------------------------------------------------------
    def __init__(self, parent=None, width=5, height=4, dpi=80):
    #-----------------------------------------------------------------------
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        #self.axes = fig.add_subplot(subplot)
        super(Mpl_canvas, self).__init__(self.fig)


                
#===========================================================================
class User_input(QDialog):                                              
#===========================================================================
    #-----------------------------------------------------------------------
    def __init__(self, parent=None, title=None, head=None, label=None, text=None, delete_src=False, size=(400, 150)):
    #-----------------------------------------------------------------------
        #super(User_input, self).__init__(*args, **kwargs)
        super(User_input, self).__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumSize(*size)
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.func = None
        intro = QLabel()
        intro.setText(head)
        intro.setWordWrap(True)
        self.layout.addWidget(intro)
        ### input
        if label and text:
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
class Plot(QGroupBox):                                              
#===========================================================================
    #-----------------------------------------------------------------------
    def __init__(self, parent=None):
    #-----------------------------------------------------------------------
        super(Plot, self).__init__(parent)
        self.setTitle('Plot')
        self.name = 'plot'
        self.setObjectName('plot')
        layout = QVBoxLayout()
        self.setLayout(layout)
        self.canvas = Mpl_canvas(self)
        toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(toolbar)
        layout.addWidget(self.canvas)

    #-----------------------------------------------------------------------
    def file_is_open(self, filename):
    #-----------------------------------------------------------------------
        return False

#===========================================================================
class Menu(QGroupBox):                                              
#===========================================================================
    #-----------------------------------------------------------------------
    def __init__(self, parent=None, title='', ncol=2):
    #-----------------------------------------------------------------------
        super(Menu, self).__init__(parent)
        self.setTitle(title)
        self.setLayout(QHBoxLayout())
        for i in range(ncol):
            self.layout().addLayout(QVBoxLayout())
            self.layout().itemAt(i).setAlignment(Qt.AlignTop)

    #-----------------------------------------------------------------------
    def column(self, nr):
    #-----------------------------------------------------------------------
        return self.layout().itemAt(nr)


# #===========================================================================
# class Log_viewer_with_level(QPlainTextEdit):
# #===========================================================================
#     #-----------------------------------------------------------------------
#     def contextMenuEvent(self, e: QContextMenuEvent) -> None:
#     #-----------------------------------------------------------------------
#         menu = self.createStandardContextMenu(e.globalPos())
#         menu.addSeparator()
#         level = menu.addMenu('Log level')
#         actions = [QAction(str(l+1), self) for l in range(3)]
#         group = QActionGroup(self)
#         for act in actions:
#             level.addAction(act)
#             group.addAction(act)
#         actions[0].triggered.connect(lambda: print(1))
#         actions[1].triggered.connect(lambda: print(2))
#         actions[2].triggered.connect(lambda: print(3))
#         actions[2].setChecked(True)
#         menu.popup(e.globalPos())
#         # if group.checkedAction():
#         #    group.checkedAction().trigger()

#     def set_log_value_getter(self, func):
#         self.get_log_value = func

#     # def change_log_level(self):
#     #     win = QDialog()
#     #     layout = QVBoxLayout()
#     #     win.setLayout(layout)
#     #     text = QLabel(f'Log level is {self.get_log_value()}')
#     #     layout.addWidget(text)
#     #     win.open()

#===========================================================================
class Editor(QGroupBox):                                              
#===========================================================================
    #-----------------------------------------------------------------------
    def __init__(self, parent=None, name='', read_only=False, save=True, save_func=None, browser=False,
                 top=True, top_name='Top', end=True, search=True, search_width=None,
                 refresh=True, space=0, match_case=False):
    #-----------------------------------------------------------------------
        super(Editor, self).__init__(parent)
        self.btn_width = 60
        self.btn_height = 25
        self.vscroll = {}
        self.name = name
        self.file = None
        self.search = search
        self.browser = browser
        self.refresh = refresh
        self.read_only = read_only
        self.save = save
        self.save_func = save_func
        self.top = top
        self.top_name = top_name
        self.end = end
        self.search_width = search_width
        self.space = space
        self.match_case = match_case
        # if name == 'app_log_viewer':
        #     self.editor_ = Log_viewer_with_level()
        # else:
        self.editor_ = QPlainTextEdit()
        self.editor_.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.set_text = self.editor_.setPlainText
        self.init_UI()

    #-----------------------------------------------------------------------
    def init_UI(self):                                           # Editor
    #-----------------------------------------------------------------------
        layout = QVBoxLayout()
        self.setLayout(layout)
        buttons = QHBoxLayout()
        layout.addLayout(buttons)
        layout.addWidget(self.editor_)
        if self.refresh:
            ### Refresh button
            self.refresh_btn = self.new_button(text='Refresh', func=self.refresh_func)
            buttons.addWidget(self.refresh_btn)
        if self.read_only:
            self.editor_.setReadOnly(True)
        if not self.read_only and self.save:
            self.editor_.textChanged.connect(self.activate_save)
            ### Save button
            if self.save_func:
                self.save_btn = self.new_button(text='Save', func=partial(self.save_text, save_func=self.save_func))            
            else:
                self.save_btn = self.new_button(text='Save', func=self.save_text)
            buttons.addWidget(self.save_btn)
            ### Undo button
            self.undo_btn = self.new_button(text='Undo', func=self.undo)
            buttons.addWidget(self.undo_btn)
            ### Redo button
            self.redo_btn = self.new_button(text='Redo', func=self.redo)
            buttons.addWidget(self.redo_btn)
        if self.top:
            ### Top button
            L = len(self.top_name)
            width = self.btn_width
            if L > 6:
                width = L*10
            self.top_btn = self.new_button(text=self.top_name, width=width, func=self.goto_top)
            buttons.addWidget(self.top_btn)
        if self.end:
            ### End button
            self.end_btn = self.new_button(text='End', func=self.goto_end)
            buttons.addWidget(self.end_btn)
        if self.space:
            lbl = QLabel(' ')
            lbl.setFixedWidth(self.space)
            buttons.addWidget(lbl)
        if self.search:
            ### Search field
            self.search_pos = []
            self.search_field = QLineEdit()
            self.search_field.setFixedHeight(self.btn_height)
            if self.search_width:
                self.search_field.setFixedWidth(self.search_width)
            self.search_field.setClearButtonEnabled(True) 
            self.search_field.setPlaceholderText('Search text')
            self.search_field.textChanged.connect(self.search_text)
            buttons.addWidget(self.search_field)
            ### Match case
            if self.match_case:
                lbl = QLabel()
                lbl.setText('Match case')
                buttons.addWidget(lbl)
                self.case_box = QCheckBox()
                buttons.addWidget(self.case_box)
            ### Prev button
            self.prev_btn = self.new_button(text='Prev', func=self.search_prev)
            buttons.addWidget(self.prev_btn)
            ### Next button
            self.next_btn = self.new_button(text='Next', func=self.search_next)
            buttons.addWidget(self.next_btn)
            ### Change cursor color
            palette = self.editor_.palette()
            palette.setColor(QPalette.Highlight, QColor(*[int(i*255) for i in (0,0,1,1)]))
            palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
            self.editor_.setPalette(palette)

    #-----------------------------------------------------------------------
    def update(self, file):
    #-----------------------------------------------------------------------
        #print(file)
        self.set_text(read_file(file))
        self.editor_.moveCursor(QTextCursor.End)

    #-----------------------------------------------------------------------
    def file_is_open(self, filename):
    #-----------------------------------------------------------------------
        if str(filename).lower() == str(self.file).lower():
            return True
        return False

    #-----------------------------------------------------------------------
    def document(self):                                              # Editor
    #-----------------------------------------------------------------------
        return self.editor_.document()

    #-----------------------------------------------------------------------
    def new_button(self, text=None, icon=None, func=None, width=None):          # Editor
    #-----------------------------------------------------------------------
        btn = None
        if icon:
            #btn = QPushButton(icon=QIcon(':'+icon))
            btn = QPushButton(icon=QIcon(f'icons:{icon}'))
        else:
            btn = QPushButton(text=text)
        if not width:
            width = self.btn_width
        btn.setFixedWidth(width)
        btn.setFixedHeight(self.btn_height)
        btn.clicked.connect(func)
        return btn

    #-----------------------------------------------------------------------
    def goto_end(self):                                            # Editor
    #-----------------------------------------------------------------------
        self.refresh_func()
        self.editor_.moveCursor(QTextCursor.End)

    #-----------------------------------------------------------------------
    def goto_top(self):                                            # Editor
    #-----------------------------------------------------------------------
        self.editor_.moveCursor(QTextCursor.Start)
        
    #-----------------------------------------------------------------------
    def search_next(self):                                            # Editor
    #-----------------------------------------------------------------------
        #print(self.search_string+' '+str(self.search_pos))
        start = 0
        if self.search_pos:
            start = self.search_pos[-1]
        self.search_text(self.search_field.text(), start=start)
        
    #-----------------------------------------------------------------------
    def search_prev(self):                                           # Editor
    #-----------------------------------------------------------------------
        #print(self.search_string+' '+str(self.search_pos))
        if self.search_pos:
            string = self.search_field.text()
            if len(self.search_pos)==1:
                pos = self.search_pos[0]
            else:
                pos = self.search_pos.pop()
                if self.editor_.textCursor().position() == pos:
                    pos = self.search_pos.pop()
            self.set_cursor(pos-len(string), string)
        
    #-----------------------------------------------------------------------
    def search_text(self, string, start=0, ignore_case=True):   # Editor
    #-----------------------------------------------------------------------
        #print('search_text: '+string+' '+str(start))
        if start==0:
            self.search_pos = []          
        text = self.editor_.toPlainText()
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
    def set_cursor(self, start, string, center=True):           # Editor
    #-----------------------------------------------------------------------
        cursor = self.editor_.textCursor()
        cursor.setPosition(start)
        cursor.setPosition(start+len(string), QTextCursor.KeepAnchor)   
        #print(cursor.blockNumber(), cursor.columnNumber())
        self.editor_.setTextCursor(cursor)
        if center:
            cursor = self.editor_.cursorRect()
            cursor_line = int(cursor.y()/cursor.height())
            page_height = self.editor_.geometry().height()-self.editor_.horizontalScrollBar().height()
            mid_line = int(0.5*page_height/cursor.height())
            shift = cursor_line-mid_line
            vbar = self.editor_.verticalScrollBar()
            newpos = shift + vbar.value()
            if newpos>0:
                vbar.setValue(newpos)            
                #print(newpos)
        
    #-----------------------------------------------------------------------
    def save_text(self, save_func=None):            # Editor
    #-----------------------------------------------------------------------
        #print('save_text')
        write_file(self.file, self.editor_.toPlainText())
        self.save_btn.setEnabled(False)
        if save_func:
            save_func()
        #self.prepare_case(self.case)

    #-----------------------------------------------------------------------
    def activate_save(self):                                           # Editor
    #-----------------------------------------------------------------------
        #print(self.sender())
        self.save_btn.setEnabled(True)
        
    #-----------------------------------------------------------------------
    def undo(self):                                         # Editor
    #-----------------------------------------------------------------------
        self.editor_.undo()
            
    #-----------------------------------------------------------------------
    def redo(self):                                           # Editor
    #-----------------------------------------------------------------------
        self.editor_.redo()

    #-----------------------------------------------------------------------
    def clear(self):                                             # Editor
    #-----------------------------------------------------------------------
        self.view_file(None)

    #-----------------------------------------------------------------------
    def clear_search(self):
    #-----------------------------------------------------------------------
        if self.search:
            self.search_field.setText('')
            self.search_field.setPlaceholderText('Search text')

    #-----------------------------------------------------------------------
    def view_file(self, file, title=''):                            # Editor
    #-----------------------------------------------------------------------
        curr_file = self.file
        if curr_file:
            self.vscroll[curr_file] = self.editor_.verticalScrollBar().value()
        # Clear search field
        self.clear_search()
        # Avoid re-opening file after it is saved
        if self.file_is_open(file):
            return
        #self.setObjectName(str(file))
        self.file = str(file)
        self.highlight = None
        text = ''
        if file and Path(file).is_file():
            text = read_file(file)
        self.set_text(text)
        vscroll = self.vscroll.get(str(file)) or 0
        self.editor_.verticalScrollBar().setValue(vscroll)
        self.setTitle(title)

    #-----------------------------------------------------------------------
    def refresh_func(self):                                           # Editor
    #-----------------------------------------------------------------------
        #file = self.objectName()
        self.vscroll[self.file] = self.editor_.verticalScrollBar().value()
        if self.file and Path(self.file).is_file():
            #self.setObjectName(str(file))
            self.set_text(read_file(self.file))
        vscroll = self.vscroll.get(str(self.file)) or 0
        self.editor_.verticalScrollBar().setValue(vscroll)


#===========================================================================
class PDF_viewer(Editor):                                              
#===========================================================================
    #-----------------------------------------------------------------------
    def __init__(self, *args, **kwargs):
    #-----------------------------------------------------------------------
        super().__init__(*args, search=True, save=False, refresh=False, top=False, end=False, **kwargs)

    #-----------------------------------------------------------------------
    def init_editor(self, layout):                          # PDF_viewer
    #-----------------------------------------------------------------------
        self.editor_ = QWebEngineView(self)
        self.editor_.settings().setAttribute(QWebEngineSettings.PluginsEnabled, True)
        self.editor_.settings().setAttribute(QWebEngineSettings.PdfViewerEnabled, True)
        layout.addWidget(self.editor_)

    #-----------------------------------------------------------------------
    def view_file(self, file, title=''):                    # PDF_viewer
    #-----------------------------------------------------------------------
        self.clear_search()
        self.file = str(file)
        self.editor_.setUrl(QUrl(file))
        self.editor_.setWindowTitle(title)

    #-----------------------------------------------------------------------
    def search_text(self, string, start=0, ignore_case=True):    # PDF_viewer
    #-----------------------------------------------------------------------
        #flags = QWebEnginePage.FindFlag(0)
        #if self.case_box.isChecked():
        #    flags = QWebEnginePage.FindCaseSensitively
        #print(flags)
        self.editor_.findText(string)

    #-----------------------------------------------------------------------
    def search_next(self):                                      # PDF_viewer
    #-----------------------------------------------------------------------
        self.search_text(self.search_field.text())

    #-----------------------------------------------------------------------
    def search_prev(self):                                      # PDF_viewer
    #-----------------------------------------------------------------------
        flags = QWebEnginePage.FindBackward
        #if self.case_box.isChecked():
        #    flags = flags or QWebEnginePage.FindCaseSensitively
        self.editor_.findText(self.search_field.text(), flags)

    #-----------------------------------------------------------------------
    def refresh_func(self):                                      # PDF_viewer
    #-----------------------------------------------------------------------
        pass

#===========================================================================
class Window(QMainWindow):                                              
#===========================================================================
    #-----------------------------------------------------------------------
    def __init__(self, parent=None, widget=None, title='', size=(1000, 500)):
    #-----------------------------------------------------------------------
        super(Window, self).__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumSize(*size)
        self.widget = widget
        self.setCentralWidget(widget)

#===========================================================================
class Settings(QDialog):                                              
#===========================================================================
    #-----------------------------------------------------------------------
    def __init__(self, parent=None, file=None):                   # settings
    #-----------------------------------------------------------------------
        super(Settings, self).__init__(parent)
        self.parent = parent
        self.file = Path(file) 
        self.setWindowTitle('Settings')        
        self.setWindowFlag(Qt.WindowTitleHint)
        #self.setMinimumSize(400,400)
        self.setObjectName('settings_window')
        self.line = -1
        self._get = {}
        self._set = {}
        variable = namedtuple('variable',        'text              default tip                            required ')
        self.vars = {'iorsim'         : variable('IORSim program' , None, 'Path to the IORSim executable', True),
                     'eclrun'         : variable('Eclipse program', 'eclrun', "Eclipse command, default is 'eclrun'", True), 
                     'workdir'        : variable('Case-folder', str(default_casedir),'Path to GUI-folder', True),
                     'savedir'        : variable('Download-folder', None, 'Download location for updates', False),
                     'check_input_kw' : variable('Check IORSim input file', False, 'Check IORSim input file keywords', False),
                     'convert'        : variable('Convert to unformatted output', True, 'Convert IORSim formatted output to unformatted format (readable by ResInsight)', False),
                     'del_convert'    : variable('Delete original after convert', True, 'Delete the FUNRST-file if it is successfully converted to an UNRST-file', False),
                     'merge'          : variable('Merge Eclipse and IORSim output', True, 'Merge the unformatted output from Eclipse and IORSim into one file', False),
                     'del_merge'      : variable('Delete originals after merge', True, 'Delete the original UNRST-files from Elipse and IORSim if successfully merged', False),
                     'unrst'          : variable('Confirm flushed UNRST-file during Eclipse step', True, 'Check that the UNRST-file is properly flushed before suspending Eclipse', False), 
                     'rft'            : variable('Confirm flushed RFT-file during Eclipse step', True, 'Check that the RFT-file is properly flushed before suspending Eclipse', False),
                     'ecl_keep_alive' : variable(f'Eclipse process not paused if idle time is less than', False, f'Delay pausing the Eclipse process during idle time between steps (Expert mode)', False),
                     'ecl_alive_limit': variable(f'seconds', str(ECL_ALIVE_LIMIT), f'Set this limit lower than 100 seconds to avoid unexpected Eclipse termination (Expert mode)', False),
                     'ior_keep_alive' : variable(f'IORSim process not paused when idle', False, f'Never pause IORSim during idle time between steps (Expert mode)', False),
                     'log_level'      : variable('Detail level of the application log', str(DEFAULT_LOG_LEVEL), 'A higher value gives a more detailed application log', False)}
        self.required = [k for k,v in self.vars.items() if v.required]
        self.expert = []
        self.abs_path = False
        self.initUI()
        self.set_expert_mode(False)
        self.set_default()
        self.load()

    #-----------------------------------------------------------------------
    def set_expert_mode(self, enabled):
    #-----------------------------------------------------------------------
        self.expert_mode = enabled
        [var.setEnabled(self.expert_mode) for var in self.expert]


    #-----------------------------------------------------------------------
    def keyPressEvent(self, event):
    #-----------------------------------------------------------------------
        """ Catch pressed e-key to enable/disable expert settings """
        if not event.isAutoRepeat() and event.key() == Qt.Key_E:
            self.set_expert_mode(not self.expert_mode)
        #if event.key() == Qt.Key_F:
        #    self.parent.show_message_text(f'INFO Settings are saved in {self.file}')

    #-----------------------------------------------------------------------
    def get(self, kw):                                            # settings
    #-----------------------------------------------------------------------
        return self._get[kw]()


    #-----------------------------------------------------------------------
    def set(self, kw, value):                                     # settings
    #-----------------------------------------------------------------------
        return self._set[kw](value)


    #-----------------------------------------------------------------------
    def initUI(self):                                             # settings
    #-----------------------------------------------------------------------
        self.grid = QGridLayout()
        self.setLayout(self.grid)
        self.grid.setColumnStretch(0,15) 
        self.grid.setColumnStretch(1,70) 
        self.grid.setColumnStretch(2,15) 

        ### IORSim executable
        self.add_line_with_button(var='iorsim', open_func=self.open_ior_prog)
        ### Eclipse executable
        self.add_line_with_button(var='eclrun', open_func=self.open_ecl_prog)
        ### Workdir        
        self.add_line_with_button(var='workdir', open_func=self.change_workdir)
        ### Savedir        
        self.add_line_with_button(var='savedir', open_func=self.change_savedir)

        ### Input options
        self.add_heading()
        self.add_heading('Input options')
        ### Check IORSim input format
        self.add_items([self.new_checkbox('check_input_kw')])

        ### Output options
        self.add_heading()
        self.add_heading('Output options')
        ### Convert and merge
        self.add_items([self.new_checkbox(var) for var in ('convert', 'del_convert', 'merge', 'del_merge')], nrow=2)

        ### Backward options
        self.add_heading()
        self.add_heading('Backward options')
        # UNRST and RFT checks
        self.add_items([self.new_checkbox(var) for var in ('unrst', 'rft')], nrow=2)
        # Keep processes alive between steps? 
        # Expert mode: Type 'e' to edit  
        cb = [self.new_checkbox(var) for var in ('ecl_keep_alive', 'ior_keep_alive')]
        le = self.new_lineedit('ecl_alive_limit', width=30)
        # Add widget in layout to expert mode
        #self.expert.append(le.itemAt(0).widget())
        self.expert.extend( (le.itemAt(i).widget() for i in range(le.count())) )
        self.expert.extend(cb)
        self.add_items([cb[0], le])
        self.add_items([cb[1]])

        ### Log options
        self.add_heading()
        self.add_heading('Log options')
        ### Log verbosity level 
        self.add_items([self.new_combobox(var='log_level', width=50, values=[str(i) for i in range(LOG_LEVEL_MIN, LOG_LEVEL_MAX+1)])])

        ### OK / Cancel buttons
        self.add_heading()
        self.line += 1
        yes_no = QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        self.yes_no_btns = QDialogButtonBox(yes_no)
        self.grid.addWidget(self.yes_no_btns, self.line, 0, 1, 3)
        self.yes_no_btns.accepted.connect(self.on_OK_click)
        self.yes_no_btns.rejected.connect(self.reject)
        self.yes_no_btns.setFocus()


    #-----------------------------------------------------------------------
    def add_heading(self, text=''):
    #-----------------------------------------------------------------------
        self.line += 1
        self.grid.addWidget(QLabel(text), self.line, 0)


    #-----------------------------------------------------------------------
    def add_items(self, items, nrow=1):
    #-----------------------------------------------------------------------
        self.line += 1
        layout = QGridLayout()
        self.grid.addLayout(layout, self.line, 1)
        ncol = int(len(items)/nrow)
        for i,item in enumerate(items):
            col, row = i%ncol, int(i/ncol)
            if item.isWidgetType():
                layout.addWidget(item, row, col)
            else:
                layout.addLayout(item, row, col)
        return items

    #-----------------------------------------------------------------------
    def new_combobox(self, width=None, var=None, values=[]):             # settings
    #-----------------------------------------------------------------------
        v = self.vars[var]
        layout = QHBoxLayout()
        box = QComboBox()
        box.setToolTip(v.tip)
        width and box.setFixedWidth(width)
        layout.addWidget(box)
        label = QLabel(v.text)
        label.setToolTip(v.tip)
        layout.addWidget(label)
        box.addItems(values)
        self._get[var] = box.currentText
        self._set[var] = box.setCurrentText
        return layout

    #-----------------------------------------------------------------------
    def new_checkbox(self, var=None):             # settings
    #-----------------------------------------------------------------------
        v = self.vars[var]
        box = QCheckBox(v.text)
        box.setToolTip(v.tip)
        self._get[var] = box.isChecked
        self._set[var] = box.setChecked
        return box

    #-----------------------------------------------------------------------
    def new_lineedit(self, var=None, width=None):             # settings
    #-----------------------------------------------------------------------
        v = self.vars[var]
        layout = QHBoxLayout()
        line = QLineEdit()
        width and line.setFixedWidth(width)
        line.setToolTip(v.tip)
        layout.addWidget(line)
        layout.addWidget(QLabel(v.text))
        self._get[var] = line.text
        self._set[var] = line.setText
        return layout

    #-----------------------------------------------------------------------
    def add_line_with_button(self, var=None, open_func=None, button_text='Change', read_only=True):    # settings
    #-----------------------------------------------------------------------
        self.line += 1
        v = self.vars[var]
        label = QLabel(v.text)
        self.grid.addWidget(label, self.line, 0)
        label.setToolTip(v.tip)
        line = QLineEdit()
        self.grid.addWidget(line, self.line, 1)
        line.setToolTip(v.tip)
        line.setReadOnly(read_only)
        self._get[var] = line.text 
        self._set[var] = line.setText
        if open_func:
            button = QPushButton(button_text)
            self.grid.addWidget(button, self.line, 2)
            button.clicked.connect(open_func)
        return line

        
    #-----------------------------------------------------------------------
    def set_default(self):                                # settings
    #-----------------------------------------------------------------------
        for k,v in self.vars.items():
            self._set[k](v.default)
        
    #-----------------------------------------------------------------------
    def change_workdir(self):                                  # settings
    #-----------------------------------------------------------------------
        dirname = QFileDialog.getExistingDirectory(self, 'Locate or create a directory for the case-files', 
                                                        str(Path.cwd()), QFileDialog.ShowDirsOnly)
        if dirname:
            self._set['workdir'](dirname)
            self.parent.update_casedir()

    #-----------------------------------------------------------------------
    def change_savedir(self):                             # settings
    #-----------------------------------------------------------------------
        title = 'Choose a download location for the new version'
        default = self.get('savedir') or str(Path.cwd())
        folder = QFileDialog.getExistingDirectory(self, title, default, QFileDialog.ShowDirsOnly)
        if folder:
            self._set['savedir'](folder)

    #-----------------------------------------------------------------------
    def restart_now(self):                                 # settings
    #-----------------------------------------------------------------------
        msg = f'The application must restart to apply the new case directory. Restart now?'
        restart = User_input(self, title='Restart application?', head=msg)
        restart.set_func(self.parent.reboot)
        restart.open()

    #-----------------------------------------------------------------------
    def open_ior_prog(self):                                 # settings
    #-----------------------------------------------------------------------
        fname = open_file_dialog(self, 'Locate IORSim program', 'All Files (*)')
        if fname:
            if not self.abs_path:
                try:
                    fname = str(Path(fname).relative_to(Path.cwd()))
                except ValueError:
                    pass
            self._set['iorsim'](fname)
            
    #-----------------------------------------------------------------------
    def open_ecl_prog(self):                                  # settings
    #-----------------------------------------------------------------------
        fname = open_file_dialog(self, 'Locate eclrun program', 'All Files (*)')
        if fname:
            self._set['eclrun'](fname)

    #-----------------------------------------------------------------------
    def on_OK_click(self):                                # settings
    #-----------------------------------------------------------------------
        self.done(1)
        if not self.save():
            self.parent.show_message_text(f"WARNING Unable to save settings-file {self.file}")
        else:
            self.parent.update_message(f'Settings saved in {self.file}')

    #-----------------------------------------------------------------------
    def done(self, value):                                # settings
    #-----------------------------------------------------------------------
        self.set_expert_mode(False)
        super().done(value)


    #-----------------------------------------------------------------------
    def save(self):                                      # settings
    #-----------------------------------------------------------------------
        self.file.touch(exist_ok=True)
        with open(self.file, 'w') as f:
            f.write('# This is a settings-file for ior2ecl_GUI.py, do not edit.\n')
            for var,val in self._get.items():
                # if var in self.required and len(val())==0:
                #     show_message(self, 'error', text=var+' cannot be empty!')
                #     return False
                f.write(f'{var} {val()}\n')
                #print(f'{var} {val()}')
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
                        try:
                            self._set[var.strip()]( str_to_bool(val.strip()) )
                        except KeyError:
                            pass

                        
        
#===========================================================================
class main_window(QMainWindow):                                    # main_window 
#===========================================================================
    EXIT_CODE_REBOOT = -123

    #-----------------------------------------------------------------------
    def reboot(self):                                          # main_window
    #-----------------------------------------------------------------------
        #for child in qApp.allWidgets():
        #    if child.objectName()=='settings_window':
        #        child.close()
        QCoreApplication.exit(main_window.EXIT_CODE_REBOOT)

    #-----------------------------------------------------------------------
    def __init__(self, *args, settings_file=None, **kwargs):                       # main_window
    #-----------------------------------------------------------------------
        super(main_window, self).__init__(*args, **kwargs)
        self.setWindowTitle('IORSim') 
        screen = self.screen().geometry() #size()
        saved_geo = get_keyword(settings_file, 'geometry', comment='#')
        if saved_geo:
            geo = QRect(*saved_geo[0])
        else:
            size = QSize(900, 600)
            position = QPoint(int(0.5*(screen.width()-size.width())), int(0.5*(screen.height()-size.height())))
            geo = QRect(position, size)
        self.setGeometry(geo)
        self.setObjectName('main_window')
        #self.setContentsMargins(2,2,2,2)
        self.setWindowIcon(QIcon('icons:ior2ecl_icon.svg'))
        self.font = QFont().defaultFamily()
        self.menu_fontsize = 7
        self.plot_lines = None
        self.data = {}
        self.plot_ref_data = {}
        self.ecl_boxes = {}
        self.ior_boxes = {}
        #self.current_view = None
        self.days = None
        self.log_file = None
        self.unsmry = None
        self.worker = None
        self.download_worker = None
        self.convert = None
        self.max_3_checked = []
        self.plot_prop = {}
        self.checked_boxes = {}
        self.plotted_lines = {}
        self.view = False
        self.plot_ref = None
        self.progress = None
        self.casedir = None
        self.input_file = None
        # User guide window
        self.pdf_view = None #PDF_viewer()
        self.user_guide = None #Window(widget=self.pdf_view, title='IORSim User Guide', size=(1000, 800))
        self.case = None
        self.input = {'root':None, 'ecl_days':None, 'days':100, 'step':None, 'species':[], 'mode':None}
        self.input_to_save = ['root','days','mode']
        self.settings = Settings(self, file=str(settings_file))
        self.initUI()
        self.update_casedir()
        self.threadpool = QThreadPool()
        self.show()
        # Move window if upper left corner is outside limits
        # This check must come after self.show()
        pos = self.frameGeometry().topLeft().toTuple()
        if any([p<0 for p in pos]):
            x, y = [-min(p,0) for p in pos]
            self.setGeometry(self.geometry().adjusted(x, y, x, y))            
                

    #-----------------------------------------------------------------------
    def update_casedir(self):
    #-----------------------------------------------------------------------
        self.casedir = Path(self.settings.get('workdir'))
        self.casedir.mkdir(exist_ok=True)
        self.input_file = self.casedir/'.cache.txt' #input_file #gui_dir/'input.txt'
        self.load_input()
        self.set_input_field()

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
        self.set_act = create_action(self, text='&Settings', icon='gear.png', shortcut='Ctrl+S',
                                     tip='Edit settings', func=self.settings.open)
        self.start_act = create_action(self, text=None, icon='start.svg', shortcut='Ctrl+R',
                                       tip='Run simulation', func=self.run_sim)
        self.stop_act = create_action(self, text=None, icon='stop.svg', shortcut='Ctrl+E',
                                      tip='Stop simulation', func=self.killsim)
        self.iorsim_guide_act = create_action(self, text='IORSim User Guide', icon='lifebuoy.png', shortcut='',
                                      tip='IORSim User Guide', func=self.show_iorsim_guide)
        self.script_guide_act = create_action(self, text='GUI User Guide', icon='lifebuoy.png', shortcut='',
                                      tip='User guide for this application', func=self.show_script_guide)
        self.download_act = create_action(self, text='Check for updates', icon='drive-download.png', shortcut='',
                                      tip='Check if a new version is avaliable', func=self.check_version)
        self.about_act = create_action(self, text='About', icon='question.png', shortcut='',
                                      tip='Application details', func=self.about_app)
        self.exit_act = create_action(self, text='&Exit', icon='control-power.png', shortcut='Ctrl+Q',
                                      tip='Exit application', func=self.quit)
        self.add_case_act = create_action(self, text='Import case...', icon='document--plus.png',
                                          func=self.add_case_from_file)
        self.dupl_case_act = create_action(self, text='Duplicate current case...', icon='document-copy.png',
                                           func=self.duplicate_current_case)
        self.rename_case_act = create_action(self, text='Rename current case...', icon='document-rename.png',
                                             func=self.rename_current_case)
        self.clear_case_act = create_action(self, text='Clear current case', icon='document.png',
                                            func=self.clear_current_case)
        self.delete_case_act = create_action(self, text='Delete current case', icon='document--minus.png',
                                             func=self.delete_current_case)
        self.plot_act = create_action(self, text='Plot', icon='guide.png', func=self.view_plot, checkable=True)
        self.plot_act.setChecked(True)
        self.ecl_inp_act = create_action(self, text='Eclipse input file', icon='document-attribute-e.png',
                                         func=self.view_eclipse_input, checkable=True)
        self.ior_inp_act = create_action(self, text='IORSim input file', icon='document-attribute-i.png',
                                         func=self.view_iorsim_input, checkable=True)
        self.schedule_file_act = create_action(self, text='Schedule file', icon='document-attribute-s.png',
                                          func=self.view_schedule_file, checkable=True)
        self.ecl_log_act = create_action(self, text='Eclipse log', icon='script-attribute-e.png',
                                         func=self.view_eclipse_log, checkable=True)
        self.ior_log_act = create_action(self, text='IORSim log', icon='script-attribute-i.png',
                                         func=self.view_iorsim_log, checkable=True)
        self.py_log_act = create_action(self, text='Application log', icon='script-attribute.png',
                                        func=self.view_program_log, checkable=True)
        #self.log_level_act = [create_action(self, text=str(i), func=lambda o=self: print(o.text()), checkable=True) for i in range(3)]
        #self.log_level_act = []
        #self.log_level_act.append( create_action(self, text='1', func=lambda: print(1), checkable=True) )
        #self.log_level_act.append( create_action(self, text='2', func=lambda: print(2), checkable=True) )
        #self.log_level_act.append( create_action(self, text='3', func=lambda: print(3), checkable=True) )
                
        
    #-----------------------------------------------------------------------
    def create_menus(self):                                          # main_window
    #-----------------------------------------------------------------------
        ### menu
        menu = self.menuBar()
        file_menu = menu.addMenu('&File')
        file_menu.addAction(self.add_case_act)
        file_menu.addAction(self.dupl_case_act) 
        file_menu.addAction(self.rename_case_act)
        file_menu.addAction(self.clear_case_act)
        file_menu.addAction(self.delete_case_act)
        file_menu.addSeparator()
        file_menu.addAction(self.exit_act)
        edit_menu = menu.addMenu('&Edit')
        self.view_ag = QActionGroup(self)
        for act in (self.ecl_inp_act, self.ior_inp_act, self.schedule_file_act):
            edit_menu.addAction(act)
            self.view_ag.addAction(act)
        #self.chem_menu = edit_menu.addMenu(QIcon(':documents-stack'), 'Chemistry files')
        self.chem_menu = edit_menu.addMenu(QIcon('icons:documents-stack.png'), 'Chemistry files')
        edit_menu.addSeparator()
        edit_menu.addAction(self.set_act)
        view_menu = menu.addMenu('&View')
        self.view_ag.addAction(self.plot_act)
        view_menu.addAction(self.plot_act)
        view_menu.addSeparator()
        for act in (self.ecl_log_act, self.ior_log_act, self.py_log_act):
            view_menu.addAction(act)
            self.view_ag.addAction(act) 
        # view_menu.addSeparator()
        # log_level_menu = view_menu.addMenu('Log level')
        # self.log_level_group = QActionGroup(self)
        # for act in self.log_level_act:
        #     log_level_menu.addAction(act)
        #     self.log_level_group.addAction(act)
        help_menu = menu.addMenu('&Help')
        help_menu.addAction(self.iorsim_guide_act)
        help_menu.addAction(self.script_guide_act)
        help_menu.addSeparator()
        help_menu.addAction(self.download_act)
        # # Only allow download if we are running the 'compiled' version
        # if this_file.suffix == '.py':
        #     self.download_act.setEnabled(False)
        #     self.download_act.setStatusTip(self.download_act.statusTip() + ' (only available for compiled version)')
        help_menu.addSeparator()
        help_menu.addAction(self.about_act)


    #-----------------------------------------------------------------------
    def about_app(self):
    #-----------------------------------------------------------------------
        about = f'IORSim brings chemistry-based IOR methods to existing reservoir simulators with minor modifications of the original input files. IORSim currently cooperate with Eclipse 100, but can be made available for other reservoir simulators with some adaptations.'
        self.show_message_text('INFO ' + about, extra=f'Version : {__version__}')

    #-----------------------------------------------------------------------
    def check_version(self):
    #-----------------------------------------------------------------------
        r = requests_get(latest_release, verify=False, timeout=10)
        self.new_version = r.url.split('/')[-1].replace('v','')
        if self.new_version[0] == '.':
            self.new_version = self.new_version[1:]
        if float(self.new_version) > float(__version__):
            button = ('Download', self.download)
            msg = f'INFO The latest version is {self.new_version}, you are running {__version__}. Download latest version?'
            self.show_message_text(msg, button=button, ok_text='Not now')
        else:
            self.show_message_text(f'INFO No update available, {__version__} is the latest version')


    #-----------------------------------------------------------------------
    def download_finished(self):
    #-----------------------------------------------------------------------
        self.download_worker = None

    #-----------------------------------------------------------------------
    def download_error(self, values):
    #-----------------------------------------------------------------------
        exctype, value, trace = values
        self.show_message_text(f'ERROR Download of version {self.new_version} failed!\n\n{value}')

    #-----------------------------------------------------------------------
    def download_success(self):
    #-----------------------------------------------------------------------
        self.update_message('Download complete')
        button = ('Quit', self.close)
        #msg = f'INFO Download of version {self.new_version} completed, restart application to use it.'
        dest = self.download_worker.savename
        msg = f'INFO {dest.name} is now available in {dest.parent}.\nStop the application, copy the new version over the current one, and start again.'
        self.show_message_text(msg, button=button, ok_text='Not now')

    #-----------------------------------------------------------------------
    def download(self):
    #-----------------------------------------------------------------------
        old_folder = self.settings.get('savedir')
        self.settings.change_savedir()
        folder = self.settings.get('savedir')
        if not folder:
            self.show_message_text('INFO Download aborted due to missing save location')
            return None
        if folder != old_folder:
            self.settings.save()
        self.reset_progress_and_message()
        self.download_worker = download_worker(self.new_version, folder)
        self.download_worker.signals.status_message.connect(self.update_message)
        self.download_worker.signals.show_message.connect(self.show_message_text)
        self.download_worker.signals.progress.connect(self.update_progress)
        self.download_worker.signals.finished.connect(self.download_finished) 
        self.download_worker.signals.error.connect(self.download_error) 
        self.download_worker.signals.result.connect(self.download_success) 
        self.threadpool.start(self.download_worker)

    #-----------------------------------------------------------------------
    def show_guide(self, guide, title=''):
    #-----------------------------------------------------------------------
        if not self.pdf_view:
            self.pdf_view = PDF_viewer()
        self.pdf_view.view_file(guide, title=title)
        if not self.user_guide:
            self.user_guide = Window(widget=self.pdf_view, title='IORSim User Guide', size=(1000, 800))
        self.user_guide.show()
    
    #-----------------------------------------------------------------------
    def show_iorsim_guide(self):
    #-----------------------------------------------------------------------
        self.show_guide(iorsim_guide, title='IORSim User Guide')

    #-----------------------------------------------------------------------
    def show_script_guide(self):
    #-----------------------------------------------------------------------
        self.show_guide(script_guide, title='GUI User Guide')

    #-----------------------------------------------------------------------
    def create_toolbar(self):                                  # main_window
    #-----------------------------------------------------------------------
        ### toolbar
        self.toolbar = QToolBar('Toolbar')
        self.toolbar.setStyleSheet('QToolBar{spacing:15px; padding:5px;}')
        #self.toolbar.setIconSize(QSize(24, 24))
        self.toolbar.setIconSize(QSize(20, 20))
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
        self.case_cb.setStyleSheet('QComboBox {min-width: 100px;}')
        self.case_cb.currentIndexChanged[int].connect(self.on_case_select)
        # steps
        self.days_box = widgets['days']
        self.days_box.setFixedWidth(60)
        self.days_box.setObjectName('days')
        self.days_box.textChanged[str].connect(self.on_input_change)
        # reference
        self.ref_case = widgets['compare']
        self.ref_case.setObjectName('compare')
        self.ref_case.setStyleSheet('QComboBox {min-width: 100px;}')
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
        ### Central widget
        # Layout position given as (row, col, rowspan, colspan)
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
        widget = QWidget()
        #widget.setStyleSheet('QWidget {border: 1px solid black}')
        widget.setLayout(self.layout)
        self.setCentralWidget(widget)
        # Create IORSim plot menu
        self.ior_menu = Menu(title='IORSim plot options') 
        self.layout.addWidget(self.ior_menu, *self.position['ior_menu']) # 
        # Create ECLIPSE plot menu
        self.ecl_menu = Menu(title='ECLIPSE plot options')
        self.layout.addWidget(self.ecl_menu, *self.position['ecl_menu']) # 
        # Create plot- and file-view area
        self.plot = Plot() 
        self.editor = Editor(name='editor', save_func=self.prepare_case)
        self.log_viewer = Editor(name='log_viewer', read_only=True)
        #self.app_log_viewer = Editor(name='app_log_viewer', read_only=True)
        #self.app_log_viewer.editor_.set_log_value_getter(self.settings._get['log_level'])
        # Plot is the default view at startup
        self.layout.addWidget(self.plot, *self.position['plot'])



    #-----------------------------------------------------------------------
    def set_input_field(self):
    #-----------------------------------------------------------------------
        ### set values from input-file or default
        days = 100
        ### case
        self.cases = self.read_case_dir()
        self.case = self.input.get('root')
        self.create_caselist(choose=self.case)
        ### number of days
        self.days_box.setText(str(self.input.get('days') or days))
        
        
    #-----------------------------------------------------------------------
    def save_input_values(self):                                   # main_window
    #-----------------------------------------------------------------------
        """ Save the current input values in a cache-file """
        self.input_file.touch(exist_ok=True)
        #print(self.input_file)
        with open(self.input_file, 'w') as f:
            f.write('# This is an input-file for ior2ecl_GUI.py, do not edit.\n')
            for var in self.input_to_save:
                line = f'{var} {self.input.get(var) or ""}'
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

                            
    #-----------------------------------------------------------------------
    def set_variables_from_casefiles(self):                # main_window
    #-----------------------------------------------------------------------
        inp = self.input
        inp['ecl_days'] = inp['species'] = inp['tracers'] = None
        if inp['root']:
            tsteps = ECL_input(f'{inp["root"]}.DATA').get('TSTEP')
            inp['ecl_days'] = int(sum(tsteps))
            inp['species'] = get_species_iorsim(inp['root'], raise_error=False)
            inp['tracers'] = get_tracers_iorsim(inp['root'], raise_error=False)
            inp['species'] += inp['tracers']


    #-----------------------------------------------------------------------
    def set_plot_properties(self):                # main_window
    #-----------------------------------------------------------------------
        colors = color.as_tuple
        species = enumerate(self.input['species'] or [])
        prop = {}
        prop['color'] = {} #None
        prop['line'] = {} #None
        prop['alpha'] = {} #None
        species = self.input['species']
        if species:
            prop['color'] = {specie:colors[i%len(colors)].as_hex() for i,specie in enumerate(species)}
            prop['line'] = {specie:'-' for i,specie in enumerate(species)}
            prop['alpha'] = {specie:1.0 for i,specie in enumerate(species)}
        for var in ('Temp','Temp_ecl'):
            prop['color'][var] = color.black.as_hex() #'#000000' 
            prop['line'][var] = '--' 
            prop['alpha'][var] = 0.5
        var = 'Oil'
        prop['color'][var] = color.red.as_hex() #'#d62728' # red
        prop['line'][var] = '-' 
        prop['alpha'][var] = 1.0
        var = 'Water'
        prop['color'][var] = color.blue.as_hex() #'#1f77b4' # blue
        prop['line'][var] = '-' 
        prop['alpha'][var] = 1.0
        var = 'Gas'
        prop['color'][var] = color.green.as_hex() #'#2ca02c' # green
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
        nr = 0
        if choose:
            nr = self.case_nr(choose)
            if nr < 0:
                nr = 0
        self.case_cb.setCurrentIndex(nr)

    #-----------------------------------------------------------------------
    def missing_case_error(self, tag=''):
    #-----------------------------------------------------------------------
        show_message(self, 'warning', text='No case selected!\nAdd a case from the File-menu')

    #-----------------------------------------------------------------------
    def missing_file_error(self, tag=''):
    #-----------------------------------------------------------------------
        show_message(self, 'warning', text=f'The file {Path(tag).name} is missing for the {Path(self.case).name} case')

        
    #-----------------------------------------------------------------------
    def add_case(self, case, rename=False, choose_new=True):   # main_window
    #-----------------------------------------------------------------------
        self.case = self.copy_case(case, rename=rename)
        if not self.case:
            return None


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
        '''
        Copy case input-files and files included by the input-files.
        The file suffix match is not sensitive to case.
        '''
        src = Path(from_root)
        dst = Path(to_root)
        # Input files, change name
        inp_files = [(src.with_suffix(ext), dst.with_suffix(ext)) for ext in ('.DATA','.trcinp')]
        # Included files, same name but different folders
        ecl_inc_files = ECL_input(src.with_suffix('.DATA')).get('INCLUDE') # NB! Returns full paths
        ior_inc_files = flat_list(get_keyword(src.with_suffix('.trcinp'), '\*CHEMFILE', end='\*'))
        inc_files = [(src.parent/file.name, dst.parent/file.name) for file in ecl_inc_files+ior_inc_files]
        for src_fil, dst_fil in inp_files + inc_files:
            if src_fil.is_file():
                # print(f'copy_case_files: {src_fil} -> {dst_fil}')
                shutil_copy(src_fil, dst_fil)
        

    #-----------------------------------------------------------------------
    def read_case_dir(self):                                   # main_window
    #-----------------------------------------------------------------------
        cases = []
        if self.casedir.is_dir():
            for d in Path(self.casedir).glob('*'):
                if d.is_dir():
                    cases.append(str(d/d.name))
        else:
            create = User_input(self, title='Missing case directory', head=f'The case directory {self.casedir} does not exist. Create now?')
            def func():
                Path(self.casedir).mkdir()
            create.set_func(func)
            create.open()
            #show_message(self, 'error', text=f'The directory')
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
            #show_message(self, 'warning', text="Case '{}' not found!".format(case))
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
        if nr<0 or not self.case:
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
        #print('on_input_change: ', text)
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
                if file_contains(self.case+'.DATA', text='READDATA', comment='--', end='END'):
                    mode = 'backward'
                    self.days_box.setEnabled(False)
            except (FileNotFoundError, SystemError) as e:
                show_message(self, 'error', text='The Eclipse DATA-file is missing for this case')
            # set mode from file
            mode_file = Path(self.case).parent/'mode.gui'
            if mode_file.is_file():
                with open(mode_file) as f:
                    mode = f.readline().strip()
            # on_mode_select() is called in prepare_case() and 
            # dont need to be triggered here    
            self.mode_cb.blockSignals(True)
            self.mode_cb.setCurrentIndex(self.modes.index(mode))
            self.mode_cb.blockSignals(False)
            #self.prepare_case(self.input['root'])
            self.prepare_case()
                
    #-----------------------------------------------------------------------
    def on_compare_select(self, nr):                              # main_window
    #-----------------------------------------------------------------------
        #self.reset_progress_and_message()
        self.update_message()
        if nr>0:
            ior = ecl = False
            case = str(self.cases[nr-1])
            data = deepcopy(self.data)
            #print('BEFORE')
            #print_dict(self.data)
            # Eclipse
            if self.read_ecl_data(case=case, reinit=True):
                self.plot_ref_data['ecl'] = deepcopy(self.data['ecl'])
                ecl = True
            self.unsmry = None
            # IOR
            if self.read_ior_data(case=case):
                self.plot_ref_data['ior'] = deepcopy(self.data['ior'])
                ior = True
            # copy original data back
            self.data = data
            if (self.mode=='eclipse' and not ecl) or (self.mode=='iorsim' and not ior) or (not ecl or not ior):
                self.ref_case.setCurrentIndex(self.ref_case.property('lastitem'))
                msg = f'Cannot compare {Path(case).name} against {Path(self.case).name}'
                if not self.unsmry or not self.unsmry.is_file():
                    msg += f': {Path(case).name+".UNSMRY"} is missing'
                #self.ref_case.setCurrentIndex(0)
                #self.update_message(msg)
                self.show_message_text('WARNING '+msg)
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
        dirname = '.clear_current_case_tmp'
        try:
            if not self.case:
                self.missing_case_error(tag='clear_case: ')
                return False
            case = Path(self.case)
            for fil in case.parent.glob('*UNRST'):
                fil.unlink()
            clean_dir = case.parents[1]/dirname
            # Make sure the tmp-folder is deleted
            delete_all(clean_dir)
            # Create folder
            clean_dir.mkdir()
            # Copy case-files to the tmp-folder
            self.copy_case_files(case, clean_dir/case.stem)
            # Delete all files and folders in the case-folder
            delete_all(case.parent, keep_folder=True)
            # Copy case-files back from tmp-folder
            self.copy_case_files(clean_dir/case.stem, case)
            # Delete tmp-folder
            delete_all(clean_dir)
            self.data = {}
            self.unsmry = None # read again
            self.create_plot()
        except PermissionError as e:
            show_message(self, 'error', text='Unable to clear case, '+str(e))
            return False

    #-----------------------------------------------------------------------
    def current_viewer(self):                              # main_window
    #-----------------------------------------------------------------------
        item = self.layout.itemAtPosition(0,1)
        if item:
            return item.widget()
        return None

    #-----------------------------------------------------------------------
    def delete_current_case(self):                              # main_window
    #-----------------------------------------------------------------------
        self.reset_progress_and_message()
        #print(self.case)
        if not self.case:
            self.missing_case_error(tag='delete: ')
            return False
        case = self.case
        self.input['root'] = self.case = None
        self.max_3_checked = []
        # if self.current_viewer() in (self.editor, self.log_viewer, self.app_log_viewer):
        if self.current_viewer() in (self.editor, self.log_viewer):
            self.view_file(None)
        # Delete case folder
        delete_all(Path(case).parent)
        # Remove case from caselist
        self.create_caselist(remove=str(case))


        
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
    def get_current_mode(self):
    #-----------------------------------------------------------------------
        ind = self.mode_cb.currentIndex()
        if ind >= 0:
            return self.modes[ind].lower()

    #-----------------------------------------------------------------------
    def is_eclipse_mode(self):
    #-----------------------------------------------------------------------
        return self.get_current_mode() == 'eclipse'

    #-----------------------------------------------------------------------
    def is_iorsim_mode(self):
    #-----------------------------------------------------------------------
        return self.get_current_mode() == 'iorsim'

    #-----------------------------------------------------------------------
    def prepare_case(self):
    #-----------------------------------------------------------------------
        root = self.case or self.input['root']
        #print('prepare_case: ',root)
        self.reset_progress_and_message()
        self.plot_lines = {}
        self.ref_plot_lines = {}
        self.max_3_checked = []
        try:
            self.ref_case.setCurrentIndex(0)
            self.out_wells, self.in_wells = get_wells_iorsim(root)
            self.set_variables_from_casefiles()
            if root:
                self.on_mode_select(self.mode_cb.currentIndex())
            self.set_plot_properties()
        except SystemError as e:
            self.show_message_text(str(e))
        self.data = {}
        self.unsmry = None  # Signals to re-read Eclipse data
        # IORSim data and menu
        self.read_ior_data()
        # Add iorsim menu boxes
        self.update_ior_menu(checked = not self.is_eclipse_mode())
        # Eclipse data and menu
        self.read_ecl_data()
        # Add eclipse menu boxes
        # Check boxes only if this an eclipse-only run
        self.update_ecl_menu(checked=self.is_eclipse_mode())
        self.create_plot()
        if self.view_ag.checkedAction():
           self.view_ag.checkedAction().trigger()
        self.update_edit_menu()


    #-----------------------------------------------------------------------
    def update_edit_menu(self):
    #-----------------------------------------------------------------------
        '''
        Only enable file-actions in the Edit-menu if the file exists
        '''
        if not self.input['root']:
            return
        suffixes = ('.trcinp',       '.data',          '.sch')
        actions =  (self.ior_inp_act, self.ecl_inp_act, self.schedule_file_act)
        #files = []
        for act, ext in zip(actions, suffixes):
            enable = False
            if is_file_ignore_suffix_case( self.input['root']+ext ):
                #files.append(ext)
                enable = True
            act.setEnabled(enable)
        # Chemistry files are put in a separate menu, clear menu before adding actions
        self.chem_menu.clear()
        #if '.trcinp' in files:
        chemfiles = flat_list(get_keyword(self.input['root']+'.trcinp', '\*CHEMFILE', end='\*'))
        #self.chemfile_act = []
        for name in chemfiles:
            filename = Path(self.case).parent/name
            act = create_action(self, text=name, func=partial(self.view_input_file, name=filename, title='Chemistry input file'), icon='document-c')
            self.chem_menu.addAction(act)
            #self.chemfile_act.append(act)

        
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
        self.plot.canvas.draw()


    #-----------------------------------------------------------------------
    def new_box_with_line_layout(self, name, boxname=None, func=None, linestyle='solid', color=None):
    #-----------------------------------------------------------------------
        if not boxname:
            boxname=name
        box = self.new_checkbox(name=boxname, toggle=True, func=func, pad_left=15) 
        #box.setStyleSheet('padding-left: 15px ')
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        if not color and self.plot_prop.get('color'):
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
        menu = self.ior_menu
        delete_all_widgets_in_layout(menu.layout())
        if not self.input['root'] or not self.input['species']:
            return False
        self.ior_boxes = {}
        # Add conc / prod boxes
        lbl = QLabel()
        lbl.setText('Y-axis')
        lbl.setStyleSheet('padding-left: 10px')
        menu.column(0).addWidget(lbl)
        self.ior_boxes['yaxis'] = {}
        for text in ('Prod', 'Conc'):
            box = self.new_checkbox(text=text+'.', name='yaxis '+text.lower()+' ior', func=self.on_ior_menu_click)
            menu.column(0).addWidget(box, alignment=Qt.AlignTop)
            self.ior_boxes['yaxis'][text.lower()] = box
        # Add well boxes
        lbl = QLabel()
        lbl.setText('Wells')
        lbl.setStyleSheet('padding-top: 10px; padding-left: 10px')
        menu.column(0).addWidget(lbl)
        self.ior_boxes['well'] = {}
        for i,well in enumerate(self.out_wells or []):
            box = self.new_checkbox(text=well, name='well '+well+' ior', func=self.on_ior_menu_click)
            menu.column(0).addWidget(box, alignment=Qt.AlignTop)
            self.ior_boxes['well'][well] = box
        # Add specie boxes
        box = QCheckBox('Variables')
        box.setChecked(True)
        box.stateChanged.connect(self.set_ior_variable_boxes)
        box.setStyleSheet('padding-left: 15px')
        menu.column(1).addWidget(box)
        self.ior_var_box = box
        self.ior_boxes['var'] = {}
        for i,specie in enumerate(self.input['species'] or []):
            layout, box = self.new_box_with_line_layout(specie, func=self.on_specie_click)
            self.ior_boxes['var'][specie] = box
            menu.column(1).addLayout(layout)
        # Add temperature box
        layout, box = self.new_box_with_line_layout('Temp', linestyle='dotted', color='#707070', func=self.on_specie_click)
        self.ior_boxes['var']['Temp'] = box
        menu.column(1).addLayout(layout)
        # Set default checked boxes and add them to the checked list
        if checked:
            self.update_menu_boxes('ior')
        # Disable prod box for tracer cases 
        if self.input['tracers']:
            box = self.ior_boxes['yaxis']['prod']
            box.setEnabled(False)
            box.setChecked(False)
            
    #-----------------------------------------------------------------------
    def set_ior_variable_boxes(self):
    #-----------------------------------------------------------------------
        checked = self.ior_var_box.isChecked()
        for box in self.ior_boxes['var'].values():
            box.setChecked(checked)


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
            #block.print(details=True)
            #print(block.data())
            if block.key() == 'KEYWORDS':
                varnames = get_substrings(block.data()[0], 8)
            elif block.key() == 'WGNAMES':
                ecl_data.wells = [s for s in get_substrings(block.data()[0], 8)]
            elif block.key() == 'MEASRMNT':
                data = block.data()[0].lower()
                width = len(data)/max(len(varnames), 1)
                measure = get_substrings(data, width or 1)
            elif block.key() == 'UNITS':
                ecl_data.units = get_substrings(block.data()[0], 8)                
        #except (SystemError,TypeError) as e:
        #    print(e)
        #    self.unsmry = None # so that we call this function again
        #    return
        #if not varnames:
        if any([not v for v in (varnames, ecl_data.wells, measure, ecl_data.units)]):
            self.unsmry = None # so that we call this function again
            #print('return in ecl_init_data()')
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
        #print('read_ecl_data', case, reinit, self.unsmry)
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
    def update_ecl_menu(self, case=None, checked=False):         # main_window
    #-----------------------------------------------------------------------
        menu = self.ecl_menu
        root = self.input['root']
        if case:
            root = case
        # Delete checkboxes before creating new
        delete_all_widgets_in_layout(menu.layout())
        if not root: 
            return False
        try:
            wells, yaxis, fluids = get_eclipse_well_yaxis_fluid(root)
        except SystemError as e:
            lbl = QLabel()
            lbl.setText(str(e))
            menu.layout().addWidget(lbl)
            return
        #print(wells, yaxis, fluids)
        self.ecl_boxes = {}
        # prod/rate
        lbl = QLabel()
        lbl.setText('Y-axis')
        lbl.setStyleSheet('padding-left: 10px')
        menu.column(0).addWidget(lbl)
        self.ecl_boxes['yaxis'] = {}
        for i,name in enumerate(yaxis): 
            box = self.new_checkbox(text=name.capitalize(), name='yaxis '+name+' ecl',
                                    func=self.on_ecl_plot_click)
            self.ecl_boxes['yaxis'][name] = box
            menu.column(0).addWidget(box, alignment=Qt.AlignTop)
        # wells
        lbl = QLabel()
        lbl.setText('Wells')
        lbl.setStyleSheet('padding-top: 10px; padding-left: 10px')
        menu.column(0).addWidget(lbl)
        self.ecl_boxes['well'] = {}
        for i,well in enumerate(wells):
            box = self.new_checkbox(text=well, name='well '+well+' ecl', func=self.on_ecl_plot_click)
            if well=='Field':
                box.setObjectName('well FIELD ecl')
            menu.column(0).addWidget(box, alignment=Qt.AlignTop)
            self.ecl_boxes['well'][well] = box
        # variables
        box = QCheckBox('Variables')
        box.setChecked(True)
        box.stateChanged.connect(self.set_ecl_variable_boxes)
        box.setStyleSheet('padding-left: 15px')
        menu.column(1).addWidget(box)
        self.ecl_var_box = box
        self.ecl_boxes['var'] = {}
        for var in fluids: 
            layout, box = self.new_box_with_line_layout(var, func=self.on_ecl_var_click)
            self.ecl_boxes['var'][var] = box
            menu.column(1).addLayout(layout)
        layout, box = self.new_box_with_line_layout('Temp', boxname='Temp_ecl', linestyle='dotted',
                                                    color='#707070', func=self.on_ecl_var_click)
        self.ecl_boxes['var']['Temp_ecl'] = box
        menu.column(1).addLayout(layout)
        if checked:
            self.update_menu_boxes('ecl')

    #-----------------------------------------------------------------------
    def set_ecl_variable_boxes(self):
    #-----------------------------------------------------------------------
        checked = self.ecl_var_box.isChecked()
        for box in self.ecl_boxes['var'].values():
            box.setChecked(checked)

        
    #-----------------------------------------------------------------------
    def on_ecl_plot_click(self):
    #-----------------------------------------------------------------------
        self.update_checked_list(self.sender())
        self.create_plot()

                    
    #-----------------------------------------------------------------------
    def view_file(self, file, viewer=None, title='', reset=True):           # main_window
    #-----------------------------------------------------------------------
        if not viewer:
            viewer = self.editor
        viewer.view_file(file, title=title)
        self.current_viewer().setParent(None)
        self.layout.addWidget(viewer, *self.position['plot'])
        if reset:
            self.reset_progress_and_message()

    #-----------------------------------------------------------------------
    def view_input_file(self, name=None, ext=None, title=None, comment='#', keywords=[]):                                # main_window
    #-----------------------------------------------------------------------
        # Avoid re-opening file after it is saved
        self.log_file = None
        if self.input['root']:
            if not name:
                name = self.input['root']+ext
            # if self.current_view and self.current_view.file_is_open(name):
            if self.current_viewer().file_is_open(name):
                return
            fil = is_file_ignore_suffix_case(name)
            if fil: 
                self.view_file(fil, viewer=self.editor, title=f'{title}: {fil.name}')        
                self.highlight = Highlighter(self.editor.document(), comment=comment, keywords=keywords)
            else:
                self.sender().setChecked(False)
                self.sender().parent().missing_file_error(tag=name)
                self.editor.clear()
                return False
        
        else:
            self.sender().setChecked(False)
            self.sender().parent().missing_case_error(tag='input: ')
            return False


    #-----------------------------------------------------------------------
    def view_eclipse_input(self):                 # main_window
    #-----------------------------------------------------------------------
        if not self.input['root']:
            return
        ext='.DATA'
        title='Eclipse input file'
        comment='--'
        # Avoid re-opening file after it is saved
        #if self.editor.file_is_open(self.input['root']+ext):
        # if self.current_view and self.current_view.file_is_open(self.input['root']+ext):
        if self.current_viewer().file_is_open(self.input['root']+ext):
            return
        # Sections
        #sections = [color.red, QFont.Bold, Qt.CaseInsensitive, '\\b','\\b','RUNSPEC','GRID','EDIT','PROPS' ,'REGIONS',
        sections = [color.red, QFont.Bold, QRegularExpression.CaseInsensitiveOption, '\\b','\\b','RUNSPEC','GRID','EDIT','PROPS' ,'REGIONS',
                    'SOLUTION','SUMMARY','SCHEDULE','OPTIMIZE']
        # Global keywords
        #globals = [color.blue, QFont.Normal, Qt.CaseSensitive, '\\b','\\b','COLUMNS','DEBUG','DEBUG3','ECHO','END',
        globals = [color.blue, QFont.Normal, QRegularExpression.NoPatternOption, '\\b','\\b','COLUMNS','DEBUG','DEBUG3','ECHO','END',
                    'ENDINC','ENDSKIP','SKIP','SKIP100','SKIP300','EXTRAPMS','FORMFEED','GETDATA',
                    'INCLUDE','MESSAGES','NOECHO','NOWARN','WARN']
        # Common keywords
        #common = [color.green, QFont.Normal, Qt.CaseSensitive, r"\b",r'\b','TITLE','CART','DIMENS','FMTIN','FMTOUT',
        common = [color.green, QFont.Normal, QRegularExpression.NoPatternOption, r"\b",r'\b','TITLE','CART','DIMENS','FMTIN','FMTOUT',
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
        #print('view_iorsim_input:', self.input['root'])
        if not self.input['root']:
            return
        ext='.trcinp'
        title='IORSim input file'
        comment='#'
        name = self.input['root']+ext
        #if self.editor.file_is_open(name):
        # if self.current_view and self.current_view.file_is_open(name):
        if self.current_viewer().file_is_open(name):
            return
        # Mandatory keywords
        #mandatory = [color.blue, QFont.Bold, Qt.CaseInsensitive, '\\', '\\b'] + iorsim.keywords.required
        mandatory = [color.blue, QFont.Bold, QRegularExpression.CaseInsensitiveOption, '\\', '\\b'] + Iorsim.keywords.required
        # Optional keywords
        #optional = [color.green, QFont.Normal, Qt.CaseInsensitive, '\\', '\\b'] + iorsim.keywords.optional
        optional = [color.green, QFont.Normal, QRegularExpression.CaseInsensitiveOption, '\\', '\\b'] + Iorsim.keywords.optional
        self.view_input_file(name=name, title=title, comment=comment, keywords=[mandatory, optional])


    #-----------------------------------------------------------------------
    def view_geochem_input(self):                                # main_window
    #-----------------------------------------------------------------------
        self.view_input_file(ext='.geocheminp', title='IORSim geochem file', comment='#')
        
    #-----------------------------------------------------------------------
    def view_schedule_file(self):                                # main_window
    #-----------------------------------------------------------------------
        #globals = [color.blue, QFont.Normal, Qt.CaseSensitive, '\\b','\\b','END']
        #common = [color.green, QFont.Normal, Qt.CaseSensitive, r"\b",r'\b','TSTEP','DATES','WCONINJE','WCONHIST']
        globals = [color.blue, QFont.Normal, QRegularExpression.NoPatternOption, '\\b','\\b','END']
        common = [color.green, QFont.Normal, QRegularExpression.NoPatternOption, r"\b",r'\b','TSTEP','DATES','WCONINJE','WCONHIST']
        self.view_input_file(ext='.SCH', title='Schedule file for backward runs', comment='--', keywords=[globals, common])
        
    #-----------------------------------------------------------------------
    def view_log(self, logfile, title=None, viewer=None):       # main_window
    #-----------------------------------------------------------------------
        if not self.case:
            self.sender().setChecked(False)
            self.sender().parent().missing_case_error(tag='log: ')
            return False
        self.log_file = Path(self.case).parent/logfile
        self.view_file(self.log_file, viewer=viewer, title=title)
        #self.editor.save_btn.setEnabled(False)
        #self.editor.undo_btn.setEnabled(False)
        #self.editor.redo_btn.setEnabled(False)
        
    #-----------------------------------------------------------------------
    def view_eclipse_log(self):                                # main_window
    #-----------------------------------------------------------------------
        self.view_log('eclipse.log', title='ECLIPSE logfile', viewer=self.log_viewer)
        
    #-----------------------------------------------------------------------
    def view_iorsim_log(self):                                # main_window
    #-----------------------------------------------------------------------
        self.view_log('iorsim.log', title='IORSim logfile', viewer=self.log_viewer)
    
    #-----------------------------------------------------------------------
    def view_program_log(self):                                # main_window
    #-----------------------------------------------------------------------
        #self.view_log('ior2ecl.log', title='Application logfile', viewer=self.app_log_viewer)
        self.view_log('ior2ecl.log', title='Application logfile', viewer=self.log_viewer)
    
    #-----------------------------------------------------------------------
    def update_log(self, viewer):                                # main_window
    #-----------------------------------------------------------------------
        if self.log_file:
            # self.log_viewer.update(self.log_file)
            viewer.update(self.log_file)
    
    #-----------------------------------------------------------------------
    def view_plot(self):                                # main_window
    #-----------------------------------------------------------------------
        if not self.case:
            self.missing_case_error()
            return False
        self.log_file = None
        self.editor.setObjectName('')
        # if self.current_view:
        #     self.current_view.setParent(None)
        self.current_viewer().setParent(None)
        self.layout.addWidget(self.plot, *self.position['plot'])
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
        self.plot.canvas.draw()

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
            try:
                data = genfromtxt(str(file))
            except FileNotFoundError:
                return False
            try:
                ior['days'] = data[1:,0]
                #print(ior['days'])
                for i,name in enumerate(inp['species']):
                    #print(well, yaxis, name)
                    ior[well][yaxis][name] = data[1:,i+1]
                if 'conc' in yaxis:
                    ior[well]['conc']['Temp'] = data[1:,-1]
                    ior[well]['prod']['Temp'] = data[1:,-1]
            except (KeyError, IndexError, TypeError) as e:
                #print(e)
                pass #return False
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
        self.plot.canvas.fig.clf()
        if not self.input['root']:
            self.plot.canvas.draw()
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
            ax = self.plot.canvas.fig.add_subplot(nplot, 1, i+1)
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
        self.plot.canvas.fig.subplots_adjust(top=.93, hspace=0.4, right=.87)
        self.plot.canvas.draw()
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
        #print('update_remaining_time: ', text)
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
            self.plot.canvas.draw()
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
    def update_progress(self, t_min_n0_tuple):
    #-----------------------------------------------------------------------
        t, min, n0 = t_min_n0_tuple
        #print('update_progress, ',t, min, n0)
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
        if self.mode == 'backward':
            self.read_ior_data()
            self.read_ecl_data()
        self.days = None
        if self.mode in ('forward','eclipse','iorsim') and self.worker and self.worker.current_run():
            run = self.worker.current_run()
            if run in ('ecl','eclipse','eclrun'):
                self.read_ecl_data()
                run = 'ecl'
            if run in ('ior','iorsim'):
                self.read_ior_data()
                run = 'ior'
            if self.data.get(run):
                self.days = (self.data[run]).get('days')
        view = self.current_viewer() #.name
        #print('view:', view)
        if view == self.plot: #'plot':
            self.update_all_plot_lines()
        #elif view in (self.log_viewer, self.app_log_viewer) : #'log_viewer':
        elif view == self.log_viewer:
            self.update_log(view)

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
        if self.max_days and i['days']>self.max_days:
            show_message(self, 'warning', text='The Eclipse output read by IORSim currently sets a limit of ' + str(self.max_days) + 
                                               ' days on the time interval. Run Eclipse with a higher TSTEP to increase the maximun time interval.')
            return False
        if not i['root']:
            show_message(self, 'warning', text='No input-case selected')
            return False
        #if not self.settings.get['iorsim']():
        if not self.settings.get('iorsim'):
            show_message(self, 'warning', text='IORSim program missing in Settings', wait=True)
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
            kwargs = {'mode':'backward', 'check_unrst':s.get('unrst'), 'check_rft':s.get('rft')}
        # forward mode
        elif self.mode in ('forward','eclipse','iorsim'):
            kwargs = {'mode':'forward', 'runs':self.run}
        # start simulation
        for opt in ('convert','del_convert','merge','del_merge','check_input_kw'):
            #kwargs[opt] = s.get[opt]()
            kwargs[opt] = s.get(opt)
        self.worker = sim_worker(root=i['root'], time=i['days'], iorexe=s.get('iorsim'), eclexe=s.get('eclrun'), 
                                 ecl_keep_alive=s.get('ecl_keep_alive') and float(s.get('ecl_alive_limit')), 
                                 ior_keep_alive=s.get('ior_keep_alive') and IOR_ALIVE_LIMIT, 
                                 days_box=self.days_box, verbose=int(s.get('log_level')),
                                 **kwargs)
        self.worker.signals.status_message.connect(self.update_message)
        self.worker.signals.show_message.connect(self.show_message_text)
        self.worker.signals.progress.connect(self.update_progress)
        self.worker.signals.plot.connect(self.update_view_area)
        self.worker.signals.finished.connect(self.run_finished)
        self.threadpool.start(self.worker)
        

    #-----------------------------------------------------------------------
    def show_message_text(self, text, **kwargs):
    #-----------------------------------------------------------------------
        if not text:
            return ''
        text = text.lstrip()
        kind = 'error'
        if text.startswith(('ERROR','WARNING','INFO')):
            kind = text.split()[0]
            text = text.replace(kind,'').lstrip()
            kind = kind.lower()
        msg = show_message(self, kind, text=text, **kwargs)
        return msg
        
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
        if self.download_worker:
            self.download_worker.running = False
            sleep(0.1)
        self.save_input_values()
        # Save window geometry in settings
        geo = f'geometry {" ".join( (str(i) for i in self.geometry().getRect()) )}\n'
        replace_line(self.settings.file, find='geometry', replace=geo)
        # Quit Qt
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
        super(Highlighter, self).__init__(parent)

        self.highlightingRules = []
        #print(keywords)
        while keywords:
            kword = keywords.pop(0)
            keywordFormat = QTextCharFormat()
            keywordFormat.setForeground(kword.pop(0))
            keywordFormat.setFontWeight(kword.pop(0))
            option = kword.pop(0)
            front = kword.pop(0)
            back = kword.pop(0)
            keywordPatterns = [front + kw + back for kw in kword]            
            self.highlightingRules.extend( [(QRegularExpression(pattern, option), keywordFormat) for pattern in keywordPatterns] )
        singleLineCommentFormat = QTextCharFormat()
        singleLineCommentFormat.setForeground(color)
        self.highlightingRules.append((QRegularExpression(comment+'[^\n]*'), singleLineCommentFormat))

    def highlightBlock(self, text):
        for pattern, format in self.highlightingRules:
            expression = QRegularExpression(pattern)
            match = expression.globalMatch(text)
            while match.hasNext():
                m = match.next()
                self.setFormat(m.capturedStart(), m.capturedLength(), format)


                
###################################
#                                 #
#            M A I N              #
#                                 #
###################################

if __name__ == '__main__':
    import os

    # Need to set the locale under Linux to avoid datetime.strptime errors
    os.putenv('LC_ALL', 'C')
    #os.putenv('QTWEBENGINE_CHROMIUM_FLAGS', '--disable-logging')
    args = []

    if len(sys.argv) > 1:
        case_dir = str(default_casedir)
        workdir = get_keyword(default_settings_file, 'workdir', comment='#')
        if any(workdir):
            case_dir = workdir[0][0]
        print()
        print('   This is the terminal-version of IORSim_GUI')
        print('   Start IORSim_GUI without arguments to open the GUI')
        print()
        ior2ecl_main(case_dir=case_dir, settings_file=default_settings_file)
    else:
        exit_code = main_window.EXIT_CODE_REBOOT
        while exit_code == main_window.EXIT_CODE_REBOOT:
            app = QApplication(sys.argv + args) 
            window = main_window(settings_file=default_settings_file)
            window.show()
            exit_code = app.exec()
            window.close()
            app = None
    

        
    
