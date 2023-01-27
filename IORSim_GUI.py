#!/usr/bin/env python3
# -*- coding: utf-8 -*-

DEBUG = False

# Options
CHECK_VERSION_AT_START = True

from datetime import datetime
from itertools import chain, cycle, product
from operator import itemgetter
import sys
import os
from zipfile import ZipFile, is_zipfile

from psutil import Popen

### Check if this is a bundle version (pyinstaller)
BUNDLE_VERSION = getattr(sys, 'frozen', False)

from pathlib import Path
from urllib.parse import urlparse

#-----------------------------------------------------------------------
def resource_path():
#-----------------------------------------------------------------------
    if BUNDLE_VERSION:
        ### Running in a bundle
        path = Path(sys._MEIPASS)
    else:
        ### Running live
        path = Path.cwd()
    return path

# Default settings
MAX_CASES = 10
IORSIM_DIR = Path.home()/'.iorsim'
DOWNLOAD_DIR = IORSIM_DIR/'download'
SETTINGS_FILE = IORSIM_DIR/'settings.dat'
SESSION_FILE = IORSIM_DIR/'session.txt'

# Update files
THIS_FILE = Path(sys.argv[0])

# Guide files
GUIDE_PATH = "file:///" + str(resource_path()).replace('\\','/')
#GUIDE_PATH = str(resource_path()) # Use this for PDF_viewer_v2
IORSIM_GUIDE = GUIDE_PATH + "/guides/IORSim_2021_User_Guide.pdf"
SCRIPT_GUIDE = GUIDE_PATH + "/guides/IORSim_GUI_guide.pdf"

# GitHub
GITHUB_REPO = "https://github.com/janlv/IORSim_GUI/"
LATEST_RELEASE = GITHUB_REPO +"releases/latest"

#-----------------------------------------------------------------------
def github_url(version):
#-----------------------------------------------------------------------
    if 'py' in THIS_FILE.suffix:
        return GITHUB_REPO + f'archive/refs/tags/{version}.zip'
    return GITHUB_REPO + f'releases/download/{version}/{THIS_FILE.name}'


# External libraries
from PySide6.QtWidgets import QScrollArea, QStatusBar, QDialog, QWidget, QMainWindow, QApplication, QLabel, QPushButton, QGridLayout, QVBoxLayout, QHBoxLayout, QLineEdit, QPlainTextEdit, QDialogButtonBox, QCheckBox, QToolBar, QProgressBar, QGroupBox, QComboBox, QFrame, QFileDialog, QMessageBox
from PySide6.QtGui import QPalette, QAction, QActionGroup, QColor, QFont, QIcon, QSyntaxHighlighter, QTextCharFormat, QTextCursor
from PySide6.QtCore import QDir, QCoreApplication, QSize, QUrl, QObject, Signal, Slot, QRunnable, QThreadPool, Qt, QRegularExpression, QRect, QPoint
#from PySide6.QtPdfWidgets import QPdfView
#from PySide6.QtPdf import QPdfDocument
from PySide6.QtWebEngineCore import QWebEnginePage, QWebEngineSettings
from PySide6.QtWebEngineWidgets import QWebEngineView

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.colors import to_rgb as colors_to_rgb
from matplotlib.figure import Figure
from matplotlib import rcParams

from numpy import genfromtxt, asarray, zeros, concatenate, sort, unique
from re import compile

# Python libraries
from traceback import format_exc, print_exc, format_exc
from time import sleep
from collections import namedtuple
from shutil import copy as shutil_copy
#import warnings
from copy import deepcopy 
from functools import partial
if BUNDLE_VERSION:
    import pip_system_certs.wrapt_requests
from requests import get as requests_get, exceptions as req_exceptions

# Local libraries
from ior2ecl import SCHEDULE_SKIP_EMPTY, IORSim_input, ECL_ALIVE_LIMIT, IOR_ALIVE_LIMIT, Simulation, main as ior2ecl_main, __version__, DEFAULT_LOG_LEVEL, LOG_LEVEL_MAX, LOG_LEVEL_MIN
from IORlib.utils import Progress, convert_float_or_str, copy_recursive, flatten, get_keyword, get_tuple, groupby_sorted, kill_process, pad_zero, read_file, remove_comments, remove_leading_nondigits, replace_line, delete_all, strip_zero, try_except_loop, unique_names, write_file
from IORlib.ECL import DATA_file, UNSMRY_file, UNRST_file

QDir.addSearchPath('icons', resource_path()/'icons/')

# Font settings
#FONT = 'Segoe UI'
#LARGE_FONT = QFont(FONT, 9)
#SMALL_FONT = QFont(FONT, 8)
FONT_LARGE = 'font-size: 9pt' # for stylesheet
FONT_SMALL = 'font-size: 8pt'
FONT_PLOT = 7


#-----------------------------------------------------------------------
def new_version(version_str):
#-----------------------------------------------------------------------
    ### Check given version string against current version
    # new_version = sub(r'^[a-zA-Z-+.]*', '', version_str)  # Remove leading letters
    version = remove_leading_nondigits(version_str)
    ver = (version, __version__)
    num = pad_zero([v.split('.') for v in ver])
    diff = [int(a)-int(b) for a,b in zip(*num)]
    ### Given version is newer if the first value different from zero is positive
    if next((d>0 for d in diff if d!=0), False):
        return version_str
    return False



#====================================================================================
class Upgrader:
#====================================================================================

    LOG_FILE = IORSIM_DIR/'upgrader.log'

    #--------------------------------------------------------------------------------
    def __init__(self, argv):
    #--------------------------------------------------------------------------------
        self.pid = int(argv[0])
        self.new_file = Path(argv[1])
        self.cmd = argv[2:]
        self.target = Path.cwd()
        self._clear_log = True

    #--------------------------------------------------------------------------------
    def __str__(self) -> str:
    #--------------------------------------------------------------------------------
        return '\n'.join(f'{k}: {v}' for k,v in self.__dict__.items() if k[0] != '_')

    #--------------------------------------------------------------------------------
    def upgrade(self, limit=100, pause=0.05):
    #--------------------------------------------------------------------------------
        self.log(f'Time: {datetime.now()}\n{self}')
        # Stop app
        procs = kill_process(self.pid)
        self.log(f'Killed {self.pid}: {procs}')
        if self.new_file.is_dir():
            # Upgrader called from python script
            dest = self.target
        else:
            # Upgrader called from bundeled version (suffix is '.exe' or '' )
            dest = self.cmd[0]
        self.log(f'Copy {self.new_file} to {dest}')
        # Move new files over old ones
        # Keep trying to overwrite if PermissionError
        try_except_loop(self.new_file, dest, log=self.log, func=copy_recursive,
            limit=limit, pause=pause, error=PermissionError)
        # Start new version 
        with Popen(self.cmd) as proc:
            self.log(f'Started {self.cmd} as {proc}')
        self.log('Upgrade complete!')


    #--------------------------------------------------------------------------------
    def log(self, text):
    #--------------------------------------------------------------------------------
        mode = 'a'
        if self._clear_log:
            self._clear_log = False
            mode = 'w'
        with open(self.LOG_FILE, mode) as file:
            file.write(f'{text}\n')


#===========================================================================
class GUI_color(QColor):
#===========================================================================
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    # def __call__(self):
    #     return self.as_hex()
    def as_hex(self):
        return '#'+hex(self.rgb()).split('0xff')[-1]

#===========================================================================
class Color:
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
    as_tuple = (black, white, blue, orange, green, red, violet, brown, pink, gray, yellow, turq)
    fluid = {'Oil':red, 'Water':blue, 'Gas':green, 'Polymer':orange}
    
    @staticmethod
    def nr(i):
        return Color.as_tuple[i%len(Color.as_tuple)].as_hex()

    @staticmethod
    def as_hex(col):
        rgb = getattr(Color, col).rgb()
        return '#'+hex(rgb).split('0xff')[-1]
        
    @staticmethod
    def loop():
        return cycle(Color.as_tuple)

    # def __init__(self) -> None:
    #     self.i = 0
    
    # def next(self):
    #     self.i += 1
    #     self.nr(self.i)

#===========================================================================
class FloatEdit(QLineEdit):
#===========================================================================
    #-----------------------------------------------------------------------
    def __init__(self, *args, **kwargs):
    #-----------------------------------------------------------------------
        super().__init__(*args, **kwargs)
        #self._value = 0

    #-----------------------------------------------------------------------
    def setText(self, value, *args, precision=2, **kwargs):
    #-----------------------------------------------------------------------
        #self._value = value
        # Convert to string with given precision and remove trailing .0
        super().setText(f'{value:.{precision}f}'.rstrip('0').rstrip('.'), *args, **kwargs)

    # #-----------------------------------------------------------------------
    # def value(self):
    # #-----------------------------------------------------------------------
    #     return self._value


#--------------------------------------------------------------------------------
def show_error(func):
#--------------------------------------------------------------------------------
    def inner(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except SystemError as e:
            args[0].show_message_text(str(e))
    return inner


#-----------------------------------------------------------------------
def open_file_dialog(win, text, filetype):
#-----------------------------------------------------------------------
    options = QFileDialog.Options() | QFileDialog.DontUseNativeDialog
    file_name, _ = QFileDialog.getOpenFileName(win, text, "", filetype, options=options)
    return file_name


#-----------------------------------------------------------------------
def create_action(win, text=None, shortcut=None, tip=None, func=None, icon=None, **kwargs):
#-----------------------------------------------------------------------
    act = QAction(text, win, **kwargs)
    if shortcut:
        act.setShortcut(shortcut)
    if tip:
        act.setStatusTip(tip)
    if icon:
        act.setIcon(QIcon(f'icons:{icon}'))
    act.triggered.connect(func)
    return act


#-----------------------------------------------------------------------
def make_scrollable(widget, resizable=True):
#-----------------------------------------------------------------------
    scroll = QScrollArea()
    scroll.setStyleSheet('QScrollArea {border: 1px solid lightgray}')
    scroll.setWidget(widget)
    scroll.setWidgetResizable(resizable)
    return scroll


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
        species = flatten(get_keyword(file, '\*SPECIES', end='\*'))
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
    tracers = flatten(get_keyword(file, '\*NAME', end='\*'))
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
    out_wells = flatten(get_keyword(file, '\*PRODUCER', end='\*'))
    in_wells = flatten(get_keyword(file, '\*INJECTOR', end='\*'))
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
    #return ['FIELD']+sorted(out_wells), sorted(in_wells)
    return sorted(out_wells), sorted(in_wells)

    
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
        msg.exec()
    else:
        msg.show()
    return msg

#-----------------------------------------------------------------------
def delete_all_widgets_in_layout(layout, recursive=True):
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
            if recursive:
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
    if not box.isEnabled():
        return
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
        super().__init__()
        self.setFrameStyle(self.Shape.VLine|self.Shadow.Sunken)

#===========================================================================
class HLine(QFrame):
#===========================================================================
    def __init__(self):
        super().__init__()
        self.setFrameStyle(self.Shape.HLine|self.Shadow.Sunken)


#===========================================================================
class worker_signals(QObject):                                              
#===========================================================================
    finished = Signal()
    result = Signal(tuple)
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
        self._logfile = IORSIM_DIR/(self.kwargs.get('log') or 'base_worker.log')
        self._clear_log = True

    #-----------------------------------------------------------------------
    def log(self, text):
    #-----------------------------------------------------------------------
        mode = 'a'
        if self._clear_log:
            self._clear_log = False
            mode = 'w'
        with open(self._logfile, mode) as file:
            file.write(text+'\n')

    #-----------------------------------------------------------------------
    def raise_error(self, error=None, msg=''):
    #-----------------------------------------------------------------------
        msg = 'ERROR ' + msg 
        self.log(msg)
        raise SystemError(msg) from error

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
            self.signals and self.signals.result.emit((result,))
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
            #if value == 0:
            if value is None:
                self.progress_min = None
            #if run and value is None:
            if run:# and value is None:
                value = run.t
            self.update_progress((value and int(value), min, n0))
        #------------------------------------
        def status(run=None, value=None, mode=None, **x):
        #------------------------------------
            if not value and run:
                t, T = strip_zero((run.t, run.T))
                if self.progress_min:
                    a, b = strip_zero((self.progress_min, run.t-self.progress_min))
                    t = f'{a} + {b}'    
                value = f'{run.name}   {t} / {T} days'
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
        self.sim.prepare()
        if self.sim.ready():
            #self.days_box.setText(str(self.sim.get_time()).rstrip('0').rstrip('.'))
            self.days_box.setText(self.sim.get_time())
            result, msg = self.sim.run()
        else:
            DEBUG and print('Simulation not ready in sim_worker!')
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
        super().__init__(print_exception=False, log='download.log')
        self.running = False
        self.new_version = new_version
        #print('new_version:',new_version)
        folder = Path(folder)
        folder.mkdir(exist_ok=True)
        self.url = github_url(new_version)
        ext = Path(urlparse(self.url).path).suffix
        stem = THIS_FILE.stem.split('_v')[0]
        self.savename = folder/f'{stem}_{new_version}{ext}'
        #self.log = folder/'download.log'
        self.log(f'Time: {datetime.now()}\nSavename: {self.savename}')


    #-----------------------------------------------------------------------
    def runnable(self):
    #-----------------------------------------------------------------------
        if self.savename.is_file():
            ### File already downloaded!
            self.log(f'{self.savename} already downloaded!')
            return
        # Remove old files
        self.running = True
        delete_all(self.savename.parent, keep_folder=True, ignore_error=PermissionError)
        try:
            response = requests_get(self.url, stream=True)
        except req_exceptions.SSLError as error:
            self.raise_error(error, 'SSL error during download of update')
        except req_exceptions.ConnectionError as error:
            self.raise_error(error, 'No internet connection, unable to download update!')
        if not response.status_code == 200:
            self.raise_error(msg=f'{self.url} not found!')            
        tot_size = int(response.headers.get('content-length', 0)) or len(response.content)
        self.update_progress((-tot_size, None, None))
        self.status_message(f'Downloading version {self.new_version}')
        size = 0
        block_size = 1024*1024 # 1 MB
        self.log(f'Size: {tot_size}\nBlocks: {tot_size/block_size}\nStart-time: {datetime.now()}')
        with open(self.savename, 'wb') as file:
            for data in response.iter_content(block_size):
                if not self.running:
                    return
                size += len(data)
                #print(f'{size/tot_size:.2f}%')
                self.update_progress((size, None, None))
                file.write(data)
        self.log(f'End-time: {datetime.now()}')
        if tot_size != 0 and tot_size != size:
            msg = f'Size mismatch when downloading {self.savename}: got {size} bytes, expected {tot_size} bytes'
            self.raise_error(msg=msg)
        # Unpack zip-file for non-bundle upgrades
        if is_zipfile(self.savename):
            savedir = self.savename.parent
            with ZipFile(self.savename, 'r') as zipfile:
                zipfile.extractall(savedir)
            # Delete zip-file
            self.savename.unlink()
            # Point to unpacked folder
            self.savename = next(savedir.iterdir(), None)


#===========================================================================
class check_version_worker(base_worker):
#===========================================================================
    #-----------------------------------------------------------------------
    def __init__(self, timeout=20):
    #-----------------------------------------------------------------------
        super().__init__(print_exception=False, log='check_version.log')
        self.timeout = timeout
        self.log(f'Time: {datetime.now()}')

    #-----------------------------------------------------------------------
    def runnable(self):
    #-----------------------------------------------------------------------
        try:
            response = requests_get(LATEST_RELEASE, timeout=self.timeout)
        except req_exceptions.SSLError as error:
            self.raise_error(error, 'SSL error during version check')
        except req_exceptions.ConnectionError as error:
            self.raise_error(error, 'No internet connection, unable to check for new versions!')
        self.log(f'Response: {response.url}')
        version_str = response.url.split('/')[-1]
        return new_version(version_str)


                
#===========================================================================
class User_input(QDialog):
#===========================================================================
    #-----------------------------------------------------------------------
    def __init__(self, parent=None, title=None, head=None, label=None, text=None, size=(400, 150)):
    #-----------------------------------------------------------------------
        super().__init__(parent)
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
        super().accept()
        if self.func:
            self.func()

#===========================================================================
class Canvas(FigureCanvasQTAgg):                                              
#===========================================================================
    #-----------------------------------------------------------------------
    def __init__(self, width=5, height=5, dpi=100):
    #-----------------------------------------------------------------------
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)

#===========================================================================
class SubPlot():
#===========================================================================
    # mark = '|'

    #-----------------------------------------------------------------------
    def __init__(self, fig=None, nrows=None, index=None, title='', labels=(), data=None, varbox=None):
    #-----------------------------------------------------------------------
        self.data = data
        self.varbox = varbox
        # Add left axis
        lax = fig.add_subplot(nrows, 1, index)
        lax.title.set_text(title)
        # Set labels
        lax.set_xlabel(labels[0])
        lax.set_ylabel(labels[1])
        lax.autoscale_view()
        # Add right axis for temperature
        rax = lax.twinx()
        rax.ticklabel_format(axis='y', style='plain', useOffset=False) #scilimits=[-1,1])
        rax.set_ylabel('Temperature [C]')
        rax.autoscale_view()
        self.axes = (lax, rax)
        self.lines = {}
        self.temp = None


    #-----------------------------------------------------------------------
    def update_axes(self):                                         # SubPlot
    #-----------------------------------------------------------------------
        for ax in self.axes:
            ax.relim(visible_only=True)
            ax.autoscale_view()
        if self.temp:
            visible = self.varbox[self.temp].isChecked()
            self.axes[1].set_visible(visible)

    #-----------------------------------------------------------------------
    def update_line(self, var, set_data=True):                                         # SubPlot
    #-----------------------------------------------------------------------
        line = self.lines[var]
        checked = self.varbox[var].isChecked()
        line.set_visible(checked)
        if checked and set_data:
            x, y = self.data
            if len(x) == len(y[var]):
                line.set_data(x, y[var])
            elif DEBUG:
                print(f'Size mismatch for {var}: {len(x)} != {len(y[var])}')
        self.update_axes()


    #-----------------------------------------------------------------------
    def update_lines(self):                                         # SubPlot
    #-----------------------------------------------------------------------
        x, y = self.data
        for var,line in self.lines.items():
            checked = self.varbox[var].isChecked()
            line.set_visible(checked)
            if checked:
                if len(x) == len(y[var]):
                    line.set_data(x, y[var])
                else:
                    if DEBUG:
                        print(f'Size mismatch for {var}: {len(x)} != {len(y[var])}')
                    continue
        self.update_axes()

    #-----------------------------------------------------------------------
    def create_lines(self, prop):                                  # SubPlot
    #-----------------------------------------------------------------------
        x, y = self.data
        if x is None or y is None:
            return
        all_vars = self.varbox.keys()
        ok_vars = [var for var in all_vars if len(x) == len(y[var])]
        if DEBUG and (not_ok := set(all_vars) - set(ok_vars)):
            print(f'Size mismatch for {not_ok}')
        self.lines = {}
        for var in ok_vars:
            ax = self.axes[0]
            if 'temp' in var.lower():
                ax = self.axes[1]
                self.temp = var
            self.lines[var], = ax.plot(x, y[var], **prop[var]._asdict())


#===========================================================================
class Plots:
#===========================================================================
    mark = '|'
    ylabel = {
        'concior' : 'concentration [mol/L]',
        'prodior' : 'production [mass/day]',
        'prodecl' : 'Cum. prod. [SM3]',
        'iprodecl' : 'Inj. prod. [SM3]',
        'rateecl' : 'Prod. rate [SM3/day]',
        'irateecl' : 'Inj. rate [SM3/day]'}

    @staticmethod
    #-----------------------------------------------------------------------
    def tag_string(var='', value='', kind=''):                       # Plots
    #-----------------------------------------------------------------------
        return Plots.mark.join((var, value, kind))

    @staticmethod
    #-----------------------------------------------------------------------
    def tag_values(tag):                                             # Plots
    #-----------------------------------------------------------------------
        return namedtuple('tag','var value kind')(*tag.split(Plots.mark))

    #-----------------------------------------------------------------------
    def __init__(self, fig=None, yaxis=None, well=None):                                    # Plots
    #-----------------------------------------------------------------------
        self.fig = fig
        self.yaxis = yaxis
        self.well = well
        self.line_prop = {}
        self.lines = {}
        self.num = 0
        self.plots = []

    # #-----------------------------------------------------------------------
    # def plot_combinations(self):
    # #-----------------------------------------------------------------------
    #     tag = namedtuple('tag','kind var value')
    #     yaxis = (tag(kind, 'yaxis', box.objectName()) for kind, boxes in self.yaxis.items() for box in boxes if box.isChecked())
    #     well = (tag(kind, 'well', box.objectName()) for kind, boxes in self.well.items() for box in boxes if box.isChecked())
    #     group = {kind:{var:[] for var in ('yaxis','well')} for kind in ('ecl','ior')}
    #     for tag in chain(yaxis, well):
    #         group[tag.kind][tag.var].append(tag.value)

    #-----------------------------------------------------------------------
    def group_tags(self, boxes):                                     # Plots
    #-----------------------------------------------------------------------
        tags = [self.tag_values(box.objectName()) for box in boxes]
        group = {kind:{var:[] for var in ('yaxis','well')} for kind in ('ecl','ior')}
        for tag in tags:
            group[tag.kind][tag.var].append(tag.value)
        return group

    # #-----------------------------------------------------------------------3281
    # def tag(self, yaxis='', well='', kind=''):                       # Plots
    # #-----------------------------------------------------------------------
    #     return self.mark.join(yaxis, well, kind)

    #-----------------------------------------------------------------------
    def create(self, data=None, checked={}, varbox={}, only_nonzero=False): # Plots
    #-----------------------------------------------------------------------
        species = (var for var in varbox['ior'] if not 'temp' in var.lower())
        self.line_prop = self.line_properties(species)
        self.plots = []
        plot_comb = []
        kind_name = {'ecl':'Eclipse', 'ior':'IORSim'}
        checked = self.group_tags(checked)
        #print(checked)
        for kind in kind_name:
            boxes = checked.get(kind)
            combinations = product((kind,), boxes.get('yaxis') or (), sorted(boxes.get('well') or ()))
            plot_comb.extend(combinations)
        #print(plot_comb)
        self.num = len(plot_comb)
        self.lines = {var:[] for boxes in varbox.values() for var in boxes}
        for i, (kind, yaxis, well) in enumerate(plot_comb):
            title = kind_name[kind] + ('well ' if well != 'FIELD' else '') + well
            labels = ('days', self.ylabel[yaxis + kind])
            welldata = data and (k:=data.get(kind)) and k.get(well)
            xydata = welldata and (welldata.get('days'), welldata.get(yaxis))
            if only_nonzero and all(sum(var)<1e-8 for var in xydata[1].values()):
                # Only create non-zero plots
                continue
            plot = SubPlot(self.fig, self.num, i+1, title, labels, xydata, varbox[kind])
            plot.create_lines(self.line_prop)
            for var in plot.lines:
                self.lines[var].append(plot)
            #tag = self.tag(yaxis, well, kind)
            #self.plots[tag] = plot
            self.plots.append(plot)

    #-----------------------------------------------------------------------
    def update_line(self, var, set_data=True):                                      # Plots
    #-----------------------------------------------------------------------
        for plot in self.lines[var]:
            plot.update_line(var, set_data)

    #-----------------------------------------------------------------------
    def update(self):                                                # Plots
    #-----------------------------------------------------------------------
        for plot in self.plots:
            plot.update_lines()

    #-----------------------------------------------------------------------
    def line_properties(self, species=()):                           # Plots
    #-----------------------------------------------------------------------
        prop = namedtuple('prop','color linestyle alpha linewidth', defaults=(None, '-', 1.0, 1.5))
        #fluid_color = {'Oil':Color.red, 'Water':Color.blue, 'Gas':Color.green, 'Polymer':Color.orange}
        #ecl_prop  = {fluid:prop(col.as_hex()) for fluid,col in fluid_color.items()}
        ecl_prop  = {fluid:prop(col.as_hex()) for fluid,col in Color.fluid.items()}
        #ior_prop  = {specie:prop(Color.nr(i)) for i,specie in enumerate(species)} 
        color = Color.loop()
        ior_prop  = {specie:prop(next(color)) for specie in species} 
        temp_prop = {temp:prop(Color.black.as_hex(), '--', 0.5) for temp in ('Temp','Temp_ecl')}
        return {**ecl_prop, **ior_prop, **temp_prop}



#===========================================================================
class PlotArea(QGroupBox):
#===========================================================================
    # mark = '|'

    # @staticmethod
    # #-----------------------------------------------------------------------
    # def name_tag(*tags):
    # #-----------------------------------------------------------------------
    #     return PlotArea.mark.join(tags)

    # @staticmethod
    # #-----------------------------------------------------------------------
    # def split_tag(tag):
    # #-----------------------------------------------------------------------
    #     return tag.split(PlotArea.mark)

    #-----------------------------------------------------------------------
    def __init__(self, parent=None, fontsize=FONT_PLOT, plot_height=250):    # PlotArea
    #-----------------------------------------------------------------------
        super().__init__(parent)
        self.setTitle('Plot')
        self.checkboxes = None
        self.name = 'plot'
        self.setObjectName('plot')
        layout = QVBoxLayout()
        self.setLayout(layout)
        self.canvas = Canvas(width=5, height=5, dpi=100)
        self.fontsize = fontsize
        self.plot_height = plot_height
        rcParams['font.size'] = self.fontsize
        self.dpi = self.canvas.fig.dpi
        # Make canvas scrollable
        self.scroll_area = make_scrollable(self.canvas, resizable=False)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        #self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.horizontalScrollBar().setEnabled(False)
        layout.addWidget(self.scroll_area)
        layout.addWidget(NavigationToolbar(self.canvas, self))
        #self.clear()
        self.min_height = self.height()
        #self.left_axes = self.right_axes = ()
        #self.prop = {}
        self.plots = Plots(self.canvas.fig)

    #-----------------------------------------------------------------------
    def adjust(self):                                             # PlotArea
    #-----------------------------------------------------------------------
        top = 1 - 30/self.height()
        bottom = 50/self.height()
        self.canvas.fig.subplots_adjust(top=top, bottom=bottom, hspace=0.4)
        
    #-----------------------------------------------------------------------
    def draw(self):                                               # PlotArea
    #-----------------------------------------------------------------------
        self.canvas.draw()


    #-----------------------------------------------------------------------
    def set_height(self, height):                                     # PlotArea
    #-----------------------------------------------------------------------
        height = max(height, self.scroll_area.height())/self.dpi
        self.canvas.fig.set_figheight(height)
        self.min_height = self.height()
        self.canvas.resize(self.scroll_area.width(), self.height())

    #-----------------------------------------------------------------------
    def height(self):                                                 # PlotArea
    #-----------------------------------------------------------------------
        return self.canvas.get_width_height()[1]

    #-----------------------------------------------------------------------
    def resizeEvent(self, event):                                     # PlotArea
    #-----------------------------------------------------------------------
        #print('resize', self.scroll_area.width(), self.canvas.height())
        #if event.oldSize()[1] > event.size()[1]:
        height = max(self.scroll_area.height(), self.min_height)
        self.canvas.resize(self.scroll_area.width(), height)
        return super().resizeEvent(event)

    #-----------------------------------------------------------------------
    def file_is_open(self, filename):                                 # PlotArea
    #-----------------------------------------------------------------------
        return False

    # #-----------------------------------------------------------------------
    # def clear(self):                                                  # Plot
    # #-----------------------------------------------------------------------
    #     self.num_plots = 0
    #     self.axes_names = ()
    #     self.axes = []
    #     self.canvas.fig.clf()


    # #-----------------------------------------------------------------------
    # #def create_axes(self, checked_boxes=(), ior={}, ecl={}, parent=None):          # Plot
    # def create_axes(self, checked_boxes=()):          # Plot
    # #-----------------------------------------------------------------------
    #     self.clear()
    #     #self.checkboxes = namedtuple('checkboxes', 'ior ecl')(ior, ecl)
    #     self.checked_boxes = [box for box in checked_boxes if box.isEnabled()]
    #     #parent and parent.enable_well_boxes(True)
    #     self.axes_names = self.get_axes_names()
    #     yaxis, well, data = self.split_tag(tag)
    #     # Only check for non-zero data if no simulation is running
    #     #if parent and not parent.worker:
    #     #    self.axes_names = self.nonzero_plots(self.axes_names, parent)
    #     sum_plots = len(self.axes_names)
    #     self.set_height(sum_plots*self.plot_height)
    #     fig = self.canvas.fig
    #     #axes = [fig.add_subplot(sum_plots, 1, i+1) for i in range(sum_plots)]
    #     #axes = [fig.add_subplot(sum_plots, 1, i+1, label=name) for i,name in enumerate(self.axes_names)]
    #     self.plots = [SubPlot(fig, sum_plots, 1, i+1, name=name) for i,name in enumerate(self.axes_names)]
    #     #self.left_axes = axes
    #     # tempax = [ax.twinx() for ax in axes]
    #     # self.right_axes = tempax
    #     # self.axes = flatten(zip(axes, tempax))
    #     # self.num_plots = sum_plots
    #     # # Fix axes
    #     # src = {'ecl':'Eclipse, ', 'ior':'IORSim, '}
    #     # for ax, tag in zip(axes, self.axes_names):
    #     #     yaxis, well, data = self.split_tag(tag)
    #     #     # Set title
    #     #     ax.title.set_text(src[data] + ('well ' if well != 'FIELD' else '') + well)
    #     #     # Set labels
    #     #     ax.set_label(tag)
    #     #     ax.set_xlabel('days')
    #     #     ax.set_ylabel(self.ylabel[yaxis+data])
    #     #     ax.autoscale_view()
    #     # # Fix right-hand temperature axis
    #     # #tempbox = {'ecl':self.checkboxes.ecl['var']['Temp_ecl'], 'ior': self.checkboxes.ior['var']['Temp']}
    #     # for ax, tag in zip(tempax, self.axes_names):
    #     #     yaxis, well, data = self.split_tag(tag)
    #     #     #ax.set_visible(tempbox[data].isChecked())
    #     #     ax.ticklabel_format(axis='y', style='plain', useOffset=False) #scilimits=[-1,1])
    #     #     ax.set_label('Temp ' + tag)
    #     #     ax.set_ylabel('Temperature [C]')
    #     #     ax.autoscale_view()
    
    # #-----------------------------------------------------------------------
    # def nonzero_plots(self, plot_names, parent=None):
    # #-----------------------------------------------------------------------
    #     # Only plot if non-zero values
    #     nonzero_plots = []
    #     nonzero_well_data = set()
    #     # Check if any data > 0
    #     for name in plot_names:
    #         yaxis, well, data = self.split_tag(name)
    #         if any(sum(var)>0 for var in parent.data[data][well][yaxis].values()):
    #             nonzero_plots.append(name)
    #             nonzero_well_data.add((well, data))
    #     # Disable checkboxes with data == 0
    #     boxes = {'ecl':parent.ecl_boxes, 'ior':parent.ior_boxes}
    #     zero_plots = set(plot_names) - set(nonzero_plots)
    #     for name in zero_plots:
    #         yaxis, well, data = self.split_tag(name)
    #         if (well, data) not in nonzero_well_data:
    #             boxes[data]['well'][well].setEnabled(False)
    #     return nonzero_plots


    # #-----------------------------------------------------------------------
    # def set_title(self, ax, tag):                                 # Plot
    # #-----------------------------------------------------------------------
    #     _, well, data = tag.split(self.mark)
    #     src = {'ecl':'Eclipse, ', 'ior':'IORSim, '}
    #     ax.title.set_text(src[data] + ('well ' if well != 'FIELD' else '') + well)

    # #-----------------------------------------------------------------------
    # def set_labels(self, ax, tag):                                 # Plot
    # #-----------------------------------------------------------------------
    #     yaxis, _, data = tag.split(self.mark)
    #     ax.set_label(tag)
    #     ax.set_xlabel('days')
    #     ax.set_ylabel(self.ylabel[yaxis+data])
    #     ax.autoscale_view()

    # #-----------------------------------------------------------------------
    # def set_temp_labels(self, axx, tag):                              # Plot
    # #-----------------------------------------------------------------------
    #     _, _, data = tag.split(self.mark)
    #     tempbox = None
    #     if data == 'ior':
    #         tempbox = self.checkboxes.ior['var']['Temp']
    #     elif data == 'ecl':
    #         tempbox = self.checkboxes.ecl['var']['Temp_ecl']
    #     axx.set_visible(tempbox.isChecked())
    #     axx.ticklabel_format(axis='y', style='plain', useOffset=False) #scilimits=[-1,1])
    #     axx.set_label('Temp ' + tag)
    #     axx.set_ylabel('Temperature [C]')
    #     axx.autoscale_view()
    # #-----------------------------------------------------------------------
    # def event(self, event):                                     # Plot
    # #-----------------------------------------------------------------------
    #     print(event)
    #     return super().event(event)

    # #-----------------------------------------------------------------------
    # def get_axes_names(self):                                         # Plot
    # #-----------------------------------------------------------------------
    #     #  Checkboxes are named 'yaxis|prod|ior', 'yaxis|conc|ior', 'yaxis|rate|ecl'
    #     #  'well|P1|ior', 'well|I1|ecl'
    #     # Get tag names from checkbox objects
    #     boxes = (box.objectName().split(self.mark) for box in self.checked_boxes)
    #     axes_names = []
    #     # Loop over ior/ecl data
    #     for data, box in groupby_sorted(boxes, key=itemgetter(2), reverse=True):
    #         # Group by yaxis and well
    #         group = {k:flatten(g) for k,g in groupby_sorted(box, key=itemgetter(0))}
    #         prod = product(group.get('yaxis') or (), sorted(group.get('well') or ()))
    #         #names = list(self.mark.join(p)+'_'+data for p in prod)
    #         axes_names.extend(self.mark.join(p)+self.mark+data for p in prod)
    #     #print('axes_names', axes_names)
    #     return axes_names


#===========================================================================
class Menu(QGroupBox):
#===========================================================================
    #-----------------------------------------------------------------------
    def __init__(self, parent=None, title='', ncol=2):
    #-----------------------------------------------------------------------
        super().__init__(parent)
        self.setStyleSheet('QGroupBox {border: none; margin-top:5px; padding: 15px 0px 0px 0px;}')
        self.setTitle(title)
        layout = QHBoxLayout()
        #layout.setSizeConstraint()
        self.setLayout(layout)
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
                 refresh=True, space=0, match_case=False, size_limit_mb=None):
    #-----------------------------------------------------------------------
        super().__init__(parent)
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
        self.size_limit = size_limit_mb*1024**2 if size_limit_mb else None
        self.editor_ = QPlainTextEdit(self)
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
        if self.space > 0:
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
            #btn = QPushButton(icon=QIcon(f'icons:{icon}'), parent=self)
            btn = QPushButton(icon=QIcon(f'icons:{icon}'))
        else:
            #btn = QPushButton(text=text, parent=self)
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
        #print('save_text', self.file)
        write_file(self.file, self.editor_.toPlainText())
        self.save_btn.setEnabled(False)
        if save_func:
            #print(save_func.__name__)
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
    def read(self):
    #-----------------------------------------------------------------------
        if self.file and Path(self.file).is_file():
            text = read_file(self.file)
            if self.size_limit and (size := Path(self.file).stat().st_size) > self.size_limit:
                text = (text[:self.size_limit]
                        + f'\n --- SKIPPED {(size-self.size_limit-100)/1024**2:.0f} MB ---\n'
                        + text[-100:])
            return text
        return ''

    #-----------------------------------------------------------------------
    #def update(self, file):
    def update(self, file):
    #-----------------------------------------------------------------------
        #print(file)
        #self.set_text(read_file(file))
        self.file = file
        self.set_text(self.read())
        self.editor_.moveCursor(QTextCursor.End)

    #-----------------------------------------------------------------------
    def view_file(self, file, title=''):                            # Editor
    #-----------------------------------------------------------------------
        #print('_view_file', file, title)
        self.setTitle(title)
        curr_file = self.file
        if curr_file:
            self.vscroll[curr_file] = self.editor_.verticalScrollBar().value()
        ### Clear search field
        self.clear_search()
        ### Avoid re-opening file after it is saved
        if self.file_is_open(file):
            return
        self.file = str(file)
        # text = ''
        # if file and Path(file).is_file():
        #     text = read_file(file)
        # self.set_text(text)
        self.set_text(self.read())
        vscroll = self.vscroll.get(str(file)) or 0
        self.editor_.verticalScrollBar().setValue(vscroll)

    #-----------------------------------------------------------------------
    def refresh_func(self):                                           # Editor
    #-----------------------------------------------------------------------
        #file = self.objectName()
        self.vscroll[self.file] = self.editor_.verticalScrollBar().value()
        self.set_text(self.read())
        # if self.file and Path(self.file).is_file():
        #     self.set_text(read_file(self.file))
        vscroll = self.vscroll.get(str(self.file)) or 0
        self.editor_.verticalScrollBar().setValue(vscroll)



#===========================================================================
class Highlight_editor(Editor):
#===========================================================================
    #-----------------------------------------------------------------------
    def __init__(self, *args, comment=None, keywords=[], **kwargs):
    #-----------------------------------------------------------------------
        super().__init__(*args, **kwargs)
        self.highlighter = Highlighter(self.document(), comment=comment, keywords=keywords)



# #===========================================================================
# class PDF_viewer_v2(QPdfView):
# #===========================================================================
#     #-----------------------------------------------------------------------
#     def __init__(self, *args, **kwargs):                        # PDF_viewer
#     #-----------------------------------------------------------------------
#         super().__init__(*args, **kwargs)
#         self.setPageMode(QPdfView.PageMode.MultiPage)
#         self._document = QPdfDocument()
#         self.setDocument(self._document)

#     #-----------------------------------------------------------------------
#     def view_file(self, file, title=''):                        # PDF_viewer
#     #-----------------------------------------------------------------------
#         self._document.load(str(file))
#         #print(err, file)
#         self.setWindowTitle(title)
#         self.show()

#===========================================================================
class PDF_viewer(Editor):
#===========================================================================
    #-----------------------------------------------------------------------
    def __init__(self, *args, **kwargs):                        # PDF_viewer
    #-----------------------------------------------------------------------
        super().__init__(*args, search=True, save=False, refresh=False, top=False, end=False, **kwargs)
        viewer = QWebEngineView(self)
        viewer.settings().setAttribute(QWebEngineSettings.PluginsEnabled, True)
        viewer.settings().setAttribute(QWebEngineSettings.PdfViewerEnabled, True)
        delete_all_widgets_in_layout(self.layout(), recursive=False)
        self.layout().addWidget(viewer)
        self.editor_ = viewer

    #-----------------------------------------------------------------------
    def view_file(self, file, title=''):                        # PDF_viewer
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
class Window(QMainWindow):                                          # Window
#===========================================================================
    #-----------------------------------------------------------------------
    def __init__(self, parent=None, widget=None, title='', geo=None): # Window
    #-----------------------------------------------------------------------
        super().__init__(parent)
        self.setWindowTitle(title)
        if geo:
            self.setGeometry(geo)
        #self.setGeometry(QRect(QPoint(10,10),QSize(*size)))
        #self.setMinimumSize(*size)
        self.widget_ = widget
        self.setCentralWidget(widget)

#===========================================================================
class Settings(QDialog):
#===========================================================================
    #-----------------------------------------------------------------------
    def __init__(self, parent=None, file=None):                   # settings
    #-----------------------------------------------------------------------
        super().__init__(parent)
        self.parent = parent
        self.file = Path(file)
        self.setWindowTitle('Settings')
        self.setWindowFlag(Qt.WindowTitleHint)
        self.col_stretch = (5, 10, 70, 15)
        self.setMinimumWidth(550)
        #self.setMinimumHeight(550)
        #self.setMaximumSize(550, 850)
        self.ncol = len(self.col_stretch)
        self.setObjectName('settings_window')
        self.line = -1
        self._get = {}
        self._set = {}
        variable = namedtuple('variable', 'text default tip required ')
        self.vars = {
            'iorsim': variable('IORSim', None, 'Path to the IORSim executable', True),
            'eclrun': 
                variable('Eclipse', 'eclrun', "Eclipse command, default is 'eclrun'", True),
            'check_input_kw': 
                variable('Check IORSim input file', False, 'Check IORSim input file keywords', False),
            'convert': variable(
                'Convert to unformatted output',
                True,
                'Make IORSim output readable by ResInsight',
                False),
            'del_convert': variable(
                'Delete original after convert',
                True,
                'Delete file if successfully converted',
                False),
            'merge': variable(
                'Merge Eclipse and IORSim output',
                True,
                'Merge the unformatted output from Eclipse and IORSim into one file',
                False),
            'del_merge': variable(
                'Delete originals after merge',
                True,
                'Delete the original UNRST-files from Elipse and IORSim if successfully merged',
                False),
            'unrst': variable(
                'Confirm flushed UNRST-file during Eclipse step',
                True,
                'Check that the UNRST-file is properly flushed before suspending Eclipse',
                False),
            'rft': variable(
                'Confirm flushed RFT-file during Eclipse step',
                True,
                'Check that the RFT-file is properly flushed before suspending Eclipse',
                False),
            'ecl_keep_alive': variable(
                f'Eclipse process not paused if idle for less than',
                False,
                "Eclipse process is running also when idle ('e' to edit)",
                False),
            'ecl_alive_limit': variable(
                f'seconds',
                str(ECL_ALIVE_LIMIT),
                "If on, set this limit lower than 100 seconds to avoid unexpected Eclipse termination ('e' to edit)",
                False),
            'ior_keep_alive': variable(
                f'IORSim process not paused when idle',
                False,
                "IORSim process is running when idle. WARNING! Consumes more CPU ('e' to edit)",
                False),
            'log_level': variable(
                'Detail level of the application log',
                str(DEFAULT_LOG_LEVEL),
                'A higher value gives a more detailed application log',
                False),
            'skip_empty': variable(
                'Skip empty DATES/TSTEP entries in the schedule-file',
                SCHEDULE_SKIP_EMPTY,
                'Skip DATES/TSTEP entries in the schedule-file with missing statements',
                False)}
        #'savedir'        : variable('Download directory', None, 'Download location for updates', False),
        self.required = [k for k,v in self.vars.items() if v.required]
        self.expert = []
        self.abs_path = False
        self.initUI()
        self.set_expert_mode(False)
        self.set_default()
        self.load()

    # def resizeEvent(self, event):
    #     print(self.height(), self.scroll.height() )
    #     return super().resizeEvent(event)


    #-----------------------------------------------------------------------
    def set_expert_mode(self, enabled):
    #-----------------------------------------------------------------------
        self.expert_mode = enabled
        for var in self.expert:
            var.setEnabled(self.expert_mode)


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
        # Scrollable settings
        layout = QVBoxLayout(self)
        self.setLayout(layout)
        self.scroll = QScrollArea()
        layout.addWidget(self.scroll)
        self.scroll.setStyleSheet('QScrollArea {border: 1px solid lightgray}')
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.horizontalScrollBar().setEnabled(False)
        self.scroll.verticalScrollBar().setMinimumHeight(550)
        content = QWidget(self.scroll)
        self.grid = QGridLayout(content)
        content.setLayout(self.grid)
        self.scroll.setWidget(content)

        for i, stretch in enumerate(self.col_stretch):
            self.grid.setColumnStretch(i, stretch)

        ### IORSim executable
        self.add_heading('IORSim and Eclipse program paths')
        self.add_line_with_button(var='iorsim', open_func=self.open_ior_prog)
        ### Eclipse executable
        self.add_line_with_button(var='eclrun', open_func=self.open_ecl_prog)

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
        self.add_heading("Backward options")
        # lbl = QLabel("  (press 'e' for expert options)")
        # self.grid.addWidget(lbl, self.line, 1)
        ### UNRST and RFT checks
        self.add_items([self.new_checkbox(var) for var in ('unrst', 'rft')], nrow=2)
        ### Merge empty actions in the schedule?
        me = self.new_checkbox('skip_empty')
        #self.expert.append(me)
        self.add_items([me])
        ### Keep processes alive between steps? 
        ### Expert mode: Type 'e' to edit  
        cb = [self.new_checkbox(var) for var in ('ecl_keep_alive', 'ior_keep_alive')]
        le = self.new_lineedit('ecl_alive_limit', width=30)
        # Add widget in layout to expert mode
        self.expert.extend( (le.itemAt(i).widget() for i in range(le.count())) )
        self.expert.extend(cb)
        self.add_items([cb[0], le])
        self.add_items([cb[1]])

        ### Log options
        self.add_heading()
        self.add_heading('Log options')
        ### Log verbosity level
        values = list(map(str, range(LOG_LEVEL_MIN, LOG_LEVEL_MAX+1)))
        log = self.new_combobox(var='log_level', width=50, values=values)
        self.add_items([log])

        ### OK / Cancel buttons
        yes_no = QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        self.yes_no_btns = QDialogButtonBox(yes_no)
        layout.addWidget(self.yes_no_btns)
        self.yes_no_btns.accepted.connect(self.on_OK_click)
        self.yes_no_btns.rejected.connect(self.reject)
        self.yes_no_btns.setFocus()


    #-----------------------------------------------------------------------
    def add_heading(self, text=''):
    #-----------------------------------------------------------------------
        col = 0
        self.line += 1
        self.grid.addWidget(QLabel(text), self.line, col, 1, self.ncol)
        if text:
            self.line += 1
            line = QFrame()
            line.setStyleSheet('color: lightGray')
            line.setFrameStyle(QFrame.Shape.HLine)
            line.setLineWidth(0)
            self.grid.addWidget(line, self.line, col, 1, self.ncol)


    #-----------------------------------------------------------------------
    def add_items(self, items, nrow=1):
    #-----------------------------------------------------------------------
        self.line += 1
        layout = QGridLayout()
        self.grid.addLayout(layout, self.line, 1, 1, 2)
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
        box = QComboBox()
        v = self.vars[var]
        self._get[var] = box.currentText
        self._set[var] = box.setCurrentText
        box.setToolTip(v.tip)
        width and box.setFixedWidth(width)
        label = QLabel(v.text)
        label.setToolTip(v.tip)
        layout = QHBoxLayout()
        layout.addWidget(box)
        layout.addWidget(label)
        box.addItems(values)
        return layout

    #-----------------------------------------------------------------------
    def new_checkbox(self, var=None, size=15):             # settings
    #-----------------------------------------------------------------------
        v = self.vars[var]
        box = QCheckBox(v.text)
        box.setStyleSheet('QCheckBox::indicator { width: '+str(size)+'px; height: '+str(size)+'px;};')
        #box.setMinimumSize(50, 50)
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
        self.grid.addWidget(label, self.line, 1)
        label.setToolTip(v.tip)
        line = QLineEdit()
        self.grid.addWidget(line, self.line, 2)
        line.setToolTip(v.tip)
        line.setReadOnly(read_only)
        self._get[var] = line.text
        self._set[var] = line.setText
        if open_func:
            button = QPushButton(button_text)
            self.grid.addWidget(button, self.line, 3)
            button.clicked.connect(open_func)
        return line

        
    #-----------------------------------------------------------------------
    def set_default(self):                                # settings
    #-----------------------------------------------------------------------
        for k,v in self.vars.items():
            self._set[k](v.default)
        
    # #-----------------------------------------------------------------------
    # def restart_now(self):                                 # settings
    # #-----------------------------------------------------------------------
    #     msg = 'The application must restart to apply the new case directory. Restart now?'
    #     restart = User_input(self, title='Restart application?', head=msg)
    #     restart.set_func(self.parent.reboot)
    #     restart.open()

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
            #self.parent.update_message(f'Settings saved in {self.file}')
            self.parent.statusBar().showMessage(f'Settings saved in {self.file}')

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
                        var, val = line.split()
                    except ValueError:
                        var = line.rstrip()
                        val = ''
                    else:
                        var = var.strip()
                        if var in self._set.keys():
                            self._set[var]( str_to_bool(val.strip()) )
                        # try:
                        #     self._set[var.strip()]( str_to_bool(val.strip()) )
                        # except KeyError:
                        #     pass

                        
        
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
        super().__init__(*args, **kwargs)
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
        self.setWindowIcon(QIcon('icons:ior2ecl_icon.svg'))
        self.setStyleSheet(FONT_LARGE)
        self.silent_upgrade = False
        self.plot_lines = None
        self.ecl_fluids = {'O':'Oil', 'W':'Water', 'G':'Gas', 'T':'Temp_ecl', 'C':'Polymer'}
        # 'PC' must be 'rate', not 'conc' to pick up temp in WTPCHEA
        self.ecl_yaxes =       {'PR':'rate',   'PT':'prod',    'PC':'rate',   'IR':'irate',        'IT':'iprod'}
        self.ecl_yaxes_names = {'rate':'Rate', 'prod':'Prod.', 'rate':'Rate', 'irate':'Inj. rate', 'iprod':'Inj. prod.'}
        #self.ecl_yaxes = {'PR':'rate', 'PT':'prod', 'PC':'rate', 'IR':'rate', 'IT':'prod'}
        self.ecl_keys = ['WTPCHEA'] + list(''.join(p) for p in product(
            ('W','F'), ('O','W','G','C'), self.ecl_yaxes.keys()))
        self.data = {}
        self.plot_ref_data = {}
        self.ecl_boxes = {}
        self.ior_boxes = {}
        self.log_file = None
        self.unsmry = None
        self.worker = None
        self.download_worker = None
        self.check_version_worker = None
        self.convert = None
        self.plot_prop = {}
        self.checked_boxes = []
        self.plotted_lines = {}
        self.view = False
        self.plot_ref = None
        self.progress = None
        # User guide window
        self.pdf_view = None
        self.user_guide = None
        self.case = None
        self.schedule = None
        self.cases = ()
        self.input = {'root':None, 'ecl_days':None, 'days':100, 'step':None, 'species':[], 'mode':None, 'cases':[]}
        self.input_to_save = ('root', 'days', 'mode', 'cases')
        self.settings = Settings(parent=self, file=str(settings_file))
        self.initUI()
        self.load_session()
        self.set_input_field()
        self.threadpool = QThreadPool()
        self.show()
        # Move window if upper left corner is outside limits
        # This check must come after self.show()
        pos = self.frameGeometry().topLeft().toTuple()
        if any([p<0 for p in pos]):
            x, y = [-min(p,0) for p in pos]
            self.setGeometry(self.geometry().adjusted(x, y, x, y))            
        #print(self.screen().geometry())
        CHECK_VERSION_AT_START and self.check_version(silent=True)
                

    # #-----------------------------------------------------------------------
    # def update_casedir(self):
    # #-----------------------------------------------------------------------
    #     #self.casedir = Path(self.settings.get('workdir'))
    #     #self.casedir.mkdir(exist_ok=True)
    #     #self.input_file = self.casedir/'.cache.txt' #input_file #gui_dir/'input.txt'
    #     self.load_input()
    #     self.set_input_field()

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
        ### Check updates
        self.download_act = create_action(self, text='Check for updates', icon='drive-download.png', shortcut='',
                                      tip='Check if a new version is avaliable', func=self.check_version)
        #has_updates = default_savedir.is_dir() and next(default_savedir.iterdir(), None) or None
        #has_updates and self.download_act_upgrade()
        ### About
        self.about_act = create_action(self, text='About', icon='question.png', shortcut='',
                                      tip='Application details', func=self.about_app)
        ### Quit
        self.exit_act = create_action(self, text='&Exit', icon='control-power.png', shortcut='Ctrl+Q',
                                      tip='Exit application', func=self.quit)
        ### Add case
        self.open_case_act = create_action(self, text='Open case...', icon='blue-folder-open-document.png',
                                          func=self.open_case)
        self.copy_case_act = create_action(self, text='Copy current case...', icon='document-copy.png',
                                          func=self.copy_current_case)
        self.clear_case_act = create_action(self, text='Clear current case', icon='document.png',
                                            func=self.clear_current_case)
        self.remove_case_act = create_action(self, text='Remove current case', icon='document--minus.png',
                                             func=self.remove_current_case)
        # self.delete_case_act = create_action(self, text='Delete current case', icon='document--minus.png',
        #                                      func=self.delete_current_case)
        self.plot_act = create_action(self, text='Plot', icon='guide.png', func=self.view_plot, checkable=True)
        self.plot_act.setChecked(True)
        self.ecl_inp_act = create_action(self, text='Input file', icon='document-attribute-e.png',
                                         func=self.view_eclipse_input, checkable=True)
        self.ior_inp_act = create_action(self, text='Input file', icon='document-attribute-i.png',
                                         func=self.view_iorsim_input, checkable=True)
        self.schedule_file_act = create_action(self, text='Schedule file', icon='document-attribute-s.png',
                                          func=self.view_schedule_file, checkable=True)
        self.ecl_log_act = create_action(self, text='Run log', icon='script-attribute-e.png',
                                         func=self.view_eclipse_log, checkable=True)
        self.ior_log_act = create_action(self, text='Run log', icon='script-attribute-i.png',
                                         func=self.view_iorsim_log, checkable=True)
        self.py_log_act = create_action(self, text='Script run log', icon='script-attribute.png',
                                        func=self.view_program_log, checkable=True)
                
        
    #-----------------------------------------------------------------------
    def create_menus(self):                                    # main_window
    #-----------------------------------------------------------------------
        ### Menu
        menu = self.menuBar()
        self.view_group = QActionGroup(self)
        ### File
        file_menu = menu.addMenu('&File')
        file_menu.addAction(self.open_case_act)
        file_menu.addAction(self.copy_case_act)
        file_menu.addAction(self.clear_case_act)
        file_menu.addAction(self.remove_case_act)
        file_menu.addSeparator()
        file_menu.addAction(self.set_act)
        file_menu.addSeparator()
        file_menu.addAction(self.download_act)
        file_menu.addSeparator()
        file_menu.addAction(self.exit_act)
        ### Eclipse
        ecl_menu = menu.addMenu('&Eclipse')
        ecl_menu.addAction(self.ecl_inp_act)
        self.view_group.addAction(self.ecl_inp_act)
        self.ecl_incl_menu = ecl_menu.addMenu(QIcon('icons:documents-stack.png'), 'Include files')
        self.ecl_incl_menu.setStyleSheet(FONT_SMALL)
        ecl_menu.addAction(self.ecl_log_act)
        self.view_group.addAction(self.ecl_log_act)
        ### IORSim
        ior_menu = menu.addMenu('&IORSim')
        ior_menu.addAction(self.ior_inp_act)
        self.view_group.addAction(self.ior_inp_act)
        self.ior_incl_menu = ior_menu.addMenu(QIcon('icons:documents-stack.png'), 'Chemistry files')
        self.ior_incl_menu.setStyleSheet(FONT_SMALL)
        ior_menu.addAction(self.schedule_file_act)
        self.view_group.addAction(self.schedule_file_act)
        ior_menu.addAction(self.ior_log_act)
        self.view_group.addAction(self.ior_log_act)
        ### View
        view_menu = menu.addMenu('&View')
        view_menu.addAction(self.plot_act)
        self.view_group.addAction(self.plot_act)
        view_menu.addSeparator()
        view_menu.addAction(self.py_log_act)
        self.view_group.addAction(self.py_log_act)
        ### Help
        help_menu = menu.addMenu('&Help')
        help_menu.addAction(self.iorsim_guide_act)
        help_menu.addAction(self.script_guide_act)
        help_menu.addSeparator()
        help_menu.addAction(self.about_act)

    #-----------------------------------------------------------------------
    def about_app(self):
    #-----------------------------------------------------------------------
        about = f'IORSim brings chemistry-based IOR methods to existing reservoir simulators with minor modifications of the original input files. IORSim currently runs with Eclipse 100, but can be made available for other reservoir simulators with some adaptations.'
        self.show_message_text('INFO ' + about, extra=f'Version : {__version__}')


    #-----------------------------------------------------------------------
    def check_version(self, silent=False):
    #-----------------------------------------------------------------------
        if self.check_version_worker or self.download_worker:
            ### Version check already in progress...
            # self.show_message_text('INFO Download of new version in progress')
            return
        self.download_act.setEnabled(False)
        self.silent_upgrade = silent
        self.check_version_worker = check_version_worker()
        signals = self.check_version_worker.signals
        signals.finished.connect(self.check_version_finished) 
        signals.result.connect(self.check_version_success) 
        if not silent:
            signals.error.connect(self.check_version_error) 
        self.threadpool.start(self.check_version_worker)

    #-----------------------------------------------------------------------
    def check_version_finished(self):
    #-----------------------------------------------------------------------
        #print('check_version finished')
        self.check_version_worker = None

    #-----------------------------------------------------------------------
    def check_version_error(self, values):
    #-----------------------------------------------------------------------
        exctype, value, trace = values
        self.show_message_text(f'ERROR Error during version check!\n\n{value}')

    #-----------------------------------------------------------------------
    def check_version_success(self, result):
    #-----------------------------------------------------------------------
        #print('check_version success')
        self.download_act.setEnabled(True)
        self.new_version = result[0]
        if self.new_version:
            if THIS_FILE.with_name('.git').is_dir():
                not self.silent_upgrade and self.show_message_text(f"INFO Version {self.new_version} is available, but this script is running in a folder under Git version control.\n\nExecute 'git pull' from the command line to upgrade.")
                return
            if self.silent_upgrade:
                self.download()
            else:   
                button = ('Download', self.download)
                ver = remove_leading_nondigits(self.new_version)
                msg = f'INFO The latest version is {ver}, you are running {__version__}. Download latest version?'
                self.show_message_text(msg, button=button, ok_text='Not now', wait=True)
        else:
            not self.silent_upgrade and self.show_message_text(f'INFO No update available, {__version__} is the latest version')


    #-----------------------------------------------------------------------
    def download_finished(self):
    #-----------------------------------------------------------------------
        #print('Download finished')
        self.download_worker = None
        self.download_act.setEnabled(True)

    #-----------------------------------------------------------------------
    def download_error(self, values):
    #-----------------------------------------------------------------------
        exctype, value, trace = values
        self.show_message_text(f'ERROR Download of version {self.new_version} failed!\n\n{value}')

    #-----------------------------------------------------------------------
    def download_act_upgrade(self):
    #-----------------------------------------------------------------------
        act = self.download_act
        act.setText('Restart to upgrade')
        act.triggered.disconnect()
        act.triggered.connect(self.upgrade)
        act.setIcon(QIcon('icons:arrow-circle-225-left.png'))
        act.setEnabled(True)

    #-----------------------------------------------------------------------
    def download_act_check_version(self):
    #-----------------------------------------------------------------------
        act = self.download_act
        act.setText('Check for updates')
        act.triggered.disconnect()
        act.triggered.connect(self.check_version)
        act.setIcon(QIcon('icons:drive-download.png'))
        act.setEnabled(True)

    #-----------------------------------------------------------------------
    def download_success(self):
    #-----------------------------------------------------------------------
        # print('Download success')
        self.download_dest = self.download_worker.savename
        self.download_act_upgrade()
        if not self.silent_upgrade:
            self.update_message('Download complete')
            button = ('Upgrade', self.upgrade)
            msg = f"INFO {self.download_dest.name} is now ready to be installed.\n\nClicking the 'Upgrade' button will install the new version in the current working directory and restart the application."
            self.show_message_text(msg, button=button, ok_text='Not now')


    #-----------------------------------------------------------------------
    def upgrade(self):
    #-----------------------------------------------------------------------
        def error(msg):
            self.download_act_check_version()
            self.show_message_text(msg)
        self.close()
        upgrader = self.download_dest
        if not Path(upgrader).exists():
            return error('WARNING Upgrade script not found!')
        pid = str(os.getpid())
        cmd = [str(upgrader), '-upgrade', pid, self.download_dest]
        if not BUNDLE_VERSION:
            cmd[0] = str(Path(cmd[0])/'IORSim_GUI.py')
            exec = [sys.executable]
            cmd = exec + cmd + exec
        # Appent arguments given to this script must be re-applied for the restart
        cmd.extend(sys.argv)
        #print(f'Calling: {cmd}')
        Popen(cmd)


    #-----------------------------------------------------------------------
    def download(self):
    #-----------------------------------------------------------------------
        if self.download_worker:
            ### Download already in progress...
            return
        self.download_act.setEnabled(False)
        # print('enabled 1:',self.download_act.isEnabled())
        self.reset_progress_and_message()
        self.download_worker = download_worker(self.new_version, DOWNLOAD_DIR)
        signals = self.download_worker.signals
        signals.finished.connect(self.download_finished)
        signals.result.connect(self.download_success)
        if not self.silent_upgrade:
            signals.status_message.connect(self.update_message)
            signals.show_message.connect(self.show_message_text)
            signals.progress.connect(self.update_progress)
            signals.error.connect(self.download_error)
        self.threadpool.start(self.download_worker)


    #-----------------------------------------------------------------------
    def show_guide(self, guide, title=''):
    #-----------------------------------------------------------------------
        if self.pdf_view is None:
            self.pdf_view = PDF_viewer()
        self.pdf_view.view_file(guide, title=title)
        if self.user_guide is None:
            geo = self.geometry()
            geo.adjust(30,30,-30,-30)
            self.user_guide = Window(widget=self.pdf_view, title=title, geo=geo)
        self.user_guide.show()


    #-----------------------------------------------------------------------
    def show_iorsim_guide(self):
    #-----------------------------------------------------------------------
        self.show_guide(IORSIM_GUIDE, title='IORSim User Guide')

    #-----------------------------------------------------------------------
    def show_script_guide(self):
    #-----------------------------------------------------------------------
        self.show_guide(SCRIPT_GUIDE, title='GUI User Guide')

    #-----------------------------------------------------------------------
    def create_toolbar(self):                                  # main_window
    #-----------------------------------------------------------------------
        ### toolbar
        self.toolbar = QToolBar('Toolbar')
        self.toolbar.setStyleSheet('QToolBar{spacing:15px; padding:5px;}')
        self.toolbar.setIconSize(QSize(20, 20))
        self.addToolBar(self.toolbar)
        self.toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toolbar.setStyleSheet('QToolButton { padding: 0px 0px 0px 0px}')
        self.create_toolbar_widgets()

    #-----------------------------------------------------------------------
    def create_toolbar_widgets(self):                           # main_window
    #-----------------------------------------------------------------------
        ### simulation controls
        widgets = {'run'    : QComboBox(),
                   'case'    : QComboBox(),
                   'days'    : FloatEdit(),
                   'compare' : QComboBox()}
        tips = ('Set running mode',
                'Select recently opened case',
                'Set total time interval',
                'Compare current case against a previous case')
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
        self.mode_cb.addItems(mode_names)
        self.mode_cb.setCurrentIndex(-1)
        self.mode_cb.currentIndexChanged[int].connect(self.on_mode_select)
        # case
        self.case_cb = widgets['case']
        self.case_cb.setMinimumWidth(120)
        #self.case_cb.setMaximumWidth(360)
        self.case_cb.setMinimumWidth(120)
        #self.case_cb.setMaximumWidth(200)
        self.case_cb.setStyleSheet('QComboBox {min-width: 100px;}')
        self.case_cb.currentIndexChanged[int].connect(self.on_case_select)
        # days
        self.days_box = widgets['days']
        self.days_box.setMinimumWidth(60)
        self.days_box.setMaximumWidth(120)
        self.days_box.setObjectName('days')
        self.days_box.textChanged[str].connect(self.on_input_change)
        # reference
        self.ref_case = widgets['compare']
        self.ref_case.setMinimumWidth(120)
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
        ### Layout position given as (row, col, rowspan, colspan)
        self.position = {'ior_menu' : (0, 0),
                         'ecl_menu' : (1, 0),
                         'plot'     : (0, 1, 2, 1)}
        self.layout = QGridLayout()
        self.layout.setContentsMargins(10,10,10,10)
        self.layout.setSpacing(10)
        self.layout.setColumnStretch(0,20)
        self.layout.setColumnStretch(1,80)
        self.layout.setRowStretch(0,50)
        self.layout.setRowStretch(1,50)
        widget = QWidget()
        #widget.setStyleSheet('QWidget {border: 1px solid black}')
        widget.setLayout(self.layout)
        self.setCentralWidget(widget)
        ### Create IORSim plot menu
        self.ior_menu = Menu(parent=self, title='IORSim plot options')
        self.layout.addWidget(make_scrollable(self.ior_menu), *self.position['ior_menu']) # 
        ### Create ECLIPSE plot menu
        self.ecl_menu = Menu(parent=self, title='ECLIPSE plot options')
        self.layout.addWidget(make_scrollable(self.ecl_menu), *self.position['ecl_menu']) # 
        ### Create plot- and file-view area
        self.plot_area = PlotArea(self)
        # Plain editor with no syntax-highlight or save-function
        self.editor = Editor(name='editor', save_func=None)
        # Schedule-file editor
        self.sch_editor = Editor(name='sch_editor', save_func=self.view_schedule_file)
        ### Eclipse editor
        rules = namedtuple('rules',('color weight option front back words'))
        sections = rules(Color.red, QFont.Bold, QRegularExpression.CaseInsensitiveOption, '\\b','\\b', DATA_file.section_names)
        globals = rules(Color.blue, QFont.Normal, QRegularExpression.NoPatternOption, '\\b','\\b', DATA_file.global_kw)
        common = rules(Color.green, QFont.Normal, QRegularExpression.NoPatternOption, r"\b",r'\b', DATA_file.common_kw)
        self.eclipse_editor = Highlight_editor(name='Eclipse editor', comment='--', keywords=(sections, globals, common), save_func=self.prepare_case)
        ### IORSim editor
        mandatory = rules(Color.blue, QFont.Bold, QRegularExpression.CaseInsensitiveOption, '\\', '\\b', IORSim_input.keywords.required)
        optional = rules(Color.green, QFont.Normal, QRegularExpression.CaseInsensitiveOption, '\\', '\\b', IORSim_input.keywords.optional)
        self.iorsim_editor = Highlight_editor(name='IORSim editor', comment='#', keywords=(mandatory, optional), save_func=self.prepare_case)
        ### Chemfile editor
        self.chem_editor = Highlight_editor(name='Chemistry editor', comment='#')
        self.log_viewer = Editor(name='log_viewer', read_only=True, size_limit_mb=5)
        self.editors = (self.eclipse_editor, self.iorsim_editor, self.editor, self.chem_editor, self.log_viewer)
        ### Plot is the default view at startup
        self.layout.addWidget(self.plot_area, *self.position['plot'])


    #-----------------------------------------------------------------------
    def set_input_field(self):
    #-----------------------------------------------------------------------
        ### set values from input-file or default
        days = 100
        ### case
        #self.cases = self.read_case_dir()
        self.cases = self.input.get('cases')
        if not isinstance(self.cases, list):
            self.cases = [self.cases]
        #for case in self.cases:
        #    if not Path(case).parent.is_file()
        self.case = self.input.get('root')
        self.create_caselist(choose=self.case)
        ### number of days
        #self.days_box.float = self.input.get('days') or days
        #self.days_box.setText(f"{self.days_box.float:.2f}".rstrip('0').rstrip('.')) # Remove trailing .0
        self.days_box.setText(self.input.get('days') or days)
        #self.days_box.setText(str(self.input.get('days') or days).rstrip('0').rstrip('.')) # Remove trailing .0
        
        
    #-----------------------------------------------------------------------
    def save_session(self, sep=','):                                   # main_window
    #-----------------------------------------------------------------------
        """ Save the current input values in a cache-file """
        #self.input_file.touch(exist_ok=True)
        #print(self.input_file)
        self.input['cases'] = sep.join(self.cases)
        #print(self.input['cases'])
        lines = [f'{var}{sep}{val}\n' for var in self.input_to_save if (val:=self.input.get(var))]
        #print(lines)
        with open(SESSION_FILE, 'w') as f:
            f.write(''.join(lines))
            #f.write('# This is an input-file for ior2ecl_GUI.py, do not edit.\n')
            # for var in self.input_to_save:
            #     line = f'{var} {self.input.get(var) or ""}'
            #     f.write(line+'\n')
            #     #print('saved input: '+line)
        #return True

    #-----------------------------------------------------------------------
    def load_session(self, sep=','):                                   # main_window
    #-----------------------------------------------------------------------
        if SESSION_FILE.is_file():
            with open(SESSION_FILE) as file:
                lines = file.readlines()
                for var, *val in (l.split(sep) for l in lines if not l.startswith('#')):
                    #valgen = convert_float_or_str(val)
                    #val1 = next(valgen)
                    #vals = [val1, *rest] if (rest:=list(valgen)) else [val1]
                    vals = list(convert_float_or_str(val))
                    self.input[var] = vals[0] if len(vals) == 1 else vals
                    #print(var, self.input[var])
                # for line in f:
                #     if line.lstrip().startswith('#'):
                #         continue
                #     try:
                #         (var, val) = line.split()
                #     except ValueError:
                #         var = line.rstrip()
                #         val = None
                #     finally:
                #         try:
                #             # v = int(val) 
                #             v = float(val) 
                #         except (TypeError,ValueError):
                #             v = val
                #         finally:
                #             self.input[var] = v


    #-----------------------------------------------------------------------
    def set_variables_from_casefiles(self):                # main_window
    #-----------------------------------------------------------------------
        inp = self.input
        inp['ecl_days'] = inp['species'] = inp['tracers'] = None
        if inp['root']:
            tsteps = DATA_file(inp['root']).tsteps(missing_ok=True, negative_ok=True)
            inp['ecl_days'] = sum(tsteps)
            inp['species'] = get_species_iorsim(inp['root'], raise_error=False)
            inp['tracers'] = get_tracers_iorsim(inp['root'], raise_error=False)
            inp['species'] += inp['tracers']


    #-----------------------------------------------------------------------
    def set_plot_properties(self):                # main_window
    #-----------------------------------------------------------------------
        colors = Color.as_tuple
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
            prop['color'][var] = Color.black.as_hex() #'#000000' 
            prop['line'][var] = '--'
            prop['alpha'][var] = 0.5
        var = 'Oil'
        prop['color'][var] = Color.red.as_hex() #'#d62728' # red
        prop['line'][var] = '-'
        prop['alpha'][var] = 1.0
        var = 'Water'
        prop['color'][var] = Color.blue.as_hex() #'#1f77b4' # blue
        prop['line'][var] = '-'
        prop['alpha'][var] = 1.0
        var = 'Gas'
        prop['color'][var] = Color.green.as_hex() #'#2ca02c' # green
        prop['line'][var] = '-'
        prop['alpha'][var] = 1.0
        var = 'Polymer'
        prop['color'][var] = Color.orange.as_hex()
        prop['line'][var] = '-'
        prop['alpha'][var] = 1.0
        self.plot_prop = prop

    # #-----------------------------------------------------------------------
    # def update_cases(self, remove=None, insert=None, sort=False, limit=10):
    # #-----------------------------------------------------------------------
    #     if remove:
    #         self.cases.pop(self.case_nr(remove))
    #     if insert:
    #         self.cases.insert(0, str(insert))
    #         if len(self.cases) > limit:
    #             self.cases.pop()
    #         if sort:
    #             self.cases = sorted(self.cases)
    #     self.input['cases'] = ' '.join(self.cases)

    #-----------------------------------------------------------------------
    def missing_case_numbers(self, message=True):
    #-----------------------------------------------------------------------
        # Check for missing case-folders
        exists = [c for c in self.cases if Path(c).with_suffix('.DATA').is_file()]
        missing = tuple(set(self.cases) - set(exists))
        if missing and message:
            self.show_message_text(f'WARNING The following cases no longer exist and have been removed: {missing}')
        return (self.case_nr(m) for m in missing)

    #-----------------------------------------------------------------------
    def create_caselist(self, remove=(), insert=(), choose=None, sort=False):
    #-----------------------------------------------------------------------
        for rem in get_tuple(remove):
            self.cases.pop(self.case_nr(rem))
        for ins in get_tuple(insert):
            if str(ins) in self.cases:
                self.show_message_text(f'INFO Case {ins} is already in the case list')
                return
            self.cases.insert(0, str(ins))
            if len(self.cases) > MAX_CASES:
                self.cases.pop()
        if sort:
            self.cases = sorted(self.cases)
        # Remove missing cases
        for nr in self.missing_case_numbers():
            self.cases.pop(nr)
        items = [Path(f).stem for f in self.cases]
        # Make unique item names
        # Reverse list because newest case is first
        unique = unique_names(items[::-1])[::-1]
        #print('unique', unique)
        #print('items', items)
        if insert:
            ind = (i for i,(a,b) in enumerate(zip(unique, items)) if a != b)
            changed = [f'{self.cases[i]} listed as {unique[i]}' for i in ind]
            if changed:
                self.show_message_text(f"INFO Duplicate case-name: Case {', '.join(changed)} in the case-list")
        items = unique
        # Create case combobox
        self.case_cb.blockSignals(True)
        self.case_cb.clear()
        self.case_cb.addItems(items)
        self.case_cb.setCurrentIndex(-1)
        self.case_cb.blockSignals(False)
        # Create compare case combobox
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

        
    # #-----------------------------------------------------------------------
    # def add_case(self, case, rename=False, choose_new=True):   # main_window
    # #-----------------------------------------------------------------------
    #     self.case = self.copy_case(case, rename=rename)
    #     if not self.case:
    #         return None

    #-----------------------------------------------------------------------
    def copy_current_case(self, dest=None, choose_new=True):
    #-----------------------------------------------------------------------
        dest = dest or QFileDialog.getExistingDirectory(self,
                'Choose destination folder',
                str(Path.cwd()), QFileDialog.ShowDirsOnly)
        if not dest:
            return
        dest_root = Path(dest)/Path(self.case).stem
        to_data_file = dest_root.with_suffix('.DATA')
        if to_data_file.is_file():
            head = (f'A file named {to_data_file.name} already exists'
                'in the given folder, please choose another folder')
            rename = User_input(self, title='Choose destination folder',
                head=head, label='Destination folder', text=str(dest_root.parent))
            def func():
                newname = rename.var.text()
                self.copy_current_case(dest=newname)
            rename.set_func(func)
            rename.open()
            return
        self.copy_case_files(self.case, dest_root)
        self.case = str(dest_root)
        self.create_caselist(insert=self.case, choose=choose_new and self.case or None)
        return self.case


    @show_error
    #-----------------------------------------------------------------------
    def copy_case_files(self, from_root, to_root):             # main_window
    #-----------------------------------------------------------------------
        '''
        Copy case input-files and files included by the input-files.
        The file suffix match is not sensitive to case.
        '''
        src = Path(from_root)
        dst = Path(to_root)
        # Create missing destination folders
        dst.parent.mkdir(parents=True, exist_ok=True)
        # Input files, change name
        mandatory = ('.DATA', '.trcinp')
        optional = ('.SCH',)
        inp_files = [(src.with_suffix(ext), dst.with_suffix(ext)) for ext in mandatory + optional]
        # Included files, same name but different folders
        include_files = chain(DATA_file(src).include_files(), IORSim_input(src).include_files())
        inc_files = [(path, dst.parent/path.name) for path in include_files]
        missing_files = []
        for src_fil, dst_fil in inp_files + inc_files:
            if src_fil.is_file():
                #print(f'copy_case_files: {src_fil} -> {dst_fil}')
                shutil_copy(src_fil, dst_fil)
            elif not src_fil.suffix in optional:
                missing_files.append(src_fil)
        ### Copy optional files
        for file in src.parent.glob('*.CFG'):
            #print(f'copy_case_files (optional): {file} -> {dst.parent/file.name}')
            shutil_copy(file, dst.parent/file.name)
        if missing_files:
            self.show_message_text(f'WARNING The following case-files are missing: {missing_files}')

        

    # #-----------------------------------------------------------------------
    # def read_case_dir(self):                                   # main_window
    # #-----------------------------------------------------------------------
    #     cases = []
    #     if self.casedir.is_dir():
    #         for d in Path(self.casedir).glob('*'):
    #             if d.is_dir():
    #                 cases.append(str(d/d.name))
    #     else:
    #         create = User_input(self, title='Missing case directory', head=f'The case directory {self.casedir} does not exist. Create now?')
    #         def func():
    #             Path(self.casedir).mkdir()
    #         create.set_func(func)
    #         create.open()
    #         #show_message(self, 'error', text=f'The directory')
    #     return cases

    #-----------------------------------------------------------------------
    def case_nr(self, case):
    #-----------------------------------------------------------------------
        if not case:
            return -1
        nr = -1
        try:
            cases = [Path(c) for c in self.cases]
            nr = cases.index(Path(case))
            return nr
        except ValueError:
            #show_message(self, 'warning', text="Case '{}' not found!".format(case))
            return nr
            
    #-----------------------------------------------------------------------
    def enable_well_boxes(self, enable): 
    #-----------------------------------------------------------------------
        boxes = flatten(box['well'].values() for box in (self.ecl_boxes, self.ior_boxes))
        for box in boxes:
            if 'FIELD' in box.objectName():
                continue
            box.setEnabled(enable)

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
        #for box in self.max_3_checked:
        for box in self.checked_boxes:
            #print('Remove:' + box.objectName())
            set_checkbox(box, False, block_signal=block_signal)
        #self.max_3_checked = []
        self.checked_boxes = []
        well = list(boxlist['well'].keys())[0]
        for box in ([val for val in boxlist['yaxis'].values()] + [boxlist['well'][well],]):
            #print('Add: '+box.objectName())
            set_checkbox(box, True, block_signal=block_signal)
            if box.isEnabled():
                #self.max_3_checked.append(box)
                self.checked_boxes.append(box)

        
    #-----------------------------------------------------------------------
    def set_mode(self, mode, box=False, tip=None, days=None, run=None):  # main_window
    #-----------------------------------------------------------------------
        if not self.case:
            self.sender().setChecked(False)
            self.missing_case_error(tag='set_mode: ')
            return False
        self.mode = self.input['mode'] = mode
        mode_tip = {'forward' : 'Eclipse completes before IORSim starts',
                    'backward': 'Eclipse and IORSim run one step in alternation',
                    'eclipse' : 'Only run Eclipse',
                    'iorsim'  : 'Only run IORSim'}
        self.mode_cb.setStatusTip(mode_tip.get(mode))
        if days:
            #self.days_box.setText(str(days).rstrip('0').rstrip('.'))
            self.days_box.setText(days)
        if tip:
            self.days_box.setStatusTip(tip)
        self.days_box.setEnabled(box)
        if not isinstance(run, tuple):
            run = (run,)
        self.run = run
        fh = open(str(Path(self.case).parent/'mode.gui'), 'w')
        fh.write(mode+'\n')
        fh.close()


    #-----------------------------------------------------------------------
    def on_mode_select(self, nr):                               # main_window
    #-----------------------------------------------------------------------
        self.reset_progress_and_message()
        if nr<0 or not self.case:
            return
        self.max_days = None
        mode = self.modes[nr]
        #print('on_mode_select', mode)
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
            #days = [1,]
            self.max_days = UNRST_file(self.input['root']).last_day() or 1
            #print(self.max_days)
            self.set_mode(mode, days=self.max_days, box=True, tip='Set total time interval, maximun is '+str(self.max_days), run=mode)
            self.update_menu_boxes('ior')
            self.create_plot()
        else:
            raise SystemError('ERROR Uknown mode: ' + mode)

            
    #-----------------------------------------------------------------------
    def on_input_change(self, text):                           # main_window
    #-----------------------------------------------------------------------
        name = self.sender().objectName()
        #print('on_input_change', name)
        var = {'days':'Number of days'}
        if not text:
            self.input[name] = 0
            return
        try:
            val = float(text)
        except:
            show_message(self, 'error', text=var[name]+' must be an number!')
        else:
            self.input[name] = val
            
    #-----------------------------------------------------------------------
    def on_case_select(self, nr):                              # main_window
    #-----------------------------------------------------------------------
        self.reset_progress_and_message()
        if self.cases:
            case = str(self.cases[nr])
            if missing := self.missing_case_numbers():
                for miss_nr in missing:
                    self.cases.pop(miss_nr)
                    self.case_cb.removeItem(miss_nr)
            nr = self.case_nr(case)
            # if nr < 0:
            #     self.show_message_text('ERROR The case is removed due to missing files')
            case = str(self.cases[nr])
            # Show full case-path in statusbar
            self.case_cb.setStatusTip(case)
            self.input['root'] = self.case = case
            # set simulation mode based on READDATA keyword in .DATA-file
            mode = 'forward'
            try:
                #if file_contains(self.case+'.DATA', text='READDATA', comment='--', end='END'):
                if 'READDATA' in DATA_file(self.case): #.data():
                    mode = 'backward'
                    self.days_box.setEnabled(False)
            except (FileNotFoundError, SystemError):
                show_message(self, 'error', text='The Eclipse DATA-file is missing for this case')
            # set mode from file
            #mode_file = Path(self.case).parent/'mode.gui'
            #if mode_file.is_file():
            #    with open(mode_file) as f:
            #        mode = f.readline().strip()
            # on_mode_select() is called in prepare_case() and 
            # dont need to be triggered here    
            self.mode_cb.blockSignals(True)
            self.mode_cb.setCurrentIndex(self.modes.index(mode))
            self.mode_cb.blockSignals(False)
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
    def open_case(self):                                       # main_window
    #-----------------------------------------------------------------------
        self.reset_progress_and_message()
        case, _ = QFileDialog.getOpenFileName(self,
            'Locate Eclipse DATA-file', str(Path.cwd()), 'DATA files (*.DATA)')
        # case = open_file_dialog(self, 'Locate Eclipse DATA-file', 'DATA files (*.DATA)')
        if case:
            case = Path(case).resolve().with_suffix('')
            self.create_caselist(insert=case, choose=case)


    # #-----------------------------------------------------------------------
    # def import_case_from_file(self):                   # main_window
    # #-----------------------------------------------------------------------
    #     self.reset_progress_and_message()
    #     case = open_file_dialog(self, 'Locate Eclipse DATA-file', 'DATA files (*.DATA)')
    #     if case:
    #         root = Path(case.split('.DATA')[0])
    #         self.add_case(root)

            
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
    def remove_case(self, case):                              # main_window
    #-----------------------------------------------------------------------
        self.reset_progress_and_message()
        #self.max_3_checked = []
        self.checked_boxes = []
        self.cases.pop()
        self.create_caselist(remove=case)

    #-----------------------------------------------------------------------
    def remove_current_case(self):
    #-----------------------------------------------------------------------
        case = str(self.case)
        self.input['root'] = self.case = None
        nr = self.case_nr(case)
        #self.remove_case(nr)
        #self.case_cb.removeItem(nr)
        self.create_caselist(remove=case)
        return case

    # #-----------------------------------------------------------------------
    # def delete_current_case(self):                              # main_window
    # #-----------------------------------------------------------------------
    #     case = self.remove_current_case()
    #     folder = Path(case).parent
    #     delete = User_input(self, title='Delete case-folder?', text=f'Folder {folder} will be deleted, continue?')
    #     delete.set_func(lambda: delete_all(folder))
    #     delete.exec()
        
    # #-----------------------------------------------------------------------
    # def duplicate_current_case(self):                              # main_window
    # #-----------------------------------------------------------------------
    #     self.reset_progress_and_message()
    #     if not self.case:
    #         self.missing_case_error(tag='duplicate: ')
    #         return False
    #     new_name = User_input(self, title='Duplicate current case', label='Name of duplicate case', text=Path(self.case).name)
    #     def func():
    #         from_case = self.case
    #         name = Path(new_name.var.text().upper()).stem
    #         to_case = self.casedir/name/name
    #         self.add_case(from_case, rename=to_case)
    #     new_name.set_func(func)
    #     new_name.open()

    # #-----------------------------------------------------------------------
    # def rename_current_case(self):                              # main_window
    # #-----------------------------------------------------------------------
    #     self.reset_progress_and_message()
    #     if not self.case:
    #         self.missing_case_error(tag='rename: ')
    #         return False
    #     rename = User_input(self, title='Rename current case', label='New case name', text=Path(self.case).name)
    #     def func():
    #         oldname = Path(self.case).stem
    #         newname = rename.var.text().upper()
    #         casedir = Path(self.casedir)
    #         newdir = casedir/newname
    #         if newdir.is_dir():
    #             self.show_message_text(f'ERROR A case named {newdir.name} already exists, choose another name')
    #             return
    #         newdir = (casedir/oldname).rename(newdir)
    #         #print(str(casedir/oldname)+' => '+str(newdir))
    #         for x in newdir.iterdir():
    #             if x.is_file() and oldname in str(x):
    #                 new = str(x.name).replace(oldname,newname)
    #                 #print(str(x)+' -> '+str(newdir/new))
    #                 x.rename(newdir/new)
    #         newroot = casedir/newname/newname
    #         self.create_caselist(remove=self.case, insert=newroot, choose=newroot)
    #     rename.set_func(func)
    #     rename.open()
        
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

    @show_error
    #-----------------------------------------------------------------------
    def prepare_case(self):
    #-----------------------------------------------------------------------
        root = self.case or self.input['root']
        #self.schedule = next(Path(root).parent.glob('*.[Ss][Cc][Hh]'), None)
        sch = Path(root).parent.glob('*.[Ss][Cc][Hh]')
        self.schedule = next((f for f in sch if f.stem == Path(root).stem), None)
        #print('prepare_case: ',root)
        self.reset_progress_and_message()
        self.plot_lines = {}
        self.ref_plot_lines = {}
        #self.max_3_checked = []
        self.checked_boxes = []
        self.ref_case.setCurrentIndex(0)
        self.out_wells, self.in_wells = get_wells_iorsim(root)
        self.set_variables_from_casefiles()
        if root:
            self.on_mode_select(self.mode_cb.currentIndex())
        #self.set_plot_properties()
        self.data = {}
        self.unsmry = None  # Signals to re-read Eclipse data
        # Add eclipse menu boxes
        self.update_ecl_menu()
        # Eclipse data
        self.read_ecl_data()
        # Check boxes only if this an eclipse-only run
        if self.is_eclipse_mode():
            self.update_menu_boxes('ecl')
        # Add iorsim menu boxes
        self.update_ior_menu()
        # IORSim data
        self.read_ior_data()
        # Check default menu boxes
        if not self.is_eclipse_mode():
            self.update_menu_boxes('ior')
        self.create_plot()
        self.update_include_menus()
        self.update_schedule_act()
        if self.view_group.checkedAction():
           self.view_group.checkedAction().trigger()


    #-----------------------------------------------------------------------
    def get_checked_act(self):
    #-----------------------------------------------------------------------
        return next((act for act in self.view_group.actions() if act.isChecked()), None)

    #-----------------------------------------------------------------------
    def update_schedule_act(self):
    #-----------------------------------------------------------------------
        # self.schedule = None
        # if self.input['root']:
        #     data = Path(self.input['root']+'.DATA')
        #     self.schedule = next(data.parent.glob('*.[Ss][Cc][Hh]'), None)
        self.schedule_file_act.setEnabled(bool(self.schedule))
        ### Uncheck schedule act if it was checked but is no longer available
        if not self.schedule and self.schedule_file_act is self.get_checked_act():
            self.plot_act.setChecked(True)



    #-----------------------------------------------------------------------
    def update_file_menu(self, files, menu, viewer=None, editor=None, title=''):
    #-----------------------------------------------------------------------
        ### Clear menu
        menu.clear()
        ### Disable if empty 
        enable = False
        for file in files:
            enable = True
            act = create_action(self, text=file.name, checkable=True, func=partial(viewer, file, title=f'{title} {Path(file).name}', editor=editor), icon='document-c')
            act.setIconText('include')
            self.view_group.addAction(act)
            ### Disable for non-existing file
            act.setEnabled(file.is_file())
            menu.addAction(act)
            #self.view_group.addAction(act)
        menu.setEnabled(enable)


    @show_error
    #-----------------------------------------------------------------------
    def update_include_menus(self):
    #-----------------------------------------------------------------------
        root = self.input['root']
        if not root:
            return
        ### Check if any include-files are checked, i.e. displayed. If checked, show plot instead
        checked_act = self.get_checked_act()
        include_act = [act for act in self.view_group.actions() if act.iconText()=='include']
        if checked_act and include_act and checked_act in include_act:
            self.plot_act.setChecked(True)
        ### Remove old include-files from the view-group
        [self.view_group.removeAction(act) for act in include_act]
        ### Add case-specific include files
        self.update_file_menu(IORSim_input(root).include_files(), self.ior_incl_menu, viewer=self.view_input_file, title='Chemistry file', editor=self.chem_editor)
        self.update_file_menu(DATA_file(root).include_files(), self.ecl_incl_menu, viewer=self.view_input_file, title='Include file', editor=self.editor)

        
    #-----------------------------------------------------------------------
    def on_ecl_var_click(self):
    #-----------------------------------------------------------------------
        #print('on_ecl_var_click', self.sender())
        name = self.sender().objectName()
        is_checked = self.sender().isChecked()
        self.update_plot_line(name, is_checked)
        if self.plot_ref_data:
            #print('ref_data')
            self.update_plot_line(name, is_checked, lines=self.ref_plot_lines, set_data=False)
        self.plot_area.draw()


    #-----------------------------------------------------------------------
    def plot_menu_box_variable(self, name, boxname=None, func=None, linestyle='solid', color=None):
    #-----------------------------------------------------------------------
        if not boxname:
            boxname=name
        #box = self.plot_menu_checkbox(name=boxname, toggle=True, func=func, pad_left=15) 
        box = self.plot_menu_checkbox(name=boxname, toggle=True, func=func)
        box.setFixedSize(QSize(15,15))
        #box.setStyleSheet('padding-left: 15px ')
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        # if not color and self.plot_prop.get('color'):
        #     color = to_rgb(self.plot_prop['color'][name])
        # print(self.plot_area.plots.line_prop)
        # if not color and (prop:=self.plot_area.plots.line_prop.get(name)):
        #     color = to_rgb(prop['color'])
        if color:
            line.setStyleSheet('border: 3px '+linestyle+' '+color)
        line.setFixedWidth(25)
        label = QLabel(name)
        label.setStyleSheet(FONT_SMALL)
        layout = QHBoxLayout()
        layout.addWidget(box)#, 1)
        layout.addWidget(line)#, 1)
        layout.addWidget(label)#, 1)
        return layout, box

    #-----------------------------------------------------------------------
    #def plot_menu_checkbox(self, text='', name='', func=None, toggle=False, pad_left=10, size=15):
    def plot_menu_checkbox(self, text='', name='', func=None, toggle=False, pad_right=10): #, size=15):
    #-----------------------------------------------------------------------
        box = QCheckBox(text)
        box.setObjectName(name)
        box.setStyleSheet(FONT_SMALL)
        if toggle:
            box.toggle()
        if func:
            box.stateChanged.connect(func)
        return box
        
    #-----------------------------------------------------------------------
    def update_ior_menu(self):                   # main_window
    #-----------------------------------------------------------------------
        #print('update_ior_menu')
        menu = self.ior_menu
        delete_all_widgets_in_layout(menu.layout())
        if not self.input['root'] or not self.input['species']:
            return False
        self.ior_boxes = {}
        # Add conc / prod boxes
        self.ior_boxes['yaxis'] = {}
        menu.column(0).addWidget(QLabel('Y-axis'))
        for text in ('Prod', 'Conc'):
            #box = self.plot_menu_checkbox(text=text+'.', name='yaxis_'+text.lower()+'_ior', func=self.on_ior_menu_click)
            #name = 'yaxis'+Plot.mark+text.lower()+Plot.mark+'ior'
            #name = PlotArea.name_tag('yaxis', text.lower(), 'ior')
            tag = Plots.tag_string(var='yaxis', value=text.lower(), kind='ior')
            box = self.plot_menu_checkbox(text=text+'.', name=tag, func=self.on_ior_menu_click)
            box.setStyleSheet(FONT_SMALL)
            menu.column(0).addWidget(box)#, alignment=Qt.AlignTop)
            self.ior_boxes['yaxis'][text.lower()] = box
        # Add well boxes
        space = QLabel()
        space.setStyleSheet('font: 1pt')
        menu.column(0).addWidget(space)
        menu.column(0).addWidget(QLabel('Wells'))
        self.ior_boxes['well'] = {}
        for well in self.out_wells:
            #box = self.plot_menu_checkbox(text=well, name='well_'+well+'_ior', func=self.on_ior_menu_click)
            #name = PlotArea.name_tag('well', well, 'ior')
            tag = Plots.tag_string(var='well', value=well, kind='ior')
            box = self.plot_menu_checkbox(text=well, name=tag, func=self.on_ior_menu_click)
            box.setStyleSheet(FONT_SMALL)
            box.setEnabled(False)
            menu.column(0).addWidget(box, alignment=Qt.AlignTop)
            self.ior_boxes['well'][well] = box
        #self.ior_boxes['well']['FIELD'].setEnabled(True)
        # Add specie boxes
        box = self.plot_menu_checkbox(text='Variables', name='ior_var')
        box.setStyleSheet(FONT_LARGE)
        box.setChecked(True)
        box.stateChanged.connect(self.set_ior_variable_boxes)
        menu.column(1).addWidget(box)
        self.ior_var_box = box
        self.ior_boxes['var'] = {}
        for i,specie in enumerate(self.input['species'] or []):
            layout, box = self.plot_menu_box_variable(specie, func=self.on_specie_click)
            self.ior_boxes['var'][specie] = box
            menu.column(1).addLayout(layout)
        # Add temperature box
        layout, box = self.plot_menu_box_variable('Temp', linestyle='dotted', color='#707070', func=self.on_specie_click)
        self.ior_boxes['var']['Temp'] = box
        menu.column(1).addLayout(layout)
        # Set default checked boxes and add them to the checked list
        # if checked:
        #     self.update_menu_boxes('ior')
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
    def get_eclipse_well_yaxis_fluid(self, case=None, raise_error=True):    # main_window
    #-----------------------------------------------------------------------
        ecl = DATA_file(case or self.input['root'])
        ecl.check(include=False)
        #vars = [line for line in ecl.section('SUMMARY').lines() if line[0] in ('W','F','R')]
        vars = ecl.summary_keys(matching=self.ecl_keys)
        #print('vars', set(vars))
        if not vars and raise_error:
            raise SystemError('SUMMARY keywords missing,\n\nEclipse plotting disabled.')
        fy = ((f,y) for v in set(vars) if (f:=self.ecl_fluids.get(v[1])) and (y:=self.ecl_yaxes.get(v[2:4])))
        fluids, yaxis = zip(*fy)
        wells = sorted(ecl.wellnames())
        if any(v[0]=='F' and v[2:4] in ('PR','PT') for v in vars):
            wells.insert(0, 'FIELD')
        fluids = list(set(fluids)-set(['Temp_ecl']))
        #print(fluids)
        #print(yaxis)
        return wells, sorted(set(yaxis)), fluids
        #return wells, ('prod','rate'), fluids

    #-----------------------------------------------------------------------
    def init_ecl_data_v2(self, case=None):            # main_window
    #-----------------------------------------------------------------------
        #  WOPR    - well oil prod rate
        #  WCPR    - well polymer prod rate
        #  WWPR    - well water prod rate
        #  WTPCHEA - well temp (Temp_ecl)
        #  WOPT    - well oil prod tot
        #  WWCT    - well water cut (prod)
        #  WWIR    - well water injection rate
        #  WWIT    - well water injection prod
        #  FOPT    - field oil prod total
        #  FWIT    - field water injection total
        #  FWCT    - field water cut total (prod)
        #  ROIP    - Reservoir oil in place

        case = case or self.input['root']
        self.unsmry = UNSMRY_file(case)
        ### Create dict of format [well][yaxis][fluid] = []
        data_file = DATA_file(case)

        wellnames = data_file.wellnames()
        field_wells = ('FIELD',) + wellnames
        ecl = {w:{'days':[]} for w in field_wells}
        [ecl[w].update({y:{f:[] for f in self.ecl_fluids.values()} for y in self.ecl_yaxes.values()}) for w in field_wells]
        ecl['days'] = []
        # Prod temp refers to rate temp
        for w in wellnames:
            for yaxis in (set(self.ecl_yaxes.values()) - set('rate')):
                ecl[w][yaxis]['Temp_ecl'] = ecl[w]['rate']['Temp_ecl']
        self.data['ecl'] = ecl
        return True


    #-----------------------------------------------------------------------
    def read_ecl_data(self, case=None, reinit=False, skip_zero=True):   # main_window
    #-----------------------------------------------------------------------
        #print('read_ecl_data START')
        datafile = case or self.input['root']
        if not datafile:
            self.data['ecl'] = {}
            return False
        ### read data
        if reinit or not self.unsmry:
            if not self.init_ecl_data_v2(case=case):
                return False
        data = self.unsmry.data(keys=self.ecl_keys)
        #self.active_wells = set(data.wells)
        # Enable menu-well-boxes for active wells
        for well in set(data.wells):
            self.ecl_boxes['well'][well].setEnabled(True)
        if data.days:
            ### Skip zero-time data
            start = 0
            if skip_zero and data.days[0] < 1e-8:
                start = 1
            ecl = self.data['ecl']
            ecl['days'].extend(data.days[start:])
            for w in set(data.wells):
                ecl[w]['days'].extend(data.days[start:])
            ### Index to map read data to ecl[well][yaxis][fluid]
            wyf_index = [(w, self.ecl_yaxes[k[2:4]], self.ecl_fluids[k[1]]) for w,k in zip(data.wells, data.keys)]
            for (w,y,f),*d in zip(wyf_index, *data.welldata[start:]):
                ecl[w][y][f].extend(d)

    #-----------------------------------------------------------------------
    def data_is_zero(self, well, yaxis, kind):
    #-----------------------------------------------------------------------
        return all(sum(var)<1e-8 for var in self.data[kind][well][yaxis].values())
        #return any(sum(var)>0 for var in self.data[data][well][yaxis].values())

    #-----------------------------------------------------------------------
    def update_ecl_menu(self, case=None):         # main_window
    #-----------------------------------------------------------------------
        menu = self.ecl_menu
        case = case or self.input['root']
        delete_all_widgets_in_layout(menu.layout())
        if not case:
            return False
        try:
            wells, yaxis, fluids = self.get_eclipse_well_yaxis_fluid(case)
        except SystemError as e:
            lbl = QLabel(parent=self)
            lbl.setText(str(e))
            menu.layout().addWidget(lbl)
            return
        #print(wells, yaxis, fluids)
        self.ecl_boxes = {}
        # prod/rate
        menu.column(0).addWidget(QLabel('Y-axis'))
        self.ecl_boxes['yaxis'] = {}
        for i,name in enumerate(yaxis):
            text = self.ecl_yaxes_names[name]
            #name = PlotArea.name_tag('yaxis', name, 'ecl')
            tag = Plots.tag_string(var='yaxis', value=name, kind='ecl')
            box = self.plot_menu_checkbox(text, tag, self.on_ecl_yaxis_click)
            box.setStyleSheet(FONT_SMALL)
            self.ecl_boxes['yaxis'][name] = box
            menu.column(0).addWidget(box)
        # space
        space = QLabel()
        space.setStyleSheet('font-size: 1pt')
        menu.column(0).addWidget(space)
        # wells
        menu.column(0).addWidget(QLabel('Wells'))
        pos = 1 if 'FIELD' in wells else 0
        self.wellnames = wells[pos:]
        wells = wells[:pos] + ['All wells'] + wells[pos:]
        #print(wells)
        self.ecl_boxes['well'] = {}
        for well in wells:
            #name = PlotArea.name_tag('well', well, 'ecl')
            tag = Plots.tag_string(var='well', value=well, kind='ecl')
            box = self.plot_menu_checkbox(well, tag, self.on_ecl_well_click)
            #box.setEnabled(False)
            box.setEnabled(True)
            box.setStyleSheet(FONT_SMALL)
            menu.column(0).addWidget(box)
            self.ecl_boxes['well'][well] = box
        field_box = self.ecl_boxes['well'].get('FIELD')
        if field_box:
            field_box.setEnabled(True)
        all_wells = self.ecl_boxes['well'].get('All wells')
        if all_wells:
            all_wells.setEnabled(True)
            all_wells.setChecked(False)
            all_wells.stateChanged.disconnect()
            all_wells.stateChanged.connect(self.set_ecl_well_boxes)
            self.ecl_well_box = all_wells
        # variables
        box = self.plot_menu_checkbox(text='Variables', name='ecl_var')
        box.setStyleSheet(FONT_LARGE)
        box.setChecked(True)
        box.stateChanged.connect(self.set_ecl_variable_boxes)
        menu.column(1).addWidget(box)
        self.ecl_var_box = box
        self.ecl_boxes['var'] = {}
        for var in fluids:
            layout, box = self.plot_menu_box_variable(var, func=self.on_ecl_var_click)
            self.ecl_boxes['var'][var] = box
            menu.column(1).addLayout(layout)
        layout, box = self.plot_menu_box_variable('Temp', boxname='Temp_ecl', linestyle='dotted',
                                                    color='#707070', func=self.on_ecl_var_click)
        self.ecl_boxes['var']['Temp_ecl'] = box
        menu.column(1).addLayout(layout)
        # if checked:
        #     self.update_menu_boxes('ecl')

    #-----------------------------------------------------------------------
    def set_ecl_variable_boxes(self):
    #-----------------------------------------------------------------------
        checked = self.ecl_var_box.isChecked()
        for box in self.ecl_boxes['var'].values():
            box.setChecked(checked)

    #-----------------------------------------------------------------------
    def set_ecl_well_boxes(self):
    #-----------------------------------------------------------------------
        checked = self.ecl_well_box.isChecked()
        boxes = self.ecl_boxes['well'].values()
        checked_boxes = []
        for box in boxes:
            boxname = box.objectName()
            if any(name in boxname for name in ('FIELD', 'All wells')):# in :
                continue
            checked_boxes.append(box)
            box.blockSignals(True)
            box.setChecked(checked)
            box.blockSignals(False)
        self.update_checked_list(*checked_boxes)
        self.create_plot()

    #-----------------------------------------------------------------------
    def disable_empty_wellboxes(self, yboxes, kind):
    #-----------------------------------------------------------------------
        nonzero = []
        #data = self.data[kind]
        yaxiss =  [PlotArea.split_tag(box.objectName())[1] for box in yboxes]
        for yaxis in yaxiss:
            #non = [any(sum(var)>0 for var in data[well][yaxis].values()) for well in self.wellnames]
            non = [not self.data_is_zero(well, yaxis, kind) for well in self.wellnames]
            nonzero.append(non)
        plot_list = []
        for well, *notzero in zip(self.wellnames, *nonzero):
            self.ecl_boxes['well'][well].setEnabled(any(notzero))
            #print(well, notzero, yaxiss, self.ecl_boxes['well'][well].isEnabled())
            yax = [y for y,nz in zip(yaxiss, notzero) if nz]
            if yax:
                plot_list.append((well, yax))
        print(plot_list)

    #-----------------------------------------------------------------------
    def on_ecl_yaxis_click(self):
    #-----------------------------------------------------------------------
        #print('on_ecl_yaxis_click', self.sender())
        box = self.sender()
        self.update_checked_list(box)
        #checked = box.isChecked()
        checked = [b for b in self.ecl_boxes['yaxis'].values() if b.isChecked()]
        if checked and not self.worker:
            self.disable_empty_wellboxes(checked, 'ecl')
        if not checked:
            self.enable_well_boxes(True)
        self.create_plot()

    #-----------------------------------------------------------------------
    def on_ecl_well_click(self):
    #-----------------------------------------------------------------------
        #print('on_ecl_well_click', self.sender())
        self.update_checked_list(self.sender())
        self.create_plot()

    #-----------------------------------------------------------------------
    def view_file(self, file, viewer=None, title='', reset=True):           # main_window
    #-----------------------------------------------------------------------
        #print(f'view_file({file}, viewer={viewer}, title={title}, reset={reset}')
        if not viewer:
            viewer = self.editor
        viewer.view_file(file, title=title)
        self.current_viewer().setParent(None)
        self.layout.addWidget(viewer, *self.position['plot'])
        if reset:
            self.reset_progress_and_message()

    #-----------------------------------------------------------------------
    def view_input_file(self, name, editor=None, title=None):       # main_window
    #-----------------------------------------------------------------------
        #print(self.sender())
        #print(self.view_group)
        self.log_file = None
        if self.input['root']:
            if self.current_viewer().file_is_open(name):
                # Do not read a saved file, but update title
                self.current_viewer().setTitle(title)
                return
            if Path(name).is_file():
                # print('view_file',title, name)
                self.view_file(name, viewer=editor, title=title)
            else:
                self.sender().setChecked(False)
                self.sender().parent().missing_file_error(tag=name)
                #self.editor.clear()
                return False
        
        else:
            self.sender().setChecked(False)
            self.sender().parent().missing_case_error(tag='input: ')
            return False


    #-----------------------------------------------------------------------
    def view_eclipse_input(self, name=None, title=None):                 # main_window
    #-----------------------------------------------------------------------
        if not self.input['root']:
            return
        ext='.DATA'
        if name is None:
            name = self.input['root']+ext
        if title is None:
            title = f'Eclipse input file {Path(name).name}'
        # Avoid re-opening file after it is saved
        if self.current_viewer().file_is_open(name):
            return
        self.view_input_file(name, title=title, editor=self.eclipse_editor)


    #-----------------------------------------------------------------------
    def view_iorsim_input(self):                                # main_window
    #-----------------------------------------------------------------------
        #print('view_iorsim_input:', self.input['root'])
        if not self.input['root']:
            return
        ext='.trcinp'
        name = self.input['root']+ext
        title = f'IORSim input file {Path(name).name}'
        if self.current_viewer().file_is_open(name):
            return
        self.view_input_file(name, title=title, editor=self.iorsim_editor)
  
        
    #-----------------------------------------------------------------------
    def view_schedule_file(self):                                # main_window
    #-----------------------------------------------------------------------
        #days = self.schedule and (sum(DATA_file(name).tsteps(missing_ok=True)) for name in (self.input['root'], self.schedule)) or 0
        days = DATA_file(self.input['root']).including(self.schedule).tsteps(missing_ok=True)
        self.view_input_file(self.schedule, title=f'Schedule file {self.schedule.name}, total days = {sum(days):.0f}', editor=self.sch_editor)
        
    #-----------------------------------------------------------------------
    def view_log(self, logfile, title=None, viewer=None):       # main_window
    #-----------------------------------------------------------------------
        if not self.case:
            self.sender().setChecked(False)
            self.sender().parent().missing_case_error(tag='log: ')
            return False
        self.log_file = Path(self.case).parent/logfile
        self.view_file(self.log_file, viewer=viewer, title=title)
        
    #-----------------------------------------------------------------------
    def view_eclipse_log(self):                                # main_window
    #-----------------------------------------------------------------------
        self.view_log('eclipse.log', title='ECLIPSE run log', viewer=self.log_viewer)
        
    #-----------------------------------------------------------------------
    def view_iorsim_log(self):                                # main_window
    #-----------------------------------------------------------------------
        self.view_log('iorsim.log', title='IORSim run log', viewer=self.log_viewer)
    
    #-----------------------------------------------------------------------
    def view_program_log(self):                                # main_window
    #-----------------------------------------------------------------------
        self.view_log('ior2ecl.log', title='Script run log', viewer=self.log_viewer)
    
    #-----------------------------------------------------------------------
    def update_log(self, viewer):                                # main_window
    #-----------------------------------------------------------------------
        if self.log_file:
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
        self.layout.addWidget(self.plot_area, *self.position['plot'])
        #self.layout.addWidget(self.scrollplot, *self.position['plot'])
        if not self.worker:# and self.case:
            self.read_ior_data()
            self.read_ecl_data()
            self.update_all_plot_lines()

            
    #-----------------------------------------------------------------------
    def update_checked_list(self, *boxes):
    #-----------------------------------------------------------------------
        for box in boxes:
            if box.isChecked():
                self.checked_boxes.insert(0, box)
            else:
                if box in self.checked_boxes:
                    self.checked_boxes.remove(box)


    # #-----------------------------------------------------------------------
    # def update_checked_list(self, box): 
    # #-----------------------------------------------------------------------
    #     #return
    #     # print('update_checked_list: '+self.sender().objectName())
    #     max_3 = self.max_3_checked
    #     if box.isChecked():
    #         max_3.append(box)
    #     else:
    #         box in max_3 and max_3.remove(box)
    #     names = [b.objectName().split() for b in max_3]
    #     data = [n[2] for n in names]
    #     if len(set(data)) > 1:
    #         # both ecl and ior are checked
    #         for ie in ('ior','ecl'):
    #             if data.count(ie)>2:
    #                 # check if 2 yaxis or 2 wells are checked
    #                 for yw in ('yaxis','well'):
    #                     ind = [i for i,(a,b,c) in enumerate(names) if a==yw and c==ie]
    #                     if len(ind)>1:
    #                         box = max_3.pop(min(ind))
    #                         set_checkbox(box, False)
    #     else:
    #         # only ecl or ior is checked
    #         if len(max_3)>3:
    #             box = max_3.pop(0)
    #             set_checkbox(box, False)
    #         names = [b.objectName().split()[0] for b in max_3]
    #         for key in ('yaxis','well'):
    #             if names.count(key)>2:
    #                 box = max_3.pop(names.index(key))
    #                 set_checkbox(box, False)
                
            
    #-----------------------------------------------------------------------
    def on_specie_click(self):
    #-----------------------------------------------------------------------
        #print('on_specie_click')
        specie = self.sender().objectName()
        is_checked = self.sender().isChecked()
        self.update_plot_line(specie, is_checked)
        if self.plot_ref_data:
            self.update_plot_line(self.sender().objectName(), is_checked, lines=self.ref_plot_lines, set_data=False)
        self.plot_area.draw()

    #-----------------------------------------------------------------------
    def update_plot_line(self, var, is_checked, lines=None, set_data=True):
    #-----------------------------------------------------------------------
        self.plot_area.plots.update_line(var, set_data)

    #-----------------------------------------------------------------------
    def update_plot_line_OLD(self, name, is_checked, lines=None, set_data=True):
    #-----------------------------------------------------------------------
        #print('update_plot_line')#, name, is_checked, lines, set_data)
        if not lines:
            lines = self.plot_lines
        if not lines:
            #print('call create_plot_lines from update_plot_line') #, name, is_checked, lines, set_data)
            self.create_plot_lines()
        if not name in lines:
            #print('return')
            return
        #print('inside', name, is_checked, lines, set_data)
        for ax,line in lines[name].items():
            #print(line)
            line.set_visible(is_checked)
            if is_checked and set_data:
                well, yaxis, var, data = PlotArea.split_tag(line.get_label())
                if not self.data[data]:
                    #print('return')
                    return
                xdata = self.data[data][well]['days']
                ydata = self.data[data][well][yaxis][var]
                if len(xdata) == len(ydata):
                    line.set_data(xdata, ydata)
                #print(well, yaxis, var, xdata, ydata)
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
        #for ax in self.plot_axes:
        for ax in self.plot_area.axes:
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
        # active_wells = [f.stem.split('W_')[-1] for f in files]
        # print(active_wells)
        inp = self.input
        ior = {}
        for w in self.out_wells:
            ior[w] = {}
            ior[w]['conc'] = {}
            ior[w]['prod'] = {}
        numbers = compile(r'[0-9]+')
        for file in files:
            try:
                if not file.exists() or numbers.search(remove_comments(file, comment='#', raise_error=False) or '') is None:
                    continue
            except PermissionError:
                continue
            # read data
            #print('Reading', file.name)
            well, yaxis = file.name.split('_W_')[-1].split('.trc')
            if yaxis == 'prd':
                yaxis = 'prod'
            try:
                data = genfromtxt(str(file))
            except (PermissionError, FileNotFoundError):
                continue
            try:
                #data = genfromtxt(str(file))
                if data.ndim > 1:
                    self.ior_boxes['well'][well].setEnabled(True)
                    ior[well]['days'] = data[1:,0]
                    #ior['FIELD']['days'] = data[1:,0]
                    #print(ior['days'])
                    for i,name in enumerate(inp['species']):
                        #print(well, yaxis, name)
                        ior[well][yaxis][name] = data[1:,i+1]
                        #ior['FIELD'][yaxis][name] = data[1:,i+1]
                    if 'conc' in yaxis:
                        ior[well]['conc']['Temp'] = data[1:,-1]
                        ior[well]['prod']['Temp'] = data[1:,-1]
                    #print(len(ior['days']), [len(ior[well][yaxis][s]) for s in inp['species']])
            except (KeyError, IndexError, TypeError) as e:
                DEBUG and print('ERROR in read_ior_data:', e)
                pass
        if all(ior[w]['conc']=={} for w in self.out_wells):
            return False
        self.data['ior'] = ior
        #self.add_ior_field_data()
        #print(self.data['ior']['FIELD'])
        return True
        
    
    #-----------------------------------------------------------------------
    def add_ior_field_data(self):
    #-----------------------------------------------------------------------
        # Sum well data as FIELD data
        ior = self.data['ior']
        days = (ior[w]['days'] for w in self.out_wells)
        days = sort(unique(concatenate(list(days))))
        #print(days)
        ior['FIELD'] = {'days':days, 'conc':{}, 'prod':{}}
        for var in self.input['species']:
            #ior['FIELD']['conc'][var] = zeros(days.shape)
            ior['FIELD']['prod'][var] = zeros(days.shape)
        for w in self.out_wells:
            size = ior[w]['days'].size
            for var in self.input['species']:
                #ior['FIELD']['conc'][var][-size:] += ior[w]['conc'][var]
                ior['FIELD']['prod'][var][-size:] += ior[w]['prod'][var]


    # #-----------------------------------------------------------------------
    # def update_axes_names(self):                   # main_window
    # #-----------------------------------------------------------------------
    # #
    # #  Checkboxes are named 'yaxis prod ior', 'yaxis conc ior', 'yaxis rate ecl'
    # #  'well P1 ior', 'well I1 ecl'
    # #
    #     #print('update_axes_names')
    #     #print(self.max_3_checked)
    #     name0 = [b.objectName().split()[0] for b in self.max_3_checked]    
    #     y_idx = [i for i,x in enumerate(name0) if x=='yaxis']
    #     w_idx = [i for i,x in enumerate(name0) if x=='well']
    #     name1 = [b.objectName().split()[1] for b in self.max_3_checked]            
    #     name2 = [b.objectName().split()[2] for b in self.max_3_checked]            
    #     self.ioraxes_names = []
    #     for y in y_idx:
    #         for w in w_idx:
    #             if name2[y] == name2[w]:
    #                 self.ioraxes_names.append(name1[y]+' '+name1[w]+' '+name2[w])
    

    #-----------------------------------------------------------------------
    def create_plot(self):                         # main_window
    #-----------------------------------------------------------------------                
        varbox = {'ecl': self.ecl_boxes['var'], 'ior': self.ior_boxes['var']}
        only_nonzero = not self.worker
        self.plot_area.plots.create(self.data, self.checked_boxes, varbox, only_nonzero)


    # #-----------------------------------------------------------------------
    # def create_plot_OLD(self):                         # main_window
    # #-----------------------------------------------------------------------                
    #     if self.plot_ref_data:
    #         self.plot_ref = True
    #     self.plot_area.clear()
    #     if not self.input['root']:
    #         self.plot_area.draw()
    #         return False
    #     #self.plot.create_axes(checked_boxes=self.checked_boxes, ior=self.ior_boxes, ecl=self.ecl_boxes)
    #     self.plot_area.create_axes(parent=self, checked_boxes=self.checked_boxes) #, ior=self.ior_boxes, ecl=self.ecl_boxes)
    #     #tempbox = {'ecl':self.ecl_boxes['var']['Temp_ecl'], 'ior': self.ior_boxes['var']['Temp']}
    #     for plot in self.plot_area.plots:
    #         if plot.data == 'ecl':# in ax.get_label():
    #             box = self.ecl_boxes['var']['Temp_ecl']
    #         else:
    #             box = self.ior_boxes['var']['Temp']
    #         ax.set_visible(box.isChecked())
    #     self.create_plot_lines()
    #     self.plot_area.adjust()
    #     self.plot_area.draw()
    #     return True

    
    # #-----------------------------------------------------------------------
    # def create_plot_lines(self):
    # #-----------------------------------------------------------------------
    #     linewidth = 1.5
    #     self.plot_lines = {}
    #     lines = {}
    #     self.ref_plot_lines = {}
    #     ref_lines = {}
    #     #ax = self.plot.axes   # or self.canvas.fig.axes
    #     #for i, name in enumerate(self.plot.axes_names):
    #     for i, plot in enumerate(self.plot_area.plots):
    #         #yaxis, well, data = Plot.split_tag(name)
    #         if plot.data=='ior':
    #             var_box = self.ior_boxes['var']
    #         elif plot.data=='ecl':
    #             var_box = self.ecl_boxes['var']
    #         var_list = var_box.keys()
    #         the_data = self.data.get(data)
    #         for var in var_list:
    #             ax = plot.left_axis
    #             if 'temp' in var.lower():
    #                 #ind = i*2+1 # extra right axis for temperature
    #                 ax = plot.right_axis
    #             #else:
    #                 #ind = i*2   # main axes

    #             line = None
    #             if the_data:
    #                 try:
    #                     xdata = the_data[plot.well]['days']
    #                     ydata = the_data[plot.well][plot.yaxis][var]
    #                     #print(yaxis, well, data, var)
    #                     #print(xdata)
    #                     #print(ydata)
    #                     if len(xdata) != len(ydata):
    #                         DEBUG and print('Size mismatch:', len(xdata), len(ydata), well, yaxis, var)
    #                         continue
    #                 except KeyError as e:
    #                     continue
    #                 #line, = ax[ind].plot(xdata, ydata, color=self.plot_prop['color'][var],
    #                 line, = ax.plot(xdata, ydata, color=self.plot_prop['color'][var],
    #                                      linestyle=self.plot_prop['line'][var],
    #                                      alpha=self.plot_prop['alpha'][var],
    #                                      lw=linewidth, label=PlotArea.name_tag(well, yaxis, var, data))
    #                                     # lw=linewidth, label=well+' '+yaxis+' '+var+' '+data)
    #             if line:
    #                 if var not in lines:
    #                     lines[var] = {}
    #                 line.set_visible(var_box[var].isChecked())
    #                 lines[var][ax[ind]] = line
    #                 #print(line, line.get_visible() )
    #             # if we have a compare case
    #             ref_line = None
    #             if self.plot_ref:
    #                 refdata = self.plot_ref_data.get(data)
    #                 if refdata: # and var_box[var].isChecked():
    #                     try:
    #                         xdata = refdata[well]['days']
    #                         ydata = refdata[well][yaxis][var]
    #                         if len(xdata) != len(ydata):
    #                             continue
    #                     except KeyError as e:
    #                         continue                        
    #                     #ref_line, = ax[ind].plot(xdata, ydata, color=self.plot_prop['color'][var],
    #                     ref_line, = ax.plot(xdata, ydata, color=self.plot_prop['color'][var],
    #                                              linestyle=self.plot_prop['line'][var], alpha=0.4,
    #                                              lw=linewidth, label=PlotArea.name_tag(well, yaxis, var, data))
    #                                             #  lw=linewidth, label=well+' '+yaxis+' '+var+' '+data)
    #             if ref_line:
    #                 if var not in ref_lines:
    #                     ref_lines[var] = {}
    #                 ref_line.set_visible(var_box[var].isChecked())
    #                 ref_lines[var][ax[ind]] = ref_line
    #             ax[ind].relim(visible_only=True)
    #             ax[ind].autoscale_view()
    #         self.plot_lines = lines
    #         self.ref_plot_lines = ref_lines
    #     if self.plot_ref:
    #         self.plot_ref = None



            
    #-----------------------------------------------------------------------
    def update_remaining_time(self, text='0:00:00'):           # main_window
    #-----------------------------------------------------------------------
        #print('update_remaining_time: ', text)
        #self.remaining_time.setText('Remaining time:  ' + text)
        #print('remaining time:',text)
        self.remaining_time.setText(text)
        self.remaining_time.repaint()


    #-----------------------------------------------------------------------
    def update_all_plot_lines(self): #, read_data=True):
    #-----------------------------------------------------------------------
        self.plot_area.plots.update()
        self.plot_area.draw()

    #-----------------------------------------------------------------------
    def update_all_plot_lines_OLD(self): #, read_data=True):
    #-----------------------------------------------------------------------
        #print('update_all_plot_lines')
        # update plot-lines
        #if read_data:
        #    self.read_ior_data()
        #    self.read_ecl_data()
        #print(f'update_all_plot_lines: {self.plot_lines}')
        if not self.plot_lines:
            #print('call from update_all_plot_lines')
            self.create_plot_lines()
        for name,ax_line in self.plot_lines.items():
            for ax,line in ax_line.items():
                well, yaxis, var, data = PlotArea.split_tag(line.get_label())
                # print(name, well, yaxis, var, data)
                if not self.data.get(data):
                    return
                if data=='ior':
                    check_box = self.ior_boxes['var'][name]
                    # is_checked = self.ior_boxes['var'][name].isChecked()
                else: # data=='ecl'
                    check_box = self.ecl_boxes['var'][name]
                    # is_checked = self.ecl_boxes['var'][name].isChecked()
                line.set_visible(check_box.isChecked())
                if check_box.isChecked():
                    try:
                        xdata = self.data[data][well]['days']
                        ydata = self.data[data][well][yaxis][var]
                        if (len(xdata) == len(ydata)):
                            line.set_data(xdata, ydata)
                        else:
                            set_checkbox(check_box, False)
                        #print(len(xdata), len(ydata))
                    except KeyError as e:
                        DEBUG and print(f'KeyError in update_all_plot_lines: {e}')
                        pass
                #try:
                #    ax.relim(visible_only=True)
                #    ax.autoscale_view()
                #except ValueError:
                #    print('ValueError in update_all_plot_lines()')
                #    #pass
        try:
            self.update_axes_limits()
            self.plot_area.draw()
        except ValueError as e:
            DEBUG and print(f'ValueError in update_all_plot_lines:: {e}')
            pass


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
            #print('t',t)
            self.update_progressbar(t)
            self.update_remaining_time(text=self.progress.remaining_time(t))
        #print(self.progressbar.minimum(), self.progressbar.maximum(), self.progressbar.value())
        

    #-----------------------------------------------------------------------
    def update_view_area(self):
    #-----------------------------------------------------------------------
        #print('update_view_area')
        ok = {'ecl':True, 'ior':True}
        if self.mode == 'backward':
            ok['ior'] = self.read_ior_data()
            ok['ecl'] = self.read_ecl_data()
        # self.days = None
        if self.mode in ('forward','eclipse','iorsim') and self.worker and self.worker.current_run():
            run = self.worker.current_run()
            #print('run', run)
            if run in ('ecl','eclipse','eclrun'):
                run = 'ecl'
                ok[run] = self.read_ecl_data()
            if run in ('ior','iorsim'):
                run = 'ior'
                #print('read')
                ok[run] = self.read_ior_data()
            # if self.data.get(run):
            #     self.days = (self.data[run]).get('days')
        view = self.current_viewer() #.name
        if view == self.plot_area: #'plot':
            #self.statusBar().clearMessage()
            #if not all(list(ok.values())):
            #    self.statusBar().showMessage('No wells are currently producing')            
            #print('call from update_view_area')
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
            show_message(self, 'warning',
                text=('You need to choose a case from the case drop-down list, '
                'or add a new case from File -> Add case.'))
            return False
        if i['days']==0:
            show_message(self, 'warning', text='Total time interval is missing.')
            return False
        if self.max_days and i['days'] > self.max_days:
            show_message(self, 'info',
                text=f'The IORSim-run is limited to {self.max_days:.2f} days.')
            self.days_box.setText(self.max_days)
            #self.on_input_change
            #return False
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
    def set_toolbar_enabled(self, value):                      # main_window
    #-----------------------------------------------------------------------
        self.start_act.setEnabled(value)
        self.case_cb.setEnabled(value)
        self.mode_cb.setEnabled(value)
        if self.mode=='backward':
            self.days_box.setEnabled(value)
        #self.sim_cb.setEnabled(value)

    #-----------------------------------------------------------------------
    def run_sim(self):                                         # main_window
    #-----------------------------------------------------------------------
        #print('run_mode:',self.mode, self.run)
        if not self.input_OK():
            return
        # Clear data
        self.data = {}
        self.unsmry = None
        # Clear messages and progress
        self.reset_progress_and_message()
        # Enable all well-boxes
        self.enable_well_boxes(True)
        # # Uncheck well-boxes
        # for box in list(self.ecl_boxes['well'].values())+list(self.ior_boxes['well'].values()):
        #     self.uncheck_box(box)
        #     box.setEnabled(False)
        # Disable toolbar
        self.set_toolbar_enabled(False)
        i = self.input
        s = self.settings
        # backward mode
        if self.mode=='backward':
            kwargs = {'mode':'backward', 'check_unrst':s.get('unrst'), 'check_rft':s.get('rft')}
        # forward mode
        elif self.mode in ('forward','eclipse','iorsim'):
            kwargs = {'mode':'forward', 'run_names':self.run}
        # start simulation
        for opt in ('convert','del_convert','merge','del_merge','check_input_kw'):
            #kwargs[opt] = s.get[opt]()
            kwargs[opt] = s.get(opt)
        self.worker = sim_worker(root=i['root'], time=i['days'], iorexe=s.get('iorsim'), eclexe=s.get('eclrun'),
                                 ecl_keep_alive=s.get('ecl_keep_alive') and float(s.get('ecl_alive_limit')),
                                 ior_keep_alive=s.get('ior_keep_alive') and IOR_ALIVE_LIMIT,
                                 days_box=self.days_box, verbose=int(s.get('log_level')), skip_empty=s.get('skip_empty'),
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
        
    # #-----------------------------------------------------------------------
    # def show_message(self, par):
    # #-----------------------------------------------------------------------
    #     show_message(self, par[0], text=par[1])
        
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
        self.close()
        # Quit Qt
        QApplication.quit()

    #-----------------------------------------------------------------------
    def close(self):                                            # main_window
    #-----------------------------------------------------------------------
        self.killsim()
        if self.download_worker:
            self.download_worker.running = False
            sleep(0.1)
        self.save_session()
        # Save window geometry in settings
        geo = f'geometry {" ".join( (str(i) for i in self.geometry().getRect()) )}\n'
        replace_line(self.settings.file, find='geometry', replace=geo)
                

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
    def __init__(self, parent=None, comment='#', color=Qt.gray, keywords=()):
        super().__init__(parent)

        self.highlightingRules = []
        #print(keywords)
        #while keywords:
        for kw in keywords:
            #kword = keywords.pop(0)
            keywordFormat = QTextCharFormat()
            keywordFormat.setForeground(kw.color)
            keywordFormat.setFontWeight(kw.weight)
            #option = kword.pop(0)
            #front = kword.pop(0)
            #back = kword.pop(0)
            keywordPatterns = [kw.front + k + kw.back for k in kw.words]            
            self.highlightingRules.extend( [(QRegularExpression(pattern, kw.option), keywordFormat) for pattern in keywordPatterns] )
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

    ### Need to set the locale under Linux to avoid datetime.strptime errors
    os.putenv('LC_ALL', 'C')
    #os.environ['QT_ENABLE_HIGHDPI_SCALING'] = '0'
    #os.environ['QT_FONT_DPI'] = '96'
    #os.environ['QT_SCALE_FACTOR'] = '1'
    #os.putenv('QTWEBENGINE_CHROMIUM_FLAGS', '--disable-logging')
    args = []

    #print(sys.argv)
    if len(sys.argv) > 1:
        if sys.argv[1] == '-upgrade':
            #print(sys.argv[2:])
            Upgrader(sys.argv[2:]).upgrade()
        else:
            print()
            print('   This is the terminal-version of IORSim_GUI')
            print('   Start IORSim_GUI without arguments to open the GUI')
            print()
            ior2ecl_main(settings_file=SETTINGS_FILE)
    else:
        IORSIM_DIR.mkdir(exist_ok=True)
        exit_code = main_window.EXIT_CODE_REBOOT
        while exit_code == main_window.EXIT_CODE_REBOOT:
            app = QApplication(sys.argv + args)
            #print(app.screens())
            window = main_window(settings_file=SETTINGS_FILE)
            window.show()
            exit_code = app.exec()
            window.close()
            app = None
