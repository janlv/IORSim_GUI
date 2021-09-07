#!/usr/bin/env python
# -*- coding: utf-8 -*-

#from struct import unpack
#from io import SEEK_CUR
#from os import startfile
import struct
#import os
from pathlib import Path

#from psutil import net_connections
#import traceback

#from numpy.lib.twodim_base import triu_indices
from .utils import list2str, float_or_str, remove_comments
from numpy import zeros, int32, float32, float64, ceil, bool_ as np_bool, array as nparray, append as npappend 
from mmap import mmap, ACCESS_READ
from re import finditer, compile
from copy import deepcopy
from collections import namedtuple
from datetime import datetime, timedelta
from struct import unpack
#from numba import njit, jit

#
#
#  import IORlib.ECL as ECL
#  import sys
#  fname = sys.argv[1]
#  for block in ECL.unformatted_file(fname).blocks():
#      if block.key()=='SEQNUM':
#           block.print()
#
#
#

# Maximum number of data records in one block.
# For REAL this means that 4*1000 bytes is the max size 
max_length = 1000   

endian = '>' # big-endian

unpack_char = {b'INTE' : 'i',
               b'REAL' : 'f',
               b'LOGI' : 'i',
               b'DOUB' : 'd',
               b'CHAR' : 's',
               b'MESS' : ' '}

#pack_char = {'INTE' : 'i',
#             'REAL' : 'f',
#             'LOGI' : 'i',
#             'DOUB' : 'd',
#             'CHAR' : 's',
#             'MESS' : ' '}

datasize = {b'INTE' : 4,
            b'REAL' : 4,
            b'LOGI' : 4,
            b'DOUB' : 8,
            b'CHAR' : 8,
            b'MESS' : 1}

datatype = {'INTE' : int32,
            'REAL' : float32,
            'LOGI' : np_bool,
            'DOUB' : float64,
            'CHAR' : str,
            'MESS' : str}

var2key = {'nwell':b'INTEHEAD'}
var2pos = {'nwell':16}


#-----------------------------------------------------------------------
def get_tsteps(file, raise_error=True):
#-----------------------------------------------------------------------
    # Remove comments
    if not Path(file).is_file():
        return [0]
    data = remove_comments(file)
    regex = compile(r'\bTSTEP\b\s+([0-9*.\s]+)/')
    tsteps = [t for m in regex.findall(data) for t in m.split()]
    # Process x*y statements
    mult = lambda x, y : int(x)*(' '+y) 
    tsteps = [t if not '*' in t else mult(*t.split('*')) for t in tsteps]
    tsteps = [float(t) for ts in tsteps for t in ts.split()]
    if not tsteps:
        if raise_error:
            raise SystemError(f'ERROR TSTEPS keyword not found in {file}')
        else:
            tsteps = [0]
    return tsteps

#-----------------------------------------------------------------------
def get_restart_file_step(file, unformatted=True):
#-----------------------------------------------------------------------
    # Remove comments
    file = Path(file)
    data = remove_comments(file)
    regex = compile(r'\bRESTART\b\s+([a-zA-Z0-9_-]+)\s+([0-9]+)\s*/')
    name_step = regex.findall(data)
    if name_step:    
        name, step = [list(t) for t in zip(*name_step)]
        n = int(step[0])
        if unformatted:
            ext = '.UNRST'
        else:
            ext = f'.S{n:04}'
        return file.with_name(name[0]+ext), n 
    return '', 0

#-----------------------------------------------------------------------
def get_restart_time_step(file, unformatted=True):
#-----------------------------------------------------------------------
    t = 0
    file = Path(file)
    name, n = get_restart_file_step(file, unformatted=unformatted)
    if name and n:    
        #file = file.with_name(name)
        if unformatted:
            t,nn = get_time_step_UNRST(file=name, step=n)
            if nn[-1] != n:
                raise SystemError(f'ERROR Unable to read restart time at step {n} from {file.name}')
            t = t[-1]
        else:
            t = get_time_step_UNSMRY(file=name)[0]
    #print(t, n)
    return t, n

#-----------------------------------------------------------------------
def get_date_keyword(file, keyword, raise_error=False):
#-----------------------------------------------------------------------
    # Remove comments
    data = remove_comments(file)
    regex = compile(rf'\b{keyword}\b\s+(\d+)\s+\'*(\w+)\'*\s+(\d+)')
    dates = [' '.join(m.group(1,2,3)) for m in regex.finditer(data)]
    dates = [datetime.strptime(d, '%d %b %Y').date() for d in dates]
    if not dates and raise_error:
        raise SystemError(f'WARNING {keyword} keyword not found in {file}')
    return dates


#-----------------------------------------------------------------------
def get_start(file):
#-----------------------------------------------------------------------
    start = get_date_keyword(file, 'START')
    if start:
        start = start[0]
    else:
        start = datetime.now().date()
    return start

#-----------------------------------------------------------------------
def get_dates(file):
#-----------------------------------------------------------------------
    return get_date_keyword(file, 'DATES')

#-----------------------------------------------------------------------
def get_time_step_MSG(root=None, file=None, end=True): #, date=False):
#-----------------------------------------------------------------------
    #print('get_time_step_MSG')
    if file is None:
        file = root+'.MSG'
    with open(file) as f:
        lines = f.readlines()
    lines = ''.join(lines)
    time_reg = compile(r'<\s*\bmessage\b\s+\bdate\b="[0-9/]+"\s+time="([0-9.]+)"\s*>')
    step_reg = compile(r'\bRESTART\b\s+\bFILE\b\s+\bWRITTEN\b\s+\bREPORT\b\s+([0-9]+)')
    time = time_reg.findall(lines)
    if not time:
        return 0, 0
    step = step_reg.findall(lines)
    # Convert to numbers
    time = [float(t) for t in time]
    step = [int(s) for s in step]
    #print(time, step)
    if not time or not step:
        return 0, 0
    if end:
        # Return only last entries
        return time[-1], step[-1]
    else:
        return time, step

#-----------------------------------------------------------------------
def get_time_step_UNSMRY(root=None, file=None):
#-----------------------------------------------------------------------
    '''
    Return last time and step from an UNSMRY file
    '''
    #print('get_time_step_UNSMRY')
    if file is None:
        file = str(root)+'.UNSMRY'
    t, n = [0], [0]
    for block in unfmt_file(file).tail_blocks():
        if block.key()=='PARAMS':
            t.append(block.data()[0])
        if block.key()=='MINISTEP':
            n.append(block.data()[0])
            break
    return t[-1],n[-1]

#-----------------------------------------------------------------------
def get_time_step_UNRST(root=None, file=None, end=False, step=None):
#-----------------------------------------------------------------------
    #print('get_time_step_UNRST', file)
    #start = datetime.now()
    if file is None:
        file = str(root)+'.UNRST'
    # DOUBHEAD is time, SEQNUM is step
    kw = {'DOUBHEAD':[], 'SEQNUM':[]}
    t, n = kw.values()
    for block in unfmt_file(file).blocks():
        if block.key() in kw.keys():
            kw[block.key()].append(block.data()[0])
        if n and n[-1] == step and len(t) == len(n):
            break
    if not t or not n:
        return 0, 0
    if end:
        return t[-1], n[-1]
    #print(datetime.now()-start)
    return t, n

#-----------------------------------------------------------------------
def get_start_UNRST(root=None, file=None):
#-----------------------------------------------------------------------
    if file is None:
        file = str(root)+'.UNRST'
    for block in unfmt_file(file).blocks():
        if block.key() == 'INTEHEAD':
            dmy = [str(d) for d in block.data()[64:67]]
            return datetime.strptime(' '.join(dmy), '%d %m %Y').date()


#====================================================================================
class _datablock:                                                         # datablock
#====================================================================================

    #--------------------------------------------------------------------------------
    def __init__(self):                                                   # datablock
    #--------------------------------------------------------------------------------
        self.reset()
        
    #--------------------------------------------------------------------------------
    def set_header(self, chunk, filepos):                                 # datablock
    #--------------------------------------------------------------------------------
        self._key, self.length, self.datatype = struct.unpack(endian+'8si4s', chunk)
        if self.datatype in (b'CHAR',):
            self.length *= datasize[self.datatype]
        # Set filepos to start of header.
        # The header is 4+16+4 = 24 bytes long
        self.startpos = filepos - 24 # for forward parsing!
        
    #--------------------------------------------------------------------------------
    def add_chunk(self, chunk):                                           # datablock
    #--------------------------------------------------------------------------------
        self.chunk += chunk
        #self.empty = False

    #--------------------------------------------------------------------------------
    #def get_data_array(self):                                             # datablock
    def data(self):                                             # datablock
    #--------------------------------------------------------------------------------
        #if self.chunk in (b'0',):
        if len(self.chunk)==0:
            return None
        if len(self.chunk)==1 and self.chunk[0]==0:
            raise SystemError('No data to unpack in data().\nDid you pass data=True to parse_file()?')
        #if self.datatype in (b'CHAR',):
        #    self.length *= datasize[self.datatype]
        try:
            return struct.unpack(endian + str(self.length) + unpack_char[self.datatype], self.chunk)
        except struct.error as e:
            raise SystemError(e)
        
    #--------------------------------------------------------------------------------
    def reset(self):                                                      # datablock
    #--------------------------------------------------------------------------------
        #self.chunk = b''
        self.chunk = bytearray()
        self.length = None
        self._key = ''
        #self.endpos = self.startpos = None
        
    #--------------------------------------------------------------------------------
    def ready(self):                                                      # datablock
    #--------------------------------------------------------------------------------
        #if self.chunk != b'' or self.length == 0:
        if len(self.chunk)!=0 or self.length==0:
            return True
        else:
            return False

    #--------------------------------------------------------------------------------
    def info(self):                                                       # datablock
    #--------------------------------------------------------------------------------
        return f'{self._key.decode()} block of {len(self.chunk)} bytes holding {self.length} {self.datatype.decode()} at [{self.startpos}, {self.end()}]'

    #--------------------------------------------------------------------------------
    def print(self):                                                      # datablock
    #--------------------------------------------------------------------------------
        print(self.info())
        
    #--------------------------------------------------------------------------------
    def get_nwell(self):                                                  # datablock
    #--------------------------------------------------------------------------------
        return self.get_value('nwell')
        #try:
        #    return self.get_data_array()[16]
        #except:
        #    raise ValueError('Only INTEHEAD blocks contain NWELL, this is a {:s} block'.format(self._key.decode())) 

    #--------------------------------------------------------------------------------
    def get_value(self, var):                                             # datablock
    #--------------------------------------------------------------------------------
        #if self._key == b'INTEHEAD':
        #if self._key != self.var2key[var]:
        #    raise ValueError('ERROR! {} blocks does not hold {}'.format(self._key.decode(), var))
        #return self.get_data_array()[var2pos[var]]
        return self.data()[var2pos[var]]
        
    #--------------------------------------------------------------------------------
    def key(self):                                             # datablock
    #--------------------------------------------------------------------------------
        return self._key.decode().strip()
        
    #--------------------------------------------------------------------------------
    def start(self):                                             # datablock
    #-------------------------------------------------------------------------------
        return self.startpos
        
    #--------------------------------------------------------------------------------
    def end(self):                                             # datablock
    #-------------------------------------------------------------------------------
        # A data-block is split in chunks if the length exceeds 1000 elements
        # Each data-block is sandwiched by a 4 byte int giving the size of the block
        return self.startpos + 24 + len(self.chunk) + 8*int(ceil(self.length/max_length))
    
    #--------------------------------------------------------------------------------
    def datatype(self):                                             # datablock
    #-------------------------------------------------------------------------------
        return self.datatype.decode(),
        

#====================================================================================
class blockdata:
#====================================================================================
    #--------------------------------------------------------------------------------
    def __init__(self, key=b'', length=0, type=b'', start=0, end=0, data=None, data_start=0):
    #--------------------------------------------------------------------------------
        self._key = key
        self.length = length
        self.type = type
        self.mmap = data
        self.startpos = start
        self._end = end
        self.data_start = data_start

    #--------------------------------------------------------------------------------
    def info(self, details=False):                                                       # block
    #--------------------------------------------------------------------------------
        s = f'{self._key.decode()}'
        if details:
            s += f' block of {self.length*datasize[self.type]} bytes' 
            s += f' holding {self.length*(self.type==b"CHAR" and 8 or 1)} {self.type.decode()}'
            s += f' at [{self.startpos}, {self._end}]'
        return s

    #--------------------------------------------------------------------------------
    def print(self, data=False, details=False):                                                       # block
    #--------------------------------------------------------------------------------
        print(self.info(details=details), end='')
        if data:
            print(':',self.data())
        else:
            print()

    # #--------------------------------------------------------------------------------
    # def _data(self):
    # #--------------------------------------------------------------------------------
    #     value = []
    #     length = self.length
    #     if self.type == b'CHAR':
    #         value = b''
    #         length *= 8 
    #     a = self.data_start
    #     while len(value) < length:
    #         size = struct.unpack(endian+'i',self.mmap[a:a+4])[0]
    #         a += 4
    #         if self.type == b'CHAR':
    #             value += struct.unpack(endian+f'{size}s',self.mmap[a:a+size])[0]
    #         else:      
    #             n = int(size/datasize[self.type]) 
    #             value.extend(struct.unpack(endian+f'{n}{unpack_char[self.type]}',self.mmap[a:a+size]))
    #         a += size + 4
    #     if self.type == b'CHAR':
    #         value = [value.decode()]        
    #     return value

    #--------------------------------------------------------------------------------
    def data(self):
    #--------------------------------------------------------------------------------
        if self.type == b'CHAR':
            return self._data_char()
        else:
            return self._data_not_char()

    #--------------------------------------------------------------------------------
    def _data_not_char(self):
    #-------------------------------<-------------------------------------------------
        value = []
        a = self.data_start
        while len(value) < self.length:
            size = struct.unpack(endian+'i',self.mmap[a:a+4])[0]
            a += 4
            n = int(size/datasize[self.type]) 
            value.extend(struct.unpack(endian+f'{n}{unpack_char[self.type]}',self.mmap[a:a+size]))
            a += size + 4
        return value

    #--------------------------------------------------------------------------------
    def _data_char(self):
    #--------------------------------------------------------------------------------
        value = b''
        a = self.data_start
        while len(value) < 8*self.length:
            size = struct.unpack(endian+'i',self.mmap[a:a+4])[0]
            a += 4
            value += struct.unpack(endian+f'{size}s',self.mmap[a:a+size])[0]
            a += size + 4
        return [value.decode()]

    #--------------------------------------------------------------------------------
    def get_value(self, var):                                             # block
    #--------------------------------------------------------------------------------
        return self.data()[var2pos[var]]
        
    #--------------------------------------------------------------------------------
    def get_nwell(self):                                                  # block
    #--------------------------------------------------------------------------------
        return self.get_value('nwell')

    #--------------------------------------------------------------------------------
    def key(self):                                             # block
    #--------------------------------------------------------------------------------
        return self._key.decode().strip()
        
    #--------------------------------------------------------------------------------
    def start(self):                                             # block
    #-------------------------------------------------------------------------------
        return self.startpos

    #--------------------------------------------------------------------------------
    def end(self):                                             # block
    #-------------------------------------------------------------------------------
        return self._end


#====================================================================================
class unfmt_file:
#====================================================================================
    #  | h e a d e r  |   d a t a     |
    #  |4i|8s|4i|4s|4i|4i|1000 data|4i| 
    #  |    24 bytes  |
    #  |              |4i|8d| 

    #--------------------------------------------------------------------------------
    def __init__(self, filename):                                        # unfmt_file
    #--------------------------------------------------------------------------------
        self.fileobj = None
        self._filename = Path(filename)
        self.endpos = self.startpos = 0

    #--------------------------------------------------------------------------------
    def is_file(self):                                                   # unfmt_file
    #--------------------------------------------------------------------------------
        return self._filename.is_file()

    #--------------------------------------------------------------------------------
    def set_startpos(self, pos):                                         # unfmt_file
    #--------------------------------------------------------------------------------
        self.startpos = pos
        
    #--------------------------------------------------------------------------------
    def set_endpos(self, pos):                                           # unfmt_file
    #--------------------------------------------------------------------------------
        self.endpos = pos
        
    #--------------------------------------------------------------------------------
    def end_not_reached(self):                                           # unfmt_file
    #--------------------------------------------------------------------------------
        return self.endpos < self.size()

    #--------------------------------------------------------------------------------
    def filename(self):                                                  # unfmt_file
    #--------------------------------------------------------------------------------
        return str(self._filename)
        
    #--------------------------------------------------------------------------------
    def size(self):                                                      # unfmt_file
    #--------------------------------------------------------------------------------
        return self._filename.stat().st_size
        
    #--------------------------------------------------------------------------------
    def name(self):                                                      # unfmt_file
    #--------------------------------------------------------------------------------
        return str(self._filename.name)
        
    #--------------------------------------------------------------------------------
    #def blocks(self, data=True, datalist=(), only_new=False, encode=False):   # unfmt_file
    def blocks_old(self, data=True, datalist=(), only_new=False, encode=False):   # unfmt_file
    #--------------------------------------------------------------------------------
        if not Path(self._filename).is_file():
            return
        if datalist:
            if encode:
                datalist = [d.ljust(8).encode() for d in datalist]
            data = False
        with open(self._filename, 'rb') as self.fileobj:
            if only_new:
                startpos = self.endpos
            else:
                startpos = self.startpos
            self.fileobj.seek(startpos, 1)
            db = _datablock()
            while True:
                try:
                    chunk = self._read_chunk(self._read_bytes_forward, self._safe_seek_forward)
                except EOFError:
                    self.endpos = self.fileobj.tell()
                    if db.ready():
                        yield db
                    return
                except MemoryError:
                    return False
                if self._chunk_is_header(chunk):
                    if db.ready():
                        yield db
                        db.reset()
                    db.set_header(chunk, self.get_filepos())
                else:
                    #print(db._key, datalist)
                    if not data and not db._key in datalist:
                        chunk = bytearray(1)
                    db.chunk += chunk


    #--------------------------------------------------------------------------------
    def blocks(self, only_new=False):   # unfmt_file
    #--------------------------------------------------------------------------------
        if not Path(self._filename).is_file():
            return
        if only_new:
            startpos = self.endpos
        else:
            startpos = self.startpos
        with open(self._filename, mode='rb') as file:
            with mmap(file.fileno(), length=0, access=ACCESS_READ) as data:
                data.seek(startpos, 1)
                while data.tell() < data.size():
                    start = data.tell()
                    # Header
                    data.seek(4, 1)
                    key, length, type = struct.unpack(endian+'8si4s', data.read(16))
                    data.seek(4, 1)
                    # Value array
                    data_start = data.tell()
                    max_length = 1000
                    if type==b'CHAR':
                        max_length = 105
                    bytes = length*datasize[type] + 8*int(ceil(length/max_length))
                    data.seek(bytes, 1)
                    yield blockdata(key=key, length=length, type=type, start=start, end=data.tell(), 
                                    data=data, data_start=data_start)
                self.endpos = data.tell()

    #--------------------------------------------------------------------------------
    def tail_blocks(self):                                               # unfmt_file
    #--------------------------------------------------------------------------------
        if not Path(self._filename).is_file():
            return
        with open(self._filename, mode='rb') as file:
            with mmap(file.fileno(), length=0, access=ACCESS_READ) as data:
                # Goto end of file
                data.seek(0, 2)
                while data.tell() > 0:
                    end = data.tell()
                    # Header
                    # Rewind until we find a header
                    while data.tell() > 0:
                        data.seek(-4, 1)
                        size = struct.unpack(endian+'i',data.read(4))[0]
                        data.seek(-4-size, 1)
                        if self.is_header(data, size, data.tell()):
                            start = data.tell()-4
                            key, length, type = struct.unpack(endian+'8si4s', data.read(16))
                            data.seek(4, 1)
                            break
                        else:
                            data.seek(-4, 1)
                    # Value array
                    data_start = data.tell()
                    data.seek(start, 0)
                    yield blockdata(key=key, length=length, type=type, start=start, end=end, 
                                    data=data, data_start=data_start)
                #self.endpos = data.tell()

    #--------------------------------------------------------------------------------
    def is_header(self, data, size, pos):                                # unfmt_file
    #--------------------------------------------------------------------------------
        if size==16:
            try:
                #print(data[pos+12:pos+16])
                datasize[data[pos+12:pos+16]]
                return True
            except KeyError as e:
                return False
        else:
            return False

    # #--------------------------------------------------------------------------------
    # def tail_blocks(self, data=True, datalist=(), encode=False):         # unfmt_file
    # #--------------------------------------------------------------------------------
    #     #self.set_direction('backward')
    #     if not Path(self._filename).is_file():
    #         return
    #     if datalist:
    #         if encode:
    #             datalist = [d.ljust(8).encode() for d in datalist]
    #         data = False
    #     with open(self._filename, 'rb') as self.fileobj:
    #         # Go to end of file
    #         self.fileobj.seek(0, 2)
    #         db = _datablock()
    #         while True:
    #             try:
    #                 chunk = self._read_chunk(self._read_bytes_backward, self._safe_seek_backward)
    #             except EOFError:
    #                 self.endpos = self.fileobj.tell()
    #                 return
    #             if self._chunk_is_header(chunk):
    #                 db.set_header(chunk, self.get_filepos()+24)
    #                 if db.ready():
    #                     yield db
    #                     db.reset()
    #             else:
    #                 #print('key:', db._key, 'list:', datalist)
    #                 #if not data and not db._key in datalist:
    #                 if not data:
    #                     #chunk = b'0'
    #                     chunk = bytearray(1)
    #                 #db.add_chunk(chunk)
    #                 db.chunk += chunk

                    
    #--------------------------------------------------------------------------------
    def get_filepos(self):                                               # unfmt_file
    #--------------------------------------------------------------------------------
        return self.fileobj.tell() #/self.filesize
                    
    #--------------------------------------------------------------------------------
    def _read_bytes_forward(self, nbytes):                               # unfmt_file
    #--------------------------------------------------------------------------------
        bytesread = self.fileobj.read(nbytes)
        if not bytesread:
            raise EOFError('End of {:s} reached!'.format(self.fileobj.name))
        return(bytesread)

    #--------------------------------------------------------------------------------
    def _read_bytes_backward(self, nbytes):                              # unfmt_file
    #--------------------------------------------------------------------------------
        self._safe_seek_backward(nbytes)
        bytesread = self.fileobj.read(nbytes)
        if not bytesread:
            raise EOFError('End of {:s} reached!'.format(self.fileobj.name))
        self._safe_seek_backward(nbytes)
        return(bytesread)
    
    #--------------------------------------------------------------------------------
    def _safe_seek_backward(self, nbytes):                               # unfmt_file
    #--------------------------------------------------------------------------------
        if self.fileobj.tell() >= nbytes:
            self.fileobj.seek(-nbytes, 1)
        else:
            raise EOFError('Cannot seek beyond start of file {:s}'.format(self.fileobj.name))

    #--------------------------------------------------------------------------------
    def _safe_seek_forward(self, nbytes):                                # unfmt_file
    #--------------------------------------------------------------------------------
        #try:
        self.fileobj.seek(nbytes,1)
        #except EOFError:
        #    pass

    #--------------------------------------------------------------------------------
    def _chunk_is_header(self, chunk):                                   # unfmt_file
    #--------------------------------------------------------------------------------
        if len(chunk)==16:
            try:
                datasize[chunk[-4:]]
                return True
            except KeyError as e:
                return False
        else:
            return False
        
    #--------------------------------------------------------------------------------
    def _read_chunk(self, read_func, seek_func):                         # unfmt_file
    #--------------------------------------------------------------------------------
        size = struct.unpack(endian + 'i', read_func(4))[0] # read one int
        chunk = read_func(size)
        seek_func(4) # skip trailing/leading int
        return chunk

    #--------------------------------------------------------------------------------
    def tail_block_is(self, key):                                        # unfmt_file
    #--------------------------------------------------------------------------------
        try:
            key = key.upper().strip().ljust(8).encode()
            for block in self.tail_blocks(data=False):
                if block._key == key:
                    #block.print()
                    return True
                else:
                    return False
        except (AttributeError, ValueError):
            pass
        except:
            raise
        
    #--------------------------------------------------------------------------------
    def has_new_blocks(self):                                            # unfmt_file
    #--------------------------------------------------------------------------------
        if self._filename.stat().st_size > self.startpos:
            return True            

    #--------------------------------------------------------------------------------
    def exists(self):                                                    # unfmt_file
    #--------------------------------------------------------------------------------
        #print('Checking for ' + self._filename)
        if self._filename.is_file():
            return True

    #--------------------------------------------------------------------------------
    def create(self, *args, progress=lambda x:None, cancel=lambda:None): # unfmt_file
    #--------------------------------------------------------------------------------
        sections = args
        # Make sure the sizes of each section are equal
        # Remove last unit if they are unequal
        min_size = min([sec.size() for sec in sections])
        for sec in sections:
            while sec.size() > min_size:
                sec.pop_unit(-1)
                #print('pop from '+str(sec.filename()))
        # Open files
        progress(-(min_size+1))
        out_file = open(self._filename, 'wb')
        for sec in sections:
            sec.open_file()
        # Write sections to out_file
        OK = True
        n = 0
        while OK: 
            for sec in sections:
                OK = sec.write_next_unit(out_file)
            n += 1
            progress(n)
            cancel()
        # Close files
        for sec in sections:
            sec.close_file()
        out_file.close()
        return self._filename
    


#====================================================================================
class Section:                                                              # Section
#====================================================================================
    #--------------------------------------------------------------------------------
    def __init__(self, filename, init_key='SEQNUM', start_before=None, start_after=None,
                 end_before=None, end_after=None, skip_sections=None, remove_blocks=None):
    #--------------------------------------------------------------------------------
        self._filename = Path(filename)
        self._fh = None # filehandle
        self.init_key = init_key
        self.start_before = start_before
        self.start_after = start_after
        self.end_before = end_before
        self.end_after = end_after
        if remove_blocks and not isinstance(remove_blocks, tuple):
            remove_blocks = (remove_blocks,)
        self.remove_blocks = remove_blocks
        self._units = self.startpos_and_size()
        if skip_sections != None:
            if not isinstance(skip_sections, tuple):
                skip_sections = (skip_sections,)
            for sec in skip_sections:
                #print('sec:'+str(sec))
                self._units.pop(sec)
            
    #--------------------------------------------------------------------------------
    def open_file(self):                                                    # Section
    #--------------------------------------------------------------------------------
        self._fh = open(self._filename, 'rb')
        
    #--------------------------------------------------------------------------------
    def close_file(self):                                                   # Section
    #--------------------------------------------------------------------------------
        self._fh.close()
        
    #--------------------------------------------------------------------------------
    def filename(self):                                                     # Section
    #--------------------------------------------------------------------------------
        return self._filename
        
    #--------------------------------------------------------------------------------
    def size(self):                                                         # Section
    #--------------------------------------------------------------------------------
        return len(self._units)
        
    #--------------------------------------------------------------------------------
    def pop_unit(self, nr):                                                 # Section
    #--------------------------------------------------------------------------------
        return self._units.pop(nr)
        
    #--------------------------------------------------------------------------------
    def write_next_unit(self, out_file):                                    # Section
    #--------------------------------------------------------------------------------
        if len(self._units)==0:
            return False
        unit = self._units.pop(0)
        for pos, size in zip(unit[::2],unit[1::2]):
            self._fh.seek(pos)
            out_file.write( self._fh.read(size) )
        return True
    
    #--------------------------------------------------------------------------------
    def print(self):                                                        # Section
    #--------------------------------------------------------------------------------
        for unit in self._units:
            print(unit)

                             
    #--------------------------------------------------------------------------------
    def startpos_and_size(self):                                            # Section
    #--------------------------------------------------------------------------------
        # A unit is a list of consecutive absolute filepositions and relative
        # byte chuncks corresponding to the length of the blocks to be kept.
        # A unit list always starts with a pos and ends with a size
        if not self._filename.is_file():
            raise FileNotFoundError(str(self._filename) + ' not found in Section')
        units = []
        inside = False
        #self.keys = []
        for block in unfmt_file(self._filename).blocks():
            key = block.key()
            if inside and key==self.end_before:# and len(start)>1:
                inside = False
                # size
                unit.append(block.start()-unit[-1])
                #self.keys.append('A:'+key+',size:'+str(size[-1]))
            if inside and key==self.end_after:
                inside = False
                # size
                unit.append(block.end()-unit[-1])
                #self.keys.append('B:'+key+',size:'+str(size[-1]))
            if not inside and key==self.start_before:
                inside = True
                # pos
                unit = [block.start(),]
                units.append(unit)
                #self.keys.append('C:'+key+',start:'+str(start[-1]))
            if not inside and key==self.start_after:
                inside = True
                # pos
                unit = [block.end(),]
                units.append(unit)
                #self.keys.append('D:'+key+',start:'+str(start[-1]))
            if inside and self.remove_blocks and key in self.remove_blocks:
                # size
                unit.append(block.start()-unit[-1])
                # pos
                unit.append(block.end())
                #self.keys.append('E:'+key+',size:'+str(size[-1])+',start:'+str(start[-1]))
        if self.end_before==self.init_key:
            unit.append(self._filename.stat().st_size-unit[-1])
            #self.keys.append('F:'+self.end_before+',size:'+str(size[-1]))
        return units
        

    
#====================================================================================
class check_blocks:                                                    # check_blocks
#====================================================================================

    #--------------------------------------------------------------------------------
    def __init__(self, filename, start=None, end=None, var=None):      # check_blocks
    #--------------------------------------------------------------------------------
        #self.reader = reader(file)
        self.file = unfmt_file(filename)
        #self.key = {'start':encode(start), 'end':encode(end)}
        self.key = {'start':start.ljust(8).encode(), 'end':end.ljust(8).encode()}
        self._var = var
        self.out = {k:[] for k in ('start', 'end', 'startpos')+(self._var,)}
        self.out['end'] = 0
        #self.datalist = [self.key['start']]
        #if var:
        #    self.datalist.append(var2key[var])
        self.last_block = None
    
    #--------------------------------------------------------------------------------
    def var(self, _var):                                               # check_blocks
    #--------------------------------------------------------------------------------
        return self.out[_var][0]
            
    #--------------------------------------------------------------------------------
    def filename(self):                                                # check_blocks
    #--------------------------------------------------------------------------------
        return self.file.filename

    #--------------------------------------------------------------------------------
    def start_values(self):                                            # check_blocks
    #--------------------------------------------------------------------------------
        return {'start':self.key['start'].decode(), 'values':self.out['start']}


    #--------------------------------------------------------------------------------
    def info(self):                                                    # check_blocks
    #--------------------------------------------------------------------------------
        txt = '   {} : {}'.format(self.key['start'].decode(), list2str(self.out['start']))
        if self._var:
            txt += ', {} : {}'.format(self._var, list2str(self.out[self._var]))
        return txt

    #--------------------------------------------------------------------------------
    def update_startpos(self, nblocks):                                # check_blocks
    #--------------------------------------------------------------------------------
        startpos = self.file.endpos
        # if > nblocks are read, set startpos to end of nblock timestep
        if len(self.out['start']) > nblocks: 
            startpos = self.out['startpos'][nblocks] 
        # update reader with startpos for next run 
        self.file.set_startpos(startpos)

        
    #--------------------------------------------------------------------------------
    def blocks_complete(self, nblocks=None):                           # check_blocks
    #--------------------------------------------------------------------------------
        self.reset_out()
        try:
            #for b in self.file.blocks(datalist=self.datalist):
            for b in self.file.blocks():
                if b._key == self.key['start']:
                    #b.print(data=True)
                    self.out['start'].append(b.data()[0])
                    self.out['startpos'].append(b.startpos)
                if b._key == self.key['end']:
                    #b.print()
                    self.out['end'] += 1 
                if self._var and b._key == var2key[self._var]:
                    self.out[self._var].append(b.get_value(self._var))
            #print(self.out['end'])
        except (ValueError, AttributeError) as e:
            return None
        except:  # Catch all other exceptions 
            raise
        try:
            if b._key == self.key['end'] and len(self.out['start']) >= nblocks and len(self.out['start']) == self.out['end']:
                self.update_startpos(nblocks)
                return True
        except UnboundLocalError:
            return None


    #--------------------------------------------------------------------------------
    def reset_out(self):                                               # check_blocks
    #--------------------------------------------------------------------------------
        self.out = {k:[] for k in self.out.keys()}
        self.out['end'] = 0


#====================================================================================
class fmt_block:                                                         # fmt_block
    #
    # formatted block of Eclipse data
    #
#====================================================================================
    #--------------------------------------------------------------------------------
    def __init__(self, keyword=None, length=None, datatype=None, data=zeros(0)): # fmt_block
    #--------------------------------------------------------------------------------
        self.keyword = keyword
        self.length = length
        self.datatype = datatype
        self.data = data
        self.max_length = max_length

    #--------------------------------------------------------------------------------
    def key(self):                                                        # fmt_block
    #--------------------------------------------------------------------------------
        return self.keyword.strip()
    
    #--------------------------------------------------------------------------------
    def set_key(self, keyword):                                           # fmt_block
    #--------------------------------------------------------------------------------
        self.keyword = keyword #.ljust(8)
    
    #--------------------------------------------------------------------------------
    def unformatted(self):                                                # fmt_block
    #--------------------------------------------------------------------------------
        dtype = self.datatype.encode()
        length = self.length
        bytes_ = bytearray()
        # header
        bytes_ += struct.pack(endian + 'i8si4si', 16, self.keyword.encode(), length, dtype, 16)
        # data is split in multiple records if length > 1000 
        data = self.data
        typesize = datasize[dtype]
        while data.size > 0:
            length = min(len(data), self.max_length)
            size = typesize*length
            bytes_ += struct.pack(endian + 'i{}{}i'.format(length, unpack_char[dtype]), size, *data[:length], size)
            data = data[length:]
        return bytes_
                
    #--------------------------------------------------------------------------------
    def print(self):                                                      # fmt_block
    #--------------------------------------------------------------------------------
        data = ''
        if len(self.data) > 0:
            data = 'data[0,-1] = ['+str(self.data[0])+', '+str(self.data[-1])+']'
        print(self.keyword, self.length, self.datatype, data)

        
#====================================================================================
class fmt_file:                                                            # fmt_file
    #
    # Class to handle formatted Eclipse files.
    #
    # Functions:
    #             blocks(warn_missing=False)
    #             convert()
    #             is_file()
    #
    #
#====================================================================================
    #--------------------------------------------------------------------------------
    def __init__(self, filename):                                          # fmt_file
    #--------------------------------------------------------------------------------
        self.name = Path(filename)
        self.fh = None

    #--------------------------------------------------------------------------------
    def is_file(self):                                                     # fmt_file
    #--------------------------------------------------------------------------------
        if self.name.is_file():
            return True
        else:
            print('File {} does not exist!'.format(self.name))
            return False
        
    #--------------------------------------------------------------------------------
    def blocks(self, warn_missing=False):                                  # fmt_file
    #--------------------------------------------------------------------------------
        if not self.is_file():
             return
        with open(self.name) as self.fh:
            for line in self.fh:
                try:
                    keyword, length, dtype = self.read_header(line)
                    data = self.read_data(length, dtype)
                except StopIteration:
                    if warn_missing:
                        print("\n  WARNING: Missing data in '{}' block, file {} not complete!".format(keyword,self.name.name))
                    return
                except TypeError:
                    return
                else:
                    yield fmt_block(keyword, length, dtype, data)
                           
                
    #--------------------------------------------------------------------------------
    def read_header(self, line):                                           # fmt_file
    #--------------------------------------------------------------------------------
        words = line.rstrip().split("\'")
        try:
            return words[1], int(words[2]), words[3]
        except IndexError:
            return
        
    #--------------------------------------------------------------------------------
    def read_data(self, length, dtype, skip=False):                        # fmt_file
    #--------------------------------------------------------------------------------
        if skip:
            data = None
        else:
            data = zeros(length, dtype=datatype[dtype])
        n = 0
        while n < length:
            line = next(self.fh)
            cols = line.rstrip().split()
            m = len(cols)
            if not skip:
                if dtype=='DOUB':
                    cols = [c.replace('D','E') for c in cols]
                try:
                    data[n:n+m] = cols
                except ValueError:
                    return data
            n += m
        return data
            

    #----------------------------------------------------------------------------
    def convert(self, ext='UNRST', init_key='SEQNUM', rename_duplicate=True,
                rename_key=None, echo=False, progress=lambda x:None, cancel=lambda:None):  # fmt_file
    #--------------------------------------------------------------------------------
        if rename_key and len(rename_key)<2:
            raise SystemError('ERROR in convert: Format of rename_keyword options is ' +
                              "('old name', 'new name'), but "+
                              '{} were given'.format(rename_key))
        stem = self.name.stem.upper()
        fname = str(self.name.parent/stem)+'.'+ext
        out_file = open(fname, 'wb')
        bytes_ = bytearray()
        n = 0
        count = {}
        for block in self.blocks():
            key = block.key()
            if key==init_key and len(bytes_)>0:
                # write previous block to file, and reset bytes_
                n += 1
                out_file.write(bytes_)
                # reset bytes for next section
                bytes_ = bytearray()
                progress(n)
                cancel()
                count = {}
            if rename_duplicate:
                if count.get(key):
                    # duplicate keyname, rename key
                    block.set_key(key[:-1]+str(count[key]+1))
                else:
                    # create new entry
                    count[key] = 0
                count[key] += 1
            if rename_key and key==rename_key[0]:
                block.set_key(rename_key[1])
            bytes_ += block.unformatted()
        out_file.close()
        if echo:
            print('{} converted to {}'.format(self.name.name,Path(fname)))
        return Path(fname)

    #----------------------------------------------------------------------------
    def get_blocks(self, filemap, init_key, rename_duplicate, rename_key): # fmt_file
    #----------------------------------------------------------------------------
        n = 0
        pos = {k:0 for k in datasize.keys()}
        size = {'blocks':0, 'bytes':0}
        num = {'chunks':0, 'blocks':0}
        Blocks = namedtuple('Blocks',['format', 'type', 'head', 'tail', 'slice', 'stride','size','num'])
        blocks = Blocks(format=[], type=[], head=[], tail=[], slice=[], stride=pos, size=size, num=num)
        head_format='i8si4si' # 4+8+4+4+4 = 24
        count = {}
        for match in finditer(b" \'(.{8})\'([0-9 ]{13})\'(.{4})\'", filemap):
            # header
            key, length, dtype = match.groups()
            key = key.decode().strip()
            length = int(length.decode())
            if key==init_key:
                n += 1
                if n>1:
                    size['bytes'] = sum(blocks.tail) + len(blocks.format)*24
                    size['blocks'] = match.start()
                    num['blocks'] = len(blocks.format)
                    num['chunks'] = len(blocks.type)
                    return blocks
            if rename_duplicate:
                #print(key)
                if count.get(key):
                    key = key[:-1]+str(count[key]+1) #).encode()
                    count[key] = 0
                else:
                    # create new entry
                    count[key] = 0
                count[key] += 1
            if rename_key and key==rename_key[0]:
                key = rename_key[1] #.encode()
                #print(key)
            head_data = [16, key.ljust(8).encode(), length, dtype, 16]
            # split block data in chunks if max_length
            L = [min(max_length, length-n*max_length) for n in range(int(length/max_length)+1)]
            L = [l for l in L if l>0]  # Remove possible 0's at the end
            blocks.format.append( head_format+''.join(['i'+str(l)+unpack_char[dtype]+'i' for l in L]) )
            for i,l in enumerate(L):
                blocks.type.append( dtype )
                head = [l*datasize[dtype],]
                if i==0:
                    head = head_data + head  
                blocks.head.append( head )
                blocks.tail.append( l*datasize[dtype] )
                blocks.slice.append( [pos[dtype]+sum(L[:i]), pos[dtype]+sum(L[:i])+l] )
            pos[dtype] += length


    #----------------------------------------------------------------------------
    def get_data_pos(self, filemap, size):                             # fmt_file
    #----------------------------------------------------------------------------
        data = filemap[:size].split()
        dtypes = [("'"+k+"'").encode() for k in datatype.keys()]
        data_pos = {k:[] for k in datasize.keys()}
        for i in range(len(data)):
            if data[i] in dtypes:
                dty = data[i][1:-1] # remove quotes
                data_pos[dty].append([i+1, i+1+int(data[i-1])])
        return data_pos, len(data)

    #----------------------------------------------------------------------------
    def prepare_helper_arrays(self, blocks, nblocks):                  # fmt_file
    #----------------------------------------------------------------------------
        block_slices = nparray(blocks.slice)
        slices = nparray(block_slices)
        stride = nparray( [[blocks.stride[blocks.type[i]],] for i in range(blocks.num['chunks'])] )
        heads = deepcopy(blocks.head)
        tails = deepcopy(blocks.tail)
        types = deepcopy(blocks.type)
        for n in range(1,nblocks):
            slices = npappend(slices, n*stride+block_slices, axis=0)
            heads += blocks.head
            tails += blocks.tail
            types += blocks.type
        return heads, slices, tails, types

    #----------------------------------------------------------------------------
    def fast_convert(self, nblocks=1, ext='.UNRST', init_key='SEQNUM', rename_duplicate=True,
                rename_key=None, echo=False, progress=lambda x:None, cancel=lambda:None):  # fmt_file 
    #--------------------------------------------------------------------------------
        outfile = self.name.with_suffix(ext)
        with open(self.name) as f:
            with mmap(f.fileno(), length=0, offset=0, access=ACCESS_READ) as filemap:
                # prepare 
                blocks = self.get_blocks(filemap, init_key, rename_duplicate, rename_key)
                unit_format = ''.join(blocks.format) 
                data_pos, pos_stride = self.get_data_pos(filemap, blocks.size['blocks'])
                N = int(len(filemap)/blocks.size['blocks'])
                progress(-N)
                heads, slices, tails, types = self.prepare_helper_arrays(blocks, nblocks)
                # process file
                with open(outfile, 'wb') as out:
                    a = 0
                    end = len(filemap)
                    finished = False
                    n = 0
                    while not finished:
                        # convert from string to array datatype
                        b = a + nblocks*blocks.size['blocks']
                        if b>end:
                            b = end
                            nblocks = int((b-a)/blocks.size['blocks'])
                            finished = True
                        data = filemap[a:b].split()
                        a = b
                        buffer = self.string_to_num(nblocks, blocks, data_pos, data, pos_stride)
                        data_chunks = ((*heads[i], *buffer[types[i]][slices[i][0]:slices[i][1]], tails[i]) for i in range(nblocks*blocks.num['chunks']))
                        out.write(struct.pack(endian+nblocks*unit_format, *[x for y in data_chunks for x in y]))
                        n += nblocks
                        progress(n)
                        cancel()
        return outfile

    #----------------------------------------------------------------------------
    def string_to_num(self, nblocks, blocks, data_pos, data, pos_stride): # fmt_file
    #----------------------------------------------------------------------------
        buffer = {}
        # Loop over all datatypes (INTE, REAL, DOUB, etc.)
        for dtyp in blocks.stride.keys():
            dtype=datatype[dtyp.decode()]
            buf = []
            for i,j in data_pos[dtyp]:
                for nb in range(nblocks):
                    buf.append(data[i+nb*pos_stride:j+nb*pos_stride])
                    #print(f'{dtyp}: data[{i}]={data[i]}, data[{j}]={data[j-1]}')
            try: 
                buffer[dtyp] = nparray([x for y in buf for x in y], dtype=dtype)
            except ValueError:
                buffer[dtyp] = nparray([x.decode().replace('D','E') for y in buf for x in y], dtype=dtype)
            #try: buffer[dtyp] = nparray([x for y in self.get_datalist(data_pos, nblocks, data, pos_stride, dtyp) for x in y], dtype=dtype)
            #except ValueError: buffer[dtyp] = nparray([x.decode().replace('D','E') for y in self.get_datalist(data_pos, nblocks, data, pos_stride, dtyp) for x in y], dtype=dtype)
        return buffer


            
# #@njit
# #----------------------------------------------------------------------------
# def numba_convert_buffer(buffer, dtyp):
# #----------------------------------------------------------------------------
#     for i in range(len(buffer)):
#         buffer[i] = numba_str_to_num(buffer[i], dtyp)
#     return buffer
#     # if dtyp=='INTE':
#     #     return nparray(buffer, dtype=int)
#     # if dtyp=='REAL':
#     #     return nparray(buffer, dtype=float)
#     # if dtyp=='DOUB':
#     #     return nparray(buffer, dtype=float)
#     # if dtyp=='LOGI':
#     #     return nparray(buffer, dtype=bool)

# #@njit
# #----------------------------------------------------------------------------
# def numba_str_to_num(string, dtyp):
# #----------------------------------------------------------------------------
#     #print(string)
#     if dtyp=='INTE':
#         return int(string)
#     if dtyp=='REAL':
#         return float(string)
#     if dtyp=='DOUB':
#         return float(string[:15]+'E'+string[17:])
#     if dtyp=='LOGI':
#         return string=='T'

    
#====================================================================================
class RSM_block:                                                          # RSM_block
#====================================================================================
    #--------------------------------------------------------------------------------
    def __init__(self, var, unit, well, data):                            # RSM_block
    #--------------------------------------------------------------------------------
        self.var = var
        self.unit = unit
        self.well = well
        self.data = data
        self.nrow = len(self.data)
        
    #--------------------------------------------------------------------------------
    def get_data(self):                                                   # RSM_block
    #--------------------------------------------------------------------------------
        for col,(v,u,w) in enumerate(zip(self.var, self.unit, self.well)):
            yield (v, u, w, [self.data[row][col] for row in range(self.nrow)])
        
        
#====================================================================================
class RSM_file:                                                            # RSM_file
#====================================================================================
    #--------------------------------------------------------------------------------
    def __init__(self, filename):
    #--------------------------------------------------------------------------------
        self.name = Path(filename)
        self.fh = None
        self.tag = '1'
        self.nrow = self.block_length()-10
        self.colpos = None
        
    #--------------------------------------------------------------------------------
    def get_data(self):                                                    # RSM_file
    #--------------------------------------------------------------------------------
        if not self.name.is_file():
            return ()
        with open(self.name) as self.fh:
            for line in self.fh:
                # line is now at the tag-line
                for block in self.read_block():
                    for data in block.get_data():
                        yield data
                            
    #--------------------------------------------------------------------------------
    def read_block(self):                                                  # RSM_file
    #--------------------------------------------------------------------------------
        self.skip_lines(3)
        var, unit, well = self.read_var_unit_well()
        self.skip_lines(2)
        data = self.read_data(ncol=len(var))
        yield RSM_block(var, unit, well, data)
                
    #--------------------------------------------------------------------------------
    def skip_lines(self, n):                                               # RSM_file
    #--------------------------------------------------------------------------------
        for i in range(n): 
            next(self.fh) 
        
    #--------------------------------------------------------------------------------
    def read_data(self, ncol=None):                                        # RSM_file
    #--------------------------------------------------------------------------------
        data = [None]*self.nrow        
        for l in range(self.nrow):
            line = next(self.fh)
            cols = line.rstrip().split()
            if len(cols)<ncol:
                # missing column entries
                cols = self.get_columns_by_position(line=line)
            data[l] = [float_or_str(c) for c in cols]
        return data

    
    #--------------------------------------------------------------------------------
    def get_columns_by_position(self, line=None):                          # RSM_file
    #--------------------------------------------------------------------------------
        '''
        Return None if column is empty
        '''
        n = len(self.colpos)
        words = [None]*n
        for i in range(n-1):
            a, b = self.colpos[i], self.colpos[i+1]
            words[i] = line[a:b].strip() or None
        return words

    
    #--------------------------------------------------------------------------------
    def read_var_unit_well(self):                                          # RSM_file
    #--------------------------------------------------------------------------------
        line = next(self.fh)
        var = line.split()
        start = 0
        self.colpos = []
        for v in var:
            i = line.index(v,start)
            self.colpos.append(i)
            start = i+len(var)
        self.colpos.append(len(line))
        unit = self.get_columns_by_position(line=next(self.fh))
        well = self.get_columns_by_position(line=next(self.fh))
        return var, unit, well

    #--------------------------------------------------------------------------------
    def block_length(self):                                                # RSM_file
    #--------------------------------------------------------------------------------
        with open(self.name) as fh: 
            nb, n = 0, 0 
            for line in fh: 
                n += 1 
                if line[0]==self.tag: 
                    nb += 1 
                    if nb==2: 
                        return int(n) 


