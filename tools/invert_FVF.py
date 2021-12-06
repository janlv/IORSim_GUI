#!/usr/bin/env python

from pathlib import Path
from datetime import datetime
from struct import pack, unpack, error as struct_error
from mmap import mmap, ACCESS_WRITE, ACCESS_READ
from numpy import ceil

endian = '>' # big-endian

datasize = {b'INTE' : 4,
            b'REAL' : 4,
            b'LOGI' : 4,
            b'DOUB' : 8,
            b'CHAR' : 8,
            b'MESS' : 1}

unpack_char = {b'INTE' : 'i',
               b'REAL' : 'f',
               b'LOGI' : 'i',
               b'DOUB' : 'd',
               b'CHAR' : 's',
               b'MESS' : ' '}

#====================================================================================
class unfmt_block:
#====================================================================================
    #--------------------------------------------------------------------------------
    def __init__(self, key=b'', length=0, type=b'', start=0, end=0, data=None, data_start=0, file=None):
    #--------------------------------------------------------------------------------
        self._key = key
        self.length = length
        self.type = type
        self.mmap = data
        self.file = file
        self.startpos = start
        self._end = end
        self.data_start = data_start

    #--------------------------------------------------------------------------------
    def info(self, details=False):                                      # unfmt_block
    #--------------------------------------------------------------------------------
        s = f'{self._key.decode()}'
        if details:
            s += f' block of {self.length*datasize[self.type]} bytes' 
            s += f' holding {self.length*(self.type==b"CHAR" and 8 or 1)} {self.type.decode()}'
            s += f' at [{self.startpos}, {self._end}]'
        return s

    #--------------------------------------------------------------------------------
    def print(self, data=None, details=False):                          # unfmt_block
    #--------------------------------------------------------------------------------
        print(self.info(details=details), end='')
        if data is not None:
            print(':',self.data()[data])
        else:
            print()

    #--------------------------------------------------------------------------------
    def read_size_at(self, pos):                                        # unfmt_block
    #--------------------------------------------------------------------------------
        size = unpack(endian+'i',self.mmap[pos:pos+4])[0]
        if self.type == b'CHAR':
            n = size
        else:
            n = int(size/datasize[self.type])
        return size, n 

    #--------------------------------------------------------------------------------
    def pack_format(self, n):                                           # unfmt_block
    #--------------------------------------------------------------------------------
        return endian+f'{n}{unpack_char[self.type]}'

    #--------------------------------------------------------------------------------
    def replace(self, func, key=None):                                  # unfmt_block
    #--------------------------------------------------------------------------------
        with open(self.file, mode='r+') as f:
            with mmap(f.fileno(), length=0, access=ACCESS_WRITE) as mmap_write:
                pos = self.startpos + 4
                if key is None:
                    key = self.key()
                mmap_write[pos:pos+8] = f'{key:8s}'.encode()
                newdata = func(self.data())
                pos = self.startpos + 24  # header: 4+8+4+4+4 = 24, data: 4 + 1000 data + 4
                N = 0
                while N < len(newdata):
                    size, n = self.read_size_at(pos)
                    pos += 4
                    mmap_write[pos:pos+size] = pack(self.pack_format(n), *newdata[N:N+n])
                    pos += size + 4
                    N += n
                mmap_write.flush()

    #--------------------------------------------------------------------------------
    def data(self):                                                     # unfmt_block
    #--------------------------------------------------------------------------------
        value = []
        a = self.data_start
        while a < self._end:
            size, n = self.read_size_at(a)
            a += 4
            try:
                value.extend(unpack(self.pack_format(n),self.mmap[a:a+size]))
            except struct_error as e:
                raise SystemError(f'ERROR Unable to read {self.file.name}, corrupted file?')
            a += size + 4
        if self.type == b'CHAR':
            value = [b''.join(value).decode()]
        return value

    #--------------------------------------------------------------------------------
    def key(self):                                                      # unfmt_block
    #--------------------------------------------------------------------------------
        return self._key.decode().strip()
        

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
    def blocks(self, only_new=False):                                    # unfmt_file
    #--------------------------------------------------------------------------------
        if not Path(self._filename).is_file():
            return
        if only_new:
            startpos = self.endpos
        else:
            startpos = self.startpos
        with open(self._filename, mode='r') as file:
            with mmap(file.fileno(), length=0, access=ACCESS_READ) as data:
                data.seek(startpos, 1)
                while data.tell() < data.size():
                    start = data.tell()
                    # Header
                    data.seek(4, 1)
                    key, length, type = unpack(endian+'8si4s', data.read(16))
                    data.seek(4, 1)
                    # Value array
                    data_start = data.tell()
                    max_length = 1000
                    if type==b'CHAR':
                        max_length = 105
                    bytes = length*datasize[type] + 8*int(ceil(length/max_length))
                    data.seek(bytes, 1)
                    yield unfmt_block(key=key, length=length, type=type, start=start, end=data.tell(), 
                                    data=data, data_start=data_start, file=self._filename)
                self.endpos = data.tell()


#--------------------------------------------------------------------------------
def convert(file=None, fra=None, til=None, func=None):
#--------------------------------------------------------------------------------
    print(f'  Converting {fra} to {til}')
    start_time = datetime.now()
    n = 0
    for block in unfmt_file(file).blocks():
        if block.key() in fra:
            i = fra.index(block.key())
            block.replace(func, key=til[i])
            n += 1
    print(f'  Replaced {n} blocks in {(datetime.now()-start_time).total_seconds():.3f} seconds')
    print()




######################################################################################

if __name__ == '__main__':
    from argparse import ArgumentParser

    def invert(alist):
        return [1/a if a!=0 else 1 for a in alist]

    print()
    parser = ArgumentParser(description='Convert from 1/FVF* to FVF* (Formation Volume Factors) in Eclipse UNRST-files')
    parser.add_argument('file', help='Name of UNRST-file')
    parser.add_argument('-reverse', help='Reverse order: Convert from FVF* to 1/FVF*', action='store_true')
    parser.add_argument('-print', help='Print first value of data-array', action='store_true')
    args = vars(parser.parse_args())
    
    fra = ('1/FVFGAS','1/FVFOIL','1/FVFWAT')
    til = ('FVFGAS','FVFOIL','FVFWAT')
    
    if args['reverse']:
        til_copy = til
        til = fra
        fra = til_copy

    convert(file=args['file'], fra=fra, til=til, func=invert)

    if args['print']:
        for block in unfmt_file(args['file']).blocks():
            if block.key() in til:
                block.print(data=0)
