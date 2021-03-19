from datetime import datetime
from mmap import mmap, ACCESS_READ, ACCESS_WRITE
from re import finditer
from pathlib import Path
from struct import unpack, pack
from IORlib.ECL import unfmt_file as ECL_unfmt_file, fmt_file as ECL_fmt_file
from numpy import array, int32, float32, float64
from pandas import DataFrame


max_length = 1000
endian = '>' # big-endian

unpack_char = {b'INTE' : 'i',
               b'REAL' : 'f',
               b'LOGI' : 'i',
               b'DOUB' : 'd',
               b'CHAR' : 's',
               b'MESS' : ' '}

datasize = {b'INTE' : 4,
            b'REAL' : 4,
            b'LOGI' : 4,
            b'DOUB' : 8,
            b'CHAR' : 8,
            b'MESS' : 1}

datatype = {b'INTE' : int32,
            b'REAL' : float32,
            b'LOGI' : bool,
            b'DOUB' : float64,
            b'CHAR' : str,
            b'MESS' : str}

var2key = {'nwell':b'INTEHEAD'}
var2pos = {'nwell':16}

#====================================================================================
class _datablock:                                               # datablock_mmap
#====================================================================================

    #--------------------------------------------------------------------------------
    def __init__(self, start, end, data):      
    #--------------------------------------------------------------------------------
        self._start = start
        self._end = end
        self._key, self._length, self._datatype = unpack(endian+'8si4s', data[start+4:start+20])
        if self._datatype in (b'CHAR',):
            self._length *= datasize[self._datatype]
        #self._data = data[start+24:end]  # extra 4 from int prefix at next keyword
        self._data = data #[start+24:end]  # extra 4 from int prefix at next keyword
        #print(self._key, self._length, self._datatype)
        #print(self._data)
        #self._unpack = None
        #size = len(self._data)
        #size = end-(start+24)
        # A data-block is split in chunks if the length exceeds 1000 elements
        # Each data-block is sandwiched by a 4 byte int giving the size of the block
        #if size > 0:
        #    sizes = [str(min(max_size, size-n*max_size)) for n in range(int(size/max_size)+1)]
        #    dtype = unpack_char[self._datatype]
        #    self._unpack = endian+''.join(['i'+s+dtype+'i' for s in sizes])
        #print(self._unpack)

    #--------------------------------------------------------------------------------
    def info(self):                                                       # datablock
    #--------------------------------------------------------------------------------
        string = '{:s} block of {:8d} bytes holding {:8d} {} at [{}, {}]'
        return string.format(self._key.decode(), len(self._data), self._length,
                             self._datatype.decode(), self._start, self._end)

    #--------------------------------------------------------------------------------
    def data(self):                                                       # datablock
    #--------------------------------------------------------------------------------
        size = self._end-(self._start+24)
        # A data-block is split in chunks if the length exceeds 1000 elements
        # Each data-block is sandwiched by a 4 byte int giving the size of the block
        if size > 0:
            sizes = [str(min(max_size, size-n*max_size)) for n in range(int(size/max_size)+1)]
            dtype = unpack_char[self._datatype]
            return unpack(endian+''.join(['i'+s+dtype+'i' for s in sizes]), self._data[self._start+24:self._end])

    #--------------------------------------------------------------------------------
    def key(self):                                                       # datablock
    #--------------------------------------------------------------------------------
        return self._key.decode().strip()

    #--------------------------------------------------------------------------------
    def start(self):                                             # datablock
    #-------------------------------------------------------------------------------
        return self._start
        
    #--------------------------------------------------------------------------------
    def end(self):                                             # datablock
    #-------------------------------------------------------------------------------
        return self._end
    
    #--------------------------------------------------------------------------------
    def datatype(self):                                             # datablock
    #-------------------------------------------------------------------------------
        return self._datatype.decode()

    #--------------------------------------------------------------------------------
    def get_nwell(self):                                                  # datablock
    #--------------------------------------------------------------------------------
        return self.get_value('nwell')

    #--------------------------------------------------------------------------------
    def get_value(self, var):                                             # datablock
    #--------------------------------------------------------------------------------
        return self.data()[var2pos[var]]
        
    #--------------------------------------------------------------------------------
    def print(self):                                                      # datablock
    #--------------------------------------------------------------------------------
        print(self.info())
        

#====================================================================================
class unfmt_file:
#====================================================================================

    #--------------------------------------------------------------------------------
    def __init__(self, filename):                                            # reader
    #--------------------------------------------------------------------------------
        self.fileobj = None
        self._filename = Path(filename)
        self.endpos = self.startpos = 0


    #--------------------------------------------------------------------------------
    def blocks(self, only_new=False):   # reader
    #--------------------------------------------------------------------------------
        if not Path(self._filename).is_file():
            return
        with open(self._filename, 'rb') as self.fileobj:
            if only_new:
                startpos = self.endpos
            else:
                startpos = self.startpos
            with mmap(self.fileobj.fileno(), length=0, offset=startpos, access=ACCESS_READ) as data:
                prev = None
                for match in finditer(b'[A-Za-z0-9_ ]{8}', data):
                    #print(match)
                    if prev:
                        yield _datablock(prev.start()-4, match.start()-4, data)
                    prev = match


#====================================================================================
class fmt_file:
#====================================================================================

    #--------------------------------------------------------------------------------
    def __init__(self, filename):                                            # reader
    #--------------------------------------------------------------------------------
        #self._fileobj = None
        self._filename = Path(filename)
        self._endpos = self._startpos = 0
        self.width = {b'INTE':12, b'LOGI':3,  b'REAL':17, b'DOUB':23} 
        self.ncol  = {b'INTE':6,  b'LOGI':20, b'REAL':4,  b'DOUB':4}
        self.cast  = {b'INTE':int,  b'LOGI':bool, b'REAL':float,  b'DOUB':self.cast_double}

    #--------------------------------------------------------------------------------
    def cast_double(self, string):   # reader
    #--------------------------------------------------------------------------------
        return float(string.replace('D','E'))

    #--------------------------------------------------------------------------------
    def cast_logi(self, string):   # reader
    #--------------------------------------------------------------------------------
        if string=='F':
            return False
        return True

    #--------------------------------------------------------------------------------
    def convert(self, offset=0):   # reader
    #--------------------------------------------------------------------------------
        if not Path(self._filename).is_file():
            return
        bytes = bytearray()
        outfile = open(Path(self._filename).parent/'test.UNRST', 'wb')
        #outmap = mmap(outfile.fileno(), access=ACCESS_WRITE)
        with open(self._filename) as file:
            with mmap(file.fileno(), length=0, offset=offset, access=ACCESS_READ) as filemap:
                #prev = None
                n = 0
                for match in finditer(b" \'([A-Za-z0-9_/ ]{8})\'([0-9 ]{13})\'([A-Z ]{4})\'", filemap):
                    #print(match)
                    # header
                    key, length, dtype = match.groups()
                    length = int(length.decode())
                    bytes += pack(endian + 'i8si4si', 16, key, length, dtype, 16)
                    # # data
                    start = match.end()+2
                    end = start+length*self.width[dtype]+2*int(length/self.ncol[dtype])
                    tmp = filemap[start:end].split()
                    if dtype==b'DOUB':
                    #    data = [self.cast[dtype](v.decode()) for v in filemap[start:end].split()]
                        data = [self.cast[dtype](v.decode()) for v in tmp]
                    else:
                    #    data = array(filemap[start:end].split(),dtype=datatype[dtype])
                    #    data = array(tmp,dtype=datatype[dtype])
                        data = DataFrame(tmp).astype(datatype[dtype])
                    #lengths = [min(max_length, length-n*max_length) for n in range(int(length/max_length)+1)]
                    #pack_format = ['i'+str(l)+unpack_char[dtype]+'i' for l in lengths]
                    #dsize = datasize[dtype]
                    #pack_bytes = ((len*dsize, *data[sum(lengths[:i]):sum(lengths[:i])+len], len*dsize) for i,len in enumerate(lengths))
                    #bytes += pack(endian+''.join(pack_format), *[x for y in pack_bytes for x in y])
                    if key.decode().strip()=='SEQNUM' and bytes:
                        outfile.write(bytes)
                        bytes = bytearray()
                        n += 1
                        print('\r   '+str(n), end='')    
                        if n>100:
                            break

        outfile.close()
        #outfile = Path(self._filename).parent/'test.UNRST'
        #with open(outfile, 'wb') as file:
        #    file.write(bytes)
        return outfile.name


path = 'GUI/cases/SNURRE1/'
funrst = 'SNURRE1_IORSim_PLOT.FUNRST'
unrst = 'SNURRE1_IORSim_PLOT.UNRST'


a = datetime.now()
file = fmt_file(path+funrst).convert()
print()
print(datetime.now()-a)

#a = datetime.now()
#file = ECL_fmt_file(path+funrst).convert(progress=lambda n: print('\r  '+str(n), end=''))
#print()
#print(datetime.now()-a)

# n = 0
# print(unrst)
# for block in ECL_unfmt_file(path+unrst).blocks():
#     block.print()
#     n += 1
#     if n>3:
#         break

# print(file)
# for block in ECL_unfmt_file(file).blocks():
#     block.print()
