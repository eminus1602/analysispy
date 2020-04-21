import sys
import os
import re
import struct
#from conffile import *
import lcolormap
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.signal
import scipy.stats
import scipy.ndimage.filters
import scipy.ndimage
import lfilecontrol

#class CData:
class CBinaryData:
    def __init__(self):
        self.header = {}
        self.fn = ""

    def _parse_header(self,raw_header):
        pattern = "HEADER_START\\n([\\s\\S]+)HEADER_END\\n"
        m = re.search(pattern,raw_header,flags=re.MULTILINE)
        if m:
            lines = m.group(1).split("\n")
            for line in lines:
                pattern2 = "\\s*(\\S+)\\s+(\\S.*)$"
                m2 = re.search(pattern2,line)
                if m2:
                    self.header[m2.group(1)]=m2.group(2)
        
    def set_file(self,file_name):
        self.fn=file_name

    def read(self):
        if self.fn == "":
            return
        fd = open(self.fn,"rb")
        size_header = struct.unpack('i',fd.read(4)) #for 64bit
        size_data = struct.unpack('i',fd.read(4))
        print("header: {0} bytes\ndata: {1} bytes".format(size_header[0],size_data[0]))
        raw_header = fd.read(size_header[0]).decode('utf-8')
        print(raw_header)
        self._parse_header(raw_header)
        self.data = np.fromfile(fd,dtype='d',sep='')
        fd.close()

    def write(self):
        if self.fn == "":
            return
        fd = open(self.fn,"wb")
        raw_header = "HEADER_START\n"
        for k, v in self.header.items():
            raw_header = raw_header+"{0}\t{1}\n".format(k,v)
        raw_header=raw_header+"HEADER_END\n\0"
        size_header = len(raw_header)
        size_data = len(self.data)*self.data.itemsize
        print("Header size: {0} bytes\nData size: {1} bytes\nHeader:\n{2}\n".format(size_header,size_data,raw_header))
        fd.write(struct.pack("i",size_header))
        fd.write(struct.pack("i",size_data))
        fd.write(raw_header.encode('utf-8'))
        fd.write(self.data.tobytes(order="F"))
        fd.close()

class CProfile(CBinaryData):
    def __init__(self):
        super().__init__()
        self.ndims = 0
        self.resolution = 16
        self.sizes = [0, 0, 0]

    def read(self):
        super().read()
        self.ndims = int(self.header["Dimensions"])
        self.resolution = float(self.header["Resolution"])
        ss = self.header["Dimension_size"].split(",")
        for i in range(0, self.ndims):
            self.sizes[i] = int(ss[i])
        array_dims = (np.array(self.sizes)*self.resolution).astype(np.int64)
        if self.ndims == 2:
            self.data=self.data.reshape([array_dims[1], array_dims[0]],order="F")
        elif self.ndims == 3:
            self.data=self.data.reshape([array_dims[2], array_dims[1], array_dims[0]],order="F")

    def write(self):
        self.header['Dimensions']=self.ndims
        self.header['Resolution']=self.resolution
        self.header['Dimension_size']=''
        for si in self.sizes:
            if self.header['Dimension_size'] != '':
                self.header['Dimension_size']+=', '+str(si)
            else:
                self.header['Dimension_size']=str(si)
        super().write()
        
    def get_2D_slice(self,x=None,y=None,z=None):
        if self.ndims<=2:
            print("Error: The data is not 3D.")
            return None
        zcut = CProfile()
        zcut.header = self.header
        zcut.ndims = 2
        zcut.resolution = self.resolution
        if x is not None:
            zcut.data = self.data[:,:,int((x+self.sizes[0]/2)*self.resolution)]
            zcut.sizes = [self.sizes[1], self.sizes[2]]
        if y is not None:
            zcut.data = self.data[:,int((y+self.sizes[1]/2)*self.resolution),:]
            zcut.sizes = [self.sizes[0], self.sizes[2]]
        if z is not None:
            zcut.data = self.data[int((z+self.sizes[2]/2)*self.resolution),:,:]
            zcut.sizes = [self.sizes[0], self.sizes[1]]
        return zcut

    def to_CMatrix(self):
        if self.ndims != 2:
            return None
        xaxis = self.get_axis(0)
        yaxis = self.get_axis(1)
        lm = CMatrix(self.fn.replace('.bin',' Z.dat'))
        lm.set(xaxis,yaxis,self.data)
        return lm

    def get_axis(self,axis):
        return np.linspace(start=-self.sizes[axis]*0.5,stop=self.sizes[axis]*0.5,num=self.sizes[axis]*self.resolution)

    def replace_file_name(self,before,after):
        self.fn=self.fn.replace(before,after)

    def append_suffix(self,suffix,extension=".bin",delim="."):
        self.fn=self.fn.replace(extension,delim+suffix+extension)
    
    def plot_at_center(self):
        if self.ndims == 3:
            zdata = self.get_2D_slice(z=0)
            zdata.plot_at_center()
        elif self.ndims == 2:
            xaxis = self.get_axis(0)
            yaxis = self.get_axis(1)
            plt.pcolor(xaxis,yaxis,self.data,cmap=lcolormap.rwb2())
            plt.axis('equal')
            plt.show()
        else:
            print("Error: dimension should be 2 or 3")


class CMatrix:
    def __init__(self,fn,flag='r'):
        if fn==None:
            self.fz=None
            self.fx=None
            self.fy=None
            return
        self.fz=fn
        self.fx=fn.replace(' Z.dat',' X.dat')
        self.fy=fn.replace(' Z.dat',' Y.dat')

    #def setFileName(self,fn):
    def set_file_name(self,fn):
        self.fz=fn
        self.fx=fn.replace("Z.dat","X.dat")
        self.fy=fn.replace("Z.dat","Y.dat")
        
    def set(self,x,y,z):
        self.x=x
        self.y=y
        self.z=z
    
    def read(self):
        self.x=np.loadtxt(self.fx)
        self.y=np.loadtxt(self.fy)
        self.z=np.loadtxt(self.fz)
        return self

    def write(self,windows=False):
        if self.fz==None:
            return
        nlchar = '\n'
        if windows:
            nlchar='\r\n'
        np.savetxt(self.fz,self.z,delimiter='\t',newline=nlchar)
        np.savetxt(self.fx,self.x,newline='\t')
        np.savetxt(self.fy,self.y,delimiter=nlchar)

    def plot(self,cmap=lcolormap.brightjet()):
        plt.pcolor(self.x,self.y,self.z,cmap=cmap)
        plt.colorbar()

    #def getIndexByValue(self,axis,value,offset=0,interp=False):
    def get_index_by_value(self,axis,value,offset=0,interp=False):
        aarr=self.x
        if axis==1:
            aarr=self.y
        elif axis!=0:
            return None
        now=offset
        if aarr[now]==value:
            return 0
        f=(aarr[now]>=value)
        for now in range(offset,len(aarr)):
            #print(now)
            f_=(aarr[now]>=value)
            if f != f_:
                #crossing
                x0=now-1
                y0=aarr[x0]
                x1=now
                y1=aarr[x1]
                slope=y1-y0
                frac_index = (value-y0)/slope+x0
                if (not interp) and frac_index==int(frac_index):
                    return frac_index
                if interp:
                    return frac_index

        if aarr[len(aarr)-1]==value:
            return len(aarr)-1
        else:
            return None

    #def getValueByIndex(self,axis,frac_index):
    def get_value_by_index(self,axis,frac_index):
        aarr=self.x
        if axis==1:
            aarr=self.y
        elif axis!=0:
            return None
        if frac_index>=len(aarr):
            frac_index=len(aarr)-1

        i0=math.floor(frac_index)
        i1=math.ceil(frac_index)
        y0=aarr[i0]
        y1=aarr[i1]
        slope=y1-y0
        return y0+slope*(frac_index-i0)

    def cut(self,x=None,y=None,interp=False):
        if x==None and y==None:
            return None
        if x!=None and y!=None:
            return None
        if x!=None:
            axis=0
            value=x
            xt=self.y
            zt=transpose(self.z)
        if y!=None:
            axis=1
            value=y
            xt=self.x
            zt=self.z
        frac_index = self.get_index_by_value(axis,value,interp=interp)
        if frac_index==None:
            return None
        else:
            i=round(frac_index)
        yt=zt[int(i),:]
        s=CSpectrum("")
        s.set(xt.squeeze(),yt.squeeze())
        return s

    #def extractRangeByIndex(self,x0,sx,y0,sy):
    def cut_2D(self,x0,sx,y0,sy):
        x00=max(0,x0)
        y00=max(0,y0)
        x1=min(x0+sx,len(self.x)-1)
        y1=min(y0+sy,len(self.y)-1)
        x=self.x[x00:x1]
        y=self.y[y00:y1]
        z=self.z[y00:y1,x00:x1]
        r=CMatrix("")
        r.set(x,y,z)
        return r

    def stdev(self):
        return np.std(self.z.ravel())

    def median(self):
        return np.median(self.z.ravel())

    def mean(self):
        return np.mean(self.z.ravel())

    def zscore(self,mean=None,stdev=None):
        if mean!=None and stdev!=None:
            zz=(self.z-mean)/stdev
        else:
            zz=scipy.stats.zscore(self.z)
        zr=CMatrix("")
        zr.set(self.x,self.y,zz)
        return zr
    
    def medfilt(self,size):
        zmed=scipy.signal.medfilt(self.z,size)
        out=CMatrix(None)
        out.set(self.x,self.y,zmed)
        return out
    
    #def verticalMax(self):
    def vertical_max(self):
        xaxis=self.x
        yaxis=np.amax(self.z,axis=0)
        spectrum=CSpectrum("")
        spectrum.set(xaxis,yaxis)
        return spectrum
    
    #def verticalMean(self):  
    def vertical_mean(self):
        xaxis=self.x
        yaxis=np.mean(self.z,axis=0)
        spectrum=CSpectrum("")
        spectrum.set(xaxis,yaxis)
        return spectrum

    def label(self,threshold,dilation_iter=5,remove_cosmic_ray=True):
        binimage=self.z>threshold
        eroded=binimage
        if remove_cosmic_ray:
            #to remove cosmic ray - erosion in y axis
            eroded=scipy.ndimage.binary_erosion(binimage,structure=np.array([[0,1,0],[0,1,0],[0,1,0]]))
        dilated=scipy.ndimage.binary_dilation(eroded,iterations=dilation_iter)
        lb=scipy.ndimage.label(dilated)
        return lb

    def max(self,mask=None,by_index=False):
        i=0
        j=0
        if mask is None:
            i,j=np.unravel_index(np.argmax(self.z),self.z.shape)
        else:
            if not np.any(mask):
                print("All elemets are false in mask")
                return None
            ztmp=self.z.copy()
            ztmp[np.logical_not(mask)]=np.amin(ztmp)
            i,j=np.unravel_index(np.argmax(ztmp),self.z.shape)
        max_intensity = self.z[i,j]
        max_x = self.x[j]
        max_y = self.y[i]
        if by_index:
            return (j,i,max_intensity)
        return (max_x, max_y, max_intensity)

    def list_peaks(self,threshold,by_index=False):
        lb,lbn=self.label(threshold)
        peaks=[]
        for i in range(1,lbn+1):
            peaks.append(self.max(mask=(lb==i),by_index=by_index))
        return peaks

class CSpectrum:
    # Initialize object. fn: file name
    def __init__(self,fn):
        self.fn=fn
        self.x=np.array([])
        self.y=np.array([])

        #set x and y
    def set(self,x,y):
        self.x=x
        self.y=y
        return self
    
    #read dat file
    def read(self,skip_header='auto'):
        if skip_header=='auto':
            fp=open(self.fn)
            lines=fp.read().replace("\r","").split("\n")
            x=[]
            y=[]
            r=re.compile(r"(\S+)\s+(\S+)")
            r2=re.compile(r"^(?:(?:[+-]?\d*\.\d+)|(?:[+-]?\d+))(?:[Ee][+-]?\d+)?$")
            for line in lines:
                m=r.match(line)
                if m:
                    if r2.match(m[1]) and r2.match(m[2]):
                        x.append(float(m[1]))
                        y.append(float(m[2]))
            self.x=np.array(x)
            self.y=np.array(y)
        elif skip_header=='none':
            tmp = np.loadtxt(self.fn)
            self.x = tmp[:,0]
            self.y = tmp[:,1]
        else:
            print("Unknown configuration for 'skip_header'")
        return self

    def save(self,fn):
        np.savetxt(fn,np.transpose(np.array([self.x,self.y])),delimiter='\t')
    
    #plot (need to perform plt.show())
    def plot(self):
        plt.plot(self.x,self.y)
    
    #returns index where x=value
    #  -- if value is out of range, value is coerced to 0 to len(x)-1
    #  -- return value is interpolated
    def get_index_by_X_value(self, value):
        i=0
        flag=False
        for xv in self.x:
            if xv>value:
                flag=True
                break
            i=i+1
        if i==0:
            return 0
        if flag:
            x1=i
            x0=i-1
            y1=self.x[x1]
            y0=self.x[x0]
            return x0+(value-y0)/(y1-y0)
        else:
            return len(self.x)-1

    # returns part of a spectrum. range is specified by x start and end values
    def cut(self,x0,x1):
        xi0=int(self.get_index_by_X_value(x0))
        xi1=int(self.get_index_by_X_value(x1))
        s=CSpectrum("")
        s.set(self.x[xi0:xi1],self.y[xi0:xi1])
        return s

    # returns two spectra separated at x
    def split(self,x):
        x0=self.x[0]
        xi=self.get_index_by_X_value(x)
        x1=self.x[len(self.x)-1]
        s1=self.cut(x0,xi)
        s2=self.cut(xi,x1)
        return [s1, s2]

    # coerces y value 
    def coerce_min(self,value):
        out=CSpectrum("")
        x=self.x
        y=self.y
        for i in range(0,len(y)):
            y[i]=max(value,y[i])
        out.set(x,y)
        return out

    # differentiate spectrum
    def diff(self):
        out=CSpectrum("")
        x=np.zeros(len(self.x)-1)
        y=np.zeros(len(self.y)-1)
        for i in range(0,len(x)):
            x[i]=(self.x[i]+self.x[i+1])*0.5
            y[i]=(self.y[i+1]-self.y[i])/(self.x[i+1]-self.x[i])
        out.set(x,y)
        return out

    def get_peaks(self,dx=None,yth=None):
        if dx is not None:
            xstep = (self.x[-1]-self.x[0])/(len(self.x)-1)
            ndx = dx/xstep
            if ndx < 1:
                ndx=1
        else:
            ndx=None
        pks = scipy.signal.find_peaks(self.y, height=yth, distance=ndx)
        pks_x = [self.x[pki] for pki in pks[0]]
        pks_y = [self.y[pki] for pki in pks[0]]
        return (pks_x, pks_y)
    
    # set y to 0 if y<threshold
    def coerce_threshold_to_zero(self,threshold):
        out=CSpectrum("")
        x=self.x
        y=self.y
        for i in range(0,len(y)):
            if abs(y[i])<threshold:
                y[i]=0
            out.set(x,y)
        return out

    #label regions above threshold. label0 is a region below threshold
    #returns label spectrum and number of regions
    def label_regions(self,threshold):
        f = (self.y>threshold)
        regions = np.zeros(len(f))
        prev=0
        nregions=0
        for i in range(0,len(regions)):
            if not f[i]:
                prev=0
                continue
            if prev!=0:
                regions[i]=prev
                continue
            nregions=nregions+1
            prev=nregions
            regions[i]=nregions
        s=CSpectrum("")
        s.set(self.x,regions)
        return s,nregions

    def get_region_range(self,index):
        i0=-1
        i1=-1
        for i in range(0,len(self.x)):
            if self.y[i]<index:
                continue
            if self.y[i]==index and i0==-1:
                i0=i
                continue
            if self.y[i]>index and i1==-1:
                i1=i
                break
        if i1==-1:
            i1=len(self.x)-1
        return i0,i1
    
    # squares each pixel of y. If n other than 2 is given, n-th order power is calculated
    def power(self, n=2):
        s=CSpectrum("")
        x=self.x
        y=np.power(self.y,n)
        s.set(x,y)
        return s

    # log of each pixel is calculated.
    def log(self):
        s=CSpectrum("")
        x=self.x
        y=np.log(self.y)
        s.set(x,y)
        return s

class CDecayCurve(CSpectrum):
    def __init__(self,fn):
        self.fn=fn
        self.x=np.array([]) #x axis (i.e. time axis) should be evenly spaced!
        self.y=np.array([])
    
    def read(self,mode="dat"):
        if mode=="dat":
            tmp = np.loadtxt(self.fn)
            self.x = tmp[:,0]
            self.y = tmp[:,1]
        elif mode=="asc":
            #read BH ascii file, converted from SPCM
            xt=[]
            yt=[]
            f=open(self.fn)
            start=False
            for line in f:
                mark=(line[0]=='*')
                if mark and start:
                    break
                if not start and mark:
                    start=True
                if start and not mark:
                    #remove possible newline chars and split
                    cols=line.replace("\n","").replace("\r","").split(" ")
                    xt.append(float(cols[0]))
                    yt.append(float(cols[1]))
            f.close()
            #remove initial zeros
            xt=np.array(xt)
            yt=np.array(yt)
            x0=np.amin(np.where(yt>0))
            x1=np.amax(np.where(yt>0))-1
            self.x=np.array(xt[x0:x1])
            self.y=np.array(yt[x0:x1])
        if self.x[0]>self.x[1]:
            self.x=self.x[::-1]
            self.y=self.y[::-1]
        return self


    def baseline(self,t0=10,t1=110):
        #calculate average counts between 200:300-th px
        m=np.mean(self.y[t0:t1])
        s=np.std(self.y[t0:t1])
        return m,s
    
    def threshold(self):
        a,s=self.baseline()
        #16sigma
        return a+s*16

    def shift_time(self, t):
        self.x=self.x-t
    
    def get_t0(self):
        return self.x[np.argmax(self.y>self.threshold())]
    
    def generate_irf(self):
        print("IRF threshold: {}".format(self.threshold()))
        t_range=np.where(self.y>self.threshold())
        t0=np.amin(t_range)
        t1=np.amax(t_range)-1
        print(t0,t1)
        irf=CDecayCurve("")
        new_x=self.x[t0:t1]-self.x[t0]
        new_y=self.y[t0:t1]
        y0=np.amin(new_y)
        new_y=new_y-y0 #subtract baseline
        amp=np.sum(new_y)
        new_y=new_y/amp #normalize (divide by max)
        irf.set(new_x,new_y)
        return irf
    
    def cut(self,x0,x1):
        xi0=int(self.get_index_by_X_value(x0))
        xi1=int(self.get_index_by_X_value(x1))
        s=CDecayCurve("")
        s.set(self.x[xi0:xi1],self.y[xi0:xi1])
        return s

    def convolve(self,irf):
        new_y=np.convolve(self.y,irf.y,mode='same')
        #usually the real signal is longer than the irf
        new_x=self.x
        new_decay=CDecayCurve("")
        new_decay.set(new_x, new_y) #the absolute value of time axis does not mean anything in this measurement. (Only the time step per pixel matters)
        return new_decay
