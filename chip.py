import pandas as pd
import numpy as np
import os
import re
import analysispy.filecontrol

class CLayout:
    def __init__(self,fn):
        self.fn=fn
        self.areas=[]
        self.x_sizes=[]
        self.y_sizes=[]
        self.parameters=[]
        self.datafile=""
        self.df=None
        self.df_columns=['ai','xi','yi','x','y']
    
    def set_file(self,fn):
        self.fn=fn

    def set_parameters(self, params):
        self.parameters=params
        self.df_columns.extend(params)

    def add_area(self, ai, sx, sy):
        self.areas.append(ai)
        self.x_sizes.append(sx)
        self.y_sizes.append(sy)

    def initialize_dataframe(self):
        self.df=pd.DataFrame(columns=self.df_columns)
        arr_sizes=[self.y_sizes[i]*self.x_sizes[i] for i in range(0,len(self.areas))]
        tot_size=1
        for asi in arr_sizes:
            tot_size=tot_size*asi
        arr_a=np.zeros(tot_size,dtype=np.int64)
        arr_x=np.zeros(tot_size,dtype=np.int64)
        arr_y=np.zeros(tot_size,dtype=np.int64)
        k=0
        for ii,ai in enumerate(self.areas):
            for xi in range(0,self.x_sizes[ii]):
                for yi in range(0,self.y_sizes[ii]):
                    arr_a[k]=ai
                    arr_x[k]=xi
                    arr_y[k]=yi
                    k=k+1
        self.df['ai']=arr_a
        self.df['xi']=arr_x
        self.df['yi']=arr_y

    def set_device_position(self,ai,xi,yi,x,y):
        if self.df is None:
            self.initialize_dataframe()
        row=self.df[ (self.df['ai']==ai) & (self.df['xi']==xi) & (self.df['yi']==yi) ]
        row['x']=x
        row['y']=y

    def set_area_positions(self,ai,origin,step):
        aii=-1
        for i in range(0,len(self.areas)):
            if ai==self.areas[i]:
                aii=i
        if aii<0:
            return False
        sx=self.x_sizes[aii]
        sy=self.y_sizes[aii]
        x_arr=np.zeros(sx*sy)
        y_arr=np.zeros(sx*sy)
        k=0
        for xi in range(0,sx):
            for yi in range(0,sy):
                x_arr[k]=origin[0]+step[0]*xi
                y_arr[k]=origin[1]+step[1]*yi
                k=k+1
                print("({},{}) -> ({},{})".format(xi,yi,x_arr[k],y_arr[k]))
        rows=self.df[self.df['ai']==ai]
        rows['x']=x_arr
        rows['y']=y_arr
        return True

    def read(self):
        fullpath=os.path.abspath(self.fn)
        splpath=analysispy.filecontrol.split_path(fullpath)
        if not os.path.exists(fullpath):
            return False
        fp=open(fullpath,"r")
        data={}
        for line in fp:
            tmp=line.replace("\n","").split("\t")
            data[tmp[0]]=tmp[1]
        self.areas=[int(i) for i in data['Area.Items'].split(',')]
        self.x_sizes=[int(i) for i in [data["Size.A{}.X".format(ai)] for ai in self.areas]]
        self.y_sizes=[int(i) for i in [data["Size.A{}.Y".format(ai)] for ai in self.areas]]
        self.parameters=data['ParameterNames'].split(',')
        self.datafile=analysispy.filecontrol.build_path(splpath['base'],data['Layout.File'])
        tsv=np.loadtxt(self.datafile)
        cols=['ai','xi','yi','x','y']
        cols.extend(self.parameters)
        print(cols)
        self.df=pd.DataFrame(data=tsv,columns=cols)
        self.df['ai']=self.df['ai'].astype(np.int64)
        self.df['xi']=self.df['xi'].astype(np.int64)
        self.df['yi']=self.df['yi'].astype(np.int64)
        return True

    def write(self):
        fp=open("")#not implemented
    
    def get_device_position(self,ai,xi,yi):
        dev=self.df[ (self.df['ai']==ai) & (self.df['xi']==xi) & (self.df['yi']==yi) ]
        return [dev['x'].values[0],dev['y'].values[0]]

server_address={"windows":"W:/", "linux":"/home/xxx"}
database_folder="path-to-database-folder"

class CLayouts:
    def __init__(self,type="windows"):
        self.ddir=server_address[type]+database_folder
        self.open_ids=[]
        self.open_cache=[]

    def open(self,lid):
        if lid in self.open_ids:
            ii=self.open_ids.index(lid)
            return self.open_cache[ii]
        else:
            ii=len(self.open_ids)
            cl=CLayout(self.ddir+"{}.chip".format(lid))
            if not cl.read():
                return None
            self.open_ids.append(lid)
            self.open_cache.append(cl)
            return cl
    
    def get_device_position(self,li,ai,xi,yi):
        cl=self.open(li)
        if cl is None:
            return None
        return cl.get_device_position(ai,xi,yi)

    def get_device_position_by_file_name(self,fn):
        idx=parse_scan_file(fn)
        return self.get_device_position(idx[0],idx[1],idx[2],idx[3])


def parse_data_file_path(fp):
    patterns={r"a(\d+)x(\d+)y(\d+)",r"(\d+)-(\d+)-(\d+)"}
    pp=analysispy.filecontrol.split_path(fp)
    file_name=pp['file']
    pp2=analysispy.filecontrol.split_path(pp['base'])
    working_dir=pp2['file']
    ai=-1
    xi=-1
    yi=-1
    for stri in [working_dir, file_name]:
        for pati in patterns:
            m=re.search(pati,stri)
            if not m:
                continue
            ai=int(m.group(1))
            xi=int(m.group(2))
            yi=int(m.group(3))
    if ai<0 and xi<0 and yi<0:
        return None
    return [ai,xi,yi]

def parse_scan_file(fp):
    pp=analysispy.filecontrol.split_path(fp)
    file_name=pp['file'].replace("  "," ").split(" ")
    id_str=file_name[-2]
    m=re.search(r"L(\d+)A(\d+)X(\d+)Y(\d+)",id_str)
    if not m:
        return None
    li=int(m.group(1))
    ai=int(m.group(2))
    xi=int(m.group(3))
    yi=int(m.group(4))
    return [li,ai,xi,yi]
