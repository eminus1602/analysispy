from matplotlib.colors import LinearSegmentedColormap
import numpy as np

def __generate(colors,name):
    values = range(len(colors))
    vmax = np.ceil(np.max(values))
    clist = []
    for v,c in zip(values,colors):
        clist.append((v/vmax,c))
    return LinearSegmentedColormap.from_list(name,clist)

def rwb():
    return __generate(['red','white','blue'],'rwb')

def brightjet():
    return __generate(['#ffffff','#0000ff','#007fff','#00ffff','#7fff7f','#ffff00','#ff7f00','#ff0000','#7f0000','#000000'],'brightjet')

def rwb2():
    return __generate(['yellow','red','white','blue','cyan'],'rwb2')

def wry():
    return __generate(['white','red','yellow'],'wry')

def wbc():
    return __generate(['white','blue','cyan'],'wbc')

