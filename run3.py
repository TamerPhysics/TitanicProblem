
import pickle
import importlib
import copy

import numpy as np

import fisher_projection

try : 
    dir(tdata)
    del tdata
except NameError : pass


import titanic_class
import cluster6    
importlib.reload(titanic_class)
importlib.reload(cluster6)





with open('titanic_data_distmatrix_addtix.pyd', 'rb') as ff :
    tdata = pickle.load(ff)
    
    

veps = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
vmins= [1,2,3]

vt=[]
pars=[]

for ms in vmins :
    for eps in veps :
        
        vt.append( copy.deepcopy(tdata) )
        pars.append((eps,ms))
        
        vt[-1].cluster(eps, ms)
        vt[-1].compclu()
        

for t,pp in zip(vt, pars) : 
    print(pp)
    vv = t.values.ravel()
    vv = vv[np.where(np.isfinite(vv))]
    print(vv.means())
    print(vv.std())
    print()
