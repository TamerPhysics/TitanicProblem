
# add multiprocessing


import pandas as pd
import numpy as np
import multiprocessing
import os

import pdb


def calcdist(surv, pname) :

    #os.system('taskset -p 0x0000000F '+str(os.getpid()) )
    
    nsurv = len(surv.index)

    arglist = []
        
    sidx = surv.index
    sdist = pd.DataFrame(np.ones([nsurv,nsurv])*np.nan)
    sdist.index   = sidx
    sdist.columns = sidx
    for ii in range(len(sidx)) :
        for jj in range(ii+1,len(sidx)) :
            arglist.append( [ ii, jj ] )

    pool = multiprocessing.Pool()
    
    
    kk=0
    results=[]
    for ii in range(len(sidx)) :
        for jj in range(ii+1,len(sidx)) :
            results.append( pool.apply_async(thisdist, args=(arglist[kk], surv, sidx, pname) ) )
            kk=kk+1

    pool.close()        
            
    print('Pool closed')
            
    for res in results : sdist.iloc[res.get()[1],res.get()[2]] = res.get()[0]
            
    
    return sdist


def thisdist(twoelm, surv, sidx, pname) :

    ii = twoelm[0]
    jj = twoelm[1]

    return ( ( surv.loc[sidx[ii],pname] - surv.loc[sidx[jj],pname] )**2. ).sum()**0.5, ii, jj




