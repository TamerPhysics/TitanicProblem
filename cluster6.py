
# v2: look at statistic
# v4 merge dead and surv clusters simultaneously
# v5 pick next 2 pairs to merge to maximize the statistic
# v6 -visualize changes
#    -new way to populate the clus and clud dictionaries

import pandas as pd
import numpy as np
import matplotlib.pyplot as pl
from sklearn.cluster import DBSCAN

import fisher_projection

import pdb

def cluster(origsurv, sdist, origdead, ddist, pname, maxfrac, minsig) :

    # use existing keyword for the new method
    eps = maxfrac
    min_samples = minsig
    
    surv = origsurv.loc[:,pname]
    origsurv = origsurv.loc[:,pname]
    sdist= pd.DataFrame(sdist)
    clus= {} # {idx:[idx] for idx in surv.index}
    #exclus={}
    
    dead = origdead.loc[:,pname]
    origdead = origdead.loc[:,pname]
    ddist= pd.DataFrame(ddist)
    clud= {} # {idx:[idx] for idx in dead.index}
    #exclud={}

    # Last index of members, before clustering
    slastindex = sdist.index.max()
    dlastindex = ddist.index.max()
    
    
    
    
    
    # symmetrize sdist by removing NaN's
    for ii in range(sdist.shape[0]-1) :
        for jj in range(ii+1, sdist.shape[0]) :
            sdist.iloc[jj,ii] = sdist.iloc[ii,jj]
    for ii in range(sdist.shape[0]) :
        sdist.iloc[ii,ii] = 0
    
    # compute cluster
    dbclus = DBSCAN(metric='precomputed', 
                    eps=eps, min_samples=min_samples, 
                    n_jobs=3).fit(sdist)
    slbl= pd.Series( dbclus.labels_, index=surv.index)
    
    # save cluster in clus dictionary
    for ii in slbl.index :
        if   slbl[ii] == -1 :
            clus[ii] = [ii]
        elif slbl[ii] != -1 and slastindex+1+slbl[ii] in clus :
            clus[slastindex+1+slbl[ii]].append(ii)
        else : #elif slbl[ii] != -1 : 
            clus[slastindex+1+slbl[ii]] = [ii]


    # remove passengers that are now in a cluster from dist
    # matrix
    sincluster = slbl.where(slbl != -1).dropna().index
    sdist.drop( sincluster, axis=0, inplace=True)
    sdist.drop( sincluster, axis=1, inplace=True)    

    # update distance matrix and surv
    for label in set(dbclus.labels_)-{-1} :
    
        newindex = slastindex+1+label
        
#        surv.loc[newindex,:] = \
#         surv.loc[ surv.index[np.where(dbclus.labels_==iclu)] ,:].mean()

        surv.loc[newindex,:] = surv.loc[ 
                slbl.where(slbl == label).dropna().index ,:].mean()

        sdist.loc[:,newindex] = \
          ( (surv-surv.loc[newindex,:])**2. 
               ).sum(axis=1)**0.5
     
        sdist.loc[newindex,:] = np.NaN
    
    # remove passengers that are now in a cluster
    surv.drop(sincluster, axis=0, inplace=True)








    # symmetrize ddist by removing NaN's
    for ii in range(ddist.shape[0]-1) :
        for jj in range(ii+1, ddist.shape[0]) :
            ddist.iloc[jj,ii] = ddist.iloc[ii,jj]
    for ii in range(ddist.shape[0]) :
        ddist.iloc[ii,ii] = 0
    
    # compute cluster
    dbclud = DBSCAN(metric='precomputed',
                    eps=eps, min_samples=min_samples, 
                    n_jobs=3).fit(ddist)
    dlbl= pd.Series( dbclud.labels_, index=dead.index)
    
    # save cluster in clus dictionary
    for ii in dlbl.index :
        if   dlbl[ii] == -1 :
            clud[ii] = [ii]
        elif dlbl[ii] != -1 and dlastindex+1+dlbl[ii] in clud :
            clud[dlastindex+1+dlbl[ii]].append(ii)
        else : #elif dlbl[ii] != -1 : 
            clud[dlastindex+1+dlbl[ii]] = [ii]    

    # remove passengers that are now in a cluster from dist
    # matrix
    dincluster = dlbl.where(dlbl != -1).dropna().index
    ddist.drop( dincluster, axis=0, inplace=True)
    ddist.drop( dincluster, axis=1, inplace=True)    
        
    # update distance matrix and dead
    for label in set(dbclud.labels_)-{-1} :
    
        newindex = dlastindex+1+label
        
        dead.loc[newindex,:] = dead.loc[ 
                dlbl.where(dlbl == label).dropna().index ,:].mean()
        
        ddist.loc[:,newindex] = \
          ( (dead-dead.loc[newindex,:])**2. 
               ).sum(axis=1)**0.5
     
        ddist.loc[newindex,:] = np.NaN
    
    # remove passengers that are now in a cluster
    dead.drop(dincluster, axis=0, inplace=True)







    ssets = [ origsurv.loc[clus[kk], pname] for kk in clus.keys() ]
    dsets = [ origdead.loc[clud[kk], pname] for kk in clud.keys() ]

    # find the best projection on 3d
    try :
        yy, evect, ld, ax = fisher_projection.sepnd(ssets+dsets, pname, 'o'*len(ssets)+'v'*len(dsets), 3, '')

        ax.set_title('DBSCAN clustering with eps={0:3.3} and min_samples={1:d}'.format(float(eps), min_samples))
    
    except np.linalg.LinAlgError :
        print()
        print('Fisher matrix computation gave a singular matrix')
        print()
        evect=None
        ax=None


    return clus, sdist, surv, clud, ddist, dead, None, None, None, None, ax, evect

    
    
    
    
    
    








