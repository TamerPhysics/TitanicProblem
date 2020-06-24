
import pandas as pd
import numpy as np
import matplotlib.pyplot as pl

# Check that clusters are separate enough by comparing
# the distances between their centroids to their sigmas


def compareclus(clus, clud, minsig, pnorm, surv, dead) : 

    
    cludist = pd.DataFrame(index=clus.keys(), columns=clud.keys())
    
    
    means = pd.DataFrame(index = clus.keys(), columns=pnorm)
    meand = pd.DataFrame(index = clud.keys(), columns=pnorm)
    
    sigms = pd.DataFrame(index = clus.keys(), columns=pnorm)
    sigmd = pd.DataFrame(index = clud.keys(), columns=pnorm)

    for kk in clus.keys() : 
        means.loc[kk,pnorm] = surv.loc[clus[kk], pnorm].mean()
        sigms.loc[kk,pnorm] = surv.loc[clus[kk], pnorm].std()
        
        
        
    for kk in clud.keys() : 
        meand.loc[kk,pnorm] = dead.loc[clud[kk], pnorm].mean()
        sigmd.loc[kk,pnorm] = dead.loc[clud[kk], pnorm].std()
    
    
    for ii in cludist.index:
        for jj in cludist.columns:
            
            separatedclusters=0
            pname2 = list(pnorm)
            for pp in pnorm :
                
                # check if this pair of clusters ii and jj are separate.
                
                # If for each cluster the points are too close (sigma<minsig)
                # along this dimension pp
                # AND the 2 cluster are close, then ignore this dimension in
                # the computation of inter-cluster distance
                if sigms.loc[ii,pp] < minsig and \
                   sigmd.loc[jj,pp] < minsig and \
                   (means.loc[ii,pp]-meand.loc[jj,pp]) < minsig : 
                       pname2.remove(pp)
                
                # If for each cluster the points are too close (sigma<minsig)
                # along this dimension pp
                # AND clusters are far, then don't even compute distance
                # between this pair of clusters 
                # BECAUSE it may cause a large number in the distance matrix
                elif sigms.loc[ii,pp] < minsig and \
                     sigmd.loc[jj,pp] < minsig and \
                     (means.loc[ii,pp]-meand.loc[jj,pp]) > 5*minsig:
                       separatedclusters = separatedclusters+1
            
            if separatedclusters==0 and len(pname2) > 0 :
                cludist.loc[ii,jj] = (
                 ( (means.loc[ii,pname2]-meand.loc[jj,pname2])/
                   (sigms.loc[ii,pname2]+sigmd.loc[jj,pname2])  )**2. 
                                                                ).sum() / len(pname2)


            #elif separatedclusters>=1 : cludist.loc[ii,jj] = -1
            #else : pdb.set_trace()
            
    vv=  np.array( cludist.values.ravel(), dtype=np.float )
    pl.figure()
    pl.hist(np.log10(vv[np.where( (np.isfinite(vv)) & (vv!=0) )]), 20)
    
    return cludist, sigms, sigmd