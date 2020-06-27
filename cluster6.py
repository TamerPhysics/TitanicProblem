
# v2: look at statistic
# v4 merge dead and surv clusters simultaneously
# v5 pick next 2 pairs to merge to maximize the statistic
# v6 -visualize changes
#    -new way to populate the clus and clud dictionaries

import pandas as pd
import numpy as np
import matplotlib.pyplot as pl
#import time
#from matplotlib.widgets import Slider

import fisher_projection


def cluster(origsurv, sdist, origdead, ddist, pname, maxfrac, minsig) :

    surv = origsurv.loc[:,pname] #pd.DataFrame(origsurv)
    origsurv = origsurv.loc[:,pname]
    sdist= pd.DataFrame(sdist)
    clus= {idx:[idx] for idx in surv.index} #dict( zip(surv.index, surv.index) )
    exclus={}
    
    dead = origdead.loc[:,pname] #pd.DataFrame(origdead)
    origdead = origdead.loc[:,pname]
    ddist= pd.DataFrame(ddist)
    clud= {idx:[idx] for idx in dead.index} #dict( zip(dead.index, dead.index) )
    exclud={}

    # Maximum existing distance
    sdmax=sdist.max().max()
    ddmax=ddist.max().max()
    
    # Last index of members, before clustering
    slastindex = sdist.index.max()
    dlastindex = ddist.index.max()
    
    # Find the pair of ppl with minimum distance in param space
    # these will be those with indices ii and jj
    iis0 = sdist.idxmin(axis=0)
    jjs = sdist.min(axis=0).idxmin()
    iis = int(iis0[jjs])
    
    iid0 = ddist.idxmin(axis=0)
    jjd = ddist.min(axis=0).idxmin()
    iid = int(iid0[jjd])

    while sdist.loc[iis,jjs] < sdmax/maxfrac and sdist.index.min() <= slastindex :
        # find the closest pair and add to sdist and surv
        iis, jjs = add2clu(iis,jjs,sdist,clus,surv,pname,exclus)
        
    while ddist.loc[iid,jjd] < ddmax/maxfrac and ddist.index.min() <= dlastindex :
        # find the closest pair and add to ddist and dead
        iid, jjd = add2clu(iid,jjd,ddist,clud,dead,pname,exclud)
    

    ssets = [ origsurv.loc[clus[kk], pname] for kk in clus.keys() ]
    dsets = [ origdead.loc[clud[kk], pname] for kk in clud.keys() ]

    yy, evect, ld, ax = fisher_projection.sepnd(ssets+dsets, pname, 'o'*len(ssets)+'v'*len(dsets), 3, '')
    
    # means and std devs of existing clusters, to be updated        
    means = surv.loc[:,pname]
    meand = dead.loc[:,pname]
    sigms = pd.DataFrame(index = surv.index, columns=pname)
    sigmd = pd.DataFrame(index = dead.index, columns=pname)

    for kk in surv.index.difference(origsurv.index) : 
        # vector p_i
        means.loc[kk,pname] = origsurv.loc[clus[kk], pname].mean()
        # sigma_id
        sigms.loc[kk,pname] = origsurv.loc[clus[kk], pname].std()
    # sigma_i**2
    sigms2tot = (sigms**2).sum(axis=1)
    
    for kk in dead.index.difference(origdead.index) : 
        # vector p_j
        meand.loc[kk,pname] = origdead.loc[clud[kk], pname].mean()
        # sigma_jd
        sigmd.loc[kk,pname] = origdead.loc[clud[kk], pname].std()
    # sigma_j**2
    sigmd2tot = (sigmd**2).sum(axis=1)

    # For data points not in clusters, fill in sigma as minsig
    # instead of zero
    sigms2tot.loc[surv.index.intersection(origsurv.index)] = minsig**2
    sigmd2tot.loc[dead.index.intersection(origdead.index)] = minsig**2
        
#    ds2_s = pd.DataFrame(index=surv.index, columns=surv.index, dtype=np.float64)
#    ds2_d = pd.DataFrame(index=dead.index, columns=dead.index, dtype=np.float64)
        
#    vstat=[]    
#    dist2_sd = {}
#    merged=[]
#    idx2mrg = [-1,None,None]
#    while sdist.shape[0]>20 and ddist.shape[0]>20:
#
#        findnext2clus(clus, clud, surv, dead, origsurv, origdead, means, meand, sigms, sigmd, sigms2tot, sigmd2tot, pname, minsig**2, dist2_sd, ds2_s, ds2_d, idx2mrg)
#        
#        if   idx2mrg[0]==0 :
#            # add new cluste to sdist and surv, don't need return values, 
#            # since we're determining (i,j) with findnext2clus()
#            _, _ = add2clu(idx2mrg[1],idx2mrg[2],sdist,clus,surv,pname,exclus)
#            
#            # remove the two clusters which were just merged from ds2_s
#            ds2_s.drop( [idx2mrg[1],idx2mrg[2]], axis=0, inplace=True ) 
#            ds2_s.drop( [idx2mrg[1],idx2mrg[2]], axis=1, inplace=True ) 
#            
#            merged.append([idx2mrg[1],idx2mrg[2],0])
#            
#        elif idx2mrg[0]==1 :
#            # add new cluster to ddist and dead 
#            _, _ = add2clu(idx2mrg[1],idx2mrg[2],ddist,clud,dead,pname,exclud)
#            
#            # remove the two clusters which were just merged from ds2_d
#            ds2_d.drop( [idx2mrg[1],idx2mrg[2]], axis=0, inplace=True ) 
#            ds2_d.drop( [idx2mrg[1],idx2mrg[2]], axis=1, inplace=True ) 
#
#            merged.append([idx2mrg[1],idx2mrg[2],1])
#        
#        print('last merged: ', merged[-1])
#        
#        # calculate the statistic
#        vstat.append( idx2mrg[3] )
#        
#        print(sdist.shape, ddist.shape)
            
    ssets = [ origsurv.loc[clus[kk], pname] for kk in clus.keys() ]
    dsets = [ origdead.loc[clud[kk], pname] for kk in clud.keys() ]
    fisher_projection.projplot(ssets+dsets, evect, ax, pname, 'o'*len(ssets)+'v'*len(dsets), '')

    pl.subplots_adjust(left=0.25, bottom=0.25)

    return clus, sdist, surv, clud, ddist, dead, None, None, exclus, exclud, ax, evect

    
    
    
    
    
    


#def findnext2clus(clus, clud, surv, dead, origsurv, origdead, means, meand, sigms, sigmd, sigms2tot, sigmd2tot, pname, minsig2, dist2_sd, ds2_s, ds2_d, idx2mrg):
#    
#    def dist2ij(ii,jj) :
#    
#        if (ii,jj) not in dist2_sd.keys() : 
#            dist2_sd[(ii,jj)] = ((means.loc[ii,pname] - meand.loc[jj,pname])**2).sum()
#            
#        return dist2_sd[(ii,jj)]
#
#
#    def denom(ii,jj) :
#        
#        if ii not in sigms2tot.index :
#            # vector p_i
#            means.loc[ii,pname] = origsurv.loc[clus[ii], pname].mean()
#            # sigma_id
#            sigms.loc[ii,pname] = origsurv.loc[clus[ii], pname].std()
#            # sigma_i
#            sigms2tot.loc[ii] = (sigms.loc[ii,pname]**2).sum()
#
#        if jj not in sigmd2tot.index :
#            # vector p_i
#            meand.loc[jj,pname] = origdead.loc[clud[jj], pname].mean()
#            # sigma_id
#            sigmd.loc[jj,pname] = origdead.loc[clud[jj], pname].std()
#            # sigma_i
#            sigmd2tot.loc[jj] = (sigmd.loc[jj,pname]**2).sum()
#        
#        return max( sigms2tot.loc[ii], minsig2 ) + \
#               max( sigmd2tot.loc[jj], minsig2 )
#
#               
#    
#    if idx2mrg[0] == -1 :
#        for ii1 in surv.index :
#            for ii2 in surv.index[surv.index.slice_indexer(start=ii1)][1:] :
#            
#                print(ii1,ii2)
#                # vector of len(pname)
#                p_i = origsurv.loc[ clus[ii1]+clus[ii2], pname ].mean(axis=0)
#                
#                # scalar
#                sigtoti2 = max(
#                 (origsurv.loc[clus[ii1]+clus[ii2], pname].std(axis=0)**2).sum() , 
#                  minsig2 )
#                
#                
#                ds2_s.loc[ii1,ii2] = ( ((meand.loc[:,pname]-p_i)**2).sum(axis=1) / \
#                                       (sigtoti2 + sigmd2tot)               ).sum()
#                
#                for iremoved in [ii1,ii2] :
#                    for jj in clud.keys() :
#                        thisdenom = denom(iremoved,jj)
#                        ds2_s.loc[ii1,ii2] = \
#                        ds2_s.loc[ii1,ii2] - dist2ij(iremoved,jj)/thisdenom
#    
#    
#    
#    
#        for jj1 in dead.index :
#            for jj2 in dead.index[dead.index.slice_indexer(start=jj1)][1:] :
#            
#                # vector of len(pname)
#                p_j = origdead.loc[ clud[jj1]+clud[jj2], pname ].mean(axis=0)
#                
#                # scalar
#                sigtotj2 = max(
#                 (origdead.loc[clud[jj1]+clud[jj2] ,pname].std(axis=0)**2).sum() , 
#                  minsig2 )
#                
#                
#                ds2_d.loc[jj1,jj2] = ( ((means.loc[:,pname]-p_j)**2).sum(axis=1) / \
#                                       (sigtotj2 + sigms2tot)               ).sum()
#                
#                for jremoved in [jj1,jj2] :
#                    for ii in clus.keys() :
#                        thisdenom = denom(ii,jremoved)
#                        ds2_d.loc[jj1,jj2] = \
#                        ds2_d.loc[jj1,jj2] - dist2ij(ii,jremoved)/thisdenom
#
#    elif idx2mrg[0]==0:
#        
#        ii2 = surv.index[-1]
#        ds2_s.loc[ii2,:] = np.nan
#        for ii1 in surv.index[0:-1] :
#
#            # vector of len(pname)
#            p_i = origsurv.loc[ clus[ii1]+clus[ii2], pname ].mean(axis=0)
#            
#            # scalar
#            sigtoti2 = max(
#             (origsurv.loc[clus[ii1]+clus[ii2], pname].std(axis=0)**2).sum() , 
#              minsig2 )
#            
#            
#            ds2_s.loc[ii1,ii2] = ( ((meand.loc[:,pname]-p_i)**2).sum(axis=1) / \
#                                   (sigtoti2 + sigmd2tot)               ).sum()
#            
#            for iremoved in [ii1,ii2] :
#                for jj in clud.keys() :
#                    thisdenom = denom(iremoved,jj)
#                    ds2_s.loc[ii1,ii2] = \
#                    ds2_s.loc[ii1,ii2] - dist2ij(iremoved,jj)/thisdenom        
#        
#    elif idx2mrg[0]==1: 
#        
#        jj2 = dead.index[-1]
#        ds2_d.loc[jj2,:] = np.nan      
#        for jj1 in dead.index[0:-1] :
#
#            # vector of len(pname)
#            p_j = origdead.loc[ clud[jj1]+clud[jj2], pname ].mean(axis=0)
#            
#            # scalar
#            sigtotj2 = max(
#             (origdead.loc[clud[jj1]+clud[jj2] ,pname].std(axis=0)**2).sum() , 
#              minsig2 )
#            
#            
#            ds2_d.loc[jj1,jj2] = ( ((means.loc[:,pname]-p_j)**2).sum(axis=1) / \
#                                   (sigtotj2 + sigms2tot)               ).sum()
#            
#            for jremoved in [jj1,jj2] :
#                for ii in clus.keys() :
#                    thisdenom = denom(ii,jremoved)
#                    ds2_d.loc[jj1,jj2] = \
#                    ds2_d.loc[jj1,jj2] - dist2ij(ii,jremoved)/thisdenom         
#            
#            
#    ii1v = ds2_s.idxmax(axis=0)
#    ii2  = ds2_s.max(axis=0).idxmax()
#    ii1  = int(ii1v[ii2])
#    
#    jj1v = ds2_d.idxmax(axis=0)
#    jj2  = ds2_d.max(axis=0).idxmax()
#    jj1  = int(jj1v[jj2])
#
#    if ds2_s.loc[ii1,ii2] > ds2_d.loc[jj1,jj2] : 
#        idx2mrg[0:3]= 0, ii1, ii2, ds2_s.loc[ii1,ii2]
#    else : 
#        idx2mrg[0:3] = 1, jj1, jj2, ds2_d.loc[jj1,jj2]













def add2clu(ii,jj, distmatrix, clu, psgr, pname, exclu):
    
    # Drop the two closest points from the distance matrix
    #t0=time.time()
    distmatrix.drop([ii,jj], axis=0, inplace=True)
    distmatrix.drop([ii,jj], axis=1, inplace=True)
    #print('removing cols/rows', time.time()-t0)
    
    # Index of a new row to be added below to the distance matrix
    newindex = psgr.index.max()+1
    
    # Add the two points as a new point, which is a cluster    
    #t0=time.time()
    psgr.loc[newindex,pname] =  psgr.loc[[ii,jj],pname].mean(axis=0)
    #print('increase passenged DF', time.time()-t0)
 
    # remove the two points we just added as a cluster
    psgr.drop([ii,jj], axis=0, inplace=True)
    
    # Add one row and one column to distance matrix
    #t0=time.time()
    distmatrix.loc[:,newindex] = \
    ( (psgr.loc[:,pname]-psgr.loc[newindex,pname])**2. ).sum(axis=1)**0.5
    distmatrix.loc[newindex,:] = np.NaN
    #print('add row+col to dist matrix', time.time()-t0)
        
    # Keep track of cluters in the clu dictionary
    clu[newindex] = []
    exclu[jj] = []
    exclu[ii] = []

    for kk in clu[ii] : 
        clu[newindex].append(kk)
        exclu[ii].append(kk)
    del clu[ii]

    for kk in clu[jj] : 
        clu[newindex].append(kk)
        exclu[jj].append(kk)
    del clu[jj]
    

    
    # Find the new closest pair
    #t0=time.time()
    ii0 = distmatrix.idxmin(axis=0)
    jj = distmatrix.min(axis=0).idxmin()
    ii = int(ii0[jj])
    #print('find new closest pair', time.time()-t0)

    return ii, jj