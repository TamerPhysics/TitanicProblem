import pandas as pd
import numpy as np
import matplotlib.pyplot as pl

def expnd(testpoint, clus, clud, sigms, sigmd, pdisc, pdiscfull, pcontnorm, pcontnormfull, deltacont, surv, dead):
    
    sigms = pd.DataFrame(sigms)
    sigmd = pd.DataFrame(sigmd)

    surv = pd.DataFrame(surv)
    dead = pd.DataFrame(dead)
    
    clus = dict(clus)
    clud = dict(clud)
    
    minsig=1e-6
    
    # expected number of dead ppl at point testpoint
    nnd = 0.
    for kk in clud.keys() : 
        
        thiscluster = pd.DataFrame(dead.loc[clud[kk],:])
        
        # Check the expected number of dead ppl
        # from the disrete parameters first:

        # initialize the cretria to all true
        criteria = thiscluster.numsex == thiscluster.numsex
        undef_param_factor = 1.
        
        for pp in pdiscfull : 
        
            if pp in pdisc : 
            
                # criteria of selection, in order to count entries matching testpoint
                criteria = criteria & ( thiscluster[pp]==testpoint.loc[0,pp] )
                #undef_param_factor = undef_param_factor * 1.
            
            else : 
                
                # if test point does NOT have a value of this parameter pp
                # then calculate the overall fraction of dead ppl with the
                # values of pp that are the same as in this cluster kk 
                weightedfrac=0.
                for pval in thiscluster[pp].unique() : 

                    # weighted average of fractions of how many in overall
                    # dead population has each value of pp found in this 
                    # cluster. It's weighted by fraction of cluster members
                    # with this value pp
                    weightedfrac =  weightedfrac + (thiscluster[pp]==pval).sum() / len(clud[kk]) * (dead[pp]==pval).sum() / dead[pp].notnull().sum()
                        
                
                undef_param_factor = undef_param_factor * weightedfrac
    
        nndthisclu = thiscluster[criteria].shape[0] * undef_param_factor


        ################ CONTINUOUS PARAMETERS ####################
        # if we survived the discrete parameter filter, continue on..
        if nndthisclu > 0 :

            # the means of params with the selection criteria
            # of the discrete param values
            meanthisloc = thiscluster.loc[criteria, pcontnorm].mean()
            
            # For the sigma, we only use the criteria if we have 
            # enough counts to make it reliable. Otherwise, use
            # the value computed over all cluster.
            if thiscluster[criteria].shape[0] >= 5: 
                sigmthisloc = thiscluster.loc[criteria, pcontnorm].std()
            else : sigmthisloc = sigmd.loc[kk,pcontnorm]

            # If for just 1 param, the sigmas of the continuous params 
            # of this cluster are so small (all cluster points have 
            # same values of the param) AND the test point 
            # lies far from the avg value of the cluster, then 
            # there is no overlap between cluster and testpoint
            # ==> There's no value to add to nnd
            noexpectedn=False
            for pp in pcontnormfull : 
                if pp in pcontnorm and sigmthisloc[pp] < minsig and np.abs( meanthisloc[pp]-testpoint.loc[0,pp]) > minsig : noexpectedn=True


            if noexpectedn==False: 
            
                for pp in pcontnormfull : 
                    
                    if pp in pcontnorm and sigmthisloc[pp] > minsig:
                        
                        
                        sigdist = (meanthisloc[pp]-testpoint.loc[0,pp])**2 / sigmthisloc[pp]**2
                        
                        nndthisclu = nndthisclu / (sigmthisloc[pp]*(2*np.pi))**0.5 * np.exp(-sigdist/2)
                        
                        #pdb.set_trace()
    
                        # if we have a very small sigma, but the testpoint matches
                        # the cluster values, then add all cluster members. (no need
                        # to check if np.abs( meanthisloc[pp]-testpoint.loc[0,pp]) < minsig 
                        # because the other possibility was ruled out in the noexpectedn
                        # paragraph)
                        #elif sigmthisloc[pp] < minsig : nndthisclu = nndthisclu * 1. #!!!!!!!!!!
                        # no need to do anything :)
                            
                        
                        
                    # parameter value not contained in test point. Assume uniform probability
                    # distribution over all range of the parameter in the entire dataset
                    else : 
                        
                        nndthisclu = nndthisclu / deltacont.loc[pp]
            
        # nndthisclu should have now the "contrib" of both discrete and continuous params
        nnd = nnd + nndthisclu

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
            
    # expected number of surviving ppl at point testpoint
    nns = 0.
    for kk in clus.keys() : 
        
        thiscluster = pd.DataFrame(surv.loc[clus[kk],:])
        
        # Check the expected number of surv ppl
        # from the disrete parameters first:

        # initialize the cretria to all true
        criteria = thiscluster.numsex == thiscluster.numsex
        undef_param_factor = 1.
        
        for pp in pdiscfull : 
        
            if pp in pdisc : 
            
                # criteria of selection, in order to count entries matching testpoint
                criteria = criteria & ( thiscluster[pp]==testpoint.loc[0,pp] )
                #undef_param_factor = undef_param_factor * 1.
            
            else : 
                
                # if test point does NOT have a value of this parameter pp
                # then calculate the overall fraction of surv ppl with the
                # values of pp that are the same as in this cluster kk 
                weightedfrac=0.
                for pval in thiscluster[pp].unique() : 

                    # weighted average of fractions of how many in overall
                    # surv population has each value of pp found in this 
                    # cluster. It's weighted by fraction of cluster members
                    # with this value pp
                    weightedfrac =  weightedfrac + (thiscluster[pp]==pval).sum() / len(clus[kk]) * (surv[pp]==pval).sum() / surv[pp].notnull().sum()
                        
                
                undef_param_factor = undef_param_factor * weightedfrac
    
        nnsthisclu = thiscluster[criteria].shape[0] * undef_param_factor


        ################ CONTINUOUS PARAMETERS ####################
        # if we survived the discrete parameter filter, continue on..
        if nnsthisclu > 0 :

            # the means of params with the selection criteria
            # of the discrete param values
            meanthisloc = thiscluster.loc[criteria, pcontnorm].mean()
            
            # For the sigma, we only use the criteria if we have 
            # enough counts to make it reliable. Otherwise, use
            # the value computed over all cluster.
            if thiscluster[criteria].shape[0] >= 5: 
                sigmthisloc = thiscluster.loc[criteria, pcontnorm].std()
            else : sigmthisloc = sigms.loc[kk,pcontnorm]

            # If for just 1 param, the sigmas of the continuous params 
            # of this cluster are so small (all cluster points have 
            # same values of the param) AND the test point 
            # lies far from the avg value of the cluster, then 
            # there is no overlap between cluster and testpoint
            # ==> There's no value to add to nns
            noexpectedn=False
            for pp in pcontnormfull : 
                if pp in pcontnorm and sigmthisloc[pp] < minsig and np.abs( meanthisloc[pp]-testpoint.loc[0,pp]) > minsig : noexpectedn=True


            if noexpectedn==False: 
            
                for pp in pcontnormfull : 
                    
                    if pp in pcontnorm and sigmthisloc[pp] > minsig:
                        
                        
                        sigdist = (meanthisloc[pp]-testpoint.loc[0,pp])**2 / sigmthisloc[pp]**2
                        
                        nnsthisclu = nnsthisclu / (sigmthisloc[pp]*(2*np.pi))**0.5 * np.exp(-sigdist/2)
                        
                        #pdb.set_trace()
    
                        # if we have a very small sigma, but the testpoint matches
                        # the cluster values, then add all cluster members. (no need
                        # to check if np.abs( meanthisloc[pp]-testpoint.loc[0,pp]) < minsig 
                        # because the other possibility was ruled out in the noexpectedn
                        # paragraph)
                        #elif sigmthisloc[pp] < minsig : nnsthisclu = nnsthisclu * 1. #!!!!!!!!!!
                        # no need to do anything :)
                            
                        
                        
                    # parameter value not contained in test point. Assume uniform probability
                    # distribution over all range of the parameter in the entire dataset
                    else : 
                        
                        nnsthisclu = nnsthisclu / deltacont.loc[pp]
            
        # nnsthisclu should have now the "contrib" of both discrete and continuous params
        nns = nns + nnsthisclu



    ################################################
    ############# finish and return
    #################################################

            
    if   nns > nnd : survvar = 1
    elif nns < nnd : survvar = 0
    else : survvar = int( np.round( np.random.rand() ) )
    
    return nns, nnd, survvar