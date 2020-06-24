

import pandas as pd
import numpy as np
import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D
import numpy.random as npr

import pdb

pl.ion()


def projdata(traindata1, traindata2, testdata, pname, numdim) :
    
    yy, evect, lmbd, ax = sepnd(traindata1 + traindata2, pname, 'o'*len(traindata1)+'v'*len(traindata2), numdim, '')
    
    # create new object to not modify the input variable
    testdata = testdata.loc[:,pname]
    
    # create dataframe which will contain projected
    # test data
    yytest = pd.DataFrame(columns=yy.columns)

    # columns of new data frame:
    #dfcol = ['y'+str(kk) for kk in range(numdim) ]
    #yytest = pd.DataFrame(columns=dfcol)
    
    for ii in testdata.index :
        
        yytest.loc[ii,0:numdim] = np.dot(evect[:,np.abs(lmbd).argsort()[-1:len(pname)-1-numdim:-1]].T, testdata.loc[ii,:])


    ############### plotting #######################
            
    if numdim==2 :
        ax.plot( yytest.y0, yytest.y1, '+')
            
            
    elif numdim==3 : 
        crit = yytest.y0.notnull() & yytest.y1.notnull() & yytest.y2.notnull()
        ax.scatter( yytest.y0[crit], yytest.y1[crit], yytest.y2[crit], '+')
            


    elif numdim==1 :
        ax.hist(yytest.y0, bins=80, alpha=0.6) # range=[yy.y0.min(),yy.y0.max()], 
        
        
        
        
        
    return yy, evect, lmbd, yytest
    































    
def sepnd(datalist, pname, symstr, numdim, colstr, ax=None) :
    
    
    ndata = len(datalist)

    means = []
    sws=np.zeros([len(pname), len(pname), ndata])
    sbs=np.zeros([len(pname), len(pname), ndata])
    
    for idata in range(ndata) :

        surv = datalist[idata].loc[:,pname]
        
        # Mean vectors
        means.append( surv.mean() )

    # Overall mean of means
    ovrlmean = means[0] / ndata
    for idata in range(1,ndata): ovrlmean = ovrlmean + means[idata]/ndata
    
    for idata in range(ndata) :
        
        surv = datalist[idata].loc[:,pname]

        # Scatter within each class
        for ii in range( surv[pname[0]].count() ) :
            sws[:,:,idata] = sws[:,:,idata] + np.outer( surv.iloc[ii,:]-means[idata] , surv.iloc[ii,:]-means[idata] )


        # Scatter between each classes and the overall mean
        sbs[:,:,idata] = surv.shape[0] * np.outer(means[idata]-ovrlmean, means[idata]-ovrlmean)

            
    # Scatter within classes
    sw = np.sum(sws, axis=2)

    # Scatter between classes
    sb = np.sum(sbs, axis=2)

    # Matrix to be eigen-solved
    mm = np.matmul(np.linalg.inv(sw), sb)
    #return mm

    # eigen values and vectors
    lmbd , evect = np.linalg.eig(mm)
    
    # columns of new data frame:
    dfcol = ['y'+str(kk) for kk in range(numdim) ]
    dfcol.append('clu')
    yy = pd.DataFrame(columns=dfcol)

    # check that there are no imaginary parts to the highest eigenvalues
    for ii in np.abs(lmbd).argsort()[-1:len(pname)-1-numdim:-1] : 
        if np.imag( lmbd[ii] ) != 0 or (np.imag(evect[:,ii])**2).sum() != 0: 
            raise Exception('Imaginary solution to eigenproblem')
    
    # If function hasn't returned from the above check,
    # then use real values of eivenvectors which will be used
    # for the transformation
    evectreal = np.real( evect[:,np.abs(lmbd).argsort()[-1:len(pname)-1-numdim:-1]] )

            
    # Project data on the eigenvectors with highest eigenvalues
    # This gives the new variable yy, which has numdim dimensions
    # and will maximize separation between the classes
    for idata in range(ndata) :
        
        surv = datalist[idata].loc[:,pname]

        for ii in surv.index :
            
            yy.loc[ii,0:numdim] = np.dot(evectreal.T, surv.loc[ii,:])
            yy.loc[ii, 'clu'] = idata
    
    # colors
    maxndata=200
    colfunc = pl.get_cmap('gist_ncar')
    col_list = [colfunc(icol/maxndata) for icol in range(maxndata)]
    npr.seed(0)
    npr.shuffle(col_list)


        
    ######### Plotting ########################

    if numdim==2 :
        #cols = 'bgrcmykw'
        
        if not ax :
            fig = pl.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.set_facecolor((.8,.8,.8))
        
        for idata in range(ndata) : 
            if symstr and len(symstr)==ndata : 
                ax.plot( yy.y0[yy.clu==idata], yy.y1[yy.clu==idata], symstr[idata], color=col_list[idata]) #, color = cols[idata % 8] )
            else : 
                ax.plot( yy.y0[yy.clu==idata], yy.y1[yy.clu==idata], 'o', color=col_list[idata])
            
            
    elif numdim==3 : 
        if not ax :
            fig = pl.figure()
            ax = fig.add_subplot(111, projection='3d') 
        for idata in range(ndata) : 
            if symstr and len(symstr)==ndata : 
                ax.scatter( yy.y0[yy.clu==idata], yy.y1[yy.clu==idata], yy.y2[yy.clu==idata], marker=symstr[idata], color=col_list[idata]) #, color = cols[idata % 8] )
            else :
                ax.scatter( yy.y0[yy.clu==idata], yy.y1[yy.clu==idata], yy.y2[yy.clu==idata], marker='o', color=col_list[idata])
            


    elif numdim==1 :
        if not ax :
            fig = pl.figure()
            ax = fig.add_subplot(111)
        for idata in range(ndata) :
            if colstr and len(colstr)==ndata: 
                ax.hist(yy.y0[yy.clu==idata], bins=80, range=[yy.y0.min(),yy.y0.max()], color=colstr[idata], alpha=0.6)
            else : 
                ax.hist(yy.y0[yy.clu==idata], bins=80, range=[yy.y0.min(),yy.y0.max()], alpha=0.6)

                
                
    ######################################
                
                
                
                

    
    return yy , evectreal, lmbd, ax
    


def projplot(datalist, evectreal, ax, pname, symstr, colstr) :

    ndata = len(datalist)
    
    numdim = evectreal.shape[1]

    # columns of new data frame:
    dfcol = ['y'+str(kk) for kk in range(numdim) ]
    dfcol.append('clu')
    yy = pd.DataFrame(columns=dfcol) #list(range(numdim))+['clu'] )


    # Project data on the eigenvectors with highest eigenvalues
    # This gives the new variable yy, which has numdim dimensions
    # and will maximize separation between the classes
    for idata in range(ndata) :
        
        surv = datalist[idata].loc[:,pname]

        for ii in surv.index :
            
            yy.loc[ii,0:numdim] = np.dot(evectreal.T, surv.loc[ii,:])
            yy.loc[ii, 'clu'] = idata

    ######### Plotting ########################

    # colors
    maxndata=200
    colfunc = pl.get_cmap('gist_ncar')
    col_list = [colfunc(icol/maxndata) for icol in range(maxndata)]
    npr.seed(0)
    npr.shuffle(col_list)

    if numdim==2 :
        #cols = 'bgrcmykw'
        
        if not ax :
            fig = pl.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.set_facecolor((.8,.8,.8))
        
        for idata in range(ndata) : 
            if symstr and len(symstr)==ndata : 
                ax.plot( yy.y0[yy.clu==idata], yy.y1[yy.clu==idata], symstr[idata], color=col_list[idata]) #, color = cols[idata % 8] )
            else : 
                ax.plot( yy.y0[yy.clu==idata], yy.y1[yy.clu==idata], 'o', color=col_list[idata])
            
            
    elif numdim==3 : 
        if not ax :
            fig = pl.figure()
            ax = fig.add_subplot(111, projection='3d')
        scatpl=[]
        for idata in range(ndata) : 
            if symstr and len(symstr)==ndata : 
                scatpl.append( ax.scatter( yy.y0[yy.clu==idata], yy.y1[yy.clu==idata], yy.y2[yy.clu==idata], marker=symstr[idata], color=col_list[idata]) )#, color = cols[idata % 8] )
            else :
                scatpl.append( ax.scatter( yy.y0[yy.clu==idata], yy.y1[yy.clu==idata], yy.y2[yy.clu==idata], marker='o', color=col_list[idata]) )
            


    elif numdim==1 :
        if not ax :
            fig = pl.figure()
            ax = fig.add_subplot(111)
        for idata in range(ndata) :
            if colstr and len(colstr)==ndata: 
                ax.hist(yy.y0[yy.clu==idata], bins=80, range=[yy.y0.min(),yy.y0.max()], color=colstr[idata], alpha=0.6)
            else : 
                ax.hist(yy.y0[yy.clu==idata], bins=80, range=[yy.y0.min(),yy.y0.max()], alpha=0.6)

                
    if scatpl : return scatpl
    ######################################