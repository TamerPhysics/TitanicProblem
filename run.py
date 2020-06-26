
import pickle
import copy
import os

import matplotlib as pl

from titanic_class import *

try : 

    dir(tdata)
    
except NameError :

    if not os.path.exists('titanic_data_distmatrix_addtix.pyd') :
        
        tdata = Titanic()
        tdata.calcdists()
    
        with open('titanic_data_distmatrix_addtix.pyd', 'wb') as ff :
            pickle.dump(tdata, ff)
            
    else : 
        with open('titanic_data_distmatrix_addtix.pyd', 'rb') as ff :
            tdata = pickle.load(ff)

try :
    _ = t1
except NameError :
    t1 = copy.deepcopy(tdata)
    t1.cluster(maxfrac=3)
    t1.compclu()
    t1.classify()

try :
    _ = t2
except NameError :
    t2 = copy.deepcopy(tdata)
    t2.cluster(maxfrac=5)
    t2.compclu()
    t2.classify()

def plotmodtik(t2) :
    f1,ax1 = pl.subplots(1,2)
    tp = ax1[0].hist2d(
            t2.train.ticketnum[t2.train.ticketnum.notna()] 
            % 100000 , 
            t2.train.Survived[t2.train.ticketnum.notna()], 
            [32,2])
    
    #pl.plot( tp[0][:,0]/(tp[0][:,1]+tp[0][:,0]) , 'o')
    #pl.plot( tp[0][:,0]/(tp[0][:,1]+tp[0][:,0]) , 'b')
    ax1[1].bar( tp[1][0:-1], 
           tp[0][:,0]/(tp[0][:,1]+tp[0][:,0]), 
           tp[1][1:]-tp[1][0:-1], 
           align='edge' )
    
    
    
def plotdivtik(t2) :
    f2,ax2 = pl.subplots(1,2)
    tp2 = ax2[0].hist2d(
            t2.train.ticketnum[t2.train.ticketnum.notna()] 
            // 100000 , 
            t2.train.Survived[t2.train.ticketnum.notna()], 
            [32,2])
    
    ax2[1].bar( tp2[1][0:-1], 
           tp2[0][:,0]/(tp2[0][:,1]+tp2[0][:,0]), 
           tp2[1][1:]-tp2[1][0:-1], 
           align='edge' )
    