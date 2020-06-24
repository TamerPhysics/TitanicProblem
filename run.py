
import pickle
import copy
import os

from titanic_class import *

try : 

    dir(tdata)
    
except NameError :

    if not os.path.exists('titanic_data_distmatrix.pyd') :
        
        tdata = Titanic()
        tdata.calcdists()
    
        with open('titanic_data_distmatrix.pyd', 'wb') as ff :
            pickle.dump(tdata, ff)
            
    else : 
        with open('titanic_data_distmatrix.pyd', 'rb') as ff :
            tdata = pickle.load(ff)

t1 = copy.deepcopy(tdata)
t1.cluster(maxfrac=3)
t1.compclu()
t1.classify()

