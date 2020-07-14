
import pickle
import importlib
import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D

from sklearn.neighbors import KNeighborsClassifier as Vecino
from sklearn.model_selection import cross_val_score

try : 
    dir(t)
    del t
except NameError : pass


import titanic_class 
importlib.reload(titanic_class)


t=titanic_class.Titanic()

#X = t.surv[t.pnorm].T
cols = t.pnorm
X = t.train[cols].dropna()

U, S, Vt = np.linalg.svd(X , full_matrices=False)

princip = Vt[0:4,:]

newcols = ['c'+str(ii) for ii in range(princip.shape[0])]
projx = pd.DataFrame(index=X.index, 
                     columns=newcols, 
                     dtype=np.float)
for ii in projx.index :
    projx.loc[ii,:] = princip @ X.loc[ii,:]
    

#pl.figure()
#pl.plot(projx.c0[t.train.Survived==1], 
#        projx.c1[t.train.Survived==1], 'og', alpha=0.5)
#pl.plot(projx.c0[t.train.Survived==0], 
#        projx.c1[t.train.Survived==0], 'or', alpha=0.5)

#pl.figure()
#pl.subplot(111, projection='3d')
#pl.plot(projx.c0[t.train.Survived==1], 
#        projx.c1[t.train.Survived==1],
#        projx.c2[t.train.Survived==1], 'og', alpha=0.5)
#pl.plot(projx.c0[t.train.Survived==0], 
#        projx.c1[t.train.Survived==0],
#        projx.c2[t.train.Survived==0], 'or', alpha=0.5)

projx.loc[:, 'Survived'] = (t.train[cols+['Survived']].dropna()).Survived

t.dectree(train=projx, pars=['c0', 'c1', 'c2'], compare2=projx, predict=False)

for col in newcols :
    t.filltest.loc[:,col]=-1

for ii in t.test2.index :
    t.filltest.loc[ii,newcols] = princip @ t.filltest.loc[ii,cols]

vcn = Vecino(n_neighbors=35, algorithm='brute')
print( cross_val_score(vcn, projx[newcols], projx.Survived, cv=10) )

vcn.fit(projx[newcols], projx.Survived)

predarr = vcn.predict(t.filltest[newcols])
    
# Save predictions as pandas data frame        
pred = pd.Series(predarr, index=t.filltest.index, dtype=np.int, 
                 name='Survived')
#pred.to_csv('svd4d_kneighbor35.csv', header=True, index=True)