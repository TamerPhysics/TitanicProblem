
import pickle
import importlib
import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D

from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

try : 
    dir(t)
    del t
except NameError : pass


import titanic_class 
importlib.reload(titanic_class)

# plot 2D decision function
def plot_decision_function(classifier, X, y=None, axis=None, title=''):
    
    if len(X.shape) != 2 :
        raise Exception('this only works for 2D')
    
    if y is None :
        y = pd.Series(1, index=X.index)

    cols = list(X.columns)
    
    # plot the decision function
    xx, yy = np.meshgrid(np.linspace(X.loc[:,cols[0]].min(), 
                                     X.loc[:,cols[0]].max(), 500), 
                         np.linspace(X.loc[:,cols[1]].min(), 
                                     X.loc[:,cols[1]].max(), 500))

    Z = classifier.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    if axis==None : 
        fig, axis = pl.subplots()
        
    # plot the line, the points, and the nearest vectors to the plane
    axis.contourf(xx, yy, Z, alpha=0.75, cmap=pl.cm.bone)
    
    #axis.scatter(X.loc[:, 0], X[:, 1], c=y, alpha=0.9,
    #             cmap=pl.cm.bone) #, edgecolors='black')

    #axis.axis('off')
    axis.set_title(title)



t=titanic_class.Titanic()

#X = t.surv[t.pnorm].T
cols = t.pnorm
X = t.train[cols].dropna()

# Singular value decomposition
U, S, Vt = np.linalg.svd(X , full_matrices=False)

# Pick the first few principal axes from V matrix
princip = Vt[0:4,:]

# new column names c0, c1, c2, etc
newcols = ['c'+str(ii) for ii in range(princip.shape[0])]

# this DF will contain projected data on the first
# few principal axes
projx = pd.DataFrame(index=X.index, columns=newcols, 
                     dtype=np.float)

# Do the projection
for ii in projx.index :
    projx.loc[ii,:] = princip @ X.loc[ii,:]

# Add the Survived column to the projected DF
projx.loc[:, 'Survived'] = (t.train[cols+['Survived']].dropna()).Survived

# Add c0, c1, ... columns to test data
for col in newcols :
    t.filltest.loc[:,col]=-1

# Project test data
for ii in t.test2.index :
    t.filltest.loc[ii,newcols] = princip @ t.filltest.loc[ii,cols]




# NON-LINEAR SVM CLASSIFIER
X2 = projx.loc[:,newcols]
y2 = projx.Survived




#fig2, ax2 = pl.subplots()
#ax2.plot(projx.c0[projx.Survived==1], 
#        projx.c1[projx.Survived==1], 'og', alpha=0.5)
#ax2.plot(projx.c0[projx.Survived==0], 
#        projx.c1[projx.Survived==0], 'or', alpha=0.5)

svm = SVC(C=1, kernel='rbf', gamma=1/0.7**2)
svm.fit(X2, y2)
#plot_decision_function(svm, X2, y=y2, axis=ax2, title='rbf radius=2') 

scores = cross_val_score(svm, projx[newcols], projx.Survived, cv=10)
print( scores )
print( scores.mean() )


















