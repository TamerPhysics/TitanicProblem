
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


t=titanic_class.Titanic()

#X = t.surv[t.pnorm].T
cols = t.pnorm
X = t.train[cols].dropna()

# Singular value decomposition
U, S, Vt = np.linalg.svd(X , full_matrices=False)

# Pick the first few principal axes from V matrix
princip = Vt[0:2,:]

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



# LINEAR SVM CLASSIFIER
X2 = projx.loc[:,newcols]
y2 = projx.Survived

svmlin = LinearSVC(C=1, dual=False) #loss="hinge",

svmlin.fit(X2, y2)

fig, ax = pl.subplots()
ax.plot(projx.c0[projx.Survived==1], 
        projx.c1[projx.Survived==1], 'og', alpha=0.5)
ax.plot(projx.c0[projx.Survived==0], 
        projx.c1[projx.Survived==0], 'or', alpha=0.5)

xax = np.array([projx.c0.min(), projx.c0.max()])

ww = svmlin.coef_[0]
bb = svmlin.intercept_[0]

#ax.plot( xax, -ww[1]/ww[0]*xax - ww[0]/bb , 'c' )


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

# NON-LINEAR SVM CLASSIFIER

svm = SVC(kernel='poly', gamma=1, coef0=0.)
svm.fit(X2, y2)
plot_decision_function(svm, X2, y=y2, axis=ax, title='') 





fig2, ax2 = pl.subplots()
ax2.plot(projx.c0[projx.Survived==1], 
        projx.c1[projx.Survived==1], 'og', alpha=0.5)
ax2.plot(projx.c0[projx.Survived==0], 
        projx.c1[projx.Survived==0], 'or', alpha=0.5)

svm = SVC(kernel='rbf', gamma=0.5**2)
svm.fit(X2, y2)
plot_decision_function(svm, X2, y=y2, axis=ax2, title='rbf radius=2') 




fig3, ax3 = pl.subplots()
ax3.plot(projx.c0[projx.Survived==1], 
        projx.c1[projx.Survived==1], 'og', alpha=0.5)
ax3.plot(projx.c0[projx.Survived==0], 
        projx.c1[projx.Survived==0], 'or', alpha=0.5)

svm = SVC(C=10, kernel='rbf', gamma=0.5**2)
svm.fit(X2, y2)
plot_decision_function(svm, X2, y=y2, axis=ax3, title='C=10, rbf radius=2') 



fig4, ax4 = pl.subplots()
ax4.plot(projx.c0[projx.Survived==1], 
        projx.c1[projx.Survived==1], 'og', alpha=0.5)
ax4.plot(projx.c0[projx.Survived==0], 
        projx.c1[projx.Survived==0], 'or', alpha=0.5)

svm = SVC(C=100, kernel='rbf', gamma=0.5**2)
svm.fit(X2, y2)
plot_decision_function(svm, X2, y=y2, axis=ax4, title='C=100, rbf radius=1/2') 




fig5, ax5 = pl.subplots()
ax5.plot(projx.c0[projx.Survived==1], 
        projx.c1[projx.Survived==1], 'og', alpha=0.5)
ax5.plot(projx.c0[projx.Survived==0], 
        projx.c1[projx.Survived==0], 'or', alpha=0.5)

svm = SVC(C=100, kernel='rbf', gamma=0.2**2)
svm.fit(X2, y2)
plot_decision_function(svm, X2, y=y2, axis=ax5, title='C=100, rbf radius=5') 



fig6, ax6 = pl.subplots()
ax6.plot(projx.c0[projx.Survived==1], 
        projx.c1[projx.Survived==1], 'og', alpha=0.5)
ax6.plot(projx.c0[projx.Survived==0], 
        projx.c1[projx.Survived==0], 'or', alpha=0.5)

svm = SVC(C=100, kernel='rbf', gamma=1/0.5**2)
svm.fit(X2, y2)
plot_decision_function(svm, X2, y=y2, axis=ax6, title='C=100, rbf radius=0.5')








fig7, ax7 = pl.subplots()
ax7.plot(projx.c0[projx.Survived==1], 
        projx.c1[projx.Survived==1], 'og', alpha=0.5)
ax7.plot(projx.c0[projx.Survived==0], 
        projx.c1[projx.Survived==0], 'or', alpha=0.5)

svm = SVC(C=50, kernel='rbf', gamma=1/0.5**2)
svm.fit(X2, y2)
plot_decision_function(svm, X2, y=y2, axis=ax7, title='C=50, rbf radius=0.5')










fig8, ax8 = pl.subplots()
ax8.plot(projx.c0[projx.Survived==1], 
        projx.c1[projx.Survived==1], 'og', alpha=0.5)
ax8.plot(projx.c0[projx.Survived==0], 
        projx.c1[projx.Survived==0], 'or', alpha=0.5)

svm = SVC(C=10, kernel='rbf', gamma=1/0.5**2)
svm.fit(X2, y2)
plot_decision_function(svm, X2, y=y2, axis=ax8, title='C=10, rbf radius=0.5')




fig9, ax9 = pl.subplots()
ax9.plot(projx.c0[projx.Survived==1], 
        projx.c1[projx.Survived==1], 'og', alpha=0.5)
ax9.plot(projx.c0[projx.Survived==0], 
        projx.c1[projx.Survived==0], 'or', alpha=0.5)

svm = SVC(C=1, kernel='rbf', gamma=1/0.5**2)
svm.fit(X2, y2)
plot_decision_function(svm, X2, y=y2, axis=ax9, title='C=1, rbf radius=0.5')





















