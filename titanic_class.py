
# based on comparecluster2.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as pl

import cluster6
import compclu
import expectedn3
import distmatrix7

from sklearn.tree import DecisionTreeClassifier as DecisTree
from sklearn.ensemble import RandomForestClassifier as RndFrst

import pdb

# order of functions to run
# initialize 
# --> .calcdist() 
# --> .cluster()
# --> compclu() 
# --> classify()

class Titanic :
    
    def __init__(self, pname = ['Pclass', 'logage', 'numsex', 
                      'SibSp', 'Parch', 'logfare',
                      'ticketnum_mod', 'ticketnum_grp_clip']) :
        
        # Load the data
        self.train = pd.read_csv('train.csv')
        self.train.index = self.train.PassengerId
        
        # create numerical fields for sex column
        self.train.insert(self.train.shape[1], 'numsex', -1, True)
        self.train.loc[self.train.Sex=='male',  'numsex'] = 0.
        self.train.loc[self.train.Sex=='female','numsex'] = 1.
        
        # create log10(age) column
        self.train['logage'] = np.log10(self.train.Age)
        
        # create log10(fare) column
        self.train.insert(self.train.shape[1], 'logfare', np.log10(self.train.Fare.clip(1.)), True)
        
        # ticket number (remove letters)
        for ii in self.train.index :
            for ss in self.train.Ticket[ii].split(' ') :
                try :
                    self.train.loc[ii,'ticketnum'] = int(ss)
                    self.train.loc[ii,'ticketnum_mod'] = \
                      int(ss) % 100000
                    self.train.loc[ii,'ticketnum_grp'] = \
                      int(ss) // 100000
                    break
                except ValueError :
                    pass
        
        self.train.loc[:,'ticketnum_grp_clip'] = \
          self.train.ticketnum_grp.clip(upper=4)
        
        overallmean = self.train.mean(axis=0)
        overallstd  = self.train.std(axis=0)
        
        # Numerical parameters which define our parameter space
        self.pname = list(pname)
        
        # Normalize axes by the whole dataset std dev
        self.pnorm=[]
        for pp in list(self.pname) : 
            self.train.insert(self.train.shape[1], pp+'_norm', -1, True)
            self.train.loc[:,pp+'_norm'] = (self.train.loc[:,pp]-overallmean[pp]) / overallstd[pp]
        
            self.pnorm.append(pp+'_norm')
        
        # Clean data from NaN values in Age and ticketnum
        # The other columns of interest don't have NaN values
        self.train2 = self.train[self.train.Age.notnull() &
                                 self.train.ticketnum.notnull()]
        
        # Create 2 new dataset corresponding to surviving and dead passengers
        self.surv = self.train2.loc[self.train2.Survived==True , self.pname+self.pnorm]
        self.dead = self.train2.loc[self.train2.Survived==False, self.pname+self.pnorm]
        

        
        
        
        
        
        
        
        ###################### TEST DATA #####################
        
        # Load the data
        self.testdata = pd.read_csv('test.csv')
        self.testdata.index = self.testdata.PassengerId
        
        # create numerical fields for sex column
        self.testdata.insert(self.testdata.shape[1], 'numsex', -1, True)
        self.testdata.loc[self.testdata.Sex=='male',   'numsex']= 0.
        self.testdata.loc[self.testdata.Sex=='female', 'numsex'] = 1.
        
        
        # create log10(age) column
        self.testdata.insert(self.testdata.shape[1], 'logage', np.log10(self.testdata.Age), True)
        
        # create log10(fare) column
        self.testdata.insert(self.testdata.shape[1], 'logfare', np.log10(self.testdata.Fare.clip(1.)), True)
        
        # ticket number (remove letters)
        for ii in self.testdata.index :
            for ss in self.testdata.Ticket[ii].split(' ') :
                try :
                    self.testdata.loc[ii,'ticketnum'] = int(ss)
                    self.testdata.loc[ii,'ticketnum_mod'] = \
                      int(ss) % 100000
                    self.testdata.loc[ii,'ticketnum_grp'] = \
                      int(ss) // 100000
                    break
                except ValueError :
                    pass
                    
        self.testdata.loc[:,'ticketnum_grp_clip'] = \
          self.testdata.ticketnum_grp.clip(upper=4)
          
        # Normalize axes by the whole dataset std dev
        for pp in list(self.pname) : 
            self.testdata.insert(self.testdata.shape[1], pp+'_norm', -1, True)
            self.testdata.loc[:,pp+'_norm'] = (self.testdata.loc[:,pp]-overallmean[pp]) / overallstd[pp]
        
        
        # Clean data from NaN values in Age, Fare and ticketnum
        # The other columns of interest don't have NaN values
        self.test2 = pd.DataFrame( 
                self.testdata[self.testdata.Age.notnull() & 
                              self.testdata.ticketnum.notnull() &
                              self.testdata.Fare.notnull()] )


        self.testbadpars=['Age', 'logfare', 
                          'ticketnum_mod', 'ticketnum_grp_clip']
        
        self.testbadparsnorm=[]
        for pp in list(self.testbadpars) : 
            self.testbadparsnorm.append(pp+'_norm')
            
            
            
            
        #### Filled-in data in test data================
        self.filltest = self.testdata.copy()
        pars = self.pnorm
        for ii in self.testdata.index :
            if self.testdata.loc[ii, pars].isna().sum() > 0 :
        
                napars = list( 
                 self.testdata.loc[ii, pars].loc[
                   self.testdata.loc[ii, pars].isna()].index)
        
                goodpars=list(pars)
                for pp in napars :
                    goodpars.remove(pp)
        
                radius=0.2
                while self.filltest.loc[ii,napars].isna().sum()>0 \
                  or radius==0.2 or radius < 1.7 :

                    # data points which are close to the
                    # data point to be filled
                    wh = pd.Series(True, index=self.testdata.index)
                    for pp in goodpars :
                        wh = wh & (
                                (self.testdata.loc[:,pp]-
                                 self.testdata.loc[ii,pp]).abs()<radius) \
                             & self.testdata.loc[:,pp].notnull()
                    
                    self.filltest.loc[ii,napars] = \
                      self.testdata.loc[wh,napars].mean()

                    radius = radius * 2







    ############# DISTANCE MATRIX #################
    def calcdists(self) :
        
        self.sdist = distmatrix7.calcdist(self.surv, self.pnorm)
        self.ddist = distmatrix7.calcdist(self.dead, self.pnorm)




    ############# clustering analysis ############
    def cluster(self, maxfrac=5, minsig=1e-6) :

        self.clus, self.sdist2, self.surv2, \
        self.clud, self.ddist2, self.dead2, \
        self.vstat, self.merged, self.exclus, self.exclud, \
        self.ax, self.evect = \
          cluster6.cluster(self.surv, self.sdist, 
                           self.dead, self.ddist, self.pnorm, 
                           maxfrac, minsig)



    # compare cluster distances to check if they are 
    # separated enoug
    def compclu(self) :

        self.cludist, self.sigms, self.sigmd = \
         compclu.compareclus(self.clus, self.clud, 
                             1e-6, self.pnorm, 
                             self.surv, self.dead)

    def classify(self, filename='', 
      pcontnormfull=['Age_norm', 'logfare_norm', 'ticketnum_mod_norm'],
      pdiscfull=['Pclass', 'numsex', 'Parch', 'SibSp', 'ticketnum_grp_clip']) :

        #Continuous params: pcontnormfull
        deltapcont = pd.Series(dtype=np.float64, index=pcontnormfull)
        for par in pcontnormfull :
            deltapcont[par] = self.train.loc[:,par].max() - self.train.loc[:,par].min()
            
        #Discrete parameters: pdiscfull

        newpoint = pd.DataFrame(columns=self.pname+self.pnorm, index=[0])
        self.expn=pd.DataFrame(columns=['ns','nd','Survived'],
                               index=self.testdata.index)
        for ii in self.testdata.index:
            
            newpoint.loc[0,self.pname+self.pnorm] = self.testdata.loc[ii,self.pname+self.pnorm]
        
            pcontnorm = list( self.testdata.loc[ii,pcontnormfull].index[ self.testdata.loc[ii,pcontnormfull].notnull() ] )
            pdisc = list( self.testdata.loc[ii,pdiscfull].index[ self.testdata.loc[ii,pdiscfull].notnull() ] )
        
        
            
            self.expn.loc[ii,:] = expectedn3.expnd(newpoint, self.clus, self.clud, 
                    self.sigms, self.sigmd, 
                    pdisc, pdiscfull, pcontnorm, pcontnormfull, 
                    deltapcont, self.surv, self.dead)
        
        
        
        
        if filename != '' :
            self.expn.loc[:,'Survived'].to_csv(filename, header=True, index=True)



    # split training data in order to test the model
    # on part of it
    def splitdata(self, frac=0.2, seed=None, train=None) :

        if train is None :
            train = self.train
        
        np.random.seed(seed)
        
        # which data points to extract
        where_extracted = pd.Series( np.random.random(train.shape[0]) < frac,
                                     index=train.index)
        
        # test data set where we can verify our accuracy
        self.testsample = train.loc[where_extracted,:]
        
        # reduced size training dataset
        self.minitrain= train.loc[~where_extracted,:]
        self.minisurv = self.minitrain.loc[self.minitrain.Survived==True, 
                                           self.pname+self.pnorm]
        self.minidead = self.minitrain.loc[self.minitrain.Survived==False, 
                                           self.pname+self.pnorm]
        


    
    
    
    
    
    
    # decision tree model
    def dectree(self, train=None, pars=None, compare2=None, 
                predict=True) :
        
        if train is None : 
            train = self.train
            
        if pars is None :
            pars = self.pnorm
        
        train = train.dropna()
        
        # classifier object
        clf = DecisTree()
        
        # Learning from the data 
        clf.fit(train.loc[:, pars], train.loc[:, 'Survived'])
        
        if predict : 
            
            # Predict
            predarr = clf.predict(self.test2.loc[:, pars])
    
            # Save predictions as pandas data frame        
            pred2 = pd.Series(predarr, index=self.test2.index)
            self.pred = pd.Series(-1, dtype=np.int, index=self.testdata.index)
            self.pred.loc[pred2.index] = pred2
            
            # run prediction for data points with NaN values for
            # some parameters
            # this will be true bc test data has some
            # NaN values that were excluded in self.test2
            if (self.pred==-1).sum() > 0 :
                
                testdata = self.testdata.loc[:,pars]
                
                # data points (rows) where there's at least 1 NaN value
                # these will make up our testing dataset
                nanrow_criteria = testdata.isna().any(axis=1)
                
                # the parameters (columns) where there's at least 1 NaN
                # value. Exclude them in the training and predicting
                # the "~" character is a logical NOT
                column_criteria = ~testdata.isna().any()
                
                # select test data by excluding bad columns, and only
                # selecting rows which have not been predicted above
                testdata_wherena = testdata.loc[nanrow_criteria, 
                                                     column_criteria]
                
                # classifier object
                clf2 = DecisTree()
                
                # train the data
                clf2.fit(train.loc[:, testdata_wherena.columns], 
                         train.loc[:,'Survived'])
                
                # predict
                predna_arr = clf2.predict(testdata_wherena)
                
                # save as pandas series
                predna = pd.Series(predna_arr, index = testdata_wherena.index)
                
                # save in the original pred pandas series
                self.pred.loc[testdata_wherena.index] = predna
        
        
        # if we want to check the accuracy of the prediction
        # against a test sample
        if compare2 is not None :
            
            compare2 = compare2.dropna()
            
            predcomp_arr = clf.predict(compare2.loc[:, pars])
        
            predcomp = pd.Series(predcomp_arr, index=compare2.index)
            
            ncorrect = (predcomp==compare2.loc[:,'Survived']).sum()
            
            print('\nAcccuracy = ', ncorrect / predcomp.shape[0])
            print('+/- ', ncorrect**0.5 / predcomp.shape[0])
        
        
        
    def randomforest(self, train=None, pars=None, compare2=None, **kwargs) :
        
        if train is None : 
            train = self.train
            
        if pars is None :
            pars = self.pnorm
        
        # drop NaN from train, test data
        train = train.dropna()
        testdata = self.testdata.loc[:,pars].dropna()
        
        # classifier object
        clf = RndFrst(**kwargs)
        
        # Learning from the data 
        clf.fit(train.loc[:, pars], train.loc[:, 'Survived'])
        
        # Predict
        predarr = clf.predict(testdata)

        # Save predictions as pandas data frame        
        pred2 = pd.Series(predarr, index=testdata.index)
        self.predforest = pd.Series(-1, dtype=np.int, index=self.testdata.index)
        self.predforest.loc[pred2.index] = pred2
        
        # run prediction for data points with NaN values for
        # some parameters
        # this will be true bc test data has some
        # NaN values that were excluded in self.test2
        if (self.predforest==-1).sum() > 0 :
            
            testdata = self.testdata.loc[:,pars]
            
            # data points (rows) where there's at least 1 NaN value
            # these will make up our testing dataset
            nanrow_criteria = testdata.isna().any(axis=1)
            
            # the parameters (columns) where there's at least 1 NaN
            # value. Exclude them in the training and predicting
            # the "~" character is a logical NOT
            column_criteria = ~testdata.isna().any()
            
            # select test data by excluding bad columns, and only
            # selecting rows which have not been predicted above
            testdata_wherena = testdata.loc[nanrow_criteria, 
                                                 column_criteria]
            
            # classifier object
            clf2 = RndFrst(**kwargs)
            
            # train the data
            clf2.fit(train.loc[:, testdata_wherena.columns], 
                     train.loc[:,'Survived'])
            
            # predict
            predna_arr = clf2.predict(testdata_wherena)
            
            # save as pandas series
            predna = pd.Series(predna_arr, index = testdata_wherena.index)
            
            # save in the original pred pandas series
            self.predforest.loc[testdata_wherena.index] = predna
        
        
        # if we want to check the accuracy of the prediction
        # against a test sample
        if compare2 is not None :
            
            compare2 = compare2.dropna()
            
            predcomp_arr = clf.predict(compare2.loc[:, pars])
        
            predcomp = pd.Series(predcomp_arr, index=compare2.index)
            
            ncorrect = (predcomp==compare2.loc[:,'Survived']).sum()
            
            print('\nAcccuracy = ', ncorrect / predcomp.shape[0])
            print('+/- ', ncorrect**0.5 / predcomp.shape[0])     
        
        
        
        
        
    def pairplot(self, pars=None, train=None, hist=False,
        pdiscfull=['Pclass', 'Pclass_norm', 'numsex', 'numsex_norm', 
                   'ticketnum_grp_clip', 'ticketnum_grp_clip_norm']) :
        
        if pars is None : pars = self.pnorm
        
        fig, axs = pl.subplots(len(pars),len(pars))
        
        # data with all columns minus NaN values in Age and ticket number
        if train is None : 
            train = self.train.loc[:,pars+['Survived']] #.dropna()
        #else :
        #    train = train.dropna()

        surv = train[train.Survived==1]
        dead = train[train.Survived==0]
        
        for jj in range(len(pars)-1) :
            for ii in range(jj+1,len(pars)) :
                
                train2 = train.dropna(subset=[pars[ii],pars[jj]])
                surv2 = surv.dropna(subset=[pars[ii],pars[jj]])
                dead2 = dead.dropna(subset=[pars[ii],pars[jj]])
                
                # both parameters are discrete
                if pars[ii] in pdiscfull and pars[jj] in pdiscfull :
                    
                    binsi = np.sort(train2[pars[ii]].unique())
                    binsj = np.sort(train2[pars[jj]].unique())
                    deltai = (binsi[1:] - binsi[0:-1]).min()
                    deltaj = (binsj[1:] - binsj[0:-1]).min()
                    binsi = binsi - deltai/2
                    binsj = binsj - deltaj/2
                    binsi = np.append(binsi, binsi.max()+deltai/2)
                    binsj = np.append(binsj, binsj.max()+deltaj/2)

#                    binsi = [bb-deltai/2 for bb in binsi]
#                    binsi.append(max(binsi)+binsi/2)
#                    binsj = [bb-deltaj/2 for bb in binsj]
#                    binsj.append(max(binsj)+binsj/2)
                    
                    htot, xx, yy = np.histogram2d(
                            train2[pars[ii]], train2[pars[jj]],
                            bins=[binsi,binsj] )

                    hsurv, x2, y2 = np.histogram2d(surv2[pars[ii]], 
                                                   surv2[pars[jj]],
                                                   bins=[xx,yy])

                    axs[ii,jj].imshow((hsurv/htot), 
                              extent=(yy.min(), yy.max(), 
                                      xx.min(), xx.max()),
                              cmap='RdYlGn', aspect='auto', 
                              origin='lower')
                
                elif hist :
                    
                    if   pars[ii] in pdiscfull :
                        binsi = np.sort(train2[pars[ii]].unique())
                        binsj = np.linspace(train2[pars[jj]].min(), 
                                            train2[pars[jj]].max(), 10)
                    elif pars[jj] in pdiscfull :
                        binsj = np.sort(train2[pars[jj]].unique())
                        binsi = np.linspace(train2[pars[ii]].min(), 
                                            train2[pars[ii]].max(), 10)
                    else : 
                        binsi = np.linspace(train2[pars[ii]].min(), 
                                            train2[pars[ii]].max(), 10)
                        binsj = np.linspace(train2[pars[jj]].min(), 
                                            train2[pars[jj]].max(), 10)
                        
                    deltai = (binsi[1:] - binsi[0:-1]).min()
                    binsi = binsi - deltai/2
                    binsi = np.append(binsi, binsi.max()+deltai/2)
                        
                    deltaj = (binsj[1:] - binsj[0:-1]).min()
                    binsj = binsj - deltaj/2
                    binsj = np.append(binsj, binsj.max()+deltaj/2)
                    
                    htot, xx, yy = np.histogram2d(
                            train2[pars[ii]], train2[pars[jj]],
                            bins=[binsi,binsj] )

                    hsurv, x2, y2 = np.histogram2d(surv2[pars[ii]], 
                                                   surv2[pars[jj]],
                                                   bins=[xx,yy])

                    axs[ii,jj].imshow((hsurv/htot), 
                              extent=(yy.min(), yy.max(), 
                                      xx.min(), xx.max()),
                              cmap='RdYlGn', aspect='auto', 
                              origin='lower')
                              
                else :
                    
                    axs[ii,jj].scatter(surv2[pars[jj]], 
                                       surv2[pars[ii]], color='green', s=1,
                                       alpha=0.4)
                    axs[ii,jj].scatter(dead2[pars[jj]], 
                                       dead2[pars[ii]], color='red', s=1,
                                       alpha=0.4)
                    
                # labels
                if jj==0 : 
                    axs[ii,jj].set_ylabel(pars[ii], size='x-small', 
                                          rotation='horizontal')
                if ii==len(pars)-1 :
                    axs[ii,jj].set_xlabel(pars[jj], size='x-small')
                
                # tick label font size + parameters
                axs[ii,jj].tick_params(labelsize='x-small', direction='in')
            
            # histogram on the diagonal
            surv2 = surv.dropna(subset=[pars[jj]])
            dead2 = dead.dropna(subset=[pars[jj]])
            axs[jj,jj].hist(surv2[pars[jj]], color='green', alpha=0.5)
            axs[jj,jj].hist(dead2[pars[jj]], color='red', alpha=0.5)
            axs[jj,jj].tick_params(labelsize='x-small', direction='in')
            if jj==0 : 
                axs[jj,jj].set_ylabel(pars[jj], size='x-small', rotation='horizontal')
            
        # histogram on diagonal for the last parameter
        ii=len(pars)-1

        surv2 = surv.dropna(subset=[pars[ii]])
        dead2 = dead.dropna(subset=[pars[ii]])
        axs[ii,ii].hist(surv2[pars[ii]], color='green', alpha=0.5)
        axs[ii,ii].hist(dead2[pars[ii]], color='red', alpha=0.5)
        axs[ii,ii].set_xlabel(pars[ii], size='x-small')
        axs[ii,ii].tick_params(labelsize='x-small', direction='in')
        
        #clear empty plots
        for ii in range(len(pars)-1) :
            for jj in range(ii+1,len(pars)) :
                axs[ii,jj].axis('off')