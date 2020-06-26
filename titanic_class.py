
# based on comparecluster2.py

import pandas as pd
import numpy as np

import cluster6
import compclu
import expectedn3
import distmatrix7

# order of functions to run
# initialize 
# --> .calcdist() 
# --> .cluster()
# --> compclu() 
# --> classify()

class Titanic :
    
    def __init__(self) :
        
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
        self.pname = ['Pclass', 'Age', 'numsex', 
                      'SibSp', 'Parch', 'logfare',
                      'ticketnum_mod', 'ticketnum_grp_clip']
        
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
        
        
        # Clean data from NaN values in Age and ticketnum
        # The other columns of interest don't have NaN values
        self.test2 = pd.DataFrame( 
                self.testdata[self.testdata.Age.notnull() & 
                              self.testdata.ticketnum.notnull()] )
        
        """
        # the data points without age
        self.test3 = pd.DataFrame( self.testdata.loc[self.testdata.Age.isnull(),:] )
        self.pname3 = ['Pclass', 'numsex', 'SibSp', 'Parch', 'logfare', 'ticketnum']
        self.pnorm3 = ['Pclass_norm', 'numsex_norm', 'SibSp_norm', 
                       'Parch_norm', 'logfare_norm', 'ticketnum']
        
        # all data
        self.test4 = pd.DataFrame(self.testdata.loc[:,self.pname+self.pnorm])
        """




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



