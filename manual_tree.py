
import numpy as np

from titanic_class import Titanic
import fisher_projection

t=Titanic()

# all params
t.pairplot(pars=['Pclass', 'logage', 'numsex', 'SibSp', 'Parch_norm',
 'logfare', 'ticketnum_mod', 'ticketnum_grp_clip'], 
  pdiscfull=['Pclass', 'Pclass_norm', 'numsex',
      'numsex_norm', 'ticketnum_grp_clip',
      'ticketnum_grp_clip_norm', 'SibSp_norm',
      'Parch_norm',])

# exclude first and second class ladies
#t.pairplot( train=t.train[(t.train.numsex!=1) | (~t.train.Pclass.isin([1,2]))],
#  pars=['Pclass', 'logage', 'numsex', 'SibSp', 'Parch_norm',
# 'logfare', 'ticketnum_mod', 'ticketnum_grp_clip'], 
#  pdiscfull=['Pclass', 'Pclass_norm', 'numsex',
#      'numsex_norm', 'ticketnum_grp_clip',
#      'ticketnum_grp_clip_norm', 'SibSp_norm',
#      'Parch_norm',])


# 1st/2nd class and (woman or child)
ts1 = t.train[(t.train.Pclass.isin([1,2])) 
         & ( (t.train.numsex==1) | (t.train.logage.lt(1.15))) ]

# Not the above
t2 = t.train[(~t.train.Pclass.isin([1,2])) 
           | ((t.train.numsex!=1) & (t.train.logage.ge(1.15)))]

t.pairplot(train=t2,
  pars=['Pclass', 'logage', 'numsex', 'SibSp', 'Parch_norm',
        'logfare', 'ticketnum_mod', 'ticketnum_grp_clip'], 
  pdiscfull=['Pclass', 'Pclass_norm', 'numsex',
      'numsex_norm', 'ticketnum_grp_clip',
      'ticketnum_grp_clip_norm', 'SibSp_norm',
      'Parch_norm',])

# out of the remaining, pick the children with 2 or fewer siblings
ts2 = t2[(t2.logage.le(1))&(t2.SibSp.le(2))]

# not the above
t3 = t2[(t2.logage.gt(1)) | (t2.SibSp.gt(2))]

t.pairplot(train=t3,
  pars=['Pclass', 'logage', 'numsex', 'SibSp', 'Parch_norm',
        'logfare', 'ticketnum_mod', 'ticketnum_grp_clip'], 
  pdiscfull=['Pclass', 'Pclass_norm', 'numsex',
      'numsex_norm', 'ticketnum_grp_clip',
      'ticketnum_grp_clip_norm', 'SibSp_norm',
      'Parch_norm',])

# sinking! all second class left (all men)
td1 = t3[t3.Pclass==2]

# not the above
t4 = t3[t3.Pclass!=2]
t.pairplot(train=t4,
  pars=['Pclass', 'logage', 'numsex', 'SibSp', 'Parch',
        'logfare', 'ticketnum_mod', 'ticketnum_grp_clip'], 
  pdiscfull=['Pclass', 'Pclass_norm', 'numsex',
      'numsex_norm', 'ticketnum_grp_clip',
      'ticketnum_grp_clip_norm', 'SibSp', 'SibSp_norm',
      'Parch',])

# sinking! log(age)>1.5 & ticket mod >=34000
td2 = t4[t4.ticketnum_mod.ge(34000)&(t4.logage.ge(1.5))]

# not the above
t5 = t4[t4.ticketnum_mod.lt(34000)|(t4.logage.lt(1.5))]
t.pairplot(train=t5, hist=True,
  pars=['Pclass', 'logage', 'numsex', 'SibSp', 'Parch',
        'logfare', 'ticketnum_mod', 'ticketnum_grp_clip'], 
  pdiscfull=['Pclass', 'Pclass_norm', 'numsex',
      'numsex_norm', 'ticketnum_grp_clip',
      'ticketnum_grp_clip_norm', 'SibSp', 'SibSp_norm',
      'Parch',])

# sink
td3 = t5[t5.SibSp.ge(2)&(t5.Parch.le(2))]

# not the above
t6 = t5[t5.SibSp.lt(2)|(t5.Parch.gt(2))]
t.pairplot(train=t6, hist=True,
  pars=['Pclass', 'logage', 'numsex', 'SibSp', 'Parch',
        'logfare', 'ticketnum_mod', 'ticketnum_grp_clip'], 
  pdiscfull=['Pclass', 'Pclass_norm', 'numsex',
      'numsex_norm', 'ticketnum_grp_clip',
      'ticketnum_grp_clip_norm', 'SibSp', 'SibSp_norm',
      'Parch',])

pars = ['Pclass_norm', 'Age_norm', 'numsex_norm', 'SibSp_norm', 'Parch',
        'logfare', 'ticketnum_mod_norm', 'ticketnum_grp_clip_norm']

yy , evect, ld, ax = \
fisher_projection.sepnd( [t6[t6.Survived==1].dropna(subset=pars),
                          t6[t6.Survived==0].dropna(subset=pars)],
                          pars, '', 1, 'gr')

t6dr =t6.dropna(subset=pars)

#yy.y0 < 0.5 is mostly sunken


# writing the above as a tree

tr = Titanic().testdata

#women+children in 1st and 2nd class
s1 = (tr.Pclass.isin([1,2])) & ( (tr.numsex==1) | (tr.logage.lt(1.15)))
tr.loc[s1,'xp']=1

# children w 2 or fewer siblings
s2 = (tr.logage.le(1))&(tr.SibSp.le(2))
tr.loc[(~s1)&(s2),'xp']=1

# sinking! all second class left (all men)
d1 = (tr.Pclass==2)
tr.loc[~((s1)|(s2))&(d1), 'xp']=0

# sinking! log(age)>1.5 & ticket mod >=34000
d2 = (tr.ticketnum_mod.ge(34000))&(tr.logage.ge(1.5))
tr.loc[~((s1)|(s2)|(d1))&(d2),'xp']=0

#sinking
d3=tr.SibSp.ge(2) & (tr.Parch.le(2))
tr.loc[~((s1)|(s2)|(d1)|(d2))&(d3),'xp']=0

# apply fisher disc
therest=~((s1)|(s2)|(d1)|(d2)|(d3))
for ii in tr.index :
    if therest[ii] :
        tr.loc[ii,'y0'] = np.dot(evect.T, tr.loc[ii,pars])
        if tr.y0[ii] < 0.7 : 
            tr.loc[ii,'xp']=0
        else :
            tr.loc[ii,'xp']=1#np.random.randint(0,2)

tr.loc[:,'Survived'] = tr.loc[:,'xp']
tr.loc[:,'Survived'].astype('int32').to_csv('manual_tree3.csv', header=True, index=True)