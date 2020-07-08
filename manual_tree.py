
from titanic_class import Titanic

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
         & ( (t.train.numsex==1) | (t.train.logage.lt(1.15))) ].Survived.mean()

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
