# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 15:36:56 2017

@author: mbansa001c
"""

print ('loading libraries and data')

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

train['siteid'].fillna(-999, inplace=True)
test['siteid'].fillna(-999, inplace=True)

train['browserid'].fillna("None", inplace=True)
test['browserid'].fillna("None", inplace=True)

train['devid'].fillna("None", inplace=True)
test['devid'].fillna("None", inplace=True)

train['datetime'] = pd.to_datetime(train['datetime'])
test['datetime'] = pd.to_datetime(test['datetime'])

train['tweekday'] = train['datetime'].dt.weekday
train['thour'] = train['datetime'].dt.hour
train['tminute'] = train['datetime'].dt.minute

test['tweekday'] = test['datetime'].dt.weekday
test['thour'] = test['datetime'].dt.hour
test['tminute'] = test['datetime'].dt.minute



#### training only for the offers in test data #####
train_new=train[train['offerid'].isin(test['offerid'])]

train_a_b=train_new[train_new['countrycode'].isin(['a','b'])]
train_c_f=train_new[train_new['countrycode'].isin(['c','d','e','f'])]

test_a_b=test[test['countrycode'].isin(['a','b'])]
test_c_f=test[test['countrycode'].isin(['c','d','e','f'])]



trainY_df = train_a_b[['ID','click']]

train_a_b.drop(['click'], axis=1, inplace=True)


########join train and test########
df_all = pd.concat([train_a_b, test_a_b])


############ feature engineering, category count,offer count, offer access to users,week and hour level counts############
catg_cnt=df_all.groupby(['siteid'])['category'].nunique().reset_index().rename(columns = {'category':'catg_cnt'})
offer_cnt=df_all.groupby(['siteid'])['offerid'].nunique().reset_index().rename(columns = {'offerid':'offer_cnt'})
offer_id_cnt=df_all.groupby(['siteid','offerid'])['ID'].nunique().reset_index().rename(columns = {'ID':'offer_id_cnt'})

df1=pd.merge(df_all, catg_cnt, how='left', on=['siteid'])
df2=pd.merge(df1, offer_cnt, how='left', on=['siteid'])
df3=pd.merge(df2, offer_id_cnt, how='left', on=['siteid','offerid'])


df3['site_user_cnt'] = df3.groupby(['siteid'])['ID'].transform('count')
df3['user_week_cnt'] = df3.groupby(['siteid','tweekday'])['ID'].transform('count')
df3['user_hr_cnt'] = df3.groupby(['siteid','tweekday','thour'])['ID'].transform('count')

df3['offer_user_share']=df3['offer_id_cnt']/df3['site_user_cnt']
df3['week_user_share']=df3['user_week_cnt']/df3['site_user_cnt']
df3['hr_user_share']=df3['user_hr_cnt']/df3['user_week_cnt']
##################

cols = ['siteid','offerid','category','merchant']

for x in cols:
    df3[x] = df3[x].astype('object')
    

cols_to_use1=['siteid','user_hr_cnt','week_user_share','hr_user_share','merchant',	'catg_cnt',	'user_week_cnt',	'offer_cnt',	'site_user_cnt',		'browserid',	'tweekday',	'devid',	'offerid',	'thour',	'category','countrycode','offer_user_share','tminute']


###########splitting train test after feature creation###########
num_train = len(train_a_b)

X_train = df3[:num_train]
X_test = df3[num_train:]

train_target=pd.merge(X_train,trainY_df,how='inner',on=['ID'])

sampled=train_target.sample(frac=0.3)

trainX = sampled[cols_to_use1]
trainX.head()
trainY = sampled['click']

# catboost accepts categorical variables as indexes

cat_cols = [0,4,9,10,11,12,13,14,15]


print ('train model........')
X_train1, X_test1, y_train1, y_test1 = train_test_split(trainX, trainY, test_size = 0.40)
model = CatBoostClassifier(depth=8, iterations=200, learning_rate=0.1, eval_metric='AUC', random_seed=1,calc_feature_importance=True)

model.fit(X_train1
          ,y_train1
          ,cat_features=cat_cols
          ,eval_set = (X_test1, y_test1)
          ,use_best_model = True
         )

preds_class = model.predict(X_test1)
print("accuracy = {}".format(accuracy_score(y_test1, preds_class)))
print(model.feature_importance_)

print ('making predictions.................')
#### predictions for test #########
pred_a_b = model.predict_proba(X_test[cols_to_use1])[:,1]
sub_a_b=pd.DataFrame({'ID':test_a_b['ID'],'click':pred_a_b})


################################ Repeat from here for c,d,e,f countries again using train_c_f and test_c_f##############
trainY_df = train_c_f[['ID','click']]

train_c_f.drop(['click'], axis=1, inplace=True)


########join train and test########
df_all = pd.concat([train_c_f, test_c_f])

############ feature engineering, category count,offer count, offer access to users,week and hour level counts############
catg_cnt=df_all.groupby(['siteid'])['category'].nunique().reset_index().rename(columns = {'category':'catg_cnt'})
offer_cnt=df_all.groupby(['siteid'])['offerid'].nunique().reset_index().rename(columns = {'offerid':'offer_cnt'})
offer_id_cnt=df_all.groupby(['siteid','offerid'])['ID'].nunique().reset_index().rename(columns = {'ID':'offer_id_cnt'})

df1=pd.merge(df_all, catg_cnt, how='left', on=['siteid'])
df2=pd.merge(df1, offer_cnt, how='left', on=['siteid'])
df3=pd.merge(df2, offer_id_cnt, how='left', on=['siteid','offerid'])


df3['site_user_cnt'] = df3.groupby(['siteid'])['ID'].transform('count')
df3['user_week_cnt'] = df3.groupby(['siteid','tweekday'])['ID'].transform('count')
df3['user_hr_cnt'] = df3.groupby(['siteid','tweekday','thour'])['ID'].transform('count')

df3['offer_user_share']=df3['offer_id_cnt']/df3['site_user_cnt']
df3['week_user_share']=df3['user_week_cnt']/df3['site_user_cnt']
df3['hr_user_share']=df3['user_hr_cnt']/df3['user_week_cnt']
##################

cols = ['siteid','offerid','category','merchant']

for x in cols:
    df3[x] = df3[x].astype('object')
    

cols_to_use1=['siteid','user_hr_cnt','week_user_share','hr_user_share','merchant',	'catg_cnt',	'user_week_cnt',	'offer_cnt',	'site_user_cnt',		'browserid',	'tweekday',	'devid',	'offerid',	'thour',	'category','countrycode','offer_user_share','tminute']

###########splitting train test after feature creation###########
num_train = len(train_c_f)

X_train = df3[:num_train]
X_test = df3[num_train:]

train_target=pd.merge(X_train,trainY_df,how='inner',on=['ID'])

sampled=train_target.sample(frac=0.3)

trainX = sampled[cols_to_use1]
trainX.head()
trainY = sampled['click']

# catboost accepts categorical variables as indexes

cat_cols = [0,4,9,10,11,12,13,14,15]


X_train1, X_test1, y_train1, y_test1 = train_test_split(trainX, trainY, test_size = 0.40)
model = CatBoostClassifier(depth=8, iterations=200, learning_rate=0.1, eval_metric='AUC', random_seed=1,calc_feature_importance=True)

model.fit(X_train1
          ,y_train1
          ,cat_features=cat_cols
          ,eval_set = (X_test1, y_test1)
          ,use_best_model = True
         )

preds_class = model.predict(X_test1)
print("accuracy = {}".format(accuracy_score(y_test1, preds_class)))
print(model.feature_importance_)

#### predictions for test #########

pred_c_f = model.predict_proba(X_test[cols_to_use1])[:,1]
sub_c_f=pd.DataFrame({'ID':test_c_f['ID'],'click':pred_c_f})


##join predictions after repeating for c to f countries###
final=sub_a_b.append(sub_c_f)

#write output
final.to_csv("submission.csv",index=False)




