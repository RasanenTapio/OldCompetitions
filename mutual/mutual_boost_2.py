# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 20:13:33 2015

@author: winter
"""

import pandas as pd
import numpy as np 
from sklearn import preprocessing
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier

# set operating system path
os.chdir("/media/winter/DA70C3D670C3B791/mutualdata")

def xgboost_pred(train,labels,test, plst):

    #Add shuffle data

	#Using 5000 rows for early stopping. 
	offset = 6000

	num_rounds = 10000
	xgtest = xgb.DMatrix(test)

	#create a train and validation dmatrices 
	xgtrain = xgb.DMatrix(train[offset:,:], label=labels[offset:])
	xgval = xgb.DMatrix(train[:offset,:], label=labels[:offset])

	#train using early stopping and predict
	watchlist = [(xgtrain, 'train'),(xgval, 'val')]
	model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=1250)
	preds1 = model.predict(xgtest,ntree_limit=model.best_iteration)


	#reverse train and labels and use different 5k for early stopping. 
	# this adds very little to the score but it is an option if you are concerned about using all the data. 
	train = train[::-1,:]
	#labels = np.log1p(labels[::-1])
	labels = np.log(labels[::-1])
    
	xgtrain = xgb.DMatrix(train[offset:,:], label=labels[offset:])
	xgval = xgb.DMatrix(train[:offset,:], label=labels[:offset])

	watchlist = [(xgtrain, 'train'),(xgval, 'val')]
	model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=1250)
	preds2 = model.predict(xgtest,ntree_limit=model.best_iteration)


	#combine predictions
	#since the metric only cares about relative rank we don't need to average
	preds = (preds1)*1.4 + np.expm1(preds2)*8.6
	return preds
 
 #load train and test 
train  = pd.read_csv('./mutualdata/train.csv', index_col=0)
test  = pd.read_csv('./mutualdata/test.csv', index_col=0)
print("Data loaded")

labels = train.Hazard
labels_log = np.log1p(np.array(labels))
train.drop('Hazard', axis=1, inplace=True)

train_s = train
test_s = test


train_s.drop('T2_V10', axis=1, inplace=True)
train_s.drop('T2_V7', axis=1, inplace=True)
train_s.drop('T1_V13', axis=1, inplace=True)
train_s.drop('T1_V10', axis=1, inplace=True)

test_s.drop('T2_V10', axis=1, inplace=True)
test_s.drop('T2_V7', axis=1, inplace=True)
test_s.drop('T1_V13', axis=1, inplace=True)
test_s.drop('T1_V10', axis=1, inplace=True)

columns = train.columns
test_ind = test.index


train_s = np.array(train_s)
test_s = np.array(test_s)

# label encode the categorical variables
for i in range(train_s.shape[1]):
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(train_s[:,i]) + list(test_s[:,i]))
    train_s[:,i] = lbl.transform(train_s[:,i])
    test_s[:,i] = lbl.transform(test_s[:,i])

train_s = train_s.astype(float)
test_s = test_s.astype(float)

# Model 1 parameters
params = {}
params["objective"] = "reg:linear"
params["eta"] = 0.005
params["min_child_weight"] = 6
params["subsample"] = 0.7
params["colsample_bytree"] = 0.7
params["scale_pos_weight"] = 1
params["silent"] = 1
params["max_depth"] = 9

plst_in = list(params.items())


preds1 = xgboost_pred(train_s,labels,test_s, plst_in)

# Model 2 parameters
params = {}
params["objective"] = "reg:linear"
params["eta"] = 0.005
params["min_child_weight"] = 6
params["subsample"] = 0.7
params["colsample_bytree"] = 0.6
params["scale_pos_weight"] = 1
params["silent"] = 1
params["max_depth"] = 9

plst_in = list(params.items())


#preds2 = xgboost_pred(train_s,labels,test_s, plst_in)
#preds2a = xgboost_pred(train_s,labels_log,test_s, plst_in)

#model_3 building

#train = train.T.to_dict().values()
#test = test.T.to_dict().values()

vec = DictVectorizer()
#train = vec.fit_transform(train)
#test = vec.transform(test)
x_dv = vec.fit_transform(train.append(test).T.to_dict().values())
train = x_dv[:len(train), :]
test = x_dv[len(train):, :]

print("Data transformation done")

# Model 3 parameters
params = {}
params["objective"] = "reg:linear"
params["eta"] = 0.005
params["min_child_weight"] = 6
params["subsample"] = 0.7
params["colsample_bytree"] = 0.7
params["scale_pos_weight"] = 1
params["silent"] = 1
params["max_depth"] = 9

plst_in = list(params.items())  

preds3 = xgboost_pred(train,labels,test, plst_in)
#preds3a = xgboost_pred(train,labels,test, plst_in)

# Model 4

#rf = RandomForestClassifier(n_estimators=100, random_state=1)
#print("Fitting RF")
#rf.fit(np.array(train_s), np.array(labels))
#print("Predicting RF")
#preds4 = rf.predict_proba(np.array(test_s))[:,1]

#preds = 0.47 * (preds1**0.045+preds2**0.045+np.expm1(preds2a)**0.045)/3 + 0.53 * (preds3**0.055 + preds3a**0.055)/2
#preds = 0.24 * (preds1**0.045) + 0.23 * (preds2**0.045) + 0.53 * (preds2**0.055)
preds = 0.47 * (preds1**0.045) + 0.53 * (preds2**0.055)
#preds = 0.35 * (preds1**0.045) + 0.45 * (preds3**0.055) + preds4 * 0.2

#generate solution
preds = pd.DataFrame({"Id": test_ind, "Hazard": preds})
preds = preds.set_index('Id')
preds.to_csv('./results/xgboost_offset_test1.csv')