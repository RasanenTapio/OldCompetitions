#forked from Gilberto Titericz Junior
#data preprocessing done with Kaggle Scripts

import pandas as pd
import numpy as np
from sklearn import ensemble, preprocessing
import xgboost as xgb

# set operating system path
os.chdir("/media/winter/DA70C3D670C3B791/catedata")

# load test data to get idx
test = pd.read_csv('./input/test_set.csv', parse_dates=[3,])

# drop useless columns and create labels
idx = test.id.values.astype(int)

# convert data to numpy array
train = np.array(train)
test = np.array(test)

# Load train and test data and labels
train = genfromtxt('./data/train.csv', delimiter=',')
test = genfromtxt('./data/test.csv', delimiter=',')
labels = genfromtxt('./data/labels.csv', delimiter=',')

print(labels)
print(test)
print(train)
print("Loading data succesfull!")

# i like to train on log(1+x) for RMSLE ;) 
# The choice is yours :)
label_log = np.log1p(labels)

# fit a random forest model

params = {}
params["objective"] = "reg:linear"
params["eta"] = 0.02
params["min_child_weight"] = 6
params["subsample"] = 0.7
params["colsample_bytree"] = 0.6
params["scale_pos_weight"] = 0.8
params["silent"] = 1
params["max_depth"] = 8
params["max_delta_step"]=2

plst = list(params.items())

xgtrain = xgb.DMatrix(train, label=label_log)
xgtest = xgb.DMatrix(test)

print('2000')

num_rounds = 2000
model = xgb.train(plst, xgtrain, num_rounds)
preds1 = model.predict(xgtest)

print('3000')

num_rounds = 3000
model = xgb.train(plst, xgtrain, num_rounds)
preds2 = model.predict(xgtest)

print('4000')

num_rounds = 4000
model = xgb.train(plst, xgtrain, num_rounds)
preds4 = model.predict(xgtest)

#label_log = np.power(labels,1/16)
label_log = np.power(labels,1.0/16.0)

xgtrain = xgb.DMatrix(train, label=label_log)
xgtest = xgb.DMatrix(test)

print('power 1/16 4000')

num_rounds = 4000
model = xgb.train(plst, xgtrain, num_rounds)
preds3 = model.predict(xgtest)

#for loop in range(2):
#    model = xgb.train(plst, xgtrain, num_rounds)
#    preds1 = preds1 + model.predict(xgtest)
#preds = (0.55*np.expm1( (preds1+preds2+preds4)/3))+(0.45*np.power(preds3,16))
#preds = (0.60*np.expm1( (preds1+preds2+preds4)/3))+(0.40*np.power(preds3,16.0))
preds = 0.4*np.expm1(preds4)+.1*np.expm1(preds1)+0.1*np.expm1(preds2)+0.4*np.power(preds3,16)

preds = pd.DataFrame({"id": idx, "cost": preds})
preds.to_csv('./results/xgboost_ensebmle_test4', index=False)