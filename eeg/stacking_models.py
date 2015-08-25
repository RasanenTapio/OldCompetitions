# -*- coding: utf-8 -*-

"""
Forked from: https://www.kaggle.com/karma86/grasp-and-lift-eeg-detection/rf-lda-lr-v2-1/run/41141
@author Ajoo
forked from Adam Gągol's script based on Elena Cuoco's

"""

import numpy as np
import time
import pandas as pd
from scipy.signal import butter, lfilter
from sklearn.linear_model import LogisticRegression
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

#############function to read data###########
FNAME = "/media/winter/DA70C3D670C3B791/eegdata/data/input/{0}/subj{1}_series{2}_{3}.csv"
def load_data(subj, series=range(1,9), prefix = 'train', opts='def'):
    data = [pd.read_csv(FNAME.format(prefix,subject,s,'data'), index_col=0) for s in series]
    idx = [d.index for d in data]
    data = [d.values.astype(float) for d in data]
    if prefix == 'train' and opts == 'def':
        events = [pd.read_csv(FNAME.format(prefix,subject,s,'events'), index_col=0).values for s in series]
        return data, events, idx
    elif prefix == 'train' and opts == 'stacking':
        events = [pd.read_csv(FNAME.format(prefix,subject,s,'events'), index_col=0).values for s in series]
        return data, idx, events
    else:
        return data, idx

def compute_features(X, scale=None):
    X0 = [x[:,0] for x in X]
    X = np.concatenate(X, axis=0)
    F = [];
    for fc in np.linspace(0,1,11)[1:]:
        b,a = butter(3,fc/250.0,btype='lowpass')
        F.append(np.concatenate([lfilter(b,a,x0) for x0 in X0], axis=0)[:,np.newaxis])
    F = np.concatenate(F, axis=1)
    F = np.concatenate((X,F,F**2), axis=1)
        
    if scale is None:    
        scale = StandardScaler()
        F = scale.fit_transform(F)
        return F, scale
    else:
        F = scale.transform(F)
        return F
    



#%%########### Initialize ####################################################
os.chdir("/media/winter/DA70C3D670C3B791/eegdata")
cols = ['HandStart','FirstDigitTouch',
        'BothStartLoadPhase','LiftOff',
        'Replace','BothReleased']

subjects = range(1,2)
idx_tot = []
scores_tot1 = []
scores_tot2 = []
scores_tot3 = []
scores_tot4 = []
scores_tot5 = []

def my_func(a):
    """Average first and last element of a 1-D array"""
    return np.fft(a)

###loop on subjects and 8 series for train data + 2 series for test data
for subject in subjects:

    X_train, y, idx_train = load_data(subject)
    X_test, idx_test = load_data(subject,[9,10],'test')


################ Train classifiers ###########################################
    rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, criterion="entropy", random_state=1)
    lda = LDA()
    lr = LogisticRegression()
    clf = svm.SVC()
    
    X_train, scaler = compute_features(X_train)
    
    X_test = compute_features(X_test, scaler)   #pass the learned mean and std to normalized test data
  
    y = np.concatenate(y,axis=0)  
    idx_train = np.array(idx_train)  
  
    # Split to train and stacking sets
    X_train, X_stacking = np.array_split(X_train, [int(round(X_train.shape[0]*0.8))])
    y, y_stacking = np.array_split(y, [int(round(y.shape[0]*0.8))])
    #idx_train, idx_stack = np.array_split(idx_train, [int(round(idx_train.shape[0]*0.8))])

    #idx_train = np.asarray(idx_train)
    #idx_stack = np.asarray(idx_stack)

    scores1_test = np.empty((X_test.shape[0],6))
    scores1_stacking = np.empty((X_stacking.shape[0],6))
    scores2_test = np.empty((X_test.shape[0],6))
    scores2_stacking = np.empty((X_stacking.shape[0],6))
    scores3_test = np.empty((X_test.shape[0],6))
    scores3_stacking = np.empty((X_stacking.shape[0],6))
    scores4_test = np.empty((X_test.shape[0],6))
    scores4_stacking = np.empty((X_stacking.shape[0],6))
  
    downsample = 80
    # test SVM for 2 first subjects
    if subject in subjects:
        for i in range(6):
            print('Train subject %d, class %s' % (subject, cols[i]))
            rf.fit(X_train[::downsample,:], y[::downsample,i])
            lda.fit(X_train[::downsample,:], y[::downsample,i])
            lr.fit(X_train[::downsample,:], y[::downsample,i])
            clf.fit(X_train[::downsample,:], y[::downsample,i])  
           
            scores1_test[:,i] = rf.predict(X_test)
            scores1_stacking[:,i] = rf.predict(X_stacking)
            scores2_test[:,i] = lda.predict(X_test) 
            scores2_stacking[:,i] = lda.predict(X_stacking) 
            scores3_test[:,i] = lr.predict(X_test)
            scores3_stacking[:,i] = lr.predict(X_stacking)
            scores4_test[:,i] = clf.predict(X_test)
            scores4_stacking[:,i] = clf.predict(X_stacking)

        # write file

        submission_file1 = 'stacking/train'+str(subject)+'.csv'   # 
        submission_file2 = 'stacking/stacking'+str(subject)+'.csv'  
        submission_file3 = 'stacking/test'+str(subject)+'.csv'
        
        # Data 1st level models are trained on
        #submission1 = pd.DataFrame(data=np.concatenate((y,X_train),axis=1))
        # write file
        #submission1.to_csv(submission_file1,index_label='id',float_format='%.3f')
         
        # Data to validata and test metadata
        submission2 = pd.DataFrame(data=np.concatenate((y_stacking,scores1_stacking, scores2_stacking,
                                                        scores3_stacking, scores4_stacking,X_stacking),axis=1))
        # write file
        submission2.to_csv(submission_file2,index_label='id',float_format='%.3f')       
        
        # 
        submission3 = pd.DataFrame(index=np.concatenate(idx_test), data=np.concatenate((scores1_test,scores2_test,
                                   scores3_test, scores4_test, X_test),axis=1))
        # write file
        submission3.to_csv(submission_file3,index_label='id',float_format='%.3f')