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
#from scipy.fftpack import fft
#############function to read data###########
FNAME = "/media/winter/DA70C3D670C3B791/eegdata/data/input/{0}/subj{1}_series{2}_{3}.csv"
def load_data(subj, series=range(1,9), prefix = 'train'):
    data = [pd.read_csv(FNAME.format(prefix,subject,s,'data'), index_col=0) for s in series]
    idx = [d.index for d in data]
    data = [d.values.astype(float) for d in data]
    if prefix == 'train':
        events = [pd.read_csv(FNAME.format(prefix,subject,s,'events'), index_col=0).values for s in series]
        return data, events
    else:
        return data, idx

def transform_bandpass(column_t):
    freqs = [7, 30]
    b,a = butter(5,np.array(freqs)/250.0,btype='bandpass')
    return lfilter(b,a,column_t)
    
def transform_lowpass(column_t):
    freqs = [20] # 35–70 Hz? # recommend: 20 Hz cut off skalaarina?
    b,a = butter(3,np.array(freqs)/250.0,btype='lowpass')
    return lfilter(b,a,column_t)

def compute_features(X, scale=None):
    X = np.concatenate(X,axis=0)
    df_bandpass = np.apply_along_axis(transform_bandpass, 0, X)
    print("Bandpass filtering done")
    df_lowpass = np.apply_along_axis(transform_lowpass, 0, X)
    print("Lowpass filtering done")
    
    F = [];
    
    F = df_bandpass
    F = np.concatenate((F,df_lowpass,df_lowpass**2), axis=1)
       
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

subjects = range(1,13)
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

    X_train, y = load_data(subject)
    X_test, idx = load_data(subject,[9,10],'test')


################ Train classifiers ###########################################
    lda = LDA() # try? solver='eigen', shrinkage='auto'
    #rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, criterion="entropy", random_state=1)
    #lr = LogisticRegression()
    #clf = svm.SVC(probability=True)
    
    X_train, scaler = compute_features(X_train)
    
    #start_time = time.time()
    # other transformations:
    #for i in X_train.T:
    #    i = fft(i)

    #X_train = np.apply_along_axis(my_func, 0, X_train)
    
    #elapsed_time = time.time() - start_time
    #print(elapsed_time) 
    
    X_test = compute_features(X_test, scaler)   #pass the learned mean and std to normalized test data
    #X_test = np.apply_along_axis(my_func, 0, X_test)
  
    # Try imporvin score by droppin eye movements / blinking
    #X_train = np.delete(X_train, [0,1], 1)
    #X_test = np.delete(X_test, [0,1], 1)
  
    y = np.concatenate(y,axis=0)
    #scores1 = np.empty((X_test.shape[0],6))
    scores2 = np.empty((X_test.shape[0],6))
    #scores3 = np.empty((X_test.shape[0],6))
    #scores4 = np.empty((X_test.shape[0],6))
    #scores5 = np.empty((X_test.shape[0],6))
    
    downsample = 20
    # test SVM for 2 first subjects
    if subject in subjects:
        for i in range(6):
            print('Train subject %d, class %s' % (subject, cols[i]))
            #rf.fit(X_train[::downsample,:], y[::downsample,i])
            lda.fit(X_train[::,:], y[::,i])
            #lr.fit(X_train[::downsample,:], y[::downsample,i])
            #clf.fit(X_train[::downsample,:], y[::downsample,i])  
           
            #scores1[:,i] = rf.predict_proba(X_test)[:,1]
            scores2[:,i] = lda.predict_proba(X_test)[:,1] 
            #scores3[:,i] = lr.predict_proba(X_test)[:,1]
            #scores4[:,i] = clf.predict_proba(X_test)[:,1]
            #scores5[:,i] = clf.predict(X_test)[:,1]

    #scores_tot1.append(scores1)
    scores_tot2.append(scores2)
    #scores_tot3.append(scores3)
    #scores_tot4.append(scores4)
    #scores_tot5.append(scores4)
    idx_tot.append(np.concatenate(idx))
    
#%%########### submission file ################################################
submission_file = 'models/model2_ds0_low2_band1_test1.csv'
# create pandas object for submission
submission = pd.DataFrame(index=np.concatenate(idx_tot),
                          columns=cols,
                          data=np.concatenate(scores_tot2))

# write file
submission.to_csv(submission_file,index_label='id',float_format='%.3f')