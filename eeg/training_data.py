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
from sklearn.preprocessing import StandardScaler
import pywt

#############function to read data###########
FNAME = "/media/winter/DA70C3D670C3B791/eegdata/data/input/{0}/subj{1}_series{2}_{3}.csv"
def load_data(subj, series=range(1,9), prefix = 'train', opts='def'):
    data = [pd.read_csv(FNAME.format(prefix,subject,s,'data'), index_col=0) for s in series]
    idx = [d.index for d in data]
    data = [d.values.astype(float) for d in data]
    if prefix == 'train' and opts == 'def':
        events = [pd.read_csv(FNAME.format(prefix,subject,s,'events'), index_col=0).values for s in series]
        return data, events
    elif prefix == 'train' and opts == 'stacking':
        events = [pd.read_csv(FNAME.format(prefix,subject,s,'events'), index_col=0).values for s in series]
        return data, idx
    else:
        return data, idx

def compute_features(X, scale=None):
    X0 = [x[:,0] for x in X]
    X = np.concatenate(X, axis=0)
    F = [];
    for fc in np.linspace(0,1,41)[1:]:
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
    
def is_odd(num):
    return num & 0x1

def muunna_frame(X):
    F = [];
    for column in X.T:
        F.append(muunna(column))
    F = np.concatenate(F, axis=1)
    return F
    

def muunna(sarake):
    wave1 = pywt.Wavelet('haar')
    #wave2 = pywt.Wavelet('db4')
    aa1 = pywt.swt(np.array(sarake), wave1, level=1)
    #aa2 = pywt.swt(np.array(sarake), wave2, level=1)
    bb = []
    bb.append(np.concatenate(aa1, axis=0))
    #bb.append(np.concatenate(aa2, axis=0))
    return np.concatenate((bb), axis=0).T
    
#%%########### Initialize ####################################################
os.chdir("/media/winter/DA70C3D670C3B791/eegdata")
cols = ['HandStart','FirstDigitTouch',
        'BothStartLoadPhase','LiftOff',
        'Replace','BothReleased']

subjects = range(1,13)

###loop on subjects and 8 series for train data + 2 series for test data
for subject in subjects:

    print('subject '+str(subject)+' loading')
    X_train, y = load_data(subject)
    X_test, idx = load_data(subject,[9,10],'test')


################ Train classifiers ###########################################
    
    X_train_wave = np.concatenate(X_train,axis=0)
    X_test_wave = np.concatenate(X_test,axis=0)
    y = np.concatenate(y,axis=0)
    
    trigger = 0 
    
    if is_odd(len(X_train_wave[:,0])):
        print("Train rows "+ str(X_train_wave.shape[0])+". Train data is odd, correcting")
        X_train_wave = X_train_wave[:-1, :]
        y = y[:-1, :]
        
    if is_odd(len(X_test_wave[:,0])):
        print("Test rows "+ str(X_test_wave.shape[0])+". Test data is odd, correcting")
        X_test_wave = X_test_wave[:-1, :]
        trigger = 1

    print pywt.swt_max_level(X_test_wave.shape[0])
    print X_test_wave.shape

    print("Transforming test")    
    X_test_wave = muunna_frame(X_test_wave)

    print("Transforming train")    
    X_train_wave = muunna_frame(X_train_wave)
        

    if trigger:
        print("Test rows "+ str(X_test_wave.shape[0]))
        add_line = X_train_wave[0,:]
        X_test_wave = np.vstack((X_test_wave,add_line)) ### JATKA TASTA
        print("Added row. Test rows "+ str(X_test_wave.shape[0]))

    # Normalization?
    X_train, scaler = compute_features(X_train)  
    X_test = compute_features(X_test, scaler)   #pass the learned mean and std to normalized test data
  
   
    # test SVM for 2 first subjects
    if subject in subjects:

        # write file
        submission_file1 = 'datad/train'+str(subject)+'.csv'
        submission_file2 = 'datad/test'+str(subject)+'.csv'
        
        # create pandas object for submission
        submission1 = pd.DataFrame(data=np.concatenate((y,X_train, X_train_wave),axis=1))
        # write file
        submission1.to_csv(submission_file1,index_label='id',float_format='%.3f')
               
        submission2 = pd.DataFrame(index=np.concatenate(idx), data=np.concatenate((X_test,X_test_wave),axis=1))
        # write file
        submission2.to_csv(submission_file2,index_label='id',float_format='%.3f')
        print('subject '+str(subject)+' saved')


