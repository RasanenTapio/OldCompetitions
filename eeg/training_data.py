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
    aa = [];
    bb = []
    #pituus
    # floor (pituus/16000)
    loops = range(1,int(math.floor(len(sarake)/16000)))
    max_loops = int(len(sarake) - math.floor(len(sarake)/16000)*16000)
    
    wave2 = pywt.Wavelet('db4')
    
    sarake1 = np.array(sarake[0:16000])

    (cA6, cD6), (cA5, cD5), (cA4, cD4), (cA3, cD3), (cA2, cD2), (cA1, cD1) = pywt.swt(sarake1, wave2, level=6)
    aa = np.column_stack((cD1,cD2,cD3,cD4,cD5,cD6))
    
    for loop in loops:
    
        # loop yli len(col) mod 16000 tai 8000
        sarake1 = np.array(sarake[16000*loop:16000*(loop+1)])
    
        (cA6, cD6), (cA5, cD5), (cA4, cD4), (cA3, cD3), (cA2, cD2), (cA1, cD1) = pywt.swt(sarake1, wave2, level=6)
        bb = np.column_stack((cD1,cD2,cD3,cD4,cD5,cD6))
        aa = np.vstack((aa,bb))
    
    # lopussa valitaan viimeiset 16000
    # ja sijoitetaan kohtaan [(loops*16000):len(sarake)]
    sarake2 = np.array(sarake[(len(sarake) - 16000):len(sarake)])
    (cA6, cD6), (cA5, cD5), (cA4, cD4), (cA3, cD3), (cA2, cD2), (cA1, cD1) = pywt.swt(sarake2, wave2, level=6)
    bb = np.column_stack((cD1,cD2,cD3,cD4,cD5,cD6))[(len(cD1)-max_loops):len(cD1)]
    
    aa = np.vstack((aa,bb))
    #aa = np.concatenate((aa,bb), axis=0)
    #bb.append(np.concatenate(aa2, axis=0))
    return aa
    
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
    
    X_test = np.concatenate((X_test),axis=0)
    X_train = np.concatenate((X_train),axis=0)
    
    print X_test_wave.shape

    print("Transforming test")    
    X_test_wave = muunna_frame(X_test_wave)

    print("Transforming train")    
    X_train_wave = muunna_frame(X_train_wave)
        
    # Normalization?
    #X_train, scaler = compute_features(X_train)  
    #X_test = compute_features(X_test, scaler)   #pass the learned mean and std to normalized test data
   
    # test SVM for 2 first subjects
    if subject in subjects:

        # write file
        submission_file1 = 'datae/train'+str(subject)+'.csv'
        submission_file2 = 'datae/test'+str(subject)+'.csv'
        
        # create pandas object for submission
        #submission1 = pd.DataFrame(data=np.concatenate((y,X_train, X_train_wave),axis=1))
        # write file
        #submission1.to_csv(submission_file1,index_label='id',float_format='%.3f')
               
        #submission2 = pd.DataFrame(index=np.concatenate(idx), data=np.concatenate((X_test,X_test_wave),axis=1))
        # write file
        #submission2.to_csv(submission_file2,index_label='id',float_format='%.3f')
        print('subject '+str(subject)+' saved')

