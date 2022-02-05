# -*- coding: utf-8 -*-
"""
Module with limit state functions an a general approach for a SVM problem.
"""
import numpy as np
from sklearn.svm import SVC # this will make a SVM for classification
from sklearn.preprocessing import MaxAbsScaler

def limit_state_g1(x):
    # 2D Decision Function
    x1, x2 = np.hsplit(x, x.shape[1])
        
    return 2 - x2 + np.exp(x1**2/(-10)) + (x1/5)**4

def limit_state_g2(x):
    # 3 Span Beam
    q, E, I = np.hsplit(x, x.shape[1])
    L=5
        
    return L/360 - 0.0069*q*L**4/(E*I)

def limit_state_g3(x):
    # Non-linear Oscilator
    m, c1, c2, r, F1, t1 = np.hsplit(x, x.shape[1])
        
    w0 = np.sqrt((c1 + c2)/m)
    return 3 * r - np.abs(2*F1/(m*w0*w0) * np.sin(w0*t1/2))

#%%
def SVM(X_test, X_train, y_train, gamma, problem):
    
    #  set up the SVM and the scaler
    if problem != 1: # problem 1 is already scalated
        scaler = MaxAbsScaler()
        
    clf_svm = SVC(kernel='rbf', gamma=gamma)

    #  store indexes of the training set in the test set
    idxs = np.array([], dtype=int)
    
    calls=0 # this will count how many times the limit state function is called
    
    # Main loop
    while True:
        if problem != 1:
        # transform train and test sets
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled  = scaler.transform(X_test)
            
            # train the SVC
            clf_svm.fit(X_train_scaled, y_train)
        
            # predict the label of the testing set
            yy = clf_svm.decision_function(X_test_scaled)
        else:
          
            # train the SVC
            clf_svm.fit(X_train, y_train)
        
            # predict the label of the testing set
            yy = clf_svm.decision_function(X_test)            
        
        # do not take into account those samples that already belong to the train set
        if len(idxs) > 0:
            yy[idxs] = np.nan
            
        # select the sample that is closer to the margin
        idx = np.nanargmin(np.abs(yy))
        
        # and add it to the indexes array
        idxs = np.append(idxs, idx)
       
        # if the sample is within the margin
        if np.abs(yy[idx]) < 1:
            # evaluate the limit state function for the corresponding problem
            y_idx   = eval(f'limit_state_g{problem}(X_test[[idx]])')        
            
            calls +=1 # the limit state function was called
            # add the label to the train set
            X_train = np.vstack((X_train, X_test[idx]))
            y_train = np.append(y_train, int(y_idx <= 0))
        else:
            break
        # end if
    # end while
    if problem !=1:
        return scaler, clf_svm, calls
    else: # problem 1 (2D Decision Function) was not scalated
        return X_train, y_train, clf_svm, calls