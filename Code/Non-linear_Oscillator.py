# -*- coding: utf-8 -*-
"""
Example 3 Schueremans, L., & Van Gemert, D. (2005). Benefit of splines and 
neural networks in simulation based structural reliability analysis. Structural 
Safety, 27(3), 246–261.
"""

import numpy as np
from functions import SVM
from functions import limit_state_g3

class LN_to_N:
    '''
    Transform the mean and standard deviation of a lognormal distribution to 
    the equivalent parameters of the associated normal distribution.
    '''
    def mean(self, mu, std):
        return np.log(mu**2/np.sqrt(mu**2 + std**2))
    
    def std(self, mu, std):
        return np.sqrt(np.log(1 + std**2/mu**2))
    
# %% Generate the testing set (databank D)
np.random.seed(8)
n = 1_000 # number of samples

variables = [ 'm', 'c1', 'c2',  'r', 'F1', 't1']
means     = [1.00, 1.00, 0.10, 0.50, 1.00, 1.00]
stds      = [0.05, 0.10, 0.01, 0.05, 0.20, 0.20]

for i in range(len(variables)):
    exec(f'{variables[i]} = np.random.uniform({means[i] - 4*stds[i]}, {means[i] + 4*stds[i]}, size=n)')

X_test = np.c_[m, c1, c2, r, F1, t1]

# %% Generate the training set: a failed and a safe sample
X_train = np.array([[ 0.97672417,  0.99922057],
                    [ 0.81513031,  0.66274947],
                    [ 0.09399072,  0.09479305],
                    [ 0.70832328,  0.43579138],
                    [ 0.5830418 ,  1.73360746],
                    [ 0.39082188,  1.60327251]]).T

y_train = np.array([0, 1]) # safe=0 / failed=1

scaler, clf_svm, calls = SVM(X_test, X_train, y_train, gamma='scale', problem=3)
#%%
n = 1_000_000    
    
for i in range(len(variables)):
    exec(f'{variables[i]} = np.random.lognormal(LN_to_N().mean({means[i]}, {stds[i]}),'+
         f' LN_to_N().std({means[i]}, {stds[i]}), size=n)')
    
X_test = np.c_[m, c1, c2, r, F1, t1]
gMCS   = limit_state_g3(X_test) <= 0
Pf_MCS = np.mean(gMCS)

X_test_scaled = scaler.transform(X_test)
gSVM   = clf_svm.predict(X_test_scaled)
Pf_SVM = np.mean(gSVM)

print(f'Probability of failure MCS = {Pf_MCS}')
print(f'Probability of failure SVM = {Pf_SVM}')
print(f'Calls = {calls}')
print(f'Number of support vectors:\n Safe {clf_svm.n_support_[0]}' +
      f'\n Unsafe {clf_svm.n_support_[1]}')