# -*- coding: utf-8 -*-
"""
It calculates the failure probability of the problem seen in
Bucher, C. G., &; Bourgund, U. (1990). A fast and efficient response surface 
approach for structural reliability problems (Vol. 7).
"""
import numpy as np
from functions import SVM
from functions import limit_state_g2

#%%
np.random.seed(10)
n = 1_000 # number of samples

#           [kN/m] [kN/m2]    [m4]
variables = [ 'q',    'E',    'I']
means     = [  10,    2e7,   8e-4]
stds      = [ 0.4,  0.5e7, 1.5e-4]

# uniform variables for scanning the space: mean +- 4*standard deviation
for i in range(len(variables)):
    exec(f'{variables[i]} = np.random.uniform({means[i] - 4*stds[i]}, {means[i] + 4*stds[i]}, size=n)')
# vector of random variables
X_test = np.c_[q, E, I]

X_train = np.array([[ 1e8,  1e-8],
                    [1e-8,   1e8],
                    [1e-8,   1e8]]).T
y_train = np.array([ 0, 1])

scaler, clf_svm, calls = SVM(X_test, X_train, y_train, gamma=8, problem=2)

#%% Probability of failure
n = 1_000_000

for i in range(len(variables)):
    exec(f'{variables[i]} = np.random.normal({means[i]}, {stds[i]}, size=n)')
    
X_test = np.c_[q, E, I]
gMCS   = limit_state_g2(X_test) <= 0 # does the sample fail?
Pf_MCS = np.mean(gMCS)

# samples are evaluated in the classifier
X_test_scaled = scaler.transform(X_test)
gSVM   = clf_svm.predict(X_test_scaled)
Pf_SVM = np.mean(gSVM)

print(f'Probability of failure MCS = {Pf_MCS}')
print(f'Probability of failure SVM = {Pf_SVM}')
print(f'Calls = {calls}')
print(f'Number of support vectors:\n Safe {clf_svm.n_support_[0]}' +
      f'\n Unsafe {clf_svm.n_support_[1]}')