# -*- coding: utf-8 -*-
# pip install mlxtend
"""
It calculates the failure probability of the problem seen in Example 1 of 
Hurtado & Alvarez. Classification Approach for Reliability Analysis with 
Stochastic Finite-Element Modeling. (2003).

Also seen in Example 2 of Guo, Z., & Bai, G. (2009). Application of Least 
Squares Support Vector Machine for Regression to Reliability Analysis. Chinese 
Journal of Aeronautics, 22(2), 160–166.
through an algorithm with Support Vector Machines.

You must install mlxtend library.
"""
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
from functions import SVM
from functions import limit_state_g1
from matplotlib.lines import Line2D

#%%
np.random.seed(20)
n = 1_000 # number of samples

variables = ['x1', 'x2']
means     = [   0,    0]
stds      = [   1,    1]

# uniform variables for scanning the space: mean +- 4*standard deviation
for i in range(len(variables)):
    exec(f'{variables[i]} = np.random.uniform({means[i] - 4*stds[i]}, {means[i] + 4*stds[i]}, size=n)')

X_test = np.c_[ x1 ,  x2]
X_train = np.array([[0,0],
                    [0,4]]).T # known points to start iterating
y_train = np.array( [0, 1] )  # safe and unsafe categories, respectively

X, y, clf_svm, calls = SVM(X_test, X_train, y_train, gamma=0.07, problem=1)
#%%
n = 1_000_000
# generates random variables according to their PDF
x1, x2 = np.random.normal(size=n), np.random.normal(size=n)
# vector of random variables
X_test = np.c_[x1, x2] 
gMCS   = limit_state_g1(X_test) <= 0 # does the sample fail?
Pf_MCS = np.mean(gMCS)

# samples are evaluated in the classifier
gSVM   = clf_svm.predict(X_test)
Pf_SVM = np.mean(gSVM)

print(f'Probability of failure MCS = {Pf_MCS}')
print(f'Probability of failure SVM = {Pf_SVM}')
print(f'Calls = {calls}')
print(f'Number of support vectors:\n Safe {clf_svm.n_support_[0]}' +
                              f'\n Unsafe {clf_svm.n_support_[1]}')

#%% SVM Plot
ax = plot_decision_regions(X, y, clf=clf_svm, legend=0)
plt.xlabel('$x_1$'); plt.ylabel('$x_2$')
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, ['Safe', 'Unsafe'], framealpha=0.3, scatterpoints=1)
plt.savefig('SVM_1e4_1e6.pdf')
plt.show()

#%% MCM Plot
g = np.ndarray.flatten(limit_state_g1(X_test))

fig, ax = plt.subplots()
# Green if it is safe and red if it is unsafe
col = np.where(g<0,'r',np.where(g>0,'g','y'))
scatter = ax.scatter(x1, x2, c=col, s=1)

legend_elements = [Line2D([0], [0], marker='o', color='g', lw=0, label='Safe'),
                   Line2D([0], [0], marker='o', color='r', lw=0, label='Unsafe',
                           markerfacecolor='r', markersize=5)]
ax.legend(handles=legend_elements)
plt.savefig('MCM_2D_Decision_Function.png', dpi=300)
plt.show()