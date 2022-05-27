#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

from sklearn.neighbors import LocalOutlierFactor
from sklearn import metrics
from sklearn.model_selection import train_test_split

np.random.seed(42)


# In[2]:


data = pd.read_csv('mammography.csv', header=None)


# In[3]:


def predictAndEvaluate(X, ground_truth):
    clf = LocalOutlierFactor()#n_neighbors=20, contamination=0.1)
    y_pred = clf.fit_predict(X)
    fpr, tpr, thresholds = metrics.roc_curve(ground_truth, clf.negative_outlier_factor_, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    f1 = metrics.f1_score(ground_truth, y_pred)
    return (auc, f1)


# In[4]:


results = [] # (name, auc, f1)


# In[5]:


## Need to flip class label, LOF outputs 1 for an inlier and -1 for an outlier
ground_truth = data.iloc[:,-1].apply(lambda x: -int(x.strip("'"))).values
data.iloc[:, -1] = ground_truth
X = data.values[:,:-1]
print(data.shape, X.shape, ground_truth.shape)


# In[6]:


data


# In[7]:


counter = Counter(ground_truth)
for k,v in counter.items():
	per = v / len(ground_truth) * 100
	print('Class=%s, Count=%d, Percentage=%.3f%%' % (k, v, per))


# In[ ]:





# In[8]:


auc, f1 = predictAndEvaluate(X, ground_truth)
results.append(("Full data", auc, f1))
print(auc, f1)


# In[ ]:





# In[ ]:





# In[9]:


def dofiftyfiftyTests(data, n):
    results = []
    for i in range(n):
        # split normal dataset 50/50
        X, y = train_test_split(data[data.iloc[:,-1] == 1].values[:,:-1], 
                            train_size=(len(data) // 2), shuffle=True, random_state=42 + i)
        # append all abnormal data to the test set. 
        #No need to shuffle these, LOF does no updating based on these observations
        y_test = np.concatenate((y, data[data.iloc[:, -1] == -1].values[:,:-1]))
        # create ground truth for the test set
        ground_truth = np.concatenate((np.ones(len(y)),-np.ones(sum(data.iloc[:, -1] == -1))))
        clf = LocalOutlierFactor(novelty=True)#n_neighbors=20, contamination=0.1)
        clf.fit(X)
        y_pred = clf.predict(y_test)
        nof = clf.score_samples(y_test)
        fpr, tpr, thresholds = metrics.roc_curve(ground_truth, nof, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        f1 = metrics.f1_score(ground_truth, y_pred)
        results.append((i, auc, f1))
    return results


# In[10]:


n=10
r = dofiftyfiftyTests(data, n)
print("%-5s %5s %5s" %("Run", "AUC", "F1"))
for run, auc, f1 in r:
    print("%-5d %.3f %.3f" %(run, auc, f1))
results.append(("50/50 data avg of %d runs" %n, ) + tuple(np.array(r).mean(axis=0)[1:].tolist()))


# In[ ]:





# In[11]:


n=10
r = dofiftyfiftyTests(data.drop_duplicates(), n)
print("%-5s %5s %5s" %("Run", "AUC", "F1"))
for run, auc, f1 in r:
    print("%-5d %.3f %.3f" %(run, auc, f1))
results.append(("50/50 dedup avg of %d runs" %n, ) + tuple(np.array(r).mean(axis=0)[1:].tolist()))


# In[12]:


print("%-30s %5s %5s" %("Dataset", "AUC", "F1"))
for s, auc, f1 in results:
    print("%-30s %.3f %.3f" %(s, auc, f1))


# In[ ]:




