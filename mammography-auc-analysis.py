# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

from sklearn.neighbors import LocalOutlierFactor
from sklearn import metrics

np.random.seed(42)

# %%
data = pd.read_csv('mammography.csv', header=None)

# %%
def predictAndEvaluate(X, ground_truth):
    clf = LocalOutlierFactor()#n_neighbors=20, contamination=0.1)
    y_pred = clf.fit_predict(X)
    fpr, tpr, thresholds = metrics.roc_curve(ground_truth, clf.negative_outlier_factor_, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    f1 = metrics.f1_score(ground_truth, y_pred)
    return (auc, f1)

# %%
results = [] # (name, auc, f1)

# %%
## Need to flip class label, LOF outputs 1 for an inlier and -1 for an outlier
ground_truth = data.iloc[:,-1].apply(lambda x: -int(x.strip("'"))).values
data.iloc[:, -1] = ground_truth
X = data.values[:,:-1]
print(data.shape, X.shape, ground_truth.shape)

# %%
data

# %%
counter = Counter(ground_truth)
for k,v in counter.items():
	per = v / len(ground_truth) * 100
	print('Class=%s, Count=%d, Percentage=%.3f%%' % (k, v, per))

# %%


# %%
auc, f1 = predictAndEvaluate(X, ground_truth)
results.append(("Full data", auc, f1))
print(auc, f1)

# %%
dedupData = data.drop_duplicates()
ground_truth = dedupData.values[:, -1]
X = dedupData.values[:,:-1]
print(dedupData.shape, X.shape)

# %%
auc, f1 = predictAndEvaluate(X, ground_truth)
results.append(("Dedup data", auc, f1))
print(auc, f1)

# %%


# %%
shuffled = data.sample(frac=1).reset_index(drop=True)
ground_truth = shuffled.values[:, -1]
X = shuffled.values[:,:-1]
print(shuffled.shape, X.shape)

# %%
auc, f1 = predictAndEvaluate(X, ground_truth)
results.append(("Full data shuffled", auc, f1))
print(auc, f1)

# %%


# %%
shuffledDedup = shuffled.drop_duplicates()
ground_truth = shuffledDedup.values[:, -1]
X = shuffledDedup.values[:,:-1]
print(dedupData.shape, X.shape)

# %%
auc, f1 = predictAndEvaluate(X, ground_truth)
results.append(("Dedup data shuffled", auc, f1))
print(auc, f1)

# %%
results

# %%
print("%-20s %5s %5s" %("Dataset", "AUC", "F1"))
for s, auc, f1 in results:
    print("%-20s %.3f %.3f" %(s, auc, f1))

# %%



