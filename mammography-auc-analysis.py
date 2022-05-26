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
results = [] # (name, auc, f1)

# %%
data

# %%
ground_truth = data.iloc[:,-1].apply(lambda x: -int(x.strip("'"))).values
X = data.values[:,:-1]
print(data.shape, X.shape)

# %%
counter = Counter(ground_truth)
for k,v in counter.items():
	per = v / len(ground_truth) * 100
	print('Class=%s, Count=%d, Percentage=%.3f%%' % (k, v, per))

# %%
def evaluate(ground_truth, y_pred):
    fpr, tpr, thresholds = metrics.roc_curve(ground_truth, y_pred)
    auc = metrics.auc(fpr, tpr)
    f1 = metrics.f1_score(ground_truth, y_pred)
    return (auc, f1)

# %%
clf = LocalOutlierFactor()#n_neighbors=20, contamination=0.1)
y_pred = clf.fit_predict(X)
auc, f1 = evaluate(ground_truth, y_pred)
results.append(("Full data", auc, f1))
print(auc, f1)

# %%
dedupData = data.drop_duplicates()
ground_truth = dedupData.iloc[:,-1].apply(lambda x: -int(x.strip("'"))).values
X = dedupData.values[:,:-1]
print(dedupData.shape, X.shape)

# %%
clf = LocalOutlierFactor()#n_neighbors=20, contamination=0.1)
y_pred = clf.fit_predict(X)
auc, f1 = evaluate(ground_truth, y_pred)
results.append(("Dedup data", auc, f1))
print(auc, f1)

# %%
shuffled = data.sample(frac=1).reset_index(drop=True)

# %%
ground_truth = shuffled.iloc[:,-1].apply(lambda x: -int(x.strip("'"))).values
X = shuffled.values[:,:-1]
print(shuffled.shape, X.shape)

# %%
clf = LocalOutlierFactor()#n_neighbors=20, contamination=0.1)
y_pred = clf.fit_predict(X)
auc, f1 = evaluate(ground_truth, y_pred)
results.append(("Full data shuffled", auc, f1))
print(auc, f1)

# %%


# %%
shuffledDedup = data.sample(frac=1).reset_index(drop=True).drop_duplicates()
ground_truth = shuffledDedup.iloc[:,-1].apply(lambda x: -int(x.strip("'"))).values
X = shuffledDedup.values[:,:-1]
print(dedupData.shape, X.shape)

# %%
clf = LocalOutlierFactor()#n_neighbors=20, contamination=0.1)
y_pred = clf.fit_predict(X)
auc, f1 = evaluate(ground_truth, y_pred)
results.append(("Dedup data shuffled", auc, f1))
print(auc, f1)

# %%
results

# %%
print("%-20s %5s %5s" %("Dataset", "AUC", "F1"))
for s, auc, f1 in results:
    print("%-20s %.3f %.3f" %(s, auc, f1))

# %%



