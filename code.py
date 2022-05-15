import pandas as pd
import numpy as np
from pyod.models.abod import ABOD
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report


data = pd.read_csv('/content/drive/MyDrive/Kaggle/creditcardfraud/creditcard.csv')
X = data.drop(['Class'], axis=1)
abod = ABOD(n_neighbors=5,contamination=0.0017,method='fast')
y_pred = abod.fit_predict(X)
print(classification_report(y_true=data['Class'], y_pred=y_pred))

ifo = IsolationForest(contamination=0.002)
y_pred = ifo.fit_predict(X)
np.unique(y_pred, return_counts=True)
y_pred[y_pred == 1] = 0
y_pred[y_pred == -1] = 1
print(classification_report(y_true=data['Class'], y_pred=y_pred))
