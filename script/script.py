# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../"))
data=pd.read_csv("data/winequality-red.csv")
# Any results you write to the current directory are saved as output.
data.head()
bin = (2, 6.5, 8)
names = ['bad', 'good']
data['quality'] = pd.cut(data['quality'], bins = bin, labels = names)
data.head()
label_quality = LabelEncoder()
#data['quality'] = pd.get_dummies(data['quality'])
data['quality'] = label_quality.fit_transform(data['quality'])
data['quality'].value_counts()
X=data.drop('quality',1)
Y=data['quality']
#Y.head()
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3)
clf = RandomForestClassifier(n_jobs=2, random_state=0)
clf.fit(X_train, y_train)
preds=clf.predict(X_test)
clf.predict_proba(X_test[0:10])
pd.crosstab(y_test, preds)    
print ("Accuracy is ", accuracy_score(y_test,preds)*100)


