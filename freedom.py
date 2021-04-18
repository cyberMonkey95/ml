#!/usr/bin/python 

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import confusion_matrix,accuracy_score, precision_score, recall_score

allData=pd.read_csv("./spam.data",sep=' ')

data=allData[ allData.columns[48:] ].copy()
data=data.rename(columns={'0.40':'f1', '0.41':'f2', '0.42':'f3', '0.778':'f4','0.43':'f5', '0.44':'f6', '3.756':'f7', '61':'f8', '278':'f9','1':'spam'})

scaler = StandardScaler()
X=scaler.fit_transform(data.values)
X=pd.DataFrame(X)
X=data[data.columns[:9]]
y=data['spam']

#split data into the training set and the test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)


logistic_model=LogisticRegression(random_state=0).fit(X_train,y_train)
p=logistic_model.predict(X_test)

dummy_clf = DummyClassifier(strategy="most_frequent").fit(X_train,y_train)
pp=dummy_clf.predict(X_test)


nn_clf = MLPClassifier( solver='lbfgs', alpha= 1e-5, hidden_layer_sizes=(5,2), random_state=1)
nn_clf.fit(X_train,y_train)
nn=nn_clf.predict(X_test)


cunt=confusion_matrix(y_test, p)
accuracy = round(accuracy_score(y_test, p), 3)
precision = round(precision_score(y_test, p), 3)
recall = round(recall_score(y_test, p), 3)

#up next:

#better preprocessing

#rethink the 2 other models

#START REGRESSSION