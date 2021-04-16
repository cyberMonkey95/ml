#!/usr/bin/python 

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

allData=pd.read_csv("./spam.data",sep=' ')

data=allData[ allData.columns[48:] ].copy()
data=data.rename(columns={'0.40':'f1', '0.41':'f2', '0.42':'f3', '0.778':'f4','0.43':'f5', '0.44':'f6', '3.756':'f7', '61':'f8', '278':'f9','1':'spam'})

#Standardize
for i in data.columns[:9]:
    scale=StandardScaler().fit( data[i].values.reshape(-1,1) )
    data[i]=pd.DataFrame(scale.transform( data[i].values.reshape(-1,1) ))
    
X=data[data.columns[:9]]
y=data['spam']

#split data into the training set and the test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)


model=LogisticRegression(random_state=0).fit(X_train,y_train)


p=model.predict(X_test)
