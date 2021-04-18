#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 23:07:01 2021

@author: lastsamurai
"""

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

allData=pd.read_csv("./spam.data",sep=' ')

data=allData[ allData.columns[:10] ].copy()

scaler= StandardScaler()
X=scaler.fit_transform(data.values)
X=pd.DataFrame(X)

y=X[X.columns[9]]
X=X[X.columns[:8]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

reg=LinearRegression().fit(X_train,y_train)
p=reg.predict(X_test)

print(mean_squared_error(y_test,p))


