#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 02:16:05 2021

@author: lastsamurai
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

rng = np.random.RandomState(42)
x = 10 * rng.rand(50)
y = 2 * x - 1 + rng.randn(50)
#plt.scatter(x, y);

model = LinearRegression(fit_intercept=True)

X = x[:, np.newaxis]

model.fit(X, y)

xfit = np.linspace(-1, 11)

Xfit = xfit[:, np.newaxis]
yfit = model.predict(Xfit)


#plt.scatter(x, y)
plt.plot(xfit, yfit);


#allData=pd.read_csv(r'./spam.data',sep=' ')