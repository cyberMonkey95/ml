#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 23:17:44 2021

@author: lastsamurai
"""

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier

X = [[0., 0.], [1., 1.]]

y = [0, 1]

clf = MLPClassifer( solver='lbfgs', alpha= 1e-5, hidden_layer_sizes=(5,2), random_state=1)

clf.fit(X, y)


clf.predict([[2., 2.], [-1., -2.]])

