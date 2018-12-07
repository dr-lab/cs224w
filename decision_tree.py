#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 22:15:13 2018

@author: sullivan42
"""

import pandas as pd
import sklearn.tree
from sklearn.model_selection import train_test_split
network = "alpha"

#nodeid is column 0
gt = pd.read_csv("./rev2/data/%s_gt.csv"%(network),header=None)

#nodeid=col 0
features = pd.read_csv("./results/%s_graph_embedding_vectors.csv"%(network),header=None)

out = gt.merge(features,'inner',left_on=0,right_on=0,suffixes=("_y","_x"))

y = out["1_y"].copy()
X = out.drop('1_y', axis=1)

dt = sklearn.tree.DecisionTreeClassifier(min_samples_split=20, random_state=99)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)



dt.fit(X_train, y_train)

dt.score(X_test,y_test)

dt.decision_path(X_test)