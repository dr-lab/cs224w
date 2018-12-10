#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 22:15:13 2018

@author: sullivan42
"""

import pandas as pd
import sklearn.tree
from sklearn.model_selection import train_test_split
import subprocess
import matplotlib.pyplot as plt
import numpy as np
#import export_graphviz
network = "alpha"
"""
#nodeid is column 0
gt = pd.read_csv("./rev2/data/%s_gt.csv"%(network),header=None)

#nodeid=col 0
features = pd.read_csv("./results/%s_graph_embedding_vectors_features.csv"%(network),header=None)

out = gt.merge(features,'inner',left_on=0,right_on=0,suffixes=("_y","_x"))

y = out["1_y"].copy()
X = out.drop('1_y', axis=1)
X = X.drop(0, axis=1)

dt = sklearn.tree.DecisionTreeClassifier(min_samples_split=20, random_state=99)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)



dt.fit(X_train, y_train)

dt.score(X_test,y_test)

dt.decision_path(X_test)

export_graphviz(dt,out_file='dt.dot')
command = ["dot", "-Tpng", "dt.dot", "-o", "dt.png"]
subprocess.check_call(command)



gt = pd.read_csv("./rev2/data/%s_gt.csv"%(network),header=None)
net = pd.read_csv("./rev2/data/%s_network.csv"%(network),header=None)

net_neg = net[net[2] < 0]
net_pos = net[net[2] >= 0]
net_pos.to_csv("%s_pos_network.csv"%(network),header=False)
net_neg.to_csv("%s_neg_network.csv"%(network),header=False)


gt_net = gt.merge(net,'inner',left_on=0,right_on=0,suffixes=("_y","_x"))
"""
feats = ["features", "features_pos_neg", "features_pos_neg_rev2"]
modes = ['cosine', 'l2']
for f in feats:
    scores = np.zeros((0,1))
    for i in np.linspace(.10,.90,9):
        gt = pd.read_csv("./rev2/data/%s_gt.csv"%(network),header=None)
        
        #nodeid=col 0
        features = pd.read_csv("./results/%s_graph_embedding_vectors_%s.csv"%(network,f),header=None)
        
        out = gt.merge(features,'inner',left_on=0,right_on=0,suffixes=("_y","_x"))
        
        y = out["1_y"].copy()
        X = out.drop('1_y', axis=1)
        X = X.drop(0, axis=1)
        
        dt = sklearn.tree.DecisionTreeClassifier(min_samples_split=20, random_state=99)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-i, random_state=42)
        dt.fit(X_train, y_train)
        score = dt.score(X_test,y_test)
        #scores= np.append(scores,score)
        """print "========="
        print "Feature: %s" % f
        print "score %f" % score
        dt.decision_path(X_test)
        
        #export_graphviz(dt,out_file='dt.dot')
        command = ["dot", "-Tpng", "dt.dot", "-o", "dt.png"]
        subprocess.check_call(command)
                plt.plot(anchors,ps, label = l)"""
                
                
        predictions = dt.predict_proba(X_test)

        auc = sklearn.metrics.roc_auc_score(y_test, predictions[:,1])
        scores= np.append(scores,auc)
    plt.plot(np.linspace(.10,.90,9)*100, scores,label = f)

plt.plot(np.array([10,20,30,40,50,60,70,80,90]),[.78,.79,.8,.8,.79,.81,.82,.81,.83], label = 'REV2 Only')

plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25),
      ncol=4, fancybox=True, shadow=True)
plt.xlabel("Percent Training Data")
plt.ylabel("Average AUC") 
plt.savefig("./results/decision_tree_results.png", bbox_inches = "tight")
plt.show()

        
        
        
        
        