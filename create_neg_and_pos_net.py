#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Code ot genreate negative weight only and positive weight only network
csv files
@author: sullivan42
"""

import pandas as pd

network = "alpha"

net = pd.read_csv("./rev2/data/%s_network.csv"%(network),header=None)

net_neg = net[net[2] < 0]
net_pos = net[net[2] >= 0]
net_pos.to_csv("%s_pos_network.csv"%(network),header=False)
net_neg.to_csv("%s_neg_network.csv"%(network),header=False)


