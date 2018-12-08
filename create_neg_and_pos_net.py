#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Code ot genreate negative weight only and positive weight only network
csv files
@author: sullivan42
"""

import pandas as pd
import networkx as nx
network = "alpha"

net = pd.read_csv("./rev2/data/%s_network.csv"%(network),header=None)

net_neg = net[net[2] < 0]
net_pos = net[net[2] >= 0]
net_pos.to_csv("%s_pos_network.csv"%(network),header=False)
net_neg.to_csv("%s_neg_network.csv"%(network),header=False)

def getPosNegNet(dataset):
    G = nx.read_gpickle("./rev2/data/%s_network.pkl" % (dataset))
    Gpos = nx.read_gpickle("./rev2/data/%s_network.pkl" % (dataset))
    Gneg = nx.read_gpickle("./rev2/data/%s_network.pkl" % (dataset))

    for e in G.edges_iter(data='weight', default=1):
        #print e
        if e[2] >= 0:
            Gneg.remove_edge(e[0],e[1])
            #print "pos"
        if e[2] < 0:
            Gpos.remove_edge(e[0],e[1])  
            #print "neg"
    # Error  checking
    for e in Gpos.edges_iter(data='weight', default=1):
        if e[2]<0:
            raise ValueError('pos net has neg weight')
        
    for e in Gneg.edges_iter(data='weight', default=1):
        if e[2]>0:
            raise ValueError('neg net has pos weight')
    for n in G.nodes_iter():
        if not Gpos.has_node(n):
            Gpos.add_node(n)
        if not Gneg.has_node(n):
            Gneg.add_node(n)    
    if Gneg.number_of_nodes() != Gpos.number_of_nodes() and Gneg.number_of_nodes() != G.number_of_nodes():
        raise ValueError('network missing nodes')
    return Gpos, Gneg

Gpos,Gneg = getPosNegNet(network)


