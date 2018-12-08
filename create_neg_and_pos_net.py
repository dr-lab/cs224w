#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Code ot genreate negative weight only and positive weight only network
csv files
@author: sullivan42
"""

import pandas as pd
import networkx as nx
import numpy as np
network = "alpha"

net = pd.read_csv("./rev2/data/%s_network.csv"%(network),header=None)

net_neg = net[net[2] < 0]
net_pos = net[net[2] >= 0]
net_pos.to_csv("%s_pos_network.csv"%(network),header=False)
net_neg.to_csv("%s_neg_network.csv"%(network),header=False)


def getNodeIdList(dataset):
    df = pd.read_csv("./rev2/data/alpha_network.csv")
    df.columns = ['src','dst','weight','delta']
    node_ids = set(np.append(df['src'].unique(),df['dst'].unique()))
    return node_ids



def getPosNegNet(dataset):
    df = pd.read_csv("./rev2/data/%s_network.csv" % (dataset)) 
    df.columns = ['src','dst','weight','delta']
    G = nx.from_pandas_dataframe(df, 'src', 'dst', create_using=nx.Graph() )
    Gpos = nx.from_pandas_dataframe(df, 'src', 'dst', create_using=nx.Graph() )
    Gneg = nx.from_pandas_dataframe(df, 'src', 'dst', create_using=nx.Graph() )

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
    
    n_ids = getNodeIdList(dataset)
    for n in n_ids:
        if not Gpos.has_node(n):
            Gpos.add_node(n)
        if not Gneg.has_node(n):
            Gneg.add_node(n)    
    if Gneg.number_of_nodes() != Gpos.number_of_nodes() and Gneg.number_of_nodes() != G.number_of_nodes():
        raise ValueError('network missing nodes')
    return G, Gpos, Gneg

Gpos,Gneg,G = getPosNegNet(network)

getNodeIdList(network)
