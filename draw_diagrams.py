from __future__ import division

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import networkx_utils as utils
import cPickle

dataset = "alpha"

G = nx.read_gpickle("results/%s_graph_embedding_featured_graph.pkl" % (dataset))
gt_bad_users = cPickle.load(open("./results/%s_gt_bad_users_set.pkl" % dataset, "rb"))
gt_good_users = cPickle.load(open("./results/%s_gt_good_users_set.pkl" % dataset, "rb"))
utils.drawNxGrap(G,"%s network" % dataset, "./diagram/%s_network.png" % dataset)
