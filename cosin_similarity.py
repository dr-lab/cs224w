from __future__ import division

import networkx as nx
import numpy as np
import collections
import matplotlib.pyplot as plt
import networkx_utils as utils
import cPickle
import ground_truth_analysis as gt
dataset = "alpha"


def loadGraph():
    G = nx.read_gpickle("results/%s_graph_embedding_featured_graph.pkl" % (dataset))

    print "Total nodes=%d" % G.number_of_nodes()
    print "Total edges=%d" % G.number_of_edges()

    return G


def simScore(x, y):
    sum = np.matmul(x, y)
    x2 = np.sqrt(np.sum(np.square(x)))
    y2 = np.sqrt(np.sum(np.square(y)))

    if x2 == 0 or y2 == 0:
        return 0

    score = sum / (x2 * y2)
    return score


def dumpTopKSimNodes(feature_sim_node_score_map, base_node_id, topK):
    name_score_tuple = collections.namedtuple('sortor', 'name score')
    reverse_sorted = sorted([name_score_tuple(v, k) for (k, v) in feature_sim_node_score_map.items()], reverse=True)

    fw = open("./results/%s_%d_similarity_vectors.csv" % (dataset, base_node_id), "w")

    topKSet = []
    lastKSet = []

    len_score = len(reverse_sorted)
    for i in range(0, len_score):
        reverse_sorted[i]
        if i < topK:
            print "top k score #: %d \t %s \t%s" % (i, reverse_sorted[i].score, reverse_sorted[i].name)
            topKSet.append(reverse_sorted[i].score)

        if i> len_score - topK:
            print "last k score #: %d \t %s \t%s" % (i, reverse_sorted[i].score, reverse_sorted[i].name)
            lastKSet.append(reverse_sorted[i].score)

        fw.write("%s,%s\n" % (reverse_sorted[i].score, reverse_sorted[i].name))

    # load good users
    return topKSet, lastKSet


def calSimAndPrint(G, base_node_id):
    Feature_Sim_Node_Score_Map = {}
    base_features = G.node[base_node_id]["features"]

    nodes = G.nodes()
    for node_id in nodes:
        if node_id != base_node_id:
            features_y = G.node[node_id]["features"]
            score = simScore(base_features, features_y)
            Feature_Sim_Node_Score_Map[node_id] = score

    print "feature of node base_node_id %d" % (base_node_id)
    return Feature_Sim_Node_Score_Map


G = loadGraph()
bad_user_id = 1
sim_score_values_map = calSimAndPrint(G, bad_user_id)

topKSet, lastKSet = dumpTopKSimNodes(sim_score_values_map, bad_user_id, 200)
gt_bad_users = cPickle.load(open("./results/%s_gt_bad_users_set.pkl" % dataset, "rb"))

gt.intersectUsers(topKSet, gt_bad_users)
gt.intersectUsers(lastKSet, gt_bad_users)

# def histogram(values, bins, xlabel, ylabel, title, fileName)
utils.hist(sim_score_values_map.values(), 30, 'Sim-Score', 'Frequency',
           'Similarity Score Histogram k=3,bad userId= %d' % bad_user_id,
           "./diagram/cosin_sim_score_histogram_%d.PNG" % bad_user_id)
