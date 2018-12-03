from __future__ import division

import networkx as nx
import numpy as np
import collections

dataset = "alpha"


def loadGraph():
    G = nx.read_gpickle("./results/%s_graph_embedding_vectors.pkl" % (dataset))

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


def printTopKSimNodes(feature_sim_node_score_map, base_node_id, k):
    name_score_tuple = collections.namedtuple('sortor', 'name score')
    reverse_sorted = sorted([name_score_tuple(v, k) for (k, v) in feature_sim_node_score_map.items()], reverse=True)

    fw = open("./results/%s_%d_similarity_vectors.csv" % (dataset, base_node_id), "w")

    for i in range(0, len(reverse_sorted)):
        reverse_sorted[i]
        print "score #: %d \t %s \t%s" % (i, reverse_sorted[i].score, reverse_sorted[i].name)
        fw.write("%s,%s\n" % (reverse_sorted[i].score, reverse_sorted[i].name))


def calSimAndPrint(G, base_node_id, k):
    Feature_Sim_Node_Score_Map = {}
    base_features = G.node[base_node_id]["features"]

    nodes = G.nodes()
    for node_id in nodes:
        if node_id != base_node_id:
            features_y = G.node[node_id]["features"]
            score = simScore(base_features, features_y)
            Feature_Sim_Node_Score_Map[node_id] = score

    print "feature of node base_node_id %d" % (base_node_id)

    printTopKSimNodes(Feature_Sim_Node_Score_Map,base_node_id, k)
    return Feature_Sim_Node_Score_Map


G = loadGraph()
bad_user_id = 1
sim_score_values_map = calSimAndPrint(G, bad_user_id, 20)
