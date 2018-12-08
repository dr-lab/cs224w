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


def cosineSim(x, y):
    sum = np.matmul(x, y)
    x2 = np.sqrt(np.sum(np.square(x)))
    y2 = np.sqrt(np.sum(np.square(y)))

    if x2 == 0 or y2 == 0:
        return 0

    score = sum / (x2 * y2)
    return score


def l2NormDistance(x, y):
    return np.sqrt(np.sum(np.square(x - y)))


def dumpTopKSimNodes(feature_sim_node_score_map, base_node_id, topK, mode):
    name_score_tuple = collections.namedtuple('sortor', 'name score')
    reverse_sorted = sorted([name_score_tuple(v, k) for (k, v) in feature_sim_node_score_map.items()], reverse=True)

    # fw = open("./results/%s_%d_similarity_vectors_%s.csv" % (dataset, base_node_id, mode), "w")

    topKSet = []
    lastKSet = []

    len_score = len(reverse_sorted)
    for i in range(0, len_score):
        reverse_sorted[i]
        if i < topK:
            # print "top k score #: %d \t %s \t%s" % (i, reverse_sorted[i].score, reverse_sorted[i].name)
            topKSet.append(reverse_sorted[i].score)

        if i > len_score - topK:
            # print "last k score #: %d \t %s \t%s" % (i, reverse_sorted[i].score, reverse_sorted[i].name)
            lastKSet.append(reverse_sorted[i].score)

        # fw.write("%s,%s\n" % (reverse_sorted[i].score, reverse_sorted[i].name))

    # load good users
    return topKSet, lastKSet


def calSimAndPrint(G, base_node_id, mode):
    Feature_Sim_Node_Score_Map = {}
    base_features = G.node[base_node_id]["features"]

    nodes = G.nodes()
    for node_id in nodes:
        if node_id != base_node_id:
            features_y = G.node[node_id]["features"]
            if mode == 'cosine':
                score = cosineSim(base_features, features_y)
            elif mode == 'l2':
                score = l2NormDistance(base_features, features_y)
            else:
                raise ValueError('mode should be cosine or l2')
            Feature_Sim_Node_Score_Map[node_id] = score

    # print "feature of node base_node_id %d" % (base_node_id)
    return Feature_Sim_Node_Score_Map


G = loadGraph()
# bad_user_id = 7602
gt_bad_users = cPickle.load(
    open("./results/%s_gt_bad_users_set.pkl" % dataset, "rb"))

fw = open("./results/%s_gt_bad_users_intersect.csv" % (dataset), "w")
fw.write("bad_user_id, l2_topk, l2_lastk, cosine_topk, cosine_lastk, total_bad_user_count\n")

for bad_user_id in gt_bad_users:
    l2_topk = 0
    cosine_topk = 0
    l2_lastk = 0
    cosine_lastk = 0

    if G.has_node(bad_user_id):
        for mode in ['cosine', 'l2']:
            sim_score_values_map = calSimAndPrint(G, bad_user_id, mode)
            topKSet, lastKSet = dumpTopKSimNodes(sim_score_values_map, bad_user_id, 200, mode)

            # print "topK bad users"
            topKIntersect = gt.intersectUsers(topKSet, gt_bad_users)


            # print "topK good users"
            lastKIntersect = gt.intersectUsers(lastKSet, gt_bad_users)

            if mode=="l2":
                l2_topk = len(topKIntersect)
                l2_lastk = len(lastKIntersect)
            else:
                cosine_topk = len(topKIntersect)
                cosine_lastk = len(lastKIntersect)

            # def histogram(values, bins, xlabel, ylabel, title, fileName)
            # utils.hist(sim_score_values_map.values(), 30, 'Sim-Score', 'Frequency',
            #            '%s Similarity Score Histogram k=3,bad userId= %d' % (mode, bad_user_id),
            #            "./diagram/%s_sim_score_histogram_%d.PNG" % (mode, bad_user_id))
        print "%d\t  %d \t %d \t %d \t %d \t %d" % (bad_user_id, l2_topk, l2_lastk, cosine_topk, cosine_lastk, len(gt_bad_users))
        fw.write("%d,%d,%d,%d,%d,%d\n" % (bad_user_id, l2_topk, l2_lastk, cosine_topk, cosine_lastk, len(gt_bad_users)))
fw.close()
