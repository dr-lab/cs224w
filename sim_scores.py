from __future__ import division

import networkx as nx
import numpy as np
import collections
import matplotlib.pyplot as plt
import networkx_utils as utils
import cPickle
import ground_truth_analysis as gt
import sklearn
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


def calSimAndPrint(G, base_node_id, mode, feature_set):
    Feature_Sim_Node_Score_Map = {}
    # base node is the anchor node
    base_features = G.node[base_node_id][feature_set]

    nodes = G.nodes()
    for node_id in nodes:
        if node_id != base_node_id:
            features_y = G.node[node_id][feature_set]
            if mode == 'cosine':
                score = cosineSim(base_features, features_y)
            elif mode == 'l2':
                score = l2NormDistance(base_features, features_y)
            else:
                raise ValueError('mode should be cosine or l2')
            Feature_Sim_Node_Score_Map[node_id] = score

    # print "feature of node base_node_id %d" % (base_node_id)
    return Feature_Sim_Node_Score_Map


def generateIntersect(filePath, feature, k, iterrate_users, gt_good_users, gt_bad_users):
    fw = open(filePath, "w")
    fw.write(
        "bad_user_id, "
        "l2_topk_bad, l2_topk_good, unlabelled_l2_topk,"
        "l2_lastk_bad, l2_lastk_good, unlabelled_l2_lastk,"
        "cosine_topk_bad, cosine_topk_good, unlabelled_cosine_topk,"
        "cosine_lastk_bad, cosine_lastk_good,unlabelled_cosine_lastk"
        " \n")

    for bad_user_id in iterrate_users:
        l2_topk = 0
        cosine_topk = 0
        l2_lastk = 0
        cosine_lastk = 0

        if G.has_node(bad_user_id):
            for mode in ['cosine', 'l2']:
                sim_score_values_map = calSimAndPrint(G, bad_user_id, mode, feature)
                topKSet, lastKSet = dumpTopKSimNodes(sim_score_values_map, bad_user_id, k, mode)

                # print "topK bad users"
                topKIntersect_bad = gt.intersectUsers(topKSet, gt_bad_users)
                topKIntersect_good = gt.intersectUsers(topKSet, gt_good_users)
                # print "topK good users"
                lastKIntersect_bad = gt.intersectUsers(lastKSet, gt_bad_users)
                lastKIntersect_good = gt.intersectUsers(lastKSet, gt_good_users)

                if mode == "l2":
                    l2_topk_bad = len(topKIntersect_bad)
                    l2_lastk_bad = len(lastKIntersect_bad)
                    l2_topk_good = len(topKIntersect_good)
                    l2_lastk_good = len(lastKIntersect_good)
                else:
                    cosine_topk_bad = len(topKIntersect_bad)
                    cosine_lastk_bad = len(lastKIntersect_bad)
                    cosine_topk_good = len(topKIntersect_good)
                    cosine_lastk_good = len(lastKIntersect_good)

                # def histogram(values, bins, xlabel, ylabel, title, fileName)
                # utils.hist(sim_score_values_map.values(), 30, 'Sim-Score', 'Frequency',
                #            '%s Similarity Score Histogram k=3,bad userId= %d' % (mode, bad_user_id),
                #            "./diagram/%s_sim_score_histogram_%d.PNG" % (mode, bad_user_id))
            print "%d\t  %d \t %d \t %d \t %d \t %d \t %d \t %d \t %d \t %d \t %d \t %d \t %d" % (bad_user_id,
                                                                   l2_topk_bad,
                                                                   l2_topk_good,
                                                                   (
                                                                           k - l2_topk_bad - l2_topk_good),
                                                                   # unlabelled l2_topk
                                                                   l2_lastk_bad,
                                                                   l2_lastk_good,
                                                                   (
                                                                           k - l2_lastk_bad - l2_lastk_good),
                                                                   # unlabelled l2_lastk
                                                                   cosine_topk_bad,
                                                                   cosine_topk_good,
                                                                   (
                                                                           k - cosine_topk_bad - cosine_topk_good),

                                                                   cosine_lastk_bad,
                                                                   cosine_lastk_good,
                                                                   (
                                                                           k - cosine_lastk_bad - cosine_lastk_good)
                                                                                                  )
            fw.write("%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n" % (bad_user_id,
                                                                   l2_topk_bad,
                                                                   l2_topk_good,
                                                                   (
                                                                           k - l2_topk_bad - l2_topk_good),
                                                                   # unlabelled l2_topk
                                                                   l2_lastk_bad,
                                                                   l2_lastk_good,
                                                                   (
                                                                           k - l2_lastk_bad - l2_lastk_good),
                                                                   # unlabelled l2_lastk
                                                                   cosine_topk_bad,
                                                                   cosine_topk_good,
                                                                   (
                                                                           k - cosine_topk_bad - cosine_topk_good),

                                                                   cosine_lastk_bad,
                                                                   cosine_lastk_good,
                                                                   (
                                                                           k - cosine_lastk_bad - cosine_lastk_good)))
    fw.close()


G = loadGraph()
# bad_user_id = 7602
gt_bad_users = cPickle.load(
    open("./results/%s_gt_bad_users_set.pkl" % dataset, "rb"))

gt_good_users = cPickle.load(
    open("./results/%s_gt_good_users_set.pkl" % dataset, "rb"))

features = ["features", "features_pos_neg", "features_pos_neg_rev2"]

# k = 3
for k in range(2, 11):
    for feature in features:
        print "calculate user intersect with ground truth, %d, %s" % (k, feature)
        bad_user_file_path = "./results/%s_%s_%d_gt_bad_users_intersect.csv" % (feature, dataset, k)
        generateIntersect(bad_user_file_path, feature, k, gt_bad_users, gt_good_users, gt_bad_users)

        good_user_file_path = "./results/%s_%s_%d_gt_good_users_intersect.csv" % (feature, dataset, k)
        generateIntersect(good_user_file_path, feature, k, gt_good_users, gt_good_users, gt_bad_users)





def generateIntersect(filePath, feature, k, iterrate_users, gt_good_users, gt_bad_users):
    fw = open(filePath, "w")
    fw.write(
        "bad_user_id, "
        "l2_topk_bad, l2_topk_good, unlabelled_l2_topk,"
        "l2_lastk_bad, l2_lastk_good, unlabelled_l2_lastk,"
        "cosine_topk_bad, cosine_topk_good, unlabelled_cosine_topk,"
        "cosine_lastk_bad, cosine_lastk_good,unlabelled_cosine_lastk"
        " \n")

    for bad_user_id in iterrate_users:
        l2_topk = 0
        cosine_topk = 0
        l2_lastk = 0
        cosine_lastk = 0

        if G.has_node(bad_user_id):
            for mode in ['cosine', 'l2']:
                sim_score_values_map = calSimAndPrint(G, bad_user_id, mode, feature)
                topKSet, lastKSet = dumpTopKSimNodes(sim_score_values_map, bad_user_id, k, mode)
                
                
                
#code to get percisions               

def getScores(G, anchors, mode,feature_set, gt_good_users, gt_bad_users ):
    Feature_Sim_Node_Score_Map = {}
    # base node is the anchor node

    nodes = G.nodes()
    for node_id in nodes:
        if node_id in gt_good_users or node_id in gt_bad_users:
            features_y = G.node[node_id][feature_set]
            if node_id not in anchors:
                anchor_scores = []
                for a_id in anchors:
                        a_features = G.node[a_id][feature_set]
                        if mode == 'cosine':
                            score = cosineSim(a_features, features_y)
                            anchor_scores.append(score)
                        elif mode == 'l2':
                            score = l2NormDistance(base_features, features_y)
                            anchor_scores.append(score)
                        else:
                            raise ValueError('mode should be cosine or l2')
                Feature_Sim_Node_Score_Map[node_id] = np.max(anchor_scores)
    return Feature_Sim_Node_Score_Map
                
  

G = loadGraph()
# bad_user_id = 7602
gt_bad_users = cPickle.load(
    open("./results/%s_gt_bad_users_set.pkl" % dataset, "rb"))

gt_good_users = cPickle.load(
    open("./results/%s_gt_good_users_set.pkl" % dataset, "rb"))

features = ["features", "features_pos_neg", "features_pos_neg_rev2"]
modes = ['cosine', 'l2']
anchors = gt_bad_users[0:10] 
feature_set = features[2]            
mode = modes[0]
Feature_Sim_Node_Score_Map = getScores(G, anchors,mode, feature_set, gt_good_users, gt_bad_users )
gt_scores = np.zeros((0,2))
for nid in Feature_Sim_Node_Score_Map.keys():
    if nid in gt_good_users:
        gt_scores = np.vstack((gt_scores,[Feature_Sim_Node_Score_Map[nid],1]))
    elif nid in gt_bad_users:
        gt_scores = np.vstack((gt_scores,[Feature_Sim_Node_Score_Map[nid],0]))
    else:
        print "FUCK"




name_score_tuple = collections.namedtuple('sortor', 'name score')
reverse_sorted = sorted([name_score_tuple(v, k) for (k, v) in Feature_Sim_Node_Score_Map.items()], reverse=True)

sklearn.metrics.average_precision_score( gt_scores[:,1],  gt_scores[:,0], pos_label=1)

print ""








