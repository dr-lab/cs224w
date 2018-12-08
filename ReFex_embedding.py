from __future__ import division

import networkx as nx
import numpy as np
import cPickle
import os, sys
import create_neg_and_pos_net as negpos

dataset = "alpha"


def getMaxInOutEdges(G):
    nodes = G.nodes()
    max_in = 1
    max_out = 1
    max_ego = 1
    max_ego_out = 1
    for node_id in nodes:
        max_in = max(max_in, G.degree(node_id))
        max_out = max(max_out, G.out_degree(node_id))

        ego_net = nx.ego_graph(G, node_id)
        ego_edges = ego_net.number_of_edges()
        max_ego = max(max_ego, ego_edges)

        nbrs = nx.neighbors(G, node_id)
        nbrs_total_edges = 0
        for nbr in nbrs:
            nbrs_total_edges += G.out_degree(nbr)
            nbrs_total_edges += G.degree(nbr)

            max_ego_out = max(max_ego_out, nbrs_total_edges - ego_edges)

    return max_in, max_out, max_ego, max_ego_out


def loadRev2Score(folder, node_to_score_map):
    files = os.listdir(folder)
    file_count = len(files)
    for file in files:
        if dataset not in file:
            file_count = file_count - 1
            continue
        f = open(folder + file, "r")
        print file
        for l in f:
            ls = l.strip().split(",")

            node_id = int(ls[0][1:])
            # node_id = ls[0]
            if node_id in node_to_score_map:
                node_to_score_map[node_id].append(float(ls[1]))
            else:
                node_to_score_map[node_id] = [float(ls[1])]
        f.close()

    return node_to_score_map, file_count


def getRev2FairnessScore(G):
    """
    load the rev2 output, get the fairness scores
    :return: a map of {node_id, (fairness media, fairness score)}
    """
    node_to_fairness_map = {}
    node_to_goodness_map = {}
    nodes = G.nodes()
    for node_id in nodes:
        node_to_fairness_map[node_id] = [0]
        node_to_goodness_map[node_id] = [0]

    node_to_fairness_map, fairness_file_count = loadRev2Score("./rev2/results/fairness/", node_to_fairness_map)
    node_to_goodness_map, goodness_file_count = loadRev2Score("./rev2/results/goodness/", node_to_goodness_map)

    return (node_to_fairness_map, node_to_goodness_map, fairness_file_count, goodness_file_count)


def getEgoNetEdges(G, node_id, max_ego_out, max_ego):
    """
    Get EgoNet edge number
    :param G:
    :param node_id:
    :return: count of edge number
    """
    ego_net = nx.ego_graph(G, node_id)
    ego_edges = ego_net.number_of_edges()

    nbrs = nx.neighbors(G, node_id)
    nbrs_total_edges = 0
    for nbr in nbrs:
        nbrs_total_edges += G.out_degree(nbr)
        nbrs_total_edges += G.degree(nbr)

    ego_out_going_edges = nbrs_total_edges - ego_edges
    return ego_edges / max_ego, ego_out_going_edges / max_ego_out


def removeProductNodes(G):
    nodes = G.nodes()
    for node_id in nodes:
        if node_id not in rev2_fairness_score_map:
            G.remove_node(node_id)


def findFeatures(G, node_id, max_in, max_out, max_ego, max_ego_out):
    """
    Get 6 features for each node
    :param G:
    :param node_id:
    :return:  a tuple of 6 features
    """
    # basic network features

    # 1. out degree
    out_degree = G.out_degree(node_id) / max_out

    # 2. in degree
    in_degree = G.degree(node_id) / max_in

    # 3,4. ego net edge count, ego_out_going_edges
    ego_net_edges_count, ego_out_going_edges = getEgoNetEdges(G, node_id, max_ego, max_ego_out)

    # 5,6. Rev2 features (fairness media score, fairness)
    # if node_id not in rev2_fairness_score_map:
    #     (fairness_media_score, fairness_score) = (1.0, 1.0)
    # else:
    #     (fairness_media_score, fairness_score) = rev2_fairness_score_map[node_id]

    features = [out_degree, in_degree, ego_net_edges_count, ego_out_going_edges]

    return features


def initFeatures(G, G_pos, G_neg):
    """
    Initialize featues for each node
    :param G:
    :return:
    """

    # removeProductNodes(G)
    # removeProductNodes(G_pos)
    # removeProductNodes(G_neg)

    max_in, max_out, max_ego, max_ego_out = getMaxInOutEdges(G)
    max_in_pos, max_out_pos, max_ego_pos, max_ego_out_pos = getMaxInOutEdges(G_pos)
    max_in_neg, max_out_neg, max_ego_neg, max_ego_out_neg = getMaxInOutEdges(G_neg)

    nodes = G.nodes()
    for node_id in nodes:
        features = findFeatures(G, node_id, max_in, max_out, max_ego, max_ego_out)
        features_pos = findFeatures(G_pos, node_id, max_in_pos, max_out_pos, max_ego_pos, max_ego_out_pos)
        features_neg = findFeatures(G_neg, node_id, max_in_neg, max_out_neg, max_ego_neg, max_ego_out_neg)

        features_rev2_faireness = rev2_fairness_score_map[node_id]
        features_rev2_goodness = node_to_goodness_score_map[node_id]
        G.node[node_id]["features"] = features
        features_pos.extend(features_neg)
        G.node[node_id]["features_pos_neg"] = features_pos

        if features_pos is None:
            print("stop here")
        features_pos.extend(features_rev2_faireness)
        features_pos.extend(features_rev2_goodness)
        G.node[node_id]["features_pos_neg_rev2"] = features_pos


def augmentFeatures(G, node_id, k, features, feature_count, feature_key):
    # find its neighbors
    nbr_ids = G.adj[node_id]

    sum_features = np.zeros(feature_count * 3 ** (k - 1))
    degree = G.out_degree(node_id)
    for nbr_id in nbr_ids:
        nbr_features = G.node[nbr_id][feature_key]
        if nbr_features is None:
            print("stop here")

        sum_features = np.add(sum_features, nbr_features[0:feature_count * 3 ** (k - 1)])

    # append sum and mean only if the target nodes features length <=3*k
    new_features = features
    if degree == 0:
        new_features = np.append(features, np.zeros(feature_count * 3 ** (k - 1) * 2))
    else:
        new_features = np.append(new_features, np.divide(sum_features, degree * 1.0))
        new_features = np.append(new_features, sum_features)

    return new_features
    # G.node[node_id][feature_key] = new_features


def recursive(G, k):
    """
    Iteration k times, features augumentation by mean and sum neighbors' features.
    :param G:
    :param k:
    :return:
    """
    nodes = G.nodes()
    for node_id in nodes:
        # print G.node[node_id]["features"]
        features = G.node[node_id]["features"]
        features_pos_neg = G.node[node_id]["features_pos_neg"]
        features_pos_neg_rev2 = G.node[node_id]["features_pos_neg_rev2"]

        G_feature_count = 4
        G_neg_pos_count = 8
        G_neg_pos_rev2_count = fairness_file_count + goodness_file_count + 8

        if len(features) > G_feature_count * 3 ** (k - 1):
            return

        new_features = augmentFeatures(G, node_id, k, features, G_feature_count, "features")
        new_features_pos_neg = augmentFeatures(G, node_id, k, features_pos_neg, G_neg_pos_count, "features_pos_neg")
        new_features_pos_neg_rev2 = augmentFeatures(G, node_id, k, features_pos_neg_rev2, G_neg_pos_rev2_count,
                                                    "features_pos_neg_rev2")

        G.node[node_id]["features"] = new_features
        G.node[node_id]["features_pos_neg"] = new_features_pos_neg
        G.node[node_id]["features_pos_neg_rev2"] = new_features_pos_neg_rev2


def writeCsvEmbedding(feature_label):
    fw = open("./results/%s_graph_embedding_vectors_%s.csv" % (dataset, feature_label), "w")

    v_len = 0
    count = 0
    for node in list(G.nodes(data=feature_label)):
        node_id = node[0]
        features = node[1][feature_label]

        if v_len != len(features):
            count+=1
            if count >=2:
                print("stop here")


        fw.write("%s,%s\n" % (node_id, ",".join(str(e) for e in features)))
    fw.close()


# main code start here

G, G_pos, G_neg = negpos.getPosNegNet(dataset)
rev2_fairness_score_map, node_to_goodness_score_map, fairness_file_count, goodness_file_count = getRev2FairnessScore(G)

print "Total G nodes=%d" % G.number_of_nodes()
print "Total G_pos nodes=%d" % G_pos.number_of_nodes()
print "Total G_neg nodes=%d" % G_neg.number_of_nodes()

# drawNxGraph(getEgoNet(G, node_id), "Ego_Net_sockpuppet_{0}_{1}".format(dataset, node_id))

initFeatures(G, G_pos, G_neg)
recursive(G, 1)
recursive(G, 2)
recursive(G, 3)

writeCsvEmbedding("features")
writeCsvEmbedding("features_pos_neg")
writeCsvEmbedding("features_pos_neg_rev2")

nx.write_gpickle(G, "./results/%s_graph_embedding_featured_graph.pkl" % (dataset))

print "\nUsers only"
print "Total nodes=%d" % G.number_of_nodes()
print "Total edges=%d" % G.number_of_edges()
