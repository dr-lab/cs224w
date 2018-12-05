from __future__ import division

import networkx as nx
import numpy as np
import cPickle

dataset = "alpha"


def loadGraph():
    G = nx.DiGraph()

    # CSV file format
    # SOURCE, TARGET, RATING, TIME
    # 7188,1,10,1407470400
    # 430,1,10,1376539200

    filePath = "./rev2/data/{0}_network.csv".format(dataset)
    print "Load network from {0}".format(filePath)

    f = open(filePath, "r")
    for l in f:
        ls = l.strip().split(",")
        G.add_edge(int(ls[0]), int(ls[1]), rating=int(ls[2]),
                   time=float(ls[3]))  ## the weight should already be in the range of -1 to 1
    f.close()

    # G  = cPickle.load(open("./rev2/data/%s_network.pkl" % (dataset), "rb"))

    print "Products + users"
    print "Total nodes=%d" % G.number_of_nodes()
    print "Total edges=%d" % G.number_of_edges()

    return G


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
        max_ego = max(max_ego,ego_edges)

        nbrs = nx.neighbors(G, node_id)
        nbrs_total_edges = 0
        for nbr in nbrs:
            nbrs_total_edges += G.out_degree(nbr)
            nbrs_total_edges += G.degree(nbr)

            max_ego_out = max(max_ego_out, nbrs_total_edges - ego_edges)

    return max_in, max_out, max_ego, max_ego_out


def getRev2FairnessScore():
    """
    load the rev2 output, get the fairness scores
    :return: a map of {node_id, (fairness media, fairness score)}
    """
    node_to_fairness_map = {}
    f = open("./rev2/results/{}-fng-sorted-users-0-0-0-0-0-0-0.csv".format(dataset), "r")
    for l in f:
        ls = l.strip().split(",")
        # print ls[0][1:] , ls[0]
        node_id = int(ls[0][1:])
        # node_id = ls[0]
        node_to_fairness_map[node_id] = (float(ls[1]), float(ls[2]))
    f.close()
    return node_to_fairness_map


rev2_fairness_score_map = getRev2FairnessScore()


def getEgoNetEdges(G, node_id):
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


def findFeatures(G, node_id):
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
    ego_net_edges_count, ego_out_going_edges = getEgoNetEdges(G, node_id)

    # 5,6. Rev2 features (fairness media score, fairness)
    if node_id not in rev2_fairness_score_map:
        (fairness_media_score, fairness_score) = (1.0, 1.0)
    else:
        (fairness_media_score, fairness_score) = rev2_fairness_score_map[node_id]

    features = [out_degree, in_degree, ego_net_edges_count, ego_out_going_edges, fairness_score, fairness_media_score]
    return features


def initFeatures(G):
    """
    Initialize featues for each node
    :param G:
    :return:
    """
    nodes = G.nodes()
    for node_id in nodes:
        features = findFeatures(G, node_id)
        # delete node which is marked as Product
        if features[4] == 1:
            G.remove_node(node_id)
        else:
            # print features
            G.node[node_id]["features"] = features


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

        if len(features) > 6 * 3 ** (k - 1):
            return

        # find its neighbors
        nbr_ids = G.adj[node_id]

        sum_features = np.zeros(6 * 3 ** (k - 1))
        degree = G.out_degree(node_id)
        for nbr_id in nbr_ids:
            nbr_features = G.node[nbr_id]["features"]
            sum_features = np.add(sum_features, nbr_features[0:6 * 3 ** (k - 1)])

        # append sum and mean only if the target nodes features length <=3*k
        new_features = features
        if degree == 0:
            new_features = np.append(features, np.zeros(6 * 3 ** (k - 1) * 2))
        else:
            new_features = np.append(new_features, np.divide(sum_features, degree * 1.0))
            new_features = np.append(new_features, sum_features)

        G.node[node_id]["features"] = new_features


# main code start here

G = loadGraph()
max_in, max_out, max_ego, max_ego_out = getMaxInOutEdges(G)
# drawNxGraph(getEgoNet(G, node_id), "Ego_Net_sockpuppet_{0}_{1}".format(dataset, node_id))

initFeatures(G)
recursive(G, 1)
recursive(G, 2)
recursive(G, 3)

fw = open("./results/%s_graph_embedding_vectors.csv" % (dataset), "w")

for node in list(G.nodes(data="features")):
    node_id = node[0]
    features = node[1]["features"]
    fw.write("%s,%s\n" % (node_id, ",".join(str(e) for e in features)))
fw.close()

nx.write_gpickle(G, "./results/%s_graph_embedding_featured_graph.pkl" % (dataset))

print "\nUsers only"
print "Total nodes=%d" % G.number_of_nodes()
print "Total edges=%d" % G.number_of_edges()
