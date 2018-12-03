from __future__ import division

import networkx as nx
import numpy as np

dataset = "otc"

def loadGraph():
    G = nx.DiGraph()
    # f = open("./db/soc-sign-bitcoinalpha.csv", "r")
    f = open("./db/soc-sign-bitcoinotc.csv", "r")
    for l in f:
        ls = l.strip().split(",")
        G.add_edge(int(ls[0]), int(ls[1]), rating=int(ls[2]),
                   time=float(ls[3]))  ## the weight should already be in the range of -1 to 1
    f.close()

    print "Total nodes=%d" % G.number_of_nodes()
    print "Total edges=%d" % G.number_of_edges()
    # print list(G.nodes)

    return G



def getEgoNet(G, node_id):
    ego_net = nx.ego_graph(G, node_id)
    return ego_net


def findFeatures(G, node_id):
    out_degree = G.out_degree(node_id)
    in_degree = G.degree(node_id)
    degree_diff = in_degree - out_degree

    ego_net = nx.ego_graph(G, node_id)
    ego_net_edges_count = ego_net.number_of_edges()

    edges = list(G.edges(node_id))
    mutual_edges_count = 0
    rating_diff = 0
    for e in edges:
        # print "processing edge: {0}".format(e)
        out_rating = G[e[0]][e[1]]["rating"]

        rating_diff += out_rating

        has_in_edge = G.has_edge(e[1], e[0])

        if has_in_edge:
            in_rating = G[e[1]][e[0]]["rating"]

            mutual_edges_count += 1
            rating_diff -= in_rating

        # in_rating = G[e[1]][e[0]]["rating"]
        # print e, out_rating

    print len(edges), mutual_edges_count

    features = [out_degree, in_degree, degree_diff, ego_net_edges_count, mutual_edges_count, rating_diff]
    return features


def initFeatures(G):
    nodes = G.nodes()

    for node_id in nodes:
        features = findFeatures(G, node_id)
        print features
        # G.nodes[node_id]["features"] = features


def recursive(G, k):
    nodes = G.nodes()
    for node_id in nodes:
        print G.node[node_id]["features"]
        features = G.nodes[node_id]["features"]

        if len(features) > 6 * 3 ** (k - 1):
            return

        # find its neighbors
        nbr_ids = G.adj[node_id]

        sum_features = np.zeros(6 * 3 ** (k - 1))
        degree = G.out_degree[node_id]
        for nbr_id in nbr_ids:
            nbr_features = G.nodes[nbr_id]["features"]
            sum_features = np.add(sum_features, nbr_features[0:6 * 3 ** (k - 1)])

        # append sum and mean only if the target nodes features length <=3*k
        new_features = features
        if degree == 0:
            new_features = np.append(features, np.zeros(6 * 3 ** (k - 1) * 2))
        else:
            new_features = np.append(new_features, np.divide(sum_features, degree * 1.0))
            new_features = np.append(new_features, sum_features)

        G.nodes[node_id]["features"] = new_features




# main code start here

G = loadGraph()

# drawNxGraph(getEgoNet(G, node_id), "Ego_Net_sockpuppet_{0}_{1}".format(dataset, node_id))

initFeatures(G)
recursive(G, 1)
recursive(G, 2)
recursive(G, 3)

