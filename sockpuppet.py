from __future__ import division
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sets import Set
import collections
import networkx as nx

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

    return G


def drawNxGraph(NxG, title):
    pos = nx.spring_layout(NxG)
    nx.draw(NxG, pos)
    nx.draw_networkx_labels(NxG, pos,font_size=4)
    # plt.figure(3, figsize=(12, 12))
    plt.savefig("./project_{0}_{1}.PNG".format(dataset, title))

    plt.gcf().clear()


def drawNxGraphNodeGroups(NxG, node_groups, title):
    pos = nx.spring_layout(NxG)

    color_map = {4: 'r', 3: 'b', 2: 'y', 1: 'g', 0: 'grey'}

    keys = node_groups.keys()
    keys.sort(reverse=False)

    for key in keys:
        nx.draw_networkx_nodes(NxG, pos, node_size=3, nodelist=node_groups[key], node_color=color_map[key])
    # nx.draw_networkx_nodes(NxG, pos, node_list =[129,46], node_color='r')

    nx.draw_networkx_labels(NxG, pos, node_size=2, font_size=6)
    nx.draw_networkx_edges(NxG, pos)
    plt.figure(3, figsize=(12, 12))
    plt.savefig("./project_big_{0}_{1}.PNG".format(dataset, title))

    plt.gcf().clear()


# print G.number_of_nodes()
# print G.number_of_edges()
# print G.degree(345)
# print G.out_degree(345)
# print list(G.adj[345])
# print G.adj[345].keys()
# print G[345]

# for e in list(G.edges):
#     # print e ,
#     G[e[0]][e[1]]["rating2"] = (4,5,6,7)
#     print e,G[e[0]][e[1]]["rating"],  G[e[0]][e[1]]["rating2"]

def getEgoNet(G, node_id):
    ego_net = nx.ego_graph(G, node_id)
    return ego_net


def findFeatures(G, node_id):
    out_degree = G.out_degree[node_id]
    in_degree = G.degree[node_id]
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
    print features
    return features


def initFeatures(G):
    nodes = G.nodes

    for node_id in nodes:
        features = findFeatures(G, node_id)
        print features
        G.nodes[node_id]["features"] = features


def recursive(G, k):
    nodes = G.nodes
    for node_id in nodes:
        print G.nodes[node_id]["features"]
        features = G.nodes[node_id]["features"]

        # if the features length > 6*3**(k-1), means it is already been processed
        # print node_id
        # print features

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


def simScore(x, y):
    sum = np.matmul(x, y)
    x2 = np.sqrt(np.sum(np.square(x)))
    y2 = np.sqrt(np.sum(np.square(y)))

    if x2 == 0 or y2 == 0:
        return 0

    score = sum / (x2 * y2)
    return score


def printTop10SimNodes(feature_sim_node_score_map):
    name_score_tuple = collections.namedtuple('sortor', 'name score')

    reverse_sorted = sorted([name_score_tuple(v, k) for (k, v) in feature_sim_node_score_map.items()], reverse=True)

    for i in range(0, 10):
        reverse_sorted[i]
        print "score: %s %s" % (reverse_sorted[i].name, reverse_sorted[i].score)
        drawNxGraph(getEgoNet(G, reverse_sorted[i].score), "Ego_Net_{0}".format(reverse_sorted[i].score))


def calSimAndPrint(G, base_node_id):
    Feature_Sim_Node_Score_Map = {}
    base_features = G.nodes[base_node_id]["features"]

    nodes = G.nodes
    for node_id in nodes:
        if node_id != base_node_id:
            features_y = G.nodes[node_id]["features"]
            score = simScore(base_features, features_y)
            Feature_Sim_Node_Score_Map[node_id] = score

    print "feature of node base_node_id %d %s" % (base_node_id, base_features)

    printTop10SimNodes(Feature_Sim_Node_Score_Map)
    return Feature_Sim_Node_Score_Map


def groupNodes(sim_score_values_map, top_his, group_count):
    node_groups = {}

    for node_id in sim_score_values_map.keys():
        score = sim_score_values_map[node_id]
        group_id = int(score * 100 / (100 / 20))

        if group_id in top_his:
            index = top_his.index(group_id)

            if index in node_groups:
                node_groups[index].append(node_id)
            else:
                node_groups[index] = [node_id]

        elif (group_count-1) in node_groups:
            node_groups[group_count-1].append(node_id)
        else:
            node_groups[group_count-1] = [node_id]
    return node_groups

def findBiggestNbrNode(G):
    nodes = G.nodes
    degree = 0
    biggest_degree_node_id = 0

    node_ids = []
    node_degree = []
    for node_id in nodes:
        node_ids.append(node_id)
        out_degree = G.out_degree[node_id]
        node_degree.append(out_degree)
        if degree < out_degree:
            degree = out_degree
            biggest_degree_node_id = node_id
    print biggest_degree_node_id, degree
    return [biggest_degree_node_id, degree, node_ids, out_degree]

def mutualNbrAcrosCommunites(G, node_groups, base_node_id):
    base_adjs = list(G.adj[base_node_id])

    connected = {}
    for key in node_groups.keys():
        nbrs = node_groups[key]
        connected[key] = set(base_adjs)&set(nbrs)

    mutual_connected = {}
    for key in connected.keys():
        connected_nbrs = connected[key]

        mutual_connected[key] = []
        for connected_nbr_id in connected_nbrs:
            if G.has_edge(base_node_id, connected_nbr_id) and G.has_edge(connected_nbr_id, base_node_id):
                mutual_connected[key].append(connected_nbr_id)

    return mutual_connected

# main code start here

G = loadGraph()
(node_id, degree, node_ids, node_degree) = findBiggestNbrNode(G)

drawNxGraph(getEgoNet(G, node_id), "Ego_Net_sockpuppet_{0}_{1}".format(dataset, node_id))

initFeatures(G)
recursive(G, 1)
recursive(G, 2)
recursive(G, 3)

sim_score_values_map = calSimAndPrint(G, node_id)

n, bins, patches = plt.hist(x=sim_score_values_map.values(), bins=20, color='#0504aa',
                            alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Sim-Score')
plt.ylabel('Frequency')
plt.title('Similarity Score Histogram k=2 - {0}'.format(dataset))

maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

# plt.show()
plt.savefig("./project_sim_score_histogram_{0}.PNG".format(dataset))
plt.gcf().clear()

print n, bins

# find the index of top values
top_his = sorted(range(len(n)), key=lambda i: n[i], reverse=True)[:9]

node_groups = groupNodes(sim_score_values_map, top_his, len(top_his) + 1)

mutual_connected = mutualNbrAcrosCommunites(G, node_groups, node_id)


mutual_x = mutual_connected.keys()
mutual_y = []
for mutual_x_v in mutual_x:
    value = mutual_connected[mutual_x_v]
    mutual_y.append(len(value))

plt.bar(mutual_x, mutual_y)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Communities')
plt.ylabel('Mutual Ratings')
plt.title('Mutual Ratings within and across Communities - {0}'.format(dataset))
plt.savefig("./project_related_communites_{0}.PNG".format(dataset))
plt.gcf().clear()

# plt.bar(mutual_x, mutual_y)
# plt.grid(axis='y', alpha=0.75)
# plt.xlabel('Communities')
# plt.ylabel('Mutual Ratings')
# plt.title('Mutual Ratings within and across Communities - {0}'.format(dataset))
# plt.savefig("./project_related_communites_{0}.PNG".format(dataset))
# plt.gcf().clear()

# drawNxGraphNodeGroups(G, node_groups, "node_groups")
