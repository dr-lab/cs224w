from __future__ import division

import collections
import random

import networkx as nx
import numpy as np
import cPickle
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import hdbscan
import seaborn as sns

import ground_truth_analysis as gt

dataset = "alpha"


def load_graph():
    G = nx.read_gpickle(
        "results/%s_graph_embedding_featured_graph.pkl" % (dataset))

    print "Total nodes=%d" % G.number_of_nodes()
    print "Total edges=%d" % G.number_of_edges()

    return G


def distance(x, y):
    return np.sqrt(np.sum(np.square(x - y)))
    # x2 = np.sqrt(np.sum(np.square(x)))
    # y2 = np.sqrt(np.sum(np.square(y)))

    # if x2 == 0 or y2 == 0:
    #     return 1

    # return np.matmul(x, y)/ (x2 * y2)


def find_closest_center(score, centers):
    return min(range(len(centers)), key=lambda i: distance(score, centers[i]))


def k_means_clustering(score_map, k):
    scores = list(score_map.values())
    centers = random.sample(scores, k)

    while True:
        # 2a. Assign labels based on closest center
        labels = {
            node_id: find_closest_center(score, centers)
            for node_id, score in score_map.items()
        }

        # 2b. Find new centers from means of points
        new_centers = np.copy(centers)
        for i in range(k):
            cluster = [score_map[j] for j in score_map if labels[j] == i]
            if len(cluster) != 0:
                new_centers[i] = np.array(cluster).mean(0)

        # 2c. Check for convergence
        if np.all(centers == new_centers):
            break
        centers = new_centers

    return centers, labels