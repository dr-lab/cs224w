from __future__ import division

import collections
import random

import networkx as nx
import numpy as np
from sklearn.metrics import pairwise_distances_argmin

dataset = "alpha"


def load_graph():
    G = nx.read_gpickle("./results/%s_graph_embedding_vectors.pkl" % (dataset))

    print "Total nodes=%d" % G.number_of_nodes()
    print "Total edges=%d" % G.number_of_edges()

    return G


def k_means_clustering(scores, k):
    center_indices = np.random.choice(scores, k)
    centers = scores[center_indices]

    while True:
        # 2a. Assign labels based on closest center
        labels = pairwise_distances_argmin(scores, centers)

        # 2b. Find new centers from means of points
        new_centers = np.array([scores[labels == i].mean(0) for i in range(k)])

        # 2c. Check for convergence
        if np.all(centers == new_centers):
            break
        centers = new_centers

    return centers, labels


def main():
    graph = load_graph()
    scores = [node["features"] for node in graph.node.values()]
    centers, labels = k_means_clustering(scores, 3)


if __name__ == "__main__":
    main()