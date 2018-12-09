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


def main():
    graph = load_graph()
    feature_name = "features"
    features = np.array(
        [node[feature_name][1:] for node in graph.node.values()])
    reduced_features = TSNE(n_components=2).fit_transform(features)
    # plt.scatter(*reduced_features.T, s=50, linewidth=0, c='b', alpha=0.25)
    # plt.show()

    score_map = dict(zip(graph.node.keys(), reduced_features))

    gt_bad_users = cPickle.load(
        open("./results/%s_gt_bad_users_set.pkl" % dataset, "rb"))

    gt_good_users = cPickle.load(
        open("./results/%s_gt_good_users_set.pkl" % dataset, "rb"))
    print "Number of ground truth bad users: %d " % len(gt_bad_users)
    print "Number of ground truth good users: %d" % len(gt_good_users)


    clusterer = hdbscan.HDBSCAN(min_cluster_size=50).fit(reduced_features)
    color_palette = sns.color_palette('deep', 20)
    print clusterer.labels_.max()
    cluster_colors = [
        color_palette[x] if x >= 0 else (0.5, 0.5, 0.5) for x in clusterer.labels_
    ]
    cluster_member_colors = [
        sns.desaturate(x, p)
        for x, p in zip(cluster_colors, clusterer.probabilities_)
    ]

    plt.clf()
    plt.scatter(*reduced_features.T, s=50, linewidth=0, c=cluster_member_colors, alpha=0.25)
    plt.savefig('hdbscan.png')

    centers, labels = k_means_clustering(score_map, 10)
    cluster_colors = [
        color_palette[labels[node_id]] if x >= 0 else (0.5, 0.5, 0.5) for node_id in labels
    ]
    plt.clf()
    plt.scatter(
        *reduced_features.T,
        s=50,
        linewidth=0,
        c=cluster_colors,
        alpha=0.25)
    plt.savefig('k_means.png')

    # for k in range(2, 10):
    # print
    # print "For %d clusters" % k
    # centers, labels = k_means_clustering(score_map, k)
    # for i in range(k):
    #     cluster = [node_id for node_id in labels if labels[node_id] == i]
    #     print "Cluster ", i, " has: "
    #     print "%d users" % len(cluster)
    #     bad_users = set(gt_bad_users) & set(cluster)
    #     good_users = set(gt_good_users) & set(cluster)
    #     print "%d ground truth bad users, %.2f%% of the cluster" % (
    #         len(bad_users), len(bad_users) / len(cluster) * 100)
    #     print "%d ground truth good users, %.2f%% of the cluster" % (
    #         len(good_users), len(good_users) / len(cluster) * 100)


if __name__ == "__main__":
    main()