import networkx as nx
import numpy as np
import cPickle
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import hdbscan
import seaborn as sns

from k_means import k_means_clustering

dataset = "alpha"


def load_graph():
    G = nx.read_gpickle(
        "results/%s_graph_embedding_featured_graph.pkl" % (dataset))

    print "Total nodes=%d" % G.number_of_nodes()
    print "Total edges=%d" % G.number_of_edges()

    return G


def plot(reduced_features, cluster_colors, name, node_ids, gt_good_users, gt_bad_users):
    plt.clf()
    plt.scatter(
        *reduced_features.T,
        s=50,
        c=cluster_colors,
        alpha=0.25)

    feature_map = dict(zip(node_ids, reduced_features))
    bad_user_features = np.array(
        [feature_map[node_id] for node_id in gt_bad_users])
    plt.scatter(*bad_user_features.T, s=50, c="black")

    good_user_features = np.array(
        [feature_map[node_id] for node_id in gt_good_users])
    plt.scatter(*good_user_features.T, s=50, c="white", edgecolors="black")

    plt.savefig('diagram/%s.png' % name)


def get_hdbscan_colors(reduced_features, color_palette):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=50).fit(reduced_features)
    cluster_colors = [
        color_palette[x] if x >= 0 else (0.5, 0.5, 0.5)
        for x in clusterer.labels_
    ]
    # return [
    #     sns.desaturate(x, p)
    #     for x, p in zip(cluster_colors, clusterer.probabilities_)
    # ], clusterer.labels_.max() + 1
    return cluster_colors, clusterer.labels_.max() + 1


def get_k_means_colors(node_ids, reduced_features, color_palette, k):
    score_map = dict(zip(node_ids, reduced_features))
    centers, labels = k_means_clustering(score_map, k)
    return [
        color_palette[labels[node_id]] for node_id in labels
    ]


def main():
    graph = load_graph()
    all_features = ["features", "features_pos_neg", "features_pos_neg_rev2"]

    gt_bad_users = cPickle.load(
        open("./results/%s_gt_bad_users_set.pkl" % dataset, "rb"))

    gt_good_users = cPickle.load(
        open("./results/%s_gt_good_users_set.pkl" % dataset, "rb"))
    print "Number of ground truth bad users: %d " % len(gt_bad_users)
    print "Number of ground truth good users: %d" % len(gt_good_users)

    color_palette = sns.color_palette("Paired", 15)


    for feature_name in all_features:
        node_ids = graph.node.keys()
        features = np.array(
            [node[feature_name][1:] for node in graph.node.values()])
        reduced_features = TSNE(n_components=2).fit_transform(features)

        hdbscan_cluster_colors, k = get_hdbscan_colors(reduced_features, color_palette)
        print "Number of clusters: ", k
        plot(reduced_features, hdbscan_cluster_colors, "hdbscan_" + feature_name, node_ids, gt_good_users, gt_bad_users)


        k_means_cluster_colors = get_k_means_colors(node_ids, reduced_features, color_palette, k)
        plot(reduced_features, k_means_cluster_colors, "k_means_" + feature_name, node_ids, gt_good_users, gt_bad_users)


if __name__ == "__main__":
    main()
