from __future__ import division

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import networkx_utils as utils
import cPickle
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs

# dataset = "alpha"
#
# G = nx.read_gpickle("results/%s_graph_embedding_featured_graph.pkl" % (dataset))
# # gt_bad_users = cPickle.load(open("./results/%s_gt_bad_users_set.pkl" % dataset, "rb"))
# # gt_good_users = cPickle.load(open("./results/%s_gt_good_users_set.pkl" % dataset, "rb"))
# utils.drawNxGrap(G,"%s network" % dataset, "./diagram/%s_network.png" % dataset)


X, y_true = make_blobs(n_samples=300, centers=4,
                       cluster_std=0.60, random_state=0)
# plt.scatter(X[:, 0], X[:, 1], s=50);
# plt.show()


kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)

plt.show()