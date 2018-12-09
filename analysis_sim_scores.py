
import pandas as pd
import networkx as nx
import numpy as np

network = "alpha"
all_features = ["features", "features_pos_neg", "features_pos_neg_rev2"]

N = [2,3,4,5,6,7,8,9,10,15,20,25,30,35,40,45,50]
gt_user_types = ["good","bad"]

for feature in all_features:
    for gt_user_type in gt_user_types:
        for n in N:
            intersect = pd.read_csv("./results/%s_%s_%d_gt_%s_users_intersect.csv"%(feature, network, n, gt_user_type), delimiter = ',')
            print intersect.shape
            mean =  intersect.mean(axis=0)
            print mean
            print mean["l2_topk_good"]



            # print intersect[1].sum

            print(intersect.describe())

            # intersect = intersect.append(intersect.agg(['sum','mean']))

            # print intersect
            print sum
            # intersect.columns = ['src', 'dst', 'weight', 'delta']


