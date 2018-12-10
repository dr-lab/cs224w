
import pandas as pd
import networkx as nx
import numpy as np
import networkx_utils as utils

network = "alpha"
all_features = ["features","features_pos_neg", "features_pos_neg_rev2"]

N = [5,10,15,20,25,30,35,40,45,50]
gt_user_types = ["good","bad"]

def plot_features():
    feature_scores = {}
    for n in N:
        for feature in all_features:
            scores = 0
            for gt_user_type in gt_user_types:

                intersect = pd.read_csv("./results/%s_%s_%d_gt_%s_users_intersect.csv"%(feature, network, n, gt_user_type), delimiter = ',')
                means = intersect.mean(axis=0)
                m3 =  means/n
                l2_topk_bad = m3[1]
                l2_topk_good = m3[2]
                l2_lastk_bad = m3[4]
                l2_lastk_good = m3[5]

                cosine_topk_bad = m3[7]
                cosine_topk_good = m3[8]
                cosine_lastk_bad = m3[10]
                cosine_lastk_good = m3[11]

                if gt_user_type == "good":
                    scores += (l2_topk_good + l2_lastk_bad) / 4 + (cosine_topk_good + cosine_lastk_bad) / 4
                else:
                    scores += (l2_topk_bad +l2_lastk_good) / 4 + (cosine_topk_bad + cosine_lastk_good) / 4

            if n in feature_scores:
                feature_scores[n].append(scores)
            else:
                feature_scores[n] = [scores]

    # legends = ["L2 Similar %","Cosine Similarity %"]
    utils.plotSimScore(N, feature_scores.values(), all_features,
                       "How many nodes picked from the sorted similarity score (K)",
                       "% on K of intersected nodes within GT (the higher the better)",
                       "Naive Similarity Score comparison w/wo REV2 features",
                       "./diagram/naive_sim_score_analysis.png")
    print feature, gt_user_type, scores, N


def plot_full():
    for feature in all_features:
        for gt_user_type in gt_user_types:

            scores = []
            for n in N:
                intersect = pd.read_csv("./results/%s_%s_%d_gt_%s_users_intersect.csv"%(feature, network, n, gt_user_type), delimiter = ',')
                # print intersect.shape
                means = intersect.mean(axis=0)

                m3 =  means/n
                l2_topk_bad = m3[1]
                l2_topk_good = m3[2]
                l2_lastk_bad = m3[4]
                l2_lastk_good = m3[5]

                cosine_topk_bad = m3[7]
                cosine_topk_good = m3[8]
                cosine_lastk_bad = m3[10]
                cosine_lastk_good = m3[11]

                if gt_user_type == "good":
                    scores.append(((l2_topk_good + l2_lastk_bad) / 2, (cosine_topk_good + cosine_lastk_bad) / 2))
                else:
                    scores.append(((l2_topk_bad +l2_lastk_good) / 2, (cosine_topk_bad + cosine_lastk_good) / 2))


            legends = ["L2 Similar %","Cosine Similarity %"]
            utils.plotSimScore(N, scores, legends, "How many nodes picked from the sorted similarity score (K)",
                               "% on K of intersected nodes within GT (the higher the better)",
                               "Comparison of performance of Cosine vs. L2 %s %s" % (feature, gt_user_type),
                               "./diagram/cosine_l2_comparision_%s_%s.png" % (feature, gt_user_type))
            print feature, gt_user_type, scores, N

plot_full()
plot_features()