from __future__ import division

import networkx as nx
import numpy as np
import collections
import matplotlib.pyplot as plt
import cPickle

dataset = "alpha"


def analysisGT():
    good_users = []
    bad_users = []

    filePath = "./rev2/data/%s_gt.csv" % dataset
    print "Load gt from {0}".format(filePath)

    f = open(filePath, "r")
    for l in f:
        ls = l.strip().split(",")
        node_id = int(ls[0])
        label = int(ls[1])

        if label > 0:
            good_users.append(node_id)
        else:
            bad_users.append(node_id)
    f.close()

    with open("./results/%s_gt_good_users_set.pkl" % dataset, 'wb') as good_users_pickle_file:
        cPickle.dump(good_users, good_users_pickle_file)

    with open("./results/%s_gt_bad_users_set.pkl" % dataset, 'wb') as bad_users_pickle_file:
        cPickle.dump(bad_users, bad_users_pickle_file)

    print "good users: %d" % len(good_users)
    print "bad users: %d" % len(bad_users)


def intersectUsers(my_bad_users, gt_bad_users):
    # print my_bad_users
    # print gt_bad_users
    intersected = set(my_bad_users) & set(gt_bad_users)

    # print len(intersected)
    # print (intersected)
    return intersected


analysisGT()
