from __future__ import division

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

dataset = "alpha"


def hist(values, bins, xlabel, ylabel, title, fileName):
    n, bins, patches = plt.hist(x=values, bins=bins, color='#0504aa',
                                alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    # plt.text(23, 45, r'$k=2, N=total_nodes$')
    maxfreq = n.max()
    # Set a clean upper y-axis limit.
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

    # plt.show()
    plt.savefig(fileName)
    plt.gcf().clear()

    # print n, bins


def drawNxGraphNodeGroups(NxG, good_users, bad_users, other_users, title, fileName):
    pos = nx.spring_layout(NxG)

    color_map = {3: 'r', 2: 'b', 1: 'y'}

    nx.draw_networkx_nodes(NxG, pos, node_size=3, nodelist=other_users, node_color=color_map[1])
    nx.draw_networkx_nodes(NxG, pos, node_size=3, nodelist=bad_users, node_color=color_map[3])
    nx.draw_networkx_nodes(NxG, pos, node_size=3, nodelist=good_users, node_color=color_map[2])
    # nx.draw_networkx_nodes(NxG, pos, node_list =[129,46], node_color='r')

    nx.draw_networkx_labels(NxG, pos, node_size=2, font_size=6)
    nx.draw_networkx_edges(NxG, pos)
    plt.figure(3, figsize=(12, 12))
    plt.title(title)
    plt.savefig(fileName)

    plt.gcf().clear()


def drawNxGrap(NxG, title, fileName):
    pos = nx.spring_layout(NxG)
    nx.draw_networkx(NxG, pos)


    # nx.draw_networkx_nodes(NxG, pos, node_size=3)
    # nx.draw_networkx_labels(NxG, pos, font_size=6)
    # nx.draw_networkx_edges(NxG, pos, alpha=0.2)
    plt.figure(3, figsize=(12, 12))
    plt.title(title)
    plt.savefig(fileName)

    plt.gcf().clear()
