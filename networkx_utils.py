from __future__ import division

import matplotlib.pyplot as plt
import numpy as np

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

    print n, bins

