import itertools
import math

import pandas as pd
import networkx as nx
from scipy.cluster.hierarchy import DisjointSet
from matplotlib import pyplot as plt

df = pd.read_csv("../data/exploratory_data.csv")
nd = df.to_numpy()
x = nd[:, 0]
y = nd[:, 1]

plt.plot(x, y, 'o')
plt.show()

data = [(float(x), float(y)) for [x, y] in nd]

x_proj = lambda p: p[0]  # projection onto axis
y_proj = lambda p: p[1]  # projection onto axis


def dist(p1: tuple[float, float], p2: tuple[float, float]):  # TODO: try different dist functions?
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def centrality(p):
    return sum([dist(p, x) for x in data])


def eccentricity(p):
    return max([dist(p, x) for x in data])


f = eccentricity  # ADJUST: filter function

f_min = f(min(data, key=f))
f_max = f(max(data, key=f))
f_range = f_max - f_min
# print(f_min, f_max)

num_intervals = 20  # ADJUST: number of intervals

interval_length = f_range / num_intervals

intervals = [(f_min + i * interval_length, f_min + (i + 1) * interval_length) for i in range(num_intervals)]
# print(intervals)
gain = 0.4  # ADJUST: gain, should stay <0.5 maybe
interval_extension = gain * interval_length
intervals = [(mn - interval_extension, mx + interval_extension) for (mn, mx) in intervals]


# print(intervals)


def preimage(interval, ps):
    mn, mx = interval
    return [p for p in ps if mn <= f(p) <= mx]


for interval in intervals:
    # print(preimage(interval, data))
    pass


def single_linkage(ps, d, thresh):
    ds = DisjointSet(ps)
    curr_dist = 0
    while ds.n_subsets > 1 and curr_dist < thresh:
        s1, s2 = min(itertools.combinations(ds.subsets(), 2), key=lambda subset_pair: d(
            *min(itertools.product(*subset_pair), key=lambda point_pair: d(*point_pair))))
        ds.merge(next(iter(s1)), next(iter(s2)))
        curr_dist = d(*min(itertools.product(s1, s2), key=lambda t: d(*t)))
    return ds.subsets()


distance_threshold = 2  # ADJUST
clusters = []
for interval in intervals:
    clusters += single_linkage(preimage(interval, data), dist, distance_threshold)

g = nx.Graph()
for cluster in clusters:
    g.add_node(clusters.index(cluster))

for c1, c2 in itertools.combinations(clusters, 2):
    if not c1.isdisjoint(c2):
        g.add_edge(clusters.index(c1), clusters.index(c2))

nx.draw(g)
plt.show()
