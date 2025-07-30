import itertools
import math

import pandas as pd
import networkx as nx
from scipy.cluster.hierarchy import DisjointSet
from matplotlib import pyplot as plt

from inspect import signature


def get_data(file_path):
    df = pd.read_csv(file_path)
    nd = df.to_numpy()
    x = nd[:, 0]
    y = nd[:, 1]

    plt.plot(x, y, 'o')
    plt.show()

    data = [(float(x), float(y)) for [x, y] in nd]
    return data


def dist(p1: tuple[float, float], p2: tuple[float, float]):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def x_proj(p): return p[0]  # projection onto axis


def y_proj(p): return p[1]  # projection onto axis


def centrality(p, data=None):
    if data is not None:
        centrality.data = data
        return None
    if centrality.data is None:
        raise Exception("Must set data before computing function")
    return sum([dist(p, x) for x in centrality.data])


def eccentricity(p, data=None):
    if data is not None:
        eccentricity.data = data
        return None
    if eccentricity.data is None:
        raise Exception("Must set data before computing function")
    return max([dist(p, x) for x in eccentricity.data])


def single_linkage(ps, d, thresh):
    ds = DisjointSet(ps)
    curr_dist = 0
    while ds.n_subsets > 1 and curr_dist < thresh:
        s1, s2 = min(itertools.combinations(ds.subsets(), 2), key=lambda subset_pair: d(
            *min(itertools.product(*subset_pair), key=lambda point_pair: d(*point_pair))))
        ds.merge(next(iter(s1)), next(iter(s2)))
        curr_dist = d(*min(itertools.product(s1, s2), key=lambda t: d(*t)))
    return ds.subsets()


def preimage(interval, ps, f):
    mn, mx = interval
    return [p for p in ps if mn <= f(p) <= mx]


def create_mapper_graph(data, config):
    filter_function = config['filter_function']
    if 'data' in signature(filter_function).parameters:
        filter_function(None, data)
    f_min = filter_function(min(data, key=filter_function))
    f_max = filter_function(max(data, key=filter_function))
    f_range = f_max - f_min
    # print(f_min, f_max)

    num_intervals = config['num_intervals']

    interval_length = f_range / num_intervals

    intervals = [(f_min + i * interval_length, f_min + (i + 1) * interval_length) for i in range(num_intervals)]
    # print(intervals)
    gain = config['gain']
    interval_extension = gain * interval_length
    intervals = [(mn - interval_extension, mx + interval_extension) for (mn, mx) in intervals]
    # print(intervals)
    # for interval in intervals:
    #     print(preimage(interval, data))

    distance_threshold = config['distance_threshold']
    clusters = []
    for interval in intervals:
        clusters += single_linkage(preimage(interval, data, filter_function), dist, distance_threshold)

    g = nx.Graph()
    for cluster in clusters:
        g.add_node(clusters.index(cluster))

    for c1, c2 in itertools.combinations(clusters, 2):
        if not c1.isdisjoint(c2):
            g.add_edge(clusters.index(c1), clusters.index(c2))

    return g


def main():
    config = {  # ADJUST: to your liking
        'filter_function': eccentricity,    # the filter function that maps points to the number line
        'num_intervals': 20,                # number of intervals to take on the number line, within the range
        'gain': 0.4,                        # how much overlap is between intervals, should stay < 0.5 maybe
        'distance_threshold': 2.0           # threshold before clustering gives us
    }

    data = get_data("data/exploratory_data.csv")

    g = create_mapper_graph(data, config)

    nx.draw(g)
    plt.show()


if __name__ == '__main__':
    main()
