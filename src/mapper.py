import itertools
import math

import numpy as np
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


def dist_from_origin(p): return math.sqrt(p[0]**2 + p[1]**2)


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
    while ds.n_subsets > 1:
        s1, s2 = min(itertools.combinations(ds.subsets(), 2), key=lambda subset_pair: d(
            *min(itertools.product(*subset_pair), key=lambda point_pair: d(*point_pair))))
        if d(*min(itertools.product(s1, s2), key=lambda t: d(*t))) > thresh:
            break
        ds.merge(next(iter(s1)), next(iter(s2)))
    return ds.subsets()


def preimage(interval, ps, f):
    mn, mx = interval
    return [p for p in ps if mn <= f(p) <= mx]


def create_intervals(range_min, range_max, num_intervals, gain):
    f_range = range_max - range_min
    interval_length = f_range / num_intervals

    intervals = [(range_min + i * interval_length, range_min + (i + 1) * interval_length) for i in range(num_intervals)]
    interval_extension = gain * interval_length
    intervals = [(mn - interval_extension, mx + interval_extension) for (mn, mx) in intervals]

    return intervals


def create_mapper_graph(data, config):
    filter_function = config['filter_function']
    if 'data' in signature(filter_function).parameters:
        filter_function(None, data)
    f_min = filter_function(min(data, key=filter_function))
    f_max = filter_function(max(data, key=filter_function))
    # print(f_min, f_max)

    num_intervals = config['num_intervals']
    gain = config['gain']
    intervals = create_intervals(f_min, f_max, num_intervals, gain)
    # print(intervals)
    # for interval in intervals:
    #     print(preimage(interval, data, filter_function))

    distance_threshold = config['distance_threshold']
    clusters = []
    for interval in intervals:
        clusters += single_linkage(preimage(interval, data, filter_function), dist, distance_threshold)

    g = nx.Graph()
    for cluster in clusters:
        g.add_node(tuple(cluster))

    for c1, c2 in itertools.combinations(clusters, 2):
        if not c1.isdisjoint(c2):
            g.add_edge(tuple(c1), tuple(c2))

    return g


def main():
    data = get_data("data/exploratory_data.csv")

    # ADJUST: choose parameter possible values
    possible_filters = [x_proj, y_proj, eccentricity, centrality]
    possible_num_intervals = range(10, 21, 5)
    possible_gains = np.linspace(0.2, 0.4, 3)
    possible_distance_thresholds = np.linspace(0.3, 1, 3)
    possible_filters = [x_proj, y_proj, eccentricity, centrality, dist_from_origin]

    for filter_function, num_intervals, gain, distance_threshold \
            in itertools.product(possible_filters,
                                 possible_num_intervals,
                                 possible_gains,
                                 possible_distance_thresholds):
        config = {
            'filter_function': filter_function,  # the filter function that maps points to the number line
            'num_intervals': num_intervals,  # number of intervals to take on the number line, within range
            'gain': gain,  # how much overlap is between intervals, should stay < 0.5 probably
            'distance_threshold': distance_threshold  # threshold before clustering gives us
        }

        g = create_mapper_graph(data, config)

        f, ax = plt.subplots(1)
        ax.set_title(f"Filter: {config['filter_function'].__name__},"
                     f" # Intervals: {config['num_intervals']},"
                     f" Gain: {config['gain']},"
                     f" Threshold: {config['distance_threshold']}")
        nx.draw_networkx(g, node_size=100, pos={node: np.average(node, axis=0) for node in g.nodes}, with_labels=False, ax=ax)
        plt.show()


if __name__ == '__main__':
    main()
