import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import chain, combinations


# from line_profiler_pycharm import profile


def subsets(iterable, k=None):
    s = list(iterable)
    if k is None:
        k = len(s)
    return chain.from_iterable(combinations(s, r) for r in range(k + 1))


def dist(p_1: tuple[float, float], p_2: tuple[float, float]):
    return math.sqrt((p_1[0] - p_2[0]) ** 2 + (p_1[1] - p_2[1]) ** 2)


def dist_sq(p_1: tuple[float, float], p_2: tuple[float, float]):
    return (p_1[0] - p_2[0]) ** 2 + (p_1[1] - p_2[1]) ** 2


def vietoris_rips_complex(p: list[tuple[float, float]], r: float, k: int | None = None):
    res = []
    for s in subsets(p, k):
        for p_1, p_2 in combinations(s, 2):
            if dist_sq(p_1, p_2) > (2 * r) ** 2:
                break
        else:
            res.append(tuple(sorted(s)))
    return res


# @profile
def find_pivot(matrix, c, possible_pivots):
    column = matrix[:, c]
    possible_places = column # & possible_pivots
    if not np.any(possible_places):
        return None
    index = np.max(np.argmax(possible_places))
    return index


def get_data(file_path):
    df = pd.read_csv(file_path)
    nd = df.to_numpy()
    nd = nd[np.random.choice(len(nd), size=len(nd) // 2, replace=False)]
    x = nd[:, 0]
    y = nd[:, 1]

    plt.plot(x, y, 'o')
    plt.show()

    data = [(float(x), float(y)) for [x, y] in nd]
    data = list(set(data))  # remove repetitions
    # data = [(0.3, 1), (-0.81, 0.5879), (-0.81, -0.5877), (0.3, -0.98), (1, 0)]
    # data = [(0.01, 1), (1.02, 0), (0, -0.99), (-1, 0.02)]
    return data


# @profile
def main():
    data = get_data("../data/data_A.csv")

    # TODO: we probably have overlapping distances
    dists_sq = {}
    for x1 in data:
        for x2 in data:
            if x1 < x2:
                dists_sq[dist_sq(x1, x2)] = (x1, x2)
    # plt.hist(dists_sq, bins=1000)
    # plt.show()
    # exit(0)
    max_p = 3  # ADJUST: 3
    simplices = [[] for _ in range(max_p + 1)]  # store all simplices for each dimension by order created
    simplices_set = set()  # remember all simplices that exist
    simplices_births = [{} for _ in
                        range(max_p + 1)]  # store for each dimension for each simplex index the dist it was born
    simplices_indices = [{} for _ in range(max_p + 1)]
    simplices[0] = [tuple()]
    simplices_set.add(tuple())
    simplices_births[0][0] = math.sqrt(0)
    simplices_indices[0][tuple()] = 0
    simplices[1] = [(x,) for x in data]
    for x in data:
        simplices_set.add((x,))
    for i in range(len(simplices[1])):
        simplices_births[1][i] = math.sqrt(0)
    for i, simp in enumerate(simplices[1]):
        simplices_indices[1][simp] = i

    max_dist_sq = float('inf')  # ADJUST
    print("Calculating simplex order")
    for i, (curr_dist_sq, (x1, x2)) in enumerate(sorted(dists_sq.items())):
        if curr_dist_sq > max_dist_sq:
            break
        if len(dists_sq) < 10 or i % (len(dists_sq) // 10) == 0:
            print(f"\tOn dist_sq {i}/{len(dists_sq)} ({curr_dist_sq})")
        # ----------------------------------
        # This is a more general implementation but extremely slow
        # ----------------------------------
        # for p in range(max_p - 2 + 1):
        #     for simp in itertools.combinations(data, p):
        #         if x1 in simp or x2 in simp:
        #             continue
        #         if tuple(sorted(simp + (x1,))) in simplices[p + 1] and tuple(sorted(simp + (x2,))) in simplices[p + 1]:
        #             new_simp = tuple(sorted(simp + (x1, x2)))
        #             simplices[p + 2].append(new_simp)
        #             simplices_set.add(new_simp)
        #             simplices_births[p + 2][len(simplices[p + 2]) - 1] = math.sqrt(curr_dist_sq)
        #             simplices_indices[p + 2][new_simp] = len(simplices[p + 2]) - 1
        # ----------------------------------
        new_simp = tuple(sorted((x1, x2)))
        simplices[2].append(new_simp)
        simplices_set.add(new_simp)
        simplices_births[2][len(simplices[2]) - 1] = math.sqrt(curr_dist_sq)
        simplices_indices[2][new_simp] = len(simplices[2]) - 1
        for x in data:
            if x == x1 or x == x2:
                continue
            if dist_sq(x, x1) <= curr_dist_sq and dist_sq(x, x2) <= curr_dist_sq:
                new_simp = tuple(sorted((x1, x2, x)))
                simplices[3].append(new_simp)
                simplices_set.add(new_simp)
                simplices_births[3][len(simplices[3]) - 1] = math.sqrt(curr_dist_sq)
                simplices_indices[3][new_simp] = len(simplices[3]) - 1

    boundary_matrices = [np.zeros((len(simplices[p]), len(simplices[p + 1])), dtype=bool)
                         for p in range(max_p - 1 + 1)]
    for p in range(max_p - 1 + 1):
        print(f"Populating boundary matrix for {p=}")
        for i, simp in enumerate(simplices[p]):
            if len(simplices[p]) < 10 or i % (len(simplices[p]) // 10) == 0:
                print(f"\tOn simplex {i}/{len(simplices[p])}")
            for x in data:
                if x in simp:
                    continue
                new_simp = tuple(sorted(simp + (x,)))
                if new_simp in simplices_set:
                    boundary_matrices[p][i][simplices_indices[p + 1][new_simp]] = 1
                else:
                    # print("no")
                    ...

    birth_simplices = [set() for _ in
                       range(max_p + 1)]  # store birth simplices, remove when finding corresponding terminal
    birth_simplices[0] = {0}
    terminal_simplices = [{} for _ in range(max_p + 1)]  # store each terminal simplex with its birth simplex
    next_possible_pivots = np.ones(1, dtype=bool)
    for p, matrix in enumerate(boundary_matrices):
        if p == 0:
            continue
        print(f"Reducing matrix for {p=}")
        possible_pivots = next_possible_pivots
        next_possible_pivots = np.ones(matrix.shape[1], dtype=bool)
        pivots = {}  # for each row store the column that has a pivot in it
        for c in range(matrix.shape[1]):
            if matrix.shape[1] < 10 or c % (matrix.shape[1] // 10) == 0:
                print(f"\tOn column {c}/{matrix.shape[1]}")
            pivot = find_pivot(matrix, c, possible_pivots)
            while pivot is not None and pivot in pivots:
                zeroer = pivots[pivot]
                # np.logical_xor(matrix[:, c], matrix[:, zeroer], out=matrix[:, c])
                matrix[:, c] ^= matrix[:, zeroer]
                pivot = find_pivot(matrix, c, possible_pivots)
            if pivot is not None:
                pivots[pivot] = c
                terminal_simplices[p + 1][c] = pivot
                # try:
                birth_simplices[p].remove(pivot)
                # except KeyError:
                #     pass
                next_possible_pivots[c] = 0
            else:
                birth_simplices[p + 1].add(c)

    terminal_points = [[(simplices_births[p][birth], simplices_births[p + 1][death])
                        for death, birth in terminal_simplices[p + 1].items()]
                       for p in range(max_p - 1 + 1)]
    birth_points = [[simplices_births[p][birth]
                     for birth in birth_simplices[p]]
                    for p in range(max_p - 1 + 1)]
    f, ax = plt.subplots(1)
    ax.plot([tup[0] for tup in terminal_points[1]], [tup[1] for tup in terminal_points[1]], 'r.', markersize=5)
    xymax = max([max(tup[0], tup[1]) for tup in terminal_points[1]])
    ax.set_xlim(xmin=0, xmax=xymax)
    ax.set_ylim(ymin=0, ymax=xymax)
    ax.set_box_aspect(1)
    ax.plot([0, xymax], [0, xymax], 'k-')
    plt.show()
    f, ax = plt.subplots(1)
    ax.plot([tup[0] for tup in terminal_points[2]], [tup[1] for tup in terminal_points[2]], 'r.', markersize=5)
    xymax = max([max(tup[0], tup[1]) for tup in terminal_points[2]])
    ax.set_xlim(xmin=0, xmax=xymax)
    ax.set_ylim(ymin=0, ymax=xymax)
    ax.set_box_aspect(1)
    ax.plot([0, xymax], [0, xymax], 'k-')
    plt.show()


if __name__ == '__main__':
    main()
