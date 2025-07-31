import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import chain, combinations


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


def find_pivot(matrix, c):
    column = matrix[:, c]
    if not np.any(column):
        return None
    index = len(column) - np.max(np.argmax(column[::-1])) - 1   # argmax finds first occurrence, we reverse to find last
    return index


def get_data(file_path):
    df = pd.read_csv(file_path)
    nd = df.to_numpy()
    np.random.seed(0)
    nd = nd[np.random.choice(len(nd), size=int(len(nd) * 1), replace=False)]  # ADJUST: random drop-out

    plt.plot(nd[:, 0], nd[:, 1], 'o')
    plt.show()

    data = [(float(x), float(y)) for [x, y] in nd]
    data = list(set(data))  # remove repetitions
    return data


def show_diagram(terminal_points, birth_points, ps):
    for p in ps:
        f, ax = plt.subplots(1)
        ax.scatter([tup[0] for tup in terminal_points[p]],
                   [tup[1] for tup in terminal_points[p]],
                   [[10*(tup[1] - tup[0]) + 5 for tup in terminal_points[p]]],
                   'r')
        xy_max = max([max(tup[0], tup[1]) for tup in terminal_points[p]])
        ax.scatter([value for value in birth_points[p]],
                   [xy_max*1.1] * len(birth_points[p]),
                   [[10*(xy_max - value) + 5 for value in birth_points[p]]],
                   'r')
        ax.set_xlim(xmin=0, xmax=xy_max*1.1)
        ax.set_ylim(ymin=0, ymax=xy_max*1.1)
        ax.set_box_aspect(1)
        ax.plot([0, xy_max], [0, xy_max], 'k-')
    plt.show()


def main():
    data = get_data("data/data_A.csv")

    dists_sq = {}
    for x1 in data:
        for x2 in data:
            if x1 < x2:
                value = dist_sq(x1, x2)
                if value in dists_sq:
                    dists_sq[value] += [(x1, x2)]
                else:
                    dists_sq[value] = [(x1, x2)]
    # plt.hist(dists_sq, bins=1000)
    # plt.show()
    # exit(0)
    max_p = 3
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

    max_dist_sq = float('inf')  # if taking too long, can change to stop filtration early
    print("Calculating simplex order")
    for i, (curr_dist_sq, dist_pairs) in enumerate(sorted(dists_sq.items())):
        for (x1, x2) in dist_pairs:
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
            #         if (tuple(sorted(simp + (x1,))) in simplices[p + 1]
            #                 and tuple(sorted(simp + (x2,))) in simplices[p + 1]):
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
                    if new_simp in simplices_set:
                        continue
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
                boundary_matrices[p][i][simplices_indices[p + 1][new_simp]] = 1

    birth_simplices = [set() for _ in
                       range(max_p + 1)]  # store birth simplices, remove when finding corresponding terminal
    birth_simplices[0] = {0}
    terminal_simplices = [{} for _ in range(max_p + 1)]  # store each terminal simplex with its birth simplex
    next_impossible_pivots = set()
    for p, matrix in enumerate(boundary_matrices):
        print(f"Reducing matrix for {p=}")
        impossible_pivots = next_impossible_pivots
        for impossible_pivot in impossible_pivots:
            matrix[impossible_pivot, :] = 0
        next_impossible_pivots = set()
        pivots = {}  # for each row store the column that has a pivot in it
        for c in range(matrix.shape[1]):
            if matrix.shape[1] < 10 or c % (matrix.shape[1] // 10) == 0:
                print(f"\tOn column {c}/{matrix.shape[1]}")
            pivot = find_pivot(matrix, c)
            while pivot is not None and pivot in pivots:
                zeroer = pivots[pivot]
                matrix[:, c] ^= matrix[:, zeroer]
                pivot = find_pivot(matrix, c)
            if pivot is None or p == 0:
                birth_simplices[p + 1].add(c)
            elif pivot is not None:
                pivots[pivot] = c
                terminal_simplices[p + 1][c] = pivot
                birth_simplices[p].remove(pivot)
                # next_impossible_pivots.add(c)

    terminal_points = [[(simplices_births[p][birth], simplices_births[p + 1][death])
                        for death, birth in terminal_simplices[p + 1].items()]
                       for p in range(max_p - 1 + 1)]
    birth_points = [[simplices_births[p][birth]
                     for birth in birth_simplices[p]]
                    for p in range(max_p - 1 + 1)]

    show_diagram(terminal_points, birth_points, range(1, max_p - 1 + 1))


if __name__ == '__main__':
    main()
