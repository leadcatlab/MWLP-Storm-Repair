from graph import graph
from itertools import permutations
from collections.abc import Sequence
from collections import deque
from typing import Deque, Optional

# TODO: Comments
# TODO: Implement Floyd-Warshall
# TODO: Implement Christofides' Algorithm
# TODO: Implement Held-Karp for TSP
# TODO: Try to implement MWLP DP Algorithm
# TODO: Fix default return values to make types cleaner


def WLP(g: graph, order: Optional[Sequence[int]]) -> float:
    # TODO: Test WLP

    if order is None or len(order) <= 1:
        return 0.0

    # always start at 0
    assert order[0] == 0

    assert len(order) == g.numNodes

    n = g.numNodes
    for node in order:
        assert node < n
    wlp = 0.0
    for i, node in enumerate(order[1:]):
        # find length of path
        path: float = 0.0
        for j, before in enumerate(order[:i]):
            path += g.edgeWeight[before][order[j + 1]]

        wlp += g.nodeWeight[node] * path

    return wlp


def bruteForceMWLP(g: graph) -> float:
    # TODO: Relies on WLP which has not been tested
    # TODO: Test MWLP

    # for now assume complete
    assert graph.isComplete(g)

    mwlp = float("inf")
    nodes: list[int] = [i for i in range(1, g.numNodes)]
    for order in permutations(nodes):
        # always start at 0
        full_order: list[int] = [0] + list(order)
        mwlp = min(mwlp, WLP(g, full_order))

    return mwlp


def nearestNeighbor(g: graph) -> float:
    # TODO: Relies on WLP which has not been tested
    # TODO: Test Nearest Neighbor

    # for now assume complete
    assert graph.isComplete(g)

    visited: list[bool] = [False] * g.numNodes
    order: list[int] = [0]
    q: Deque[int] = deque()  # we assume start at 0
    q.appendleft(0)
    visited[0] = True

    while len(q) != 0:
        curr: int = q.pop()
        dist = float("inf")
        nearest: int = -1
        for n in g.adjacenList[curr]:
            if visited[n] is False and g.edgeWeight[curr][n] < dist:
                dist = g.edgeWeight[curr][n]
                nearest = n
        if nearest != -1:
            q.appendleft(nearest)
            order.append(nearest)
            visited[nearest] = True

    return WLP(g, order)


def greedy(g: graph) -> float:
    # TODO: Relies on WLP which has not been tested
    # TODO: Test Greedy

    # for now assume complete
    assert graph.isComplete(g)

    visited: list[bool] = [False] * g.numNodes
    order: list[int] = [0]
    q: Deque[int] = deque()  # we assume start at 0
    q.appendleft(0)
    visited[0] = True

    while len(q) != 0:
        curr: int = q.pop()
        weight = float("-inf")
        heaviest: int = -1
        for n in g.adjacenList[curr]:
            if visited[n] is False and g.nodeWeight[n] > weight:
                weight = g.nodeWeight[n]
                heaviest = n
        if heaviest != -1:
            q.appendleft(heaviest)
            order.append(heaviest)
            visited[heaviest] = True

    return WLP(g, order)


def TSP(g: graph) -> Optional[Sequence[int]]:
    # TODO: test TSP

    n: int = g.numNodes
    if n <= 1:
        return None

    min_dist = float("inf")
    nodes: list[int] = [i for i in range(1, n)]
    best: Optional[Sequence[int]] = None
    for order in permutations(nodes):
        # always start at 0
        full_order: list[int] = [0] + list(order)
        dist = 0.0
        for i in range(n - 1):
            dist += g.edgeWeight[full_order[i]][full_order[i + 1]]
        if dist < min_dist:
            min_dist = dist
            best = full_order

    return best
