from graph import graph
from itertools import permutations
from collections.abc import Sequence
from collections import deque
from typing import Deque

# TODO: Implement Floyd-Warshall
# TODO: Implement Christofides' Algorithm
# TODO: Implement Held-Karp for TSP
# TODO: Try to implement MWLP DP Algorithm


def WLP(g: graph, order: Sequence[int]) -> float:
    """Calculate the weighted latency of a given path

    Args:
        g: input graph
        order: sequence of nodes visited starting at 0

    Returns:
        float: the weighted latency
    """

    # TODO: Test WLP

    if order is None or len(order) <= 1:
        return 0.0

    # always start at 0
    assert order[0] == 0

    assert len(order) == g.numNodes

    n = g.numNodes
    for node in order:
        assert node < n

    # sum over sequence of w(i) * L(0, i)
    wlp = 0.0
    for i, node in enumerate(order[1:]):
        # find length of path
        path: float = 0.0
        for j, before in enumerate(order[:i]):
            path += g.edgeWeight[before][order[j + 1]]

        wlp += g.nodeWeight[node] * path

    return wlp


def bruteForceMWLP(g: graph) -> float:
    """Calculate minumum weighted latency

    Iterates over all possible paths

    Args:
        g: input graph

    Returns:
        float: the minumum weighted latency
    """

    # TODO: Relies on WLP which has not been tested
    # TODO: Test MWLP

    # for now assume complete
    assert graph.isComplete(g)

    mwlp = float("inf")
    nodes: list[int] = [i for i in range(1, g.numNodes)]
    # test every permutation
    for order in permutations(nodes):
        # always start at 0
        full_order: list[int] = [0] + list(order)
        mwlp = min(mwlp, WLP(g, full_order))

    return mwlp


def nearestNeighbor(g: graph) -> Sequence[int]:
    """Approximates MWLP using nearest neighbor heuristic

    Generates sequence starting from 0 going to the nearest node

    Args:
        g: input graph

    Returns:
        list[int]: nearest neighbor order

    """

    # TODO: Test Nearest Neighbor

    # for now assume complete
    assert graph.isComplete(g)

    visited: list[bool] = [False] * g.numNodes
    order: list[int] = [0]
    q: Deque[int] = deque()  # we assume start at 0
    q.appendleft(0)
    visited[0] = True

    # Use queue to remember current node
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

    return order


def greedy(g: graph) -> Sequence[int]:
    """Approximates MWLP using greedy heuristic

    Generates sequence starting from 0 going to the node of greatest weight

    Args:
        g: input graph

    Returns:
        list[int]: greedy order

    """

    # TODO: Test Greedy

    # for now assume complete
    assert graph.isComplete(g)

    visited: list[bool] = [False] * g.numNodes
    order: list[int] = [0]
    q: Deque[int] = deque()  # we assume start at 0
    q.appendleft(0)
    visited[0] = True

    # Use queue to remember current node
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

    return order


def TSP(g: graph) -> Sequence[int]:
    """Approximates MWLP using Travelling Salesman heuristic

    Generates sequence that solves travelling salesman

    Args:
        g: input graph

    Returns:
        lsti[int]: Solution to the Travelling Salesman Problem

    """

    # TODO: Test TSP

    n: int = g.numNodes
    if n <= 0:
        return []
    if n == 1:
        return [0]

    min_dist = float("inf")
    nodes: list[int] = [i for i in range(1, n)]
    best: Sequence[int] = []

    # test every permutation
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
