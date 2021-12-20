from graph import Graph
from itertools import permutations
from collections.abc import Sequence
from collections import deque
from typing import Deque

# TODO: Implement Floyd-Warshall
# TODO: Implement Christofides' Algorithm
# TODO: Implement Held-Karp for TSP
# TODO: Try to implement MWLP DP Algorithm


def WLP(g: Graph, order: Sequence[int]) -> float:
    """Calculate the weighted latency of a given path

    Args:
        g: input graph
        order: sequence of nodes visited starting at 0

    Returns:
        float: the weighted latency
    """

    # TODO: Test WLP

    if len(order) == 0 or (len(order) == 1 and order[0] == 0):
        return 0.0

    # always start at 0
    if order[0] != 0:
        raise ValueError(f"Passed order = {order} does not start with 0")

    n = g.numNodes
    # check nodes in order are actually valid nodes
    for node in order:
        if node >= n:
            raise ValueError(f"Node {node} is not in passed graph")

    # sum over sequence of w(i) * L(0, i)
    wlp = 0.0
    for i in range(0, len(order)):
        # find length of path
        path: float = 0.0
        for j in range(0, i):
            if order[j + 1] not in g.adjacenList[order[j]]:
                raise ValueError(f"Edge {order[j]} --> {order[j + 1]} does not exist")
            path += g.edgeWeight[order[j]][order[j + 1]]

        wlp += g.nodeWeight[order[i]] * path

    return wlp


def bruteForceMWLP(g: Graph) -> float:
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
    if not Graph.isComplete(g):
        raise ValueError("Passed graph is not complete")

    mwlp = float("inf")
    nodes: list[int] = [i for i in range(1, g.numNodes)]
    # test every permutation
    for order in permutations(nodes):
        # always start at 0
        full_order: list[int] = [0] + list(order)
        mwlp = min(mwlp, WLP(g, full_order))

    return mwlp


def nearestNeighbor(g: Graph) -> Sequence[int]:
    """Approximates MWLP using nearest neighbor heuristic

    Generates sequence starting from 0 going to the nearest node

    Args:
        g: input graph

    Returns:
        list[int]: nearest neighbor order

    """

    # TODO: Test Nearest Neighbor

    # for now assume complete
    if not Graph.isComplete(g):
        raise ValueError("Passed graph is not complete")

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


def greedy(g: Graph) -> Sequence[int]:
    """Approximates MWLP using greedy heuristic

    Generates sequence starting from 0 going to the node of greatest weight

    Args:
        g: input graph

    Returns:
        list[int]: greedy order

    """

    # TODO: Test Greedy

    # for now assume complete
    if not Graph.isComplete(g):
        raise ValueError("Passed graph is not complete")

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
            if not visited[n] and g.nodeWeight[n] > weight:
                weight = g.nodeWeight[n]
                heaviest = n
        if heaviest != -1:
            q.appendleft(heaviest)
            order.append(heaviest)
            visited[heaviest] = True

    return order


def TSP(g: Graph) -> Sequence[int]:
    """Approximates MWLP using Travelling Salesman heuristic

    Generates sequence that solves travelling salesman

    Args:
        g: input graph

    Returns:
        lsti[int]: Solution to the Travelling Salesman Problem

    """

    # TODO: Test TSP

    # for now assume complete
    if not Graph.isComplete(g):
        raise ValueError("Passed graph is not complete")

    min_dist = float("inf")
    nodes: list[int] = [i for i in range(1, g.numNodes)]
    best: Sequence[int] = []

    # test every permutation
    for order in permutations(nodes):
        # always start at 0
        full_order: list[int] = [0] + list(order)
        dist = 0.0
        for i in range(g.numNodes - 1):
            dist += g.edgeWeight[full_order[i]][full_order[i + 1]]
        if dist < min_dist:
            min_dist = dist
            best = full_order

    return best
