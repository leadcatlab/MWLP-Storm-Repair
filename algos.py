from graph import Graph
from itertools import permutations
from more_itertools import set_partitions
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

    if len(order) <= 1:
        return 0.0

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


def bruteForceMWLP(g: Graph, start: int = 0) -> Sequence[int]:
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

    if start >= g.numNodes:
        raise ValueError(f"{start = } is not in passed graph")

    mwlp = float("inf")
    nodes: list[int] = [i for i in range(g.numNodes)]
    nodes.remove(start)

    best = []
    mwlp = float("inf")
    # test every permutation
    for order in permutations(nodes):
        full_order: list[int] = [start] + list(order)
        curr: float = WLP(g, full_order)
        if curr < mwlp:
            mwlp = curr
            best = full_order

    return best


def nearestNeighbor(g: Graph, start: int = 0) -> Sequence[int]:
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

    if start >= g.numNodes:
        raise ValueError(f"{start = } is not in passed graph")

    visited: list[bool] = [False] * g.numNodes
    order: list[int] = [start]
    q: Deque[int] = deque()
    q.appendleft(start)
    visited[start] = True

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


def greedy(g: Graph, start: int = 0) -> Sequence[int]:
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

    if start >= g.numNodes:
        raise ValueError(f"{start = } is not in passed graph")

    visited: list[bool] = [False] * g.numNodes
    order: list[int] = [start]
    q: Deque[int] = deque()
    q.appendleft(start)
    visited[start] = True

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


def TSP(g: Graph, start: int = 0) -> Sequence[int]:
    """Approximates MWLP using Travelling Salesman heuristic

    Generates sequence that solves travelling salesman

    Args:
        g: input graph

    Returns:
        list[int]: Solution to the Travelling Salesman Problem

    """

    # TODO: Test TSP

    # for now assume complete
    if not Graph.isComplete(g):
        raise ValueError("Passed graph is not complete")

    if start >= g.numNodes:
        raise ValueError(f"{start = } is not in passed graph")

    min_dist = float("inf")
    nodes: list[int] = [i for i in range(g.numNodes)]
    nodes.remove(start)
    best: Sequence[int] = []

    # test every permutation
    for order in permutations(nodes):
        full_order: list[int] = [start] + list(order)
        dist = 0.0
        for i in range(g.numNodes - 1):
            dist += g.edgeWeight[full_order[i]][full_order[i + 1]]
        if dist < min_dist:
            min_dist = dist
            best = full_order

    return best


def partitionHeuristic(g: Graph, f, k: int) -> tuple[float, list[list[int]]]:
    """Bruteforce multi-agent MWLP

    Generates best partition according to passed heuristic f
    Returned partition is ordered in subgroups so the best order for each partition is returned

    Args:
        g: input graph
        f: heuristic
        k: number of agents

    Returns:
        list[list[int]: Best graph partition

    """

    # for now assume complete
    if not Graph.isComplete(g):
        raise ValueError("Passed graph is not complete")

    best: list[list[int]] = []
    mwlp_m: float = float("inf")
    # assume start is at 0
    nodes = [i for i in range(1, g.numNodes)]

    # iterate through each partition
    for part in set_partitions(nodes, k):
        curr: float = 0.0
        part_order: list[list[int]] = []
        # iterate through each group in partition
        for nodes in part:
            # assume starting at 0
            full_order: list[int] = [0] + list(nodes)
            sg, sto, ots = Graph.subgraph(g, full_order)

            heuristic_order = f(sg)
            curr += WLP(sg, heuristic_order)

            original_order = [sto[n] for n in heuristic_order]
            part_order.append(original_order)

        if curr < mwlp_m:
            mwlp_m = curr
            best = part_order

    return mwlp_m, best
