from graph import Graph
from itertools import permutations, combinations
from more_itertools import set_partitions
from collections import deque
from typing import Deque, Callable

# TODO: Implement Christofides' Algorithm
# TODO: Try to implement MWLP DP Algorithm


def floydWarshall(g: Graph) -> list[list[float]]:
    """Use Floyd-Warshall algorithm to solve all pairs shortest path

    Returns:
        2D array of distances
        dist[i][j] = distance from i -> j
        if no path exists, value is float('inf')

    """

    n: int = g.numNodes
    dist: list[list[float]] = [[float("inf") for _ in range(n)] for _ in range(n)]

    for i in range(n):
        dist[i][i] = 0.0
        for j in range(n):
            if i != j and j in g.adjacenList[i]:
                dist[i][j] = g.edgeWeight[i][j]

    for k in range(n):
        for i in range(n):
            for j in range(n):
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])

    return dist


def WLP(g: Graph, order: list[int]) -> float:
    """Calculate the weighted latency of a given path

    Args:
        g: input graph
        order: sequence of nodes visited starting at 0

    Returns:
        float: the weighted latency
    """

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


def bruteForceMWLP(g: Graph, start: int = 0) -> list[int]:
    """Calculate minumum weighted latency

    Iterates over all possible paths

    Args:
        g: input graph

    Returns:
        float: the minumum weighted latency
    """

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


def nearestNeighbor(g: Graph, start: int = 0) -> list[int]:
    """Approximates MWLP using nearest neighbor heuristic

    Generates sequence starting from 0 going to the nearest node

    Args:
        g: input graph

    Returns:
        list[int]: nearest neighbor order

    """

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


def greedy(g: Graph, start: int = 0) -> list[int]:
    """Approximates MWLP using greedy heuristic

    Generates sequence starting from 0 going to the node of greatest weight

    Args:
        g: input graph

    Returns:
        list[int]: greedy order

    """

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


def TSP(g: Graph, start: int = 0) -> list[int]:
    """Brute Force TSP

    Generates sequence that solves travelling salesman. Solved via
    pure brute force of all possible orders.

    Args:
        g: input graph

    Returns:
        list[int]: Solution to the Travelling Salesman Problem

    """

    # for now assume complete
    if not Graph.isComplete(g):
        raise ValueError("Passed graph is not complete")

    if start >= g.numNodes:
        raise ValueError(f"{start = } is not in passed graph")

    min_dist = float("inf")
    nodes: list[int] = [i for i in range(g.numNodes)]
    nodes.remove(start)
    best: list[int] = []

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


def HeldKarp(g: Graph, start: int = 0) -> list[int]:
    """TSP via Held-Karp

    Generate the solution to TSP via dynamic programming using Held-Karp
    Variable names closely follow wikipedia.org/wiki/Held-Karp_algorithm

    Args:
        g: input graph
        start: start node

    Returns:
        list[int]: Solution to the Travelling Salesman Problem

    """

    # for now assume complete
    if not Graph.isComplete(g):
        raise ValueError("Passed graph is not complete")

    if start >= g.numNodes:
        raise ValueError(f"{start = } is not in passed graph")

    completed: dict[tuple[frozenset[int], int], tuple[float, list[int]]] = dict()

    def solveTour(g: Graph, S: set[int], e: int) -> tuple[float, list[int]]:
        if len(S) == 0:
            return (g.edgeWeight[start][e], [start])

        current_min = float("inf")
        best_order: list[int] = []
        for s_i in S:
            S_i: set[int] = set(S)
            S_i.remove(s_i)
            completed_length, completed_order = completed[(frozenset(S_i), s_i)]
            subtour_length: float = completed_length + g.edgeWeight[s_i][e]
            if subtour_length < current_min:
                current_min = subtour_length
                subtour_order: list[int] = list(completed_order)
                subtour_order.append(s_i)
                best_order = subtour_order

        return current_min, best_order

    targets: set[int] = set(i for i in range(g.numNodes))
    targets.remove(start)
    for k in range(1, len(targets) + 1):
        for subset in combinations(targets, k):
            for e in subset:
                S: set[int] = set(subset)
                S.remove(e)
                completed[(frozenset(S), e)] = solveTour(g, S, e)

    tsp_sol = float("inf")
    best_order: list[int] = []
    for s_i in targets:
        S_i = set(targets)
        S_i.remove(s_i)
        if completed[(frozenset(S_i), s_i)][0] < tsp_sol:
            tsp_sol = completed[(frozenset(S_i), s_i)][0]
            best_order = completed[(frozenset(S_i), s_i)][1] + [s_i]

    return best_order


def partitionHeuristic(
    g: Graph, f: Callable[..., list[int]], k: int
) -> tuple[float, list[list[int]]]:
    """Bruteforce multi-agent MWLP

    Generates best partition according to passed heuristic f
    Returned partition is ordered in subgroups so the best order for each partition is returned

    Args:
        g: input graph
        f: heuristic
        k: number of agents

    Returns:
        float: optimal MWLP value according to heuristic
        list[list[int]: Best graph partition and order

    """

    # for now assume complete
    if Graph.isComplete(g) is False:
        raise ValueError("Passed graph is not complete")

    if k <= 0:
        raise ValueError(f"Multi-agent case must have non-zero agents ({k})")

    if k > g.numNodes:
        raise ValueError(f"Multi-agent case cannot have more agents than nodes ({k})")

    best_order: list[list[int]] = []
    minimum = float("inf")
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

        if curr < minimum:
            minimum = curr
            best_order = part_order

    return minimum, best_order


def optimalNumberOfAgents(
    g: Graph, f: Callable[..., list[int]], k_min: int, k_max: int
) -> tuple[float, list[list[int]]]:
    """Bruteforce multi-agent MWLP for variable number of agents

    Generates optimal number of agents along with best partition according to passed heuristic f
    Returned partition is ordered in subgroups so the best order for each partition is returned
    Length of optimal partition is number of agents used

    Args:
        g: input graph
        f: heuristic
        k_min: minumum number of agents (must be >= 1)
        k_max: maximum number of agents (must be <= g.numNodes - 1 since we don't count start node of 0)

    Returns:
        float: optimal MWLP value according to heuristic
        list[list[int]: Best graph partition and order for k agents where k_min <= k <= k_max

    """

    # for now assume complete
    if Graph.isComplete(g) is False:
        raise ValueError("Passed graph is not complete")

    if k_max <= k_min:
        raise ValueError(
            "Minimum number of agents {k_min} must be less than maximum number of agents {k_max}"
        )

    if k_min <= 0:
        raise ValueError(f"Multi-agent case must have non-zero agents ({k_min})")

    if k_max >= g.numNodes:
        raise ValueError(
            f"Multi-agent case cannot have more agents than non-start nodes ({k_max})"
        )

    best_order: list[list[int]] = []
    minimum = float("inf")

    for k in range(k_min, k_max + 1):
        min_for_k, order_for_k = partitionHeuristic(g, f, k)
        if min_for_k < minimum:
            minimum = min_for_k
            best_order = order_for_k

    return minimum, best_order
