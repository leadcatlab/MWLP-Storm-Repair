from graph import Graph
from itertools import permutations, combinations
from more_itertools import set_partitions
from collections import deque
from typing import Deque, Callable
import numpy as np

# TODO: Implement MWLP_DP from https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.101.7189&rep=rep1&type=pdf
# TODO: Implement Christofides' Algorithm


def FloydWarshall(g: Graph) -> list[list[float]]:
    """Use Floyd-Warshall algorithm to solve all pairs shortest path (APSP)

    Runtime: O(n^3)

    Args:
        g: input graph

    Returns:
        2D array of distances
        dist[i][j] = distance from i -> j
        if no path exists, value is float('inf')

    """

    n: int = g.numNodes
    dist: list[list[float]] = [[float("inf") for _ in range(n)] for _ in range(n)]

    # initialize dist-table
    for i in range(n):
        dist[i][i] = 0.0
        for j in range(n):
            if i != j and j in g.adjacenList[i]:
                dist[i][j] = g.edgeWeight[i][j]

    # if dist[i][k] + dist[k][j] < dist[i][j]: update
    for k in range(n):
        for i in range(n):
            for j in range(n):
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])

    return dist


def createMetricFromGraph(g: Graph) -> Graph:
    """Create metric graph from input graph
    Using Floyd-Warshall we can solve the APSP problem.
    This gives edge weights that satisfy the triangle inequality

    Runtime: O(n^3)

    Args:
        g: input graph

    Returns:
        Graph satisfying the triangle inequality (Metric Graph)

    """

    n: int = g.numNodes

    metric = Graph(n)
    metric.nodeWeight = g.nodeWeight
    metric.adjacenList = g.adjacenList

    metricWeights: list[list[float]] = FloydWarshall(g)
    for i in range(n):
        for j in range(n):
            if (
                i != j
                and g.edgeWeight[i][j] != -1
                and metricWeights[i][j] != float("inf")
            ):
                metric.edgeWeight[i][j] = metricWeights[i][j]

    return metric


def WLP(g: Graph, order: list[int]) -> float:
    """Calculate the weighted latency of a given path

    Runtime: O(n)

    Args:
        g: input graph
        order: sequence of nodes visited starting at 0

    Returns:
        float: the weighted latency
    """

    # check nodes in order are actually valid nodes
    for node in order:
        if node >= g.numNodes or node < 0:
            raise ValueError(f"Node {node} is not in passed graph")

    if len(order) <= 1:
        return 0.0

    pathLen: list[float] = [0.0] * len(order)
    for i in range(1, len(order)):
        if order[i] not in g.adjacenList[order[i - 1]]:
            raise ValueError(f"Edge {order[i - 1]} --> {order[i]} does not exist")
        pathLen[i] = pathLen[i - 1] + g.edgeWeight[order[i - 1]][order[i]]

    # sum over sequence [v_0, v_1, ..., v_n] of w(v_i) * L(0, v_i)
    wlp: float = sum(g.nodeWeight[order[i]] * pathLen[i] for i in range(len(order)))

    return wlp


def bruteForceMWLP(g: Graph, start: int = 0, seqStart: list[int] = []) -> list[int]:
    """Calculate minumum weighted latency

    Iterates over all possible paths and solves in brute force manner

    Runtime: O(n!)

    Args:
        g: input graph

    Returns:
        list[int[: Path order for the minumum weighted latency
    """

    # for now assume complete
    if not Graph.isComplete(g):
        raise ValueError("Passed graph is not complete")

    # check validity of start
    if start >= g.numNodes or start < 0:
        raise ValueError(f"{start = } is not in passed graph")
    if start in seqStart:
        raise ValueError(
            f"Cannot start ({start}) at node already visited in {seqStart = }"
        )

    # check validity of seqStart:
    for n in seqStart:
        if n >= g.numNodes or n < 0:
            raise ValueError(f"Passed {seqStart = } contains nodes not in g")

    # keep track of visited nodes
    visited: list[bool] = [False] * g.numNodes
    for n in seqStart:
        visited[n] = True
    visited[start] = True

    # valid nodes to visit
    nodes: list[int] = [i for i in range(g.numNodes) if visited[i] is False]

    best: list[int] = []
    mwlp = float("inf")

    # test every permutation
    for order in permutations(nodes):
        full_order: list[int] = seqStart + [start] + list(order)
        curr: float = WLP(g, full_order)
        if curr < mwlp:
            mwlp = curr
            best = full_order

    return best


def cost(g: Graph, order: list[int]) -> float:
    """cost function from "Polynomial time algorithms for some minimum latency problems" (Wu)

    c(order) = Latency(order) + (w(g) + w(order))*Length(order)
    This is the cost function used for MWLP_DP algorithm

    Runtime: O(n)

    Args:
        g: input graph
        order: list of nodes in subtour

    Return:
        float: output of cost function c(order)
    """

    # check nodes in order are actually valid nodes
    for node in order:
        if node >= g.numNodes:
            raise ValueError(f"Node {node} is not in passed graph")

    latency: float = WLP(g, order)
    length: float = 0.0
    for i in range(len(order) - 1):
        # Note we do not need to check if edges exist since prior call to WLP checks for us
        length += g.edgeWeight[order[i]][order[i + 1]]

    weightOrder: int = sum(g.nodeWeight[n] for n in order)
    return latency + (sum(g.nodeWeight) - weightOrder) * length


def nearestNeighbor(g: Graph, start: int = 0, seqStart: list[int] = []) -> list[int]:
    """Approximates MWLP using nearest neighbor heuristic

    Generates sequence starting from 0 going to the nearest node

    Args:
        g: input graph
        start: start node after seqStart
        seqStart: list of nodes already visited

    Returns:
        list[int]: nearest neighbor order

    """

    # for now assume complete
    if not Graph.isComplete(g):
        raise ValueError("Passed graph is not complete")

    # check validity of start
    if start >= g.numNodes or start < 0:
        raise ValueError(f"{start = } is not in passed graph")
    if start in seqStart:
        raise ValueError(
            f"Cannot start ({start}) at node already visited in {seqStart = }"
        )

    # check validity of seqStart:
    for n in seqStart:
        if n >= g.numNodes or n < 0:
            raise ValueError(f"Passed {seqStart = } contains nodes not in g")

    # keep track of visited nodes
    visited: list[bool] = [False] * g.numNodes
    for n in seqStart:
        visited[n] = True
    visited[start] = True

    # Use queue to remember current node
    order: list[int] = seqStart + [start]
    q: Deque[int] = deque()
    q.appendleft(start)

    while len(q) != 0:
        curr: int = q.pop()
        dist = float("inf")
        nearest: int = -1
        for n in g.adjacenList[curr]:
            if not visited[n] and g.edgeWeight[curr][n] < dist:
                dist = g.edgeWeight[curr][n]
                nearest = n
        if nearest != -1:
            q.appendleft(nearest)
            order.append(nearest)
            visited[nearest] = True

    return order


def greedy(g: Graph, start: int = 0, seqStart: list[int] = []) -> list[int]:
    """Approximates MWLP using greedy heuristic

    Generates sequence starting from 0 going to the node of greatest weight

    Args:
        g: input graph
        start: start node after seqStart
        seqStart: list of nodes already visited

    Returns:
        list[int]: greedy order

    """

    # for now assume complete
    if not Graph.isComplete(g):
        raise ValueError("Passed graph is not complete")

    # check validity of start
    if start >= g.numNodes or start < 0:
        raise ValueError(f"{start = } is not in passed graph")
    if start in seqStart:
        raise ValueError(
            f"Cannot start ({start}) at node already visited in {seqStart = }"
        )

    # check validity of seqStart:
    for n in seqStart:
        if n >= g.numNodes or n < 0:
            raise ValueError(f"Passed {seqStart = } contains nodes not in g")

    # keep track of visited nodes
    visited: list[bool] = [False] * g.numNodes
    for n in seqStart:
        visited[n] = True
    visited[start] = True

    # Use queue to remember current node
    order: list[int] = seqStart + [start]
    q: Deque[int] = deque()
    q.appendleft(start)

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


def randomOrder(g: Graph, start: int = 0, seqStart: list[int] = []) -> list[int]:
    """Random order generator

    Generates random sequence starting from start or starting from seqStart -> start
    Essentially a wrapper around np.random.permutation

    Args:
        g: input graph
        start: start node after seqStart
        seqStart: list of nodes already visited

    Returns:
        list[int]: greedy order

    """

    # for now assume complete
    if not Graph.isComplete(g):
        raise ValueError("Passed graph is not complete")

    # check validity of start
    if start >= g.numNodes or start < 0:
        raise ValueError(f"{start = } is not in passed graph")
    if start in seqStart:
        raise ValueError(
            f"Cannot start ({start}) at node already visited in {seqStart = }"
        )

    # check validity of seqStart:
    for n in seqStart:
        if n >= g.numNodes or n < 0:
            raise ValueError(f"Passed {seqStart = } contains nodes not in g")

    # keep track of visited nodes
    visited: list[bool] = [False] * g.numNodes
    for n in seqStart:
        visited[n] = True
    visited[start] = True

    toVisit: list[int] = [i for i in range(g.numNodes) if visited[i] is False]

    return seqStart + [start] + list(np.random.permutation(toVisit))


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

    # check validity of start
    if start >= g.numNodes:
        raise ValueError(f"{start = } is not in passed graph")

    # valid nodes to visit
    nodes: list[int] = [i for i in range(g.numNodes)]
    nodes.remove(start)

    min_dist = float("inf")
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

    # assert validity of start
    if start >= g.numNodes:
        raise ValueError(f"{start = } is not in passed graph")

    # key: tuple(set[int]: nodes, int: end)
    # value: tuple(float: path length, list[int]:  order of nodes)
    completed = dict()  # type: ignore # typing this would be too verbose

    # recursive solver
    def solveTour(S: set[int], e: int) -> tuple[float, list[int]]:
        # base case: if no in-between nodes must take edge from start -> e
        if len(S) == 0:
            return (g.edgeWeight[start][e], [start])

        min_length = float("inf")
        best_order: list[int] = []
        # otherwise iterate over S all possible second-t-last nodes
        for s_i in S:
            S_i: set[int] = set(S)
            S_i.remove(s_i)
            sublength, suborder = completed[frozenset(S_i), s_i]
            length: float = sublength + g.edgeWeight[s_i][e]
            if length < min_length:
                min_length = length
                best_order = list(suborder) + [s_i]

        return min_length, best_order

    # solve TSP over all subsets of nodes, smallest to largest
    targets: set[int] = set(i for i in range(g.numNodes))
    targets.remove(start)
    for k in range(1, len(targets) + 1):
        for subset in combinations(targets, k):
            for e in subset:
                S: set[int] = set(subset)
                S.remove(e)
                completed[frozenset(S), e] = solveTour(S, e)

    # Find best TSP over all nodes (essentially solving last case again)
    tsp_sol = float("inf")
    best_order: list[int] = []
    for s_i in targets:
        S_i = set(targets)
        S_i.remove(s_i)
        tsp, order = completed[frozenset(S_i), s_i]
        if tsp < tsp_sol:
            tsp_sol = tsp
            best_order = order + [s_i]

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

    # assume start is at 0
    nodes = [i for i in range(1, g.numNodes)]

    best_order: list[list[int]] = []
    minimum = float("inf")

    # iterate through each partition
    for part in set_partitions(nodes, k):
        curr: float = 0.0
        part_order: list[list[int]] = []

        # iterate through each group in partition
        for nodes in part:
            # assume starting at 0
            full_order: list[int] = [0] + list(nodes)
            sg, sto, ots = Graph.subgraph(g, full_order)

            # calculuate heuristic
            heuristic_order: list[int] = f(sg)
            curr += WLP(sg, heuristic_order)

            # collect orders
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

    # iterate through all possible numbers of agents
    for k in range(k_min, k_max + 1):
        min_for_k, order_for_k = partitionHeuristic(g, f, k)
        if min_for_k < minimum:
            minimum = min_for_k
            best_order = order_for_k

    return minimum, best_order


# Depreciated MWLP_DP function. Does not work
# TODO: Elaborate on weakness
# def MWLP_DP(g: Graph, start: int = 0) -> list[int]:
#     """Solve MWLP using DP
#
#     Solves MWLP using an algorithm very similar to Held-Karp for DP
#
#     Args:
#         g: Graph to solve over
#         start: optional start node
#
#     Returns:
#         order: MWLP solution order
#     """
#
#     # for now assume complete
#     if not Graph.isComplete(g):
#         raise ValueError("Passed graph is not complete")
#
#     if start >= g.numNodes:
#         raise ValueError(f"{start = } is not in passed graph")
#
#     # key: tuple(set[int]: nodes, int: end)
#     # value: tuple(float: mwlp, float: path length, list[int]: order of nodes)
#     completed = dict()  # type: ignore # typing this would be too verbose
#
#     # recursive solver
#     def solveMWLP(S: set[int], e: int) -> tuple[float, float, list[int]]:
#         # base case: if no in-between nodes must take edge from start -> e
#         if len(S) == 0:
#             path_len: float = g.edgeWeight[start][e]
#             return (path_len * g.nodeWeight[e], path_len, [start])
#
#         current_mwlp = float("inf")
#         current_length = float("inf")
#         best_order: list[int] = []
#         # otherwise iterate over S all possible second-to-last nodes
#         for s_i in S:
#             S_i: set[int] = set(S)
#             S_i.remove(s_i)
#             sub_mwlp, sublength, suborder = completed[frozenset(S_i), s_i]
#             length: float = sublength + g.edgeWeight[s_i][e]
#             mwlp: float = (g.nodeWeight[e] * length) + sub_mwlp
#             if mwlp < current_mwlp:
#                 current_mwlp = mwlp
#                 current_length = length
#                 order: list[int] = suborder + [s_i]
#                 best_order = order
#
#         return current_mwlp, current_length, best_order
#
#     # solve MWLP over all subsets of nodes, smallest to largest
#     targets: set[int] = set(i for i in range(g.numNodes))
#     targets.remove(start)
#     for k in range(1, len(targets) + 1):
#         for subset in combinations(targets, k):
#             for e in subset:
#                 S: set[int] = set(subset)
#                 S.remove(e)
#                 completed[frozenset(S), e] = solveMWLP(S, e)
#
#     # sanity check
#     sol = bruteForceMWLP(g)
#
#     # Find best MWLP over all nodes (essentially solving last case again)
#     mwlp_sol = float("inf")
#     best_order: list[int] = []
#     for s_i in targets:
#         S_i = set(targets)
#         S_i.remove(s_i)
#         mwlp, _, order = completed[frozenset(S_i), s_i]
#         if mwlp < mwlp_sol:
#             mwlp_sol = mwlp
#             best_order = order + [s_i]
#             assert mwlp == WLP(g, best_order)
#
#     if sol != best_order:
#         print(f"{sol =        }")
#         print(f"{best_order = }")
#     return best_order
