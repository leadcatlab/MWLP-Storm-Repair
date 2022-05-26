"""
Graph algorithms
"""
import random
from collections import deque
from itertools import combinations, permutations, product
from typing import Callable, Deque, Optional

import numpy as np
from more_itertools import set_partitions

from graph import Graph


def num_visited_along_path(g: Graph, path: list[int]) -> list[int]:
    """
    Presuming that node weights = people per location
    Utility function to give total visited at each position along a path

    Runtime: O(n)

    Parameters
    ----------
    g: Graph
        Input graph

    path: list[int]
        The path along the graph
        Assertions:
            Edges in path must be present in the graph/

    Returns
    -------
    list[int]
        visited[i] = sum of nodeweights of path[0..i]
        empty list => visited = []

    """

    if len(path) == 0:
        return []

    visited: list[int] = []
    visited.append(g.node_weight[path[0]])
    # Iterate through all nodes in path
    for i in range(1, len(path)):
        # Raise error if edge does not exist
        if path[i] not in g.adjacen_list[path[i - 1]]:
            raise ValueError(f"Edge {path[i - 1]} --> {path[i]} does not exist")
        visited.append(visited[-1] + g.node_weight[path[i]])
    return visited


def length_along_path(g: Graph, path: list[int]) -> list[float]:
    """
    Utility function to give total length traveled at each position along a path

    Runtime: O(n)

    Parameters
    ----------
    g: Graph
        Input graph

    path: list[int]
        The path along the graph
        Assertions:
            Edges in path must be present in the graph/

    Returns
    -------
    list[int]
        length[i] = distance traveled from from path[0] to path[i]

    """

    if len(path) <= 1:
        return [0.0]

    length: list[float] = [0.0]
    # Iterate through all nodes in path
    for i in range(1, len(path)):
        # Raise error if edge does not exist
        if path[i] not in g.adjacen_list[path[i - 1]]:
            raise ValueError(f"Edge {path[i - 1]} --> {path[i]} does not exist")
        length.append(length[-1] + g.edge_weight[path[i - 1]][path[i]])
    return length


def generate_path_function(g: Graph, path: list[int]) -> Callable[[float], int]:
    """
    Generates a function to get the number of people visited along a given path

    Parameters
    ----------
    g: Graph
        Input graph

    path: list[int]
        The path along the graph
        Assertions:
            Edges in path must be present in the graph/
            Non-empty path
    Returns
    -------
    Callable[[float], int]
        path_function(x) = the number of people visited at distance x along path
        Assertions:
            x >= 0.0
    """

    if len(path) == 0:
        raise ValueError("Passed path was empty")

    length: list[float] = length_along_path(g, path)
    visited: list[int] = num_visited_along_path(g, path)

    def path_function(x: float) -> int:
        if x < 0:
            raise ValueError("Input was a negative distance")

        # find largest index of length such that length[i] <= x
        idx: int = 0
        found: bool = False
        while not found:
            next_idx: int = idx + 1
            if next_idx < len(length) and length[next_idx] <= x:
                idx = next_idx
            else:
                found = True

        return visited[idx]

    return path_function


def generate_partition_path_function(
    g: Graph, part: list[list[int]]
) -> Callable[[float], int]:
    if Graph.is_agent_partition(g, [set(p) for p in part]) is False:
        raise ValueError("Passed partition is invalid")

    path_functions: list[Callable[[float], int]] = []
    for path in part:
        path_functions.append(generate_path_function(g, path))

    def partition_path_function(x: float) -> int:
        if x < 0:
            raise ValueError("Input was a negative distance")

        res: int = 0
        for f in path_functions:
            res += f(x)

        # Double counting start node (0) many times
        res -= g.node_weight[0] * (len(path_functions) - 1)
        return res

    return partition_path_function


def path_length(g: Graph, path: list[int]) -> float:
    """
    Get the length of a path in a graph
    Essentially just a wrapper around length_along_path

    Runtime: O(n)

    Parameters
    ----------
    g: Graph
        Input graph

    path: list[int]
        The path along the graph
        Assertions:
            Edges in path must be present in the graph/

    Returns
    -------
    float
        Path length. 0.0 if the length is less then 2 nodes

    """

    return length_along_path(g, path)[-1]


def floyd_warshall(g: Graph) -> list[list[float]]:
    """
    Use Floyd-Warshall algorithm to solve all pairs shortest path (APSP)

    Runtime: O(n^3)

    Parameters
    ----------
    g: Graph
        Input graph

    Returns
    -------
    list[list[float]]
        2D array of distances
        dist[i][j] = distance from i -> j
        if no path exists, value is float('inf')

    """

    n: int = g.num_nodes
    dist: list[list[float]] = [[float("inf") for _ in range(n)] for _ in range(n)]

    # initialize dist-table
    for i in range(n):
        dist[i][i] = 0.0
        for j in range(n):
            if i != j and j in g.adjacen_list[i]:
                dist[i][j] = g.edge_weight[i][j]

    # if dist[i][k] + dist[k][j] < dist[i][j]: update
    for (k, i, j) in product(range(n), range(n), range(n)):
        dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])

    return dist


def create_metric_from_graph(g: Graph) -> Graph:
    """
    Create metric graph from input graph
    Using Floyd-Warshall we can solve the APSP problem.
    This gives edge weights that satisfy the triangle inequality

    Runtime: O(n^3)

    Parameters
    ----------
    g: Graph
        Input graph

    Returns
    -------
    Graph
        Graph with edgeweights based off of Floyd-Warshall Algorithm
        i.e. len(u -> v) = shortest distance from u to v

    """

    n: int = g.num_nodes

    metric = Graph(n)
    metric.node_weight = g.node_weight
    metric.adjacen_list = g.adjacen_list

    metric_weights: list[list[float]] = floyd_warshall(g)
    for (i, j) in product(range(n), range(n)):
        if (
            i != j
            and g.edge_weight[i][j] != -1
            and metric_weights[i][j] != float("inf")
        ):
            metric.edge_weight[i][j] = metric_weights[i][j]

    return metric


def wlp(g: Graph, path: list[int]) -> float:
    """
    Calculate the weighted latency of a given path
    Sums of weights of node * length along path from start to node

    Runtime: O(n)

    Parameters
    ----------
    g: Graph
        Input graph

    path: list[int]
        The path along the graph
        Assertions:
            Edges in path must be present in the graph/

    Returns
    -------
    float
        Weighted Latency over the path in g

    """

    # check nodes in order are actually valid nodes
    for node in path:
        if node >= g.num_nodes or node < 0:
            raise ValueError(f"Node {node} is not in passed graph")

    if len(path) <= 1:
        return 0.0

    path_len: list[float] = [0.0] * len(path)
    for i in range(0, len(path) - 1):
        if path[i + 1] not in g.adjacen_list[path[i]]:
            raise ValueError(f"Edge {path[i]} --> {path[i + 1]} does not exist")
        path_len[i + 1] = path_len[i] + g.edge_weight[path[i]][path[i + 1]]

    # sum over sequence [v_0, v_1, ..., v_n] of w(v_i) * L(0, v_i)
    return sum(g.node_weight[path[i]] * path_len[i] for i in range(len(path)))


def brute_force_mwlp(g: Graph, start: Optional[list[int]] = None) -> list[int]:
    """
    Calculate minumum weighted latency
    Iterates over all possible paths and solves in brute force manner

    Runtime: O(n!)

    Parameters
    ----------
    g: Graph
        Input graph
        Assertions:
            g must be a complete graph

    start: list[int]
        Optional start of path (allows for partial solving)
        Assertions:
            Must contain nodes that are in the graph

    Returns
    -------
    list[int]
        Path order for minimum weighted latency

    """

    # for now assume complete
    if not Graph.is_complete(g):
        raise ValueError("Passed graph is not complete")

    if start is None:
        start = [0]

    # check validity of start:
    for n in start:
        if n >= g.num_nodes or n < 0:
            raise ValueError(f"Passed {start = } contains nodes not in g")

    # keep track of visited nodes
    visited: list[bool] = [False] * g.num_nodes
    for n in start:
        visited[n] = True

    # valid nodes to visit
    nodes: list[int] = [i for i in range(g.num_nodes) if visited[i] is False]

    best: list[int] = []
    mwlp = float("inf")

    # test every permutation
    for order in permutations(nodes):
        full_order: list[int] = start + list(order)
        curr: float = wlp(g, full_order)
        if curr < mwlp:
            mwlp = curr
            best = full_order

    return best


def nearest_neighbor(g: Graph, start: Optional[list[int]] = None) -> list[int]:
    """
    Approximates MWLP using nearest neighbor heuristic
    Starts from a node and goes to the nearest unvisited neighbor

    Runtime: O(n^2)

    Parameters
    ----------
    g: Graph
        Input graph
        Assertions:
            g must be a complete graph

    start: list[int]
        Optional start of path (allows for partial solving)
        Assertions:
            Must contain nodes that are in the graph

    Returns
    -------
    list[int]
        Path order for minimum weighted latency according to nearest neighbor

    """

    # for now assume complete
    if not Graph.is_complete(g):
        raise ValueError("Passed graph is not complete")

    if start is None:
        start = [0]

    # check validity of start:
    for n in start:
        if n >= g.num_nodes or n < 0:
            raise ValueError(f"Passed {start = } contains nodes not in g")

    # keep track of visited nodes
    visited: list[bool] = [False] * g.num_nodes
    for n in start:
        visited[n] = True

    # Use queue to remember current node
    order: list[int] = start
    q: Deque[int] = deque()
    q.appendleft(order[-1])

    while len(q) != 0:
        curr: int = q.pop()
        dist = float("inf")
        nearest: int = -1
        for n in g.adjacen_list[curr]:
            if not visited[n] and g.edge_weight[curr][n] < dist:
                dist = g.edge_weight[curr][n]
                nearest = n
        if nearest != -1:
            q.appendleft(nearest)
            order.append(nearest)
            visited[nearest] = True

    return order


def greedy(g: Graph, start: Optional[list[int]] = None) -> list[int]:
    """
    Approximates MWLP using greedy heuristic
    Starts from a node and goes to the heaviest unvisited neighbor

    Runtime: O(n^2)

    Parameters
    ----------
    g: Graph
        Input graph
        Assertions:
            g must be a complete graph

    start: list[int]
        Optional start of path (allows for partial solving)
        Assertions:
            Must contain nodes that are in the graph

    Returns
    -------
    list[int]
        Path order for minimum weighted latency according to greedy

    """

    # for now assume complete
    if not Graph.is_complete(g):
        raise ValueError("Passed graph is not complete")

    if start is None:
        start = [0]

    # check validity of start:
    for n in start:
        if n >= g.num_nodes or n < 0:
            raise ValueError(f"Passed {start = } contains nodes not in g")

    # keep track of visited nodes
    visited: list[bool] = [False] * g.num_nodes
    for n in start:
        visited[n] = True

    # Use queue to remember current node
    order: list[int] = start

    while len(order) != g.num_nodes:
        curr: int = order[-1]
        best_weight = float("-inf")
        heaviest: int = -1
        for n in g.adjacen_list[curr]:
            if not visited[n] and g.node_weight[n] > best_weight:
                best_weight = g.node_weight[n]
                heaviest = n
        if heaviest != -1:
            order.append(heaviest)
            visited[heaviest] = True

    return order


def alternate(g: Graph, start: Optional[list[int]] = None) -> list[int]:
    """
    Approximates MWLP using by alternating between two strategies (greedy + NN)

    Runtime: O(n^2)

    Parameters
    ----------
    g: Graph
        Input graph
        Assertions:
            g must be a complete graph

    start: list[int]
        Optional start of path (allows for partial solving)
        Assertions:
            Must contain nodes that are in the graph

    Returns
    -------
    list[int]
        Path order for minimum weighted latency according to alternating strategy

    """

    # for now assume complete
    if not Graph.is_complete(g):
        raise ValueError("Passed graph is not complete")

    if start is None:
        start = [0]

    # check validity of start:
    for n in start:
        if n >= g.num_nodes or n < 0:
            raise ValueError(f"Passed {start = } contains nodes not in g")

    # keep track of visited nodes
    visited: list[bool] = [False] * g.num_nodes
    for n in start:
        visited[n] = True

    # Use queue to remember current node
    order: list[int] = start
    q: Deque[int] = deque()
    q.appendleft(order[-1])

    # 0 =  Greedy, 1 = NN
    counter: int = 0
    while len(q) != 0:
        curr: int = q.pop()
        next_node: int = -1
        if counter == 0:
            best_weight = float("-inf")
            for n in g.adjacen_list[curr]:
                if not visited[n] and g.node_weight[n] > best_weight:
                    best_weight = g.node_weight[n]
                    next_node = n
        else:
            dist = float("inf")
            for n in g.adjacen_list[curr]:
                if not visited[n] and g.edge_weight[curr][n] < dist:
                    dist = g.edge_weight[curr][n]
                    next_node = n
        if next_node != -1:
            q.appendleft(next_node)
            order.append(next_node)
            visited[next_node] = True
        counter = (counter + 1) % 2

    return order


def random_order(g: Graph, start: Optional[list[int]] = None) -> list[int]:
    """
    Creates a random order of unvisited nodes

    Runtime: O(n) (?)

    Parameters
    ----------
    g: Graph
        Input graph
        Assertions:
            g must be a complete graph

    start: list[int]
        Optional start of path (allows for partial solving)
        Assertions:
            Must contain nodes that are in the graph

    Returns
    -------
    list[int]
        Random path order

    """

    # for now assume complete
    if not Graph.is_complete(g):
        raise ValueError("Passed graph is not complete")

    if start is None:
        start = [0]

    # check validity of start:
    for n in start:
        if n >= g.num_nodes or n < 0:
            raise ValueError(f"Passed {start = } contains nodes not in g")

    # keep track of visited nodes
    visited: list[bool] = [False] * g.num_nodes
    for n in start:
        visited[n] = True

    to_visit: list[int] = [i for i in range(g.num_nodes) if visited[i] is False]

    return start + list(np.random.permutation(to_visit))


def brute_force_tsp(g: Graph, start: int = 0) -> list[int]:
    """
    Bruteforce solves the Travelling Salesman Problem to generate an order
    Iterates over all possible paths and solves in brute force manner

    Runtime: O(n!)

    Parameters
    ----------
    g: Graph
        Input graph
        Assertions:
            g must be a complete graph

    start: list[int]
        Optional start of path (allows for partial solving)
        Assertions:
            Must contain nodes that are in the graph

    Returns
    -------
    list[int]
        Path order according to best TSP solution over g

    """

    # for now assume complete
    if not Graph.is_complete(g):
        raise ValueError("Passed graph is not complete")

    # check validity of start
    if start >= g.num_nodes:
        raise ValueError(f"{start = } is not in passed graph")

    # valid nodes to visit
    nodes = list(range(g.num_nodes))
    nodes.remove(start)

    min_dist = float("inf")
    best: list[int] = []

    # test every permutation
    for order in permutations(nodes):
        full_order: list[int] = [start] + list(order)
        dist = 0.0
        for i in range(g.num_nodes - 1):
            dist += g.edge_weight[full_order[i]][full_order[i + 1]]
        if dist < min_dist:
            min_dist = dist
            best = full_order

    return best


def held_karp(g: Graph, start: int = 0) -> list[int]:
    """
    Solves the Travelling Salesman Problem to generate an order
    Uses Held Karp algorithm

    Runtime: O(n^2 2^n)

    Parameters
    ----------
    g: Graph
        Input graph
        Assertions:
            g must be a complete graph

    start: int
        Optional start node of path (allows for partial solving)

    Returns
    -------
    list[int]
        Path order according to best TSP solution over g

    """

    # for now assume complete
    if not Graph.is_complete(g):
        raise ValueError("Passed graph is not complete")

    # assert validity of start
    if start >= g.num_nodes or start < 0:
        raise ValueError(f"{start = } is not in passed graph")

    # key: tuple(set[int]: nodes, int: end)
    # value: tuple(float: path length, list[int]:  order of nodes)
    completed = {}  # type: ignore # typing this would be too verbose

    # recursive solver
    def solve_tour(s: set[int], e: int) -> tuple[float, list[int]]:
        # base case: if no in-between nodes must take edge from start -> e
        if len(s) == 0:
            return (g.edge_weight[start][e], [start])

        min_length = float("inf")
        best_order: list[int] = []
        # otherwise iterate over S all possible second-t-last nodes
        for i in s:
            s_i: set[int] = set(s)
            s_i.remove(i)
            sublength, suborder = completed[frozenset(s_i), i]
            length: float = sublength + g.edge_weight[i][e]
            if length < min_length:
                min_length = length
                best_order = list(suborder) + [i]

        return min_length, best_order

    # solve TSP over all subsets of nodes, smallest to largest
    targets: set[int] = set(i for i in range(g.num_nodes))
    targets.remove(start)
    for k in range(1, len(targets) + 1):
        for subset in combinations(targets, k):
            for e in subset:
                s: set[int] = set(subset)
                s.remove(e)
                completed[frozenset(s), e] = solve_tour(s, e)

    # Find best TSP over all nodes (essentially solving last case again)
    tsp_sol = float("inf")
    best_order: list[int] = []
    for i in targets:
        s_i = set(targets)
        s_i.remove(i)
        tsp, order = completed[frozenset(s_i), i]
        if tsp < tsp_sol:
            tsp_sol = tsp
            best_order = order + [i]

    return best_order


def multi_agent_brute_force(
    g: Graph,
    k: int,
    f: Callable[..., list[int]] = brute_force_mwlp,
    max_size: int = 0,
) -> list[list[int]]:
    """
    Computes the optimal assignment of targets for a given heuristic

    Runtime: TODO

    Parameters
    ----------
    g: Graph
        Input Graph
        Assertions:
            Must be complete

    k: int
        Number of agents
        Assertions:
            0 < k <= g.num_nodes

    f: Callable[..., list[int]]
        Heuristic to optimize with
        Default:
            Brute force mwlp

    max_size: int
        Maximum number of nodes that agents can visit
        Default:
            0 which means any number is allowed
        Assertions:
            Must be >= 0

    Returns
    -------
    list[list[int]]
        Optimal assignment

    """

    # for now assume complete
    if Graph.is_complete(g) is False:
        raise ValueError("Passed graph is not complete")

    if k <= 0:
        raise ValueError(f"Multi-agent case must have non-zero agents ({k})")

    if k > g.num_nodes:
        raise ValueError(f"Multi-agent case cannot have more agents than nodes ({k})")

    if max_size < 0:
        raise ValueError(f"Maximum size of path cannot be negative ({max_size})")

    if max_size == 0:
        max_size = g.num_nodes

    # assume start is at 0
    nodes = list(range(1, g.num_nodes))

    best_order: list[list[int]] = []
    minimum = float("inf")

    # iterate through each partition
    for part in set_partitions(nodes, k):
        if any(len(subset) > max_size for subset in part):
            continue
        curr = float("-inf")
        part_order: list[list[int]] = []

        # iterate through each group in partition
        for nodes in part:
            # assume starting at 0
            full_list: list[int] = [0] + nodes
            sg, sto, _ = Graph.subgraph(g, full_list)

            # calculuate heuristic
            heuristic_order: list[int] = f(sg)
            curr = max(curr, wlp(sg, heuristic_order))

            # collect orders
            original_order = [sto[n] for n in heuristic_order]
            part_order.append(original_order)

        if curr < minimum:
            minimum = curr
            best_order = part_order

    return best_order


def greedy_assignment(g: Graph, k: int) -> list[list[int]]:
    """
    Greedy algorithm from "Predicting Outage Restoration..."
    Finds the agent with the current shortest path
    Assigns them the heaviest unvisited node

    Runtime: (if applicable)

    Parameters
    ----------
    g: Graph
        Input graph
        Assertions:
            g must be a complete graph

    k: int
        Number of agents

    Returns
    -------
    list[list[int]]
        Assigned targets and order of targets for each agent.

    """

    if Graph.is_complete(g) is False:
        raise ValueError("Passed graph is not complete")

    # The only valid nodes to visit are non-starting nodes
    nodes: list[int] = list(range(1, g.num_nodes))
    # Sort the nodes from heaviest to least heavy
    nodes = sorted(nodes, key=lambda x: g.node_weight[x], reverse=True)
    # All paths must start with the start node
    paths: list[list[int]] = [[0] for _ in range(k)]

    for node in nodes:
        # find agent with shortest path (i.e. the agent who will finish first)
        agent: int = min(range(k), key=lambda x: path_length(g, paths[x]))
        # append current node (heaviest unvisited) to agent
        paths[agent].append(node)

    return paths


def greedy_random_assignment(g: Graph, k: int, r: float) -> list[list[int]]:
    """
    Greedy + Random algorithm from "Predicting Outage Restoration..."
    Group agents into two groups:
        Group 1: Greedy
        Group 2: Random neighbor
    If the agent with the current shortest path is in Group 1, assign
    the heaviest node
    Otherwise find a random node in the radius r from the end of their path
    If no node exists, send them to the nearest neighbor

    Runtime: (if applicable)

    Parameters
    ----------
    g: Graph
        Input graph
        Assertions:
            g must be a complete graph

    k: int
        Number of agents

    r: float
        Radius of random seach

    Returns
    -------
    list[list[int]]
        Assigned targets and order of targets for each agent.

    """

    if Graph.is_complete(g) is False:
        raise ValueError("Passed graph is not complete")

    # The only valid nodes to visit are non-starting nodes
    nodes: set[int] = set(range(1, g.num_nodes))
    # Randomly divide the agents into 2 groups
    # group1: Finds the heaviest unvisited node
    # group2: Finds a random node in a certain radius
    group1: set[int] = set(random.sample(range(k), k // 2))
    # All paths must start with the start node
    paths: list[list[int]] = [[0] for _ in range(k)]

    while len(nodes) > 0:
        idx: int = min(range(k), key=lambda x: path_length(g, paths[x]))
        # Greedy agents
        if idx in group1:
            # Find heaviest node
            highest_weight: int = max(nodes, key=lambda x: g.node_weight[x])
            # append current node (heaviest unvisited) to agent
            paths[idx].append(highest_weight)
            nodes.remove(highest_weight)
        # Random destination agents
        else:
            # Find nodes in the current radius
            curr_loc: int = paths[idx][-1]
            choices: list[int] = [i for i in nodes if g.edge_weight[curr_loc][i] <= r]
            # If there are no nodes in the radius, pick nearest neighbor
            if len(choices) == 0:
                nearest: int = min(nodes, key=lambda x: g.edge_weight[curr_loc][x])
                paths[idx].append(nearest)
                nodes.remove(nearest)
            else:
                choice: int = random.choice(choices)
                paths[idx].append(choice)
                nodes.remove(choice)

    return paths


def nearest_neighbor_assignment(g: Graph, k: int) -> list[list[int]]:
    """
    Nearest Neighbor algorithm from "Agent Based Model to Estimate..."
    Find the agent with the current shortest path.
    Assign them the nerest unvisited neighbor

    Runtime: (if applicable)

    Parameters
    ----------
    g: Graph
        Input graph
        Assertions:
            g must be a complete graph

    k: int
        Number of agents

    Returns
    -------
    list[list[int]]
        Assigned targets and order of targets for each agent.

    """

    if Graph.is_complete(g) is False:
        raise ValueError("Passed graph is not complete")

    # The only valid nodes to visit are non-starting nodes
    nodes: set[int] = set(range(1, g.num_nodes))
    # All paths must start with the start node
    paths: list[list[int]] = [[0] for _ in range(k)]

    while len(nodes) > 0:
        # Find agent with the current shortest path
        idx: int = min(range(k), key=lambda x: path_length(g, paths[x]))
        # Find closest node
        curr_loc: int = paths[idx][-1]
        closest: int = min(nodes, key=lambda x: g.edge_weight[curr_loc][x])
        # append current node to agent
        paths[idx].append(closest)
        nodes.remove(closest)

    return paths


def transfers_and_swaps_mwlp(
    g: Graph, part: list[set[int]], f: Callable[..., list[int]], safe: bool = True
) -> list[set[int]]:
    """
    Algorithm 1: Improve Partition from "Balanced Task Allocation..."
    Transfers and swaps nodes from one agent to another based on the passed heuristic

    Runtime: TODO

    Parameters
    ----------
    g: Graph
        Input graph
        Assertions:
            g must be a complete graph

    part: list[set[int]]
        Starting unordered assignment of nodes for each agent
        Assertions:
            Must be an agent partition

    f: Callable[..., list[int]]
        maximum,  minimum, range, average
        Passed heuristic

    safe: bool
        If true, type checking occurs

    Returns
    -------
    list[set[int]]
        Resulting unordered assignment of nodes after transfer and swaps

    """

    if safe:
        if Graph.is_complete(g) is False:
            raise ValueError("Passed graph is not complete")

        if Graph.is_agent_partition(g, part) is False:
            raise ValueError("Passed partition is invalid")

    # creating a deep copy to be safe
    partition: list[set[int]] = [set(s) for s in part]

    # This is a deterministic algorithm
    # Thus if we get to a partition that we have seen before, we have hit a loop
    # When this occurs we return the current partition since in pratice this current
    #   was very close to the local minimum (somewhere in the loop)
    # Convert sets to frozensets for hashability
    # Convert lists to frozensets for the same reason
    seen: set[frozenset[frozenset[int]]] = set()
    no_repeats: bool = True

    m: int = len(partition)
    pairs: list[tuple[int, int]] = list(combinations(set(range(m)), 2))
    # Use these arrays as "hashmaps" of indicator variables
    # to see if a pair needs to be checked

    # determine if partition i needs to be checked for transfer
    check_transfers: list[bool] = [True] * m
    # determine if partition i needs to be checked for swap
    check_swaps: list[bool] = [True] * m
    # determine if pair i, j needs to be checked for transfer
    check_transfer_pairs: list[bool] = [True] * len(pairs)
    # determine if pair i, j needs to be checked for swap
    check_swap_pairs: list[bool] = [True] * len(pairs)
    checks: int = (
        check_transfers.count(True)
        + check_swaps.count(True)
        + check_transfer_pairs.count(True)
        + check_swap_pairs.count(True)
    )

    # while there are partitions to be checked
    while checks > 0 and no_repeats is True:
        curr_partition: frozenset[frozenset[int]] = frozenset(
            frozenset(s) for s in partition
        )
        if curr_partition in seen:
            no_repeats = False
        else:
            seen.add(curr_partition)

        for idx, (i, j) in enumerate(pairs):
            # transfers
            if check_transfer_pairs[idx] or check_transfers[i] or check_transfers[j]:
                g_i, g_j = set(partition[i]), set(partition[j])

                sub_i, _, _ = Graph.subgraph(g, g_i)
                sub_j, _, _ = Graph.subgraph(g, g_j)

                size_i: float = wlp(sub_i, f(sub_i))
                size_j: float = wlp(sub_j, f(sub_j))

                # we presume size_i => size_j
                if size_i < size_j:
                    g_i, g_j = g_j, g_i
                    sub_i, sub_j = sub_j, sub_i
                    size_i, size_j = size_j, size_i

                size_max: float = max(size_i, size_j)
                v_star: int = -1
                for v in g_i:
                    if v != 0:
                        new_i: set[int] = set(g_i)
                        new_i.remove(v)
                        new_j: set[int] = set(g_j)
                        new_j.add(v)

                        new_sub_i, _, _ = Graph.subgraph(g, new_i)
                        new_sub_j, _, _ = Graph.subgraph(g, new_j)
                        new_size_i: float = wlp(new_sub_i, f(new_sub_i))
                        new_size_j: float = wlp(new_sub_j, f(new_sub_j))

                        if (curr_max := max(new_size_i, new_size_j)) < size_max:
                            size_max = curr_max
                            v_star = v

                if v_star != -1:
                    g_i.remove(v_star)
                    g_j.add(v_star)

                    partition[i] = set(g_i)
                    partition[j] = set(g_j)

                    check_transfer_pairs[idx] = True
                    check_transfers[i] = True
                    check_transfers[j] = True
                else:
                    check_transfer_pairs[idx] = False
                    check_transfers[i] = False
                    check_transfers[j] = False

            # swaps
            elif check_swap_pairs[idx] or check_swaps[i] or check_swaps[j]:
                g_i, g_j = set(partition[i]), set(partition[j])

                sub_i, _, _ = Graph.subgraph(g, g_i)
                sub_j, _, _ = Graph.subgraph(g, g_j)

                size_i = wlp(sub_i, f(sub_i))
                size_j = wlp(sub_j, f(sub_j))

                size_max = max(size_i, size_j)
                v_i_star, v_j_star = -1, -1

                for (v, v_prime) in product(g_i, g_j):
                    if v != 0 and v_prime != 0:
                        # swap v and v_prime
                        new_i = set(g_i)
                        new_i.remove(v)
                        new_i.add(v_prime)
                        new_j = set(g_j)
                        new_j.remove(v_prime)
                        new_j.add(v)

                        new_sub_i, _, _ = Graph.subgraph(g, new_i)
                        new_sub_j, _, _ = Graph.subgraph(g, new_j)
                        new_size_i = wlp(new_sub_i, f(new_sub_i))
                        new_size_j = wlp(new_sub_j, f(new_sub_j))

                        if (curr_max := max(new_size_i, new_size_j)) < size_max:
                            size_max = curr_max
                            v_i_star, v_j_star = v, v_prime

                if v_i_star != -1 and v_j_star != -1:
                    g_i.remove(v_i_star)
                    g_i.add(v_j_star)
                    g_j.remove(v_j_star)
                    g_j.add(v_i_star)

                    partition[i] = set(g_i)
                    partition[j] = set(g_j)

                    check_swap_pairs[idx] = True
                    check_swaps[i] = True
                    check_swaps[j] = True
                else:
                    check_swap_pairs[idx] = False
                    check_swaps[i] = False
                    check_swaps[j] = False

        checks = (
            check_transfers.count(True)
            + check_swaps.count(True)
            + check_transfer_pairs.count(True)
            + check_swap_pairs.count(True)
        )

    return partition


def transfer_outliers_mwlp(
    g: Graph,
    part: list[set[int]],
    f: Callable[..., list[int]],
    alpha: float,
    safe: bool = True,
) -> list[set[int]]:
    """
    Algorithm 2: Transfer Outliers from "Balanced Task Allocation..."
    Identifies outliers in each partition and moves them to a more ideal agent
    Uses passed heuristic and alpha threshold to determine if a node needs to move

    Runtime: TBD

    Parameters
    ----------
    g: Graph
        Input graph
        Assertions:
            g must be a complete graph

    part: list[set[int]]
        Starting unordered assignment of nodes for each agent

    f: Callable[..., list[int]]
        Passed heuristic
        Assertions:
            Must be an agent partition

    alpha: float
        Threshold for detecting outliers
        Assertions:
            0 <= alpha <= 1

    safe: bool
        If true, type checking occurs

    Returns
    -------
    list[set[int]]
        Resulting unordered assignment of nodes after transferring outliers

    """

    if safe:
        if Graph.is_complete(g) is False:
            raise ValueError("Passed graph is not complete")

        if Graph.is_agent_partition(g, part) is False:
            raise ValueError("Passed partition is invalid")

        if not 0 <= alpha <= 1:
            raise ValueError("Passed alpha threshold is out of range")

    # creating a deep copy to be safe
    partition: list[set[int]] = [set(s) for s in part]

    outliers: set[int] = set()
    p_old: dict[int, int] = {}
    p_new: dict[int, int] = {}

    m: int = len(partition)
    for i in range(m):
        for node in set(v for v in partition[i] if v != 0):
            # determine if outlier
            sub_g, _, _ = Graph.subgraph(g, partition[i])
            remove_node: list[int] = [v for v in partition[i] if v != node]
            sub_g_without_node, _, _ = Graph.subgraph(g, remove_node)
            with_node: float = wlp(sub_g, f(sub_g))
            without_node: float = wlp(sub_g_without_node, f(sub_g_without_node))
            contribution: float = (with_node - without_node) / with_node
            if contribution > alpha:
                # find minimizer of this
                destination: int = -1
                min_total = float("inf")

                for j in range(m):
                    # Find where adding outlier node minimizes contribution
                    sub_g_j, _, _ = Graph.subgraph(g, partition[j] | {node})
                    total: float = wlp(sub_g_j, f(sub_g_j))
                    if total < min_total:
                        destination = j
                        min_total = total

                if destination not in {-1, i}:
                    outliers.add(node)
                    p_old[node] = i
                    p_new[node] = destination

    for outlier in outliers:
        partition[p_old[outlier]].remove(outlier)
        partition[p_new[outlier]].add(outlier)

    return partition


def evaluate_partition_heuristic(
    g: Graph,
    partition: list[set[int]],
    f: Callable[..., list[int]],
    safe: bool = True,
) -> float:
    """
    Function to evaluate a given partition of agents using a heuristic
    Returns partition with the largest weighted latency

    Runtime: TBD

    Parameters
    ----------
    g: Graph
        Input graph
        Assertions:
            g must be a complete graph

    part: list[set[int]]
        Unordered assignment of nodes for each agent

    f: Callable[..., list[int]]
        Passed heuristic
        Assertions:
            Must be an agent partition

    safe: bool
        If true, type checking occurs

    Returns
    -------
    float:
        max over all s in partition of wlp(g, f(s))

    """
    if safe:
        if Graph.is_complete(g) is False:
            raise ValueError("Passed graph is not complete")

        if Graph.is_agent_partition(g, partition) is False:
            raise ValueError("Passed partition is invalid")

    curr_max = float("-inf")
    for subset in partition:
        sub_g, _, _ = Graph.subgraph(g, subset)
        curr_max = max(curr_max, wlp(sub_g, f(sub_g)))

    return curr_max


def find_partition_with_heuristic(
    g: Graph,
    part: list[set[int]],
    f: Callable[..., list[int]],
    alpha: float,
    safe: bool = True,
) -> list[set[int]]:
    """
    Adaptation of Algorithm 3: AHP from "Balanced Task Allocation..."
    Runs iterations of Algorithm 1 and 2 using the passed heuristic
    Runs iterations until there are no more improvements to be made

    Runtime: TBD

    Parameters
    ----------
    g: Graph
        Input graph
        Assertions:
            g must be a complete graph

    part: list[set[int]]
        Starting unordered assignment of nodes for each agent

    f: Callable[..., list[int]]
        Passed heuristic
        Assertions:
            Must be an agent partition

    alpha: float
        Threshold for detecting outliers
        Assertions:
            0 <= alpha <= 1

    safe: bool
        If true, type checking occurs

    Returns
    -------
    list[set[int]]
        Resulting unordered assignment of nodes after iterations

    """

    if safe:
        if Graph.is_complete(g) is False:
            raise ValueError("Passed graph is not complete")

        if Graph.is_agent_partition(g, part) is False:
            raise ValueError("Passed partition is invalid")

        if not 0 <= alpha <= 1:
            raise ValueError("Passed alpha threshold is out of range")

    # creating a deep copy to be safe
    partition: list[set[int]] = [set(s) for s in part]
    before: float = evaluate_partition_heuristic(g, partition, f, safe)

    improved: list[set[int]] = transfers_and_swaps_mwlp(g, partition, f, safe)
    after: float = evaluate_partition_heuristic(g, improved, f, safe)

    if improvements_decreased := (after < before):
        partition = [set(subset) for subset in improved]

    while improvements_decreased:
        before = evaluate_partition_heuristic(g, partition, f, safe)

        improved = [set(subset) for subset in partition]
        improved = transfer_outliers_mwlp(g, improved, f, alpha, safe)
        improved = transfers_and_swaps_mwlp(g, improved, f, safe)
        after = evaluate_partition_heuristic(g, improved, f, safe)

        if improvements_decreased := (after < before):
            partition = [set(subset) for subset in improved]

    return partition
