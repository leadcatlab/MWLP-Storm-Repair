import random
from collections import deque
from itertools import combinations, permutations, product
from typing import Callable, Deque, Optional

import numpy as np

from graph import Graph


def path_length(g: Graph, path: list[int]) -> float:
    length: float = 0.0
    for i in range(len(path) - 1):
        if path[i + 1] not in g.adjacen_list[path[i]]:
            raise ValueError(f"Edge {path[i]} --> {path[i + 1]} does not exist")
        length += g.edge_weight[path[i]][path[i + 1]]
    return length


def floyd_warshall(g: Graph) -> list[list[float]]:
    """Use Floyd-Warshall algorithm to solve all pairs shortest path (APSP)

    Runtime: O(n^3)

    Args:
        g: input graph

    Returns:
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
    for k in range(n):
        for i in range(n):
            for j in range(n):
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])

    return dist


def create_metric_from_graph(g: Graph) -> Graph:
    """Create metric graph from input graph
    Using Floyd-Warshall we can solve the APSP problem.
    This gives edge weights that satisfy the triangle inequality

    Runtime: O(n^3)

    Args:
        g: input graph

    Returns:
        Graph satisfying the triangle inequality (Metric Graph)

    """

    n: int = g.num_nodes

    metric = Graph(n)
    metric.node_weight = g.node_weight
    metric.adjacen_list = g.adjacen_list

    metric_weights: list[list[float]] = floyd_warshall(g)
    for i in range(n):
        for j in range(n):
            if (
                i != j
                and g.edge_weight[i][j] != -1
                and metric_weights[i][j] != float("inf")
            ):
                metric.edge_weight[i][j] = metric_weights[i][j]

    return metric


def wlp(g: Graph, order: list[int]) -> float:
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
        if node >= g.num_nodes or node < 0:
            raise ValueError(f"Node {node} is not in passed graph")

    if len(order) <= 1:
        return 0.0

    path_len: list[float] = [0.0] * len(order)
    for i in range(1, len(order)):
        if order[i] not in g.adjacen_list[order[i - 1]]:
            raise ValueError(f"Edge {order[i - 1]} --> {order[i]} does not exist")
        path_len[i] = path_len[i - 1] + g.edge_weight[order[i - 1]][order[i]]

    # sum over sequence [v_0, v_1, ..., v_n] of w(v_i) * L(0, v_i)
    return sum(g.node_weight[order[i]] * path_len[i] for i in range(len(order)))


def brute_force_mwlp(g: Graph, start: Optional[list[int]] = None) -> list[int]:
    """Calculate minumum weighted latency

    Iterates over all possible paths and solves in brute force manner

    Runtime: O(n!)

    Args:
        g: input graph

    Returns:
        list[int[: Path order for the minumum weighted latency
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


def cost(g: Graph, order: list[int]) -> float:
    """cost function used for MWLP_DP algorithm

    from "Polynomial time algorithms for some minimum latency problems" (Wu)
    c(order) = Latency(order) + (w(g) + w(order))*Length(order)

    Runtime: O(n)

    Args:
        g: input graph
        order: list of nodes in subtour

    Return:
        float: output of cost function c(order)
    """

    # check nodes in order are actually valid nodes
    for node in order:
        if node >= g.num_nodes:
            raise ValueError(f"Node {node} is not in passed graph")

    latency: float = wlp(g, order)
    length: float = 0.0
    for i in range(len(order) - 1):
        length += g.edge_weight[order[i]][order[i + 1]]

    weight_order: int = sum(g.node_weight[n] for n in order)
    return latency + (sum(g.node_weight) - weight_order) * length


def nearest_neighbor(g: Graph, start: Optional[list[int]] = None) -> list[int]:
    """Approximates MWLP using nearest neighbor heuristic

    Generates sequence starting from 0 going to the nearest node

    Args:
        g: input graph

    Returns:
        list[int]: nearest neighbor order

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
    order: list[int] = list(start)
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
    """Approximates MWLP using greedy heuristic

    Generates sequence starting from 0 going to the node of greatest weight

    Args:
        g: input graph

    Returns:
        list[int]: greedy order

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
    order: list[int] = list(start)

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


def random_order(g: Graph, start: Optional[list[int]] = None) -> list[int]:
    """Random order generator

    Generates random sequence starting from start or starting from seq_start -> start
    Essentially a wrapper around np.random.permutation

    Args:
        g: input graph

    Returns:
        list[int]: greedy order

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

    """Brute Force TSP

    Generates sequence that solves travelling salesman. Solved via
    pure brute force of all possible orders.

    Args:
        g: input graph

    Returns:
        list[int]: Solution to the Travelling Salesman Problem

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


# TODO: Give these better names
def uconn_strat_1(g: Graph, k: int) -> list[list[int]]:
    if Graph.is_complete(g) is False:
        raise ValueError("Passed graph is not complete")
    if Graph.is_undirected(g) is False:
        raise ValueError("Passed graph is not undirected")

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


def uconn_strat_2(g: Graph, k: int, r: float) -> list[list[int]]:
    if Graph.is_complete(g) is False:
        raise ValueError("Passed graph is not complete")
    if Graph.is_undirected(g) is False:
        raise ValueError("Passed graph is not undirected")

    # The only valid nodes to visit are non-starting nodes
    nodes: set[int] = set(range(1, g.num_nodes))
    # Randomly divide the agents into 2 groups
    # group1: Finds the heaviest unvisited node
    # group2: Finds a random node in a certain radius
    group1: set[int] = set(random.sample(range(k), k // 2))
    # All paths must start with the start node
    paths: list[list[int]] = [[0] for _ in range(k)]

    idx: int = 0
    while len(nodes) > 0:
        # Greedy agents
        if idx in group1:
            # Find heaviest node
            highest_weight: int = max(nodes, key=lambda x: g.node_weight[x])
            # find agent in group 1 with shortest path
            agent: int = min(group1, key=lambda x: path_length(g, paths[x]))
            # append current node (heaviest unvisited) to agent
            paths[agent].append(highest_weight)
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

        idx = (idx + 1) % k

    return paths


def transfers_and_swaps_mwlp(
    g: Graph, part: list[set[int]], f: Callable[..., list[int]]
) -> list[set[int]]:

    if Graph.is_complete(g) is False:
        raise ValueError("Passed graph is not complete")

    if Graph.is_undirected(g) is False:
        raise ValueError("Passed graph is not undirected")

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
        # TODO: Determine if this check should go somewhere else
        curr_partition: frozenset[frozenset[int]] = frozenset(
            frozenset(s) for s in partition
        )
        if curr_partition in seen:
            no_repeats = False
        else:
            seen.add(curr_partition)

        # TODO: Formal justification of these conditions
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
    g: Graph, part: list[set[int]], f: Callable[..., list[int]], alpha: float
) -> list[set[int]]:

    if Graph.is_complete(g) is False:
        raise ValueError("Passed graph is not complete")

    if Graph.is_undirected(g) is False:
        raise ValueError("Passed graph is not undirected")

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
    g: Graph, partition: list[set[int]], f: Callable[..., list[int]]
) -> float:
    if Graph.is_agent_partition(g, partition) is False:
        raise ValueError("Passed partition is invalid")

    curr_max = float("-inf")
    for subset in partition:
        sub_g, _, _ = Graph.subgraph(g, subset)
        curr_max = max(curr_max, wlp(sub_g, f(sub_g)))

    return curr_max


def find_partition_with_heuristic(
    g: Graph, part: list[set[int]], f: Callable[..., list[int]], alpha: float
) -> list[set[int]]:
    if Graph.is_complete(g) is False:
        raise ValueError("Passed graph is not complete")

    if Graph.is_undirected(g) is False:
        raise ValueError("Passed graph is not undirected")

    if Graph.is_agent_partition(g, part) is False:
        raise ValueError("Passed partition is invalid")

    # creating a deep copy to be safe
    partition: list[set[int]] = [set(s) for s in part]
    before: float = evaluate_partition_heuristic(g, partition, f)

    improved: list[set[int]] = transfers_and_swaps_mwlp(g, partition, f)
    after: float = evaluate_partition_heuristic(g, improved, f)

    if improvements_decreased := (after < before):
        partition = [set(subset) for subset in improved]

    while improvements_decreased:
        before = evaluate_partition_heuristic(g, partition, f)

        improved = [set(subset) for subset in partition]
        improved = transfer_outliers_mwlp(g, improved, f, alpha)
        improved = transfers_and_swaps_mwlp(g, improved, f)
        after = evaluate_partition_heuristic(g, improved, f)

        if improvements_decreased := (after < before):
            partition = [set(subset) for subset in improved]

    return partition


def all_possible_wlp_orders_avg(g: Graph) -> float:
    """
    Heuristic for average weighted latency

    Consider all possible node orderings that start with 0
    Calculate the wlp for each order
    Sum them together, divide by (n - 1)!

    This calculates this in O(n^3) rather than O(n!)

    TODO: figure out why this shortcut works
    """

    if Graph.is_complete(g) is False:
        raise ValueError("Passed graph is not complete")

    if Graph.is_undirected(g) is False:
        raise ValueError("Passed graph is not undirected")

    n: int = g.num_nodes
    if n == 1:
        return 0.0

    pairs: list[tuple[int, int]] = list(combinations(set(range(n)), 2))
    shortcut: float = 0.0
    weight_sum: int = sum(g.node_weight[1:])
    for i, j in pairs:
        shortcut += weight_sum * g.edge_weight[i][j]
    return shortcut / (n - 1)


def transfers_and_swaps_mwlp_with_average(
    g: Graph, part: list[set[int]]
) -> list[set[int]]:

    if Graph.is_complete(g) is False:
        raise ValueError("Passed graph is not complete")

    if Graph.is_undirected(g) is False:
        raise ValueError("Passed graph is not undirected")

    if Graph.is_agent_partition(g, part) is False:
        raise ValueError("Passed partition is invalid")

    # creating a deep copy to be safe
    partition: list[set[int]] = [set(s) for s in part]

    # This is a deterministic algorithm
    # Thus if we get to a partition that we have seen before, we have hit a loop
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
        # TODO: Determine if this check should go somewhere else
        curr_partition: frozenset[frozenset[int]] = frozenset(
            frozenset(s) for s in partition
        )
        if curr_partition in seen:
            no_repeats = False
        else:
            seen.add(curr_partition)

        # TODO: Formal justification of these conditions
        for idx, (i, j) in enumerate(pairs):
            # transfers
            if check_transfer_pairs[idx] or check_transfers[i] or check_transfers[j]:
                g_i, g_j = set(partition[i]), set(partition[j])

                sub_i, _, _ = Graph.subgraph(g, g_i)
                sub_j, _, _ = Graph.subgraph(g, g_j)

                size_i: float = all_possible_wlp_orders_avg(sub_i)
                size_j: float = all_possible_wlp_orders_avg(sub_j)

                # we presume size_i => size_j
                if size_i < size_j:
                    g_i, g_j = g_j, g_i
                    sub_i, sub_j = sub_j, sub_i
                    size_i, size_j = size_j, size_i

                size_max: float = max(size_i, size_j)
                v_star: int = -1
                for v in g_i:
                    if v != 0:
                        new_i = set(g_i)
                        new_i.remove(v)
                        new_j = set(g_j)
                        new_j.add(v)

                        new_sub_i, _, _ = Graph.subgraph(g, new_i)
                        new_sub_j, _, _ = Graph.subgraph(g, new_j)

                        new_size_i = all_possible_wlp_orders_avg(new_sub_i)
                        new_size_j = all_possible_wlp_orders_avg(new_sub_j)

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

                size_i = all_possible_wlp_orders_avg(sub_i)
                size_j = all_possible_wlp_orders_avg(sub_j)

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
                        new_size_i = all_possible_wlp_orders_avg(new_sub_i)
                        new_size_j = all_possible_wlp_orders_avg(new_sub_j)

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


def transfer_outliers_mwlp_with_average(
    g: Graph, part: list[set[int]], alpha: float
) -> list[set[int]]:

    if Graph.is_complete(g) is False:
        raise ValueError("Passed graph is not complete")

    if Graph.is_undirected(g) is False:
        raise ValueError("Passed graph is not undirected")

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
            sub_g, _, _ = Graph.subgraph(g, list(partition[i]))
            remove_node: list[int] = list(v for v in partition[i] if v != node)
            sub_g_without_node, _, _ = Graph.subgraph(g, remove_node)
            with_node: float = all_possible_wlp_orders_avg(sub_g)
            without_node: float = all_possible_wlp_orders_avg(sub_g_without_node)
            contribution: float = (with_node - without_node) / with_node
            if contribution > alpha:
                # find minimizer of this
                destination: int = -1
                min_total = float("inf")

                for j in range(m):
                    sub_g_j, _, _ = Graph.subgraph(g, partition[j] | {node})
                    total: float = all_possible_wlp_orders_avg(sub_g_j)
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


def evaluate_partition_with_average(g: Graph, partition: list[set[int]]) -> float:
    if Graph.is_agent_partition(g, partition) is False:
        raise ValueError("Passed partition is invalid")

    curr_max = float("-inf")
    for subset in partition:
        sub_g, _, _ = Graph.subgraph(g, subset)
        curr_max = max(curr_max, all_possible_wlp_orders_avg(sub_g))

    return curr_max


def find_partition_with_average(
    g: Graph, part: list[set[int]], alpha: float
) -> list[set[int]]:
    if Graph.is_complete(g) is False:
        raise ValueError("Passed graph is not complete")

    if Graph.is_undirected(g) is False:
        raise ValueError("Passed graph is not undirected")

    if Graph.is_agent_partition(g, part) is False:
        raise ValueError("Passed partition is invalid")

    # creating a deep copy to be safe
    partition: list[set[int]] = [set(s) for s in part]
    before: float = evaluate_partition_with_average(g, partition)

    improved: list[set[int]] = transfers_and_swaps_mwlp_with_average(g, partition)
    after: float = evaluate_partition_with_average(g, improved)

    if improvements_decreased := (after < before):
        partition = [set(subset) for subset in improved]

    while improvements_decreased:
        before = evaluate_partition_with_average(g, partition)

        improved = list(set(subset) for subset in partition)
        improved = transfer_outliers_mwlp_with_average(g, improved, alpha)
        improved = transfers_and_swaps_mwlp_with_average(g, improved)
        after = evaluate_partition_with_average(g, improved)

        if improvements_decreased := (after < before):
            partition = [set(subset) for subset in improved]

    return partition
