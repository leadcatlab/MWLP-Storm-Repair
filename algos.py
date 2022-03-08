from collections import deque
from itertools import combinations, permutations
from typing import Callable, Deque, Optional

import numpy as np
from more_itertools import set_partitions

from graph import Graph

# TODO: Implement MWLP_DP from Exact algorithms for the minimum latency problem
# TODO: Implement Christofides' Algorithm
# TODO: better docstrings


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


# TODO: Allow for more variable start of list of nodes for all heuristic functions
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
    q: Deque[int] = deque()
    q.appendleft(order[-1])

    while len(q) != 0:
        curr: int = q.pop()
        best_weight = float("-inf")
        heaviest: int = -1
        for n in g.adjacen_list[curr]:
            if not visited[n] and g.node_weight[n] > best_weight:
                best_weight = g.node_weight[n]
                heaviest = n
        if heaviest != -1:
            q.appendleft(heaviest)
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


def partition_heuristic(
    g: Graph, f: Callable[..., list[int]], k: int
) -> tuple[float, list[list[int]]]:
    """Bruteforce multi-agent MWLP

    Generates best partition according to passed heuristic f

    Args:
        g: input graph
        f: heuristic
        k: number of agents

    Returns:
        float: optimal MWLP value according to heuristic
        list[list[int]: Best graph partition and order
            partss in each partition is ordered such tha
            the best order for each partition is maintained

    """

    # for now assume complete
    if Graph.is_complete(g) is False:
        raise ValueError("Passed graph is not complete")

    if k <= 0:
        raise ValueError(f"Multi-agent case must have non-zero agents ({k})")

    if k > g.num_nodes:
        raise ValueError(f"Multi-agent case cannot have more agents than nodes ({k})")

    # assume start is at 0
    nodes = list(range(1, g.num_nodes))

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
            sg, sto, _ = Graph.subgraph(g, full_order)

            # calculuate heuristic
            heuristic_order: list[int] = f(sg)
            curr += wlp(sg, heuristic_order)

            # collect orders
            original_order = [sto[n] for n in heuristic_order]
            part_order.append(original_order)

        if curr < minimum:
            minimum = curr
            best_order = part_order

    return minimum, best_order


def optimal_number_of_agents(
    g: Graph, f: Callable[..., list[int]], k_min: int, k_max: int
) -> tuple[float, list[list[int]]]:
    """Bruteforce multi-agent MWLP for variable number of agents

    Generates best partition according to passed heuristic f
    parts in partition maintains best order for each part
    Length of optimal partition is number of agents used

    Args:
        g: input graph
        f: heuristic
        k_min: minumum number of agents
            (must be >= 1)
        k_max: maximum number of agents
            (k_max <= g.num_nodes - 1 since we don't count start node of 0)

    Returns:
        float: optimal MWLP value according to heuristic
        list[list[int]: Optimal graph partition
            Order of each part is optimal
            number of agents = number of parts

    """

    # for now assume complete
    if Graph.is_complete(g) is False:
        raise ValueError("Passed graph is not complete")

    if k_max <= k_min:
        raise ValueError("{k_min = } >= {k_max = }")

    if k_min <= 0:
        raise ValueError(f"Multi-agent case must have non-zero agents ({k_min})")

    if k_max >= g.num_nodes:
        raise ValueError(
            f"Multi-agent case cannot have more agents than non-start nodes ({k_max})"
        )

    best_order: list[list[int]] = []
    minimum = float("inf")

    # iterate through all possible numbers of agents
    for k in range(k_min, k_max + 1):
        min_for_k, order_for_k = partition_heuristic(g, f, k)
        if min_for_k < minimum:
            minimum = min_for_k
            best_order = order_for_k

    return minimum, best_order


def choose2(n: int) -> list[tuple[int, int]]:
    """Gives all pairs (i, j) for 0 <= i, j < n without order"""

    if n <= 1:
        raise ValueError("Passed n is too small")

    pairs: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j))

    return pairs


def transfers_and_swaps_mwlp(
    g: Graph, partition: list[set[int]], f: Callable[..., list[int]]
) -> list[set[int]]:

    if Graph.is_complete(g) is False:
        raise ValueError("Passed graph is not complete")

    if Graph.is_undirected(g) is False:
        raise ValueError("Passed graph is not undirected")

    if Graph.is_agent_partition(g, partition) is False:
        raise ValueError("Passed partition is invalid")

    m: int = len(partition)
    pairs: list[tuple[int, int]] = choose2(m)
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
        sum(check_transfers)
        + sum(check_swaps)
        + sum(check_transfer_pairs)
        + sum(check_swap_pairs)
    )

    # while there are partitions to be checked
    while checks > 0:
        # print(f"{checks = }")
        # TODO: Formal justification of these conditions
        for idx, (i, j) in enumerate(pairs):
            # transfers
            if check_transfer_pairs[idx] or check_transfers[i] or check_transfers[j]:
                g_i, g_j = partition[i], partition[j]

                assert 0 in g_i
                assert 0 in g_j

                sub_i, _, _ = Graph.subgraph(g, list(g_i))
                sub_j, _, _ = Graph.subgraph(g, list(g_j))

                size_i: float = wlp(sub_i, f(sub_i))
                size_j: float = wlp(sub_j, f(sub_j))

                # we presume size_i => size_j
                if size_i <= size_j:
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

                        assert 0 in new_i
                        assert 0 in new_j

                        # NOTE: This order is random, needs to be fixed in some way
                        new_sub_i, _, _ = Graph.subgraph(g, list(new_i))
                        new_sub_j, _, _ = Graph.subgraph(g, list(new_j))

                        new_size_i = wlp(new_sub_i, f(new_sub_i))
                        new_size_j = wlp(new_sub_j, f(new_sub_j))
                        curr_max = max(new_size_i, new_size_j)

                        if curr_max < size_max:
                            size_max = curr_max
                            v_star = v

                if v_star > 0:
                    g_i.remove(v_star)
                    g_j.add(v_star)

                    partition[i] = g_i
                    partition[j] = g_j

                    check_transfer_pairs[idx] = True
                    check_transfers[i] = True
                    check_transfers[j] = True
                else:
                    check_transfer_pairs[idx] = False
                    check_transfers[i] = False
                    check_transfers[j] = False

            # swaps
            elif check_swap_pairs[idx] or check_swaps[i] or check_swaps[j]:
                g_i, g_j = partition[i], partition[j]

                assert 0 in g_i
                assert 0 in g_j

                sub_i, _, _ = Graph.subgraph(g, list(g_i))
                sub_j, _, _ = Graph.subgraph(g, list(g_j))

                size_i = wlp(sub_i, f(sub_i))
                size_j = wlp(sub_j, f(sub_j))

                size_max = max(size_i, size_j)
                v_i_star, v_j_star = -1, -1

                for v in g_i:
                    for v_prime in g_j:
                        if v != 0 and v_prime != 0:
                            # swap v and v_prime
                            new_i = set(g_i)
                            new_i.remove(v)
                            new_i.add(v_prime)
                            new_j = set(g_j)
                            new_j.remove(v_prime)
                            new_j.add(v)

                            assert 0 in new_i
                            assert 0 in new_j

                            # NOTE: This order is random, needs to be fixed in some way
                            new_sub_i, _, _ = Graph.subgraph(g, list(new_i))
                            new_sub_j, _, _ = Graph.subgraph(g, list(new_j))

                            new_size_i = wlp(new_sub_i, f(new_sub_i))
                            new_size_j = wlp(new_sub_j, f(new_sub_j))
                            curr_max = max(new_size_i, new_size_j)
                            if curr_max < size_max:
                                size_max = curr_max
                                v_i_star, v_j_star = v, v_prime

                if v_i_star > 0 and v_j_star > 0:
                    # print(f"swapping {v_i_star} to {i} and {v_j_star} to {j}")

                    g_i.remove(v_i_star)
                    g_i.add(v_j_star)
                    g_j.remove(v_j_star)
                    g_j.add(v_i_star)

                    partition[i] = g_i
                    partition[j] = g_j

                    check_swap_pairs[idx] = True
                    check_swaps[i] = True
                    check_swaps[j] = True
                else:
                    check_swap_pairs[idx] = False
                    check_swaps[i] = False
                    check_swaps[j] = False
        checks = (
            sum(check_transfers)
            + sum(check_swaps)
            + sum(check_transfer_pairs)
            + sum(check_swap_pairs)
        )
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
    pairs: list[tuple[int, int]] = choose2(n)
    shortcut: float = 0.0
    weight_sum: int = sum(g.node_weight[1:])
    for i, j in pairs:
        shortcut += weight_sum * g.edge_weight[i][j]
    return shortcut / (n - 1)


def transfers_and_swaps_mwlp_with_average(
    g: Graph, partition: list[set[int]]
) -> list[set[int]]:
    if Graph.is_complete(g) is False:
        raise ValueError("Passed graph is not complete")

    if Graph.is_undirected(g) is False:
        raise ValueError("Passed graph is not undirected")

    if Graph.is_agent_partition(g, partition) is False:
        raise ValueError("Passed partition is invalid")

    m: int = len(partition)
    pairs: list[tuple[int, int]] = choose2(m)
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
        sum(check_transfers)
        + sum(check_swaps)
        + sum(check_transfer_pairs)
        + sum(check_swap_pairs)
    )

    # while there are partitions to be checked
    while checks > 0:
        # print(f"{checks = }")
        # TODO: Formal justification of these conditions
        for idx, (i, j) in enumerate(pairs):
            # transfers
            if check_transfer_pairs[idx] or check_transfers[i] or check_transfers[j]:
                g_i, g_j = partition[i], partition[j]

                assert 0 in g_i
                assert 0 in g_j

                sub_i, _, _ = Graph.subgraph(g, list(g_i))
                sub_j, _, _ = Graph.subgraph(g, list(g_j))

                size_i: float = all_possible_wlp_orders_avg(sub_i)
                size_j: float = all_possible_wlp_orders_avg(sub_j)

                # we presume size_i => size_j
                if size_i <= size_j:
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

                        assert 0 in new_i
                        assert 0 in new_j

                        # NOTE: This order is random, needs to be fixed in some way
                        new_sub_i, _, _ = Graph.subgraph(g, list(new_i))
                        new_sub_j, _, _ = Graph.subgraph(g, list(new_j))

                        new_size_i = all_possible_wlp_orders_avg(new_sub_i)
                        new_size_j = all_possible_wlp_orders_avg(new_sub_j)

                        curr_max = max(new_size_i, new_size_j)
                        if curr_max < size_max:
                            size_max = curr_max
                            v_star = v

                if v_star > 0:
                    g_i.remove(v_star)
                    g_j.add(v_star)

                    partition[i] = g_i
                    partition[j] = g_j

                    check_transfer_pairs[idx] = True
                    check_transfers[i] = True
                    check_transfers[j] = True
                else:
                    check_transfer_pairs[idx] = False
                    check_transfers[i] = False
                    check_transfers[j] = False

            # swaps
            elif check_swap_pairs[idx] or check_swaps[i] or check_swaps[j]:
                g_i, g_j = partition[i], partition[j]

                assert 0 in g_i
                assert 0 in g_j

                sub_i, _, _ = Graph.subgraph(g, list(g_i))
                sub_j, _, _ = Graph.subgraph(g, list(g_j))

                size_i = all_possible_wlp_orders_avg(sub_i)
                size_j = all_possible_wlp_orders_avg(sub_j)

                size_max = max(size_i, size_j)
                v_i_star, v_j_star = -1, -1

                for v in g_i:
                    for v_prime in g_j:
                        if v != 0 and v_prime != 0:
                            # swap v and v_prime
                            new_i = set(g_i)
                            new_i.remove(v)
                            new_i.add(v_prime)
                            new_j = set(g_j)
                            new_j.remove(v_prime)
                            new_j.add(v)

                            assert 0 in new_i
                            assert 0 in new_j

                            # NOTE: This order is random, needs to be fixed in some way
                            new_sub_i, _, _ = Graph.subgraph(g, list(new_i))
                            new_sub_j, _, _ = Graph.subgraph(g, list(new_j))

                            new_size_i = all_possible_wlp_orders_avg(new_sub_i)
                            new_size_j = all_possible_wlp_orders_avg(new_sub_j)

                            curr_max = max(new_size_i, new_size_j)
                            if curr_max < size_max:
                                size_max = curr_max
                                v_i_star, v_j_star = v, v_prime

                if v_i_star > 0 and v_j_star > 0:
                    # print(f"swapping {v_i_star} to {i} and {v_j_star} to {j}")

                    g_i.remove(v_i_star)
                    g_i.add(v_j_star)
                    g_j.remove(v_j_star)
                    g_j.add(v_i_star)

                    partition[i] = g_i
                    partition[j] = g_j

                    check_swap_pairs[idx] = True
                    check_swaps[i] = True
                    check_swaps[j] = True
                else:
                    check_swap_pairs[idx] = False
                    check_swaps[i] = False
                    check_swaps[j] = False
        checks = (
            sum(check_transfers)
            + sum(check_swaps)
            + sum(check_transfer_pairs)
            + sum(check_swap_pairs)
        )
    return partition
