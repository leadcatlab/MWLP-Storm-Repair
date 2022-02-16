from itertools import permutations, combinations
from collections import deque
from typing import Deque, Callable, Optional
from more_itertools import set_partitions
import numpy as np
import random
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


def create_agent_partition(g: Graph, k: int) -> list[set[int]]:

    # all agent contain 0 as start
    n: int = g.num_nodes
    partition: list[set[int]] = [{0} for _ in range(k)]
    for i in range(1, n):
        partition[random.randint(0, 1000) % k].add(i)

    return partition


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
    # TODO: Allow for more variable start of list of nodes

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
    # TODO: Allow for more variable start of list of nodes

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


def weight(g: Graph, u: int, v: int) -> float:
    if u >= g.num_nodes or u < 0:
        raise ValueError(f"{u} is not in passed graph")
    if v >= g.num_nodes or v < 0:
        raise ValueError(f"{v} is not in passed graph")
    if v not in g.adjacen_list[u]:
        raise ValueError(f"{v} is not in the adjacency list of {u}")
    if u not in g.adjacen_list[v]:
        raise ValueError(f"{u} is not in the adjacency list of {v}")

    return 0.5 * (g.node_weight[u] + g.node_weight[v]) + g.edge_weight[u][v]


def total_edge_weight(g: Graph, subgraph: set[int]) -> float:
    """W(G_i) from Balanced Task Allocation by Partitioning the
       Multiple Traveling Salesperson Problem (Vandermeulen et al,)

    This is the sum of w(e) for all edges in the subgraph
    w(u <-> v)  = 1/2 * (w(u) + w(v)) + l(u <-> v)

    Passed subgraph can be represented as a set since we know g is complete

    Used for transfers and swaps algorithm

    Runtime:

    Args:
        g: input graph
        subgraph: set of nodes that induce a subgraph of g

    Returns:
        float: total edge weight of subgraph
    """

    if Graph.is_complete(g) is False:
        raise ValueError("Passed graph is not complete")

    if Graph.is_undirected(g) is False:
        raise ValueError("Passed graph is not undirected")

    for v in subgraph:
        if v >= g.num_nodes or v < 0:
            raise ValueError(f"Passed {subgraph = } contains nodes not in g")

    subgraph_list = list(subgraph)
    total: float = 0.0
    n: int = len(subgraph_list)
    for i in range(n):
        for j in range(i + 1, n):
            u, v = subgraph_list[i], subgraph_list[j]
            total += weight(g, u, v)

    return total


def marginal_edge_weight(g: Graph, subgraph: set[int], v: int) -> float:
    """∆W(G_i, v) from Balanced Task Allocation by Partitioning the
       Multiple Traveling Salesperson Problem (Vandermeulen et al,)

    This is the sum of w(v, v') for all node v' in the subgraph
    w(v -> v')  = 1/2 * (w(v) + w(v')) + l(v -> v')
    This amounts to the "contribution" of v to W(G_i)

    Passed subgraph can be represented as a set since we know g is complete

    Used for transfers and swaps algorithm

    Runtime:

    Args:
        g: input graph
        subgraph: set of nodes that induce a subgraph of g
        v: node being transferred out of the subgraph

    Returns:
        float: total edge weight of subgraph
    """

    if Graph.is_complete(g) is False:
        raise ValueError("Passed graph is not complete")

    if Graph.is_undirected(g) is False:
        raise ValueError("Passed graph is not undirected")

    if v >= g.num_nodes or v < 0:
        raise ValueError(f"{v = } is not in passed graph")

    for n in subgraph:
        if n >= g.num_nodes or n < 0:
            raise ValueError(f"Passed {subgraph = } contains nodes not in g")

    total: float = 0.0
    for v_prime in subgraph:
        if v_prime != v:
            total += weight(g, v, v_prime)

    return total


def max_average_cycle_length(g: Graph, partition: list[set[int]]) -> float:
    """Finds maximum average length of a cycle over a given partition of a graph

    C_i(P) = max over G_i in P of S_a(G_i)
    S_a(G_i) = (2 / (n_i - 1)) * W(G_i)

    Runtime:

    Args:
        g: input graph
        partition: valid partition of g

    Returns:
        float: maximum average cycle length
    """
    if Graph.is_complete(g) is False:
        raise ValueError("Passed graph is not complete")

    if Graph.is_undirected(g) is False:
        raise ValueError("Passed graph is not undirected")

    if Graph.is_partition(g, partition) is False:
        raise ValueError("Passed partition is not valid")

    c_a = float("-inf")
    for subset in partition:
        n_i: float = len(subset)
        c_a = max(c_a, (2 / (n_i - 1)) * total_edge_weight(g, subset) if n_i > 1 else 0)

    return c_a


def transfers_and_swaps(g: Graph, partition: list[set[int]]) -> list[set[int]]:
    """Algorithm 1: Improve Partition from Balanced Task Allocation by Partitioning the
       Multiple Traveling Salesperson Problem (Vandermeulen et al,)

    Uses transfers and swaps inorder to find a local minimum best partition

    Runtime:

    Args:
        g: input graph
        p: partiton of g representing nodes that make up subgraphs

    Returns:
        list[set[int]]: Local minimum partition wrt cost function

    """
    # TODO: Deal with potential divide by zero errors from size heuristic
    # TODO: Determine how to measure >> for transfers
    # TODO: Determine why c_a after improvement is sometimes worse
    # TODO: Determine why infinite loops occur sometimes

    if Graph.is_complete(g) is False:
        raise ValueError("Passed graph is not complete")

    if Graph.is_undirected(g) is False:
        raise ValueError("Passed graph is not undirected")

    if Graph.is_partition(g, partition) is False:
        raise ValueError("Passed partition is not valid")

    m: int = len(partition)

    # initialize with all possible transfers / swaps
    transfers: set[tuple[int, int]] = {
        (i, j) for i in range(m) for j in range(m) if i != j
    }
    swaps = set(transfers)

    while len(transfers) > 0 or len(swaps) > 0:
        # assert(Graph.is_partition(g, partition))
        # print([len(part) for part in partition])

        # check transfers
        if len(transfers) > 0:
            i, j = transfers.pop()
            g_i, g_j = partition[i], partition[j]

            n_i, n_j = len(g_i), len(g_j)

            total_i = total_edge_weight(g, g_i)
            total_j = total_edge_weight(g, g_j)

            size_i_init: float = (2 / (n_i - 1)) * total_i if n_i > 1 else 0
            size_j_init: float = (2 / (n_j - 1)) * total_j if n_j > 1 else 0

            if size_i_init <= size_j_init:
                continue

            size_max = max(size_i_init, size_j_init)
            v_star: int = -1
            # Try all potential transfers, transfer if max is improved
            for v in g_i:
                size_i = (
                    (2 / (n_i - 2)) * (total_i - marginal_edge_weight(g, g_i, v))
                    if n_i > 2
                    else 0
                )
                size_j = (2 / n_j) * (total_j + marginal_edge_weight(g, g_j, v))
                if curr_max := max(size_i, size_j) < size_max:
                    size_max = curr_max
                    v_star = v

            # transfer v_star from g_i to g_j if needed
            if v_star != -1:
                # print(f"Transferring {v_star} from g_{i} to g_{j}")
                # update partitions`
                g_i.remove(v_star)
                g_j.add(v_star)

                # update list of transfers:
                for k in range(m):
                    if k not in {i, j}:
                        transfers.add((j, k))
                        transfers.add((k, i))

        # check swaps
        elif len(swaps) > 0:
            i, j = swaps.pop()
            g_i, g_j = partition[i], partition[j]

            n_i, n_j = len(g_i), len(g_j)

            total_i = total_edge_weight(g, g_i)
            total_j = total_edge_weight(g, g_j)

            size_i_init = (2 / (n_i - 1)) * total_i if n_i > 1 else 0
            size_j_init = (2 / (n_j - 1)) * total_j if n_j > 1 else 0

            size_max = max(size_i_init, size_j_init)
            # v_i_star swapped from g_i to g_j
            # v_j_star swapped from g_j to g_i
            v_i_star, v_j_star = -1, -1

            # try to swap v and v_prime
            # v      : g_i -> g_j
            # v_prime: g_j -> g_i
            for v in g_i:
                for v_prime in g_j:
                    curr_weight = (0.5) * (
                        g.node_weight[v] + g.node_weight[v_prime]
                    ) + g.edge_weight[v][v_prime]

                    weight_i = (
                        total_i
                        - marginal_edge_weight(g, g_i, v)
                        + marginal_edge_weight(g, g_i, v_prime)
                        - curr_weight
                    )
                    weight_j = (
                        total_j
                        + marginal_edge_weight(g, g_j, v)
                        - marginal_edge_weight(g, g_j, v_prime)
                        - curr_weight
                    )

                    size_i = (2 / (n_i - 1)) * weight_i if n_i > 1 else 0
                    size_j = (2 / (n_j - 1)) * weight_j if n_j > 1 else 0

                    # swap is advantageous if worst case is improved
                    if curr_max := max(size_i, size_j) < size_max:
                        size_max = curr_max
                        v_i_star, v_j_star = v, v_prime

            if v_i_star != -1 and v_j_star != -1:
                # print(f"Swapping {v_i_star} to g_{j} and {v_j_star} to g_{i}")

                # update partitions`
                g_i.remove(v_i_star)
                g_i.add(v_j_star)
                g_j.remove(v_j_star)
                g_j.add(v_i_star)

                # update list of transfers:
                for k in range(m):
                    if k not in {i, j}:
                        swaps.add((i, k))
                        swaps.add((j, k))

    return partition


def transfer_outliers(
    g: Graph, partition: list[set[int]], alpha: float = 1.5
) -> list[set[int]]:
    """Algorithm 2: Transfer Outliers from Balanced Task Allocation by Partitioning the
       Multiple Traveling Salesperson Problem (Vandermeulen et al,)

    Identify outliers in subgraphs to transfer

    Runtime:

    Args:
        g: input graph
        partition: partiton of g representing nodes that make up subgraphs
        alpha: float >= 1 that affects sensitivity(?) of making a transfer

    Returns:
        list[set[int]]: better partition of nodes with outliers moved as needed

    """
    if alpha < 1.0:
        raise ValueError(f"Passed {alpha = } is too small")

    if Graph.is_complete(g) is False:
        raise ValueError("Passed graph is not complete")

    if Graph.is_partition(g, partition) is False:
        raise ValueError("Passed partition is not valid")

    m: int = len(partition)
    outliers: set[int] = set()
    new: list[int] = [-1] * g.num_nodes
    old: list[int] = [-1] * g.num_nodes
    for i in range(m):
        for v in partition[i]:
            old[v] = i
    for i, g_i in enumerate(partition):
        criterion: float = alpha * (2 / len(g_i)) * total_edge_weight(g, g_i)
        for v in g_i:
            if marginal_edge_weight(g, g_i, v) > criterion:
                j: int = -1
                curr_min = float("inf")
                for i_prime in range(m):
                    if (
                        curr := marginal_edge_weight(g, partition[i_prime], v)
                        < curr_min
                    ):
                        curr_min = curr
                        j = i_prime

                if j not in {-1, i}:
                    outliers.add(v)
                    new[v] = j

    improved: list[set[int]] = list(partition)
    for v in outliers:
        if new[v] != -1:
            improved[old[v]].remove(v)
            improved[new[v]].add(v)

    return improved


def improve_partition(g: Graph, partition: list[set[int]]) -> list[set[int]]:
    """Algorithm 3: AHP from Balanced Task Allocation by Partitioning the
       Multiple Traveling Salesperson Problem (Vandermeulen et al,)

    Use algorithm 1 and 2 to improve a partition of a graph

    Runtime:

    Args:
        g: input graph
        partition: partition of g representing nodes that make up subgraphs

    Returns:
        list[set[int]]: near-optimal solution according to algorithm 1 and 2 and C_a
    """

    if Graph.is_complete(g) is False:
        raise ValueError("Passed graph is not complete")

    if Graph.is_partition(g, partition) is False:
        raise ValueError("Passed partition is not valid")

    # initial improvement (since transfers and swaps can never return worse result)
    init: float = max_average_cycle_length(g, partition)

    print("initial transfer and swap")
    p: list[set[int]] = transfers_and_swaps(g, partition)
    curr: float = max_average_cycle_length(g, p)

    # TODO: Why does this fail sometimes
    assert curr <= init

    improved: bool = True
    while improved:
        print("transferring outliers")
        p_prime: list[set[int]] = transfer_outliers(g, p)
        print("transferring and swapping")
        p_prime = transfers_and_swaps(g, p_prime)
        if improvement := max_average_cycle_length(g, p_prime) < curr:
            p = p_prime
            curr = improvement
            print("found improvement")
        else:
            improved = False

    return p


def transfers_mwlp(
    g: Graph, part: list[set[int]], f: Callable[..., list[int]]
) -> list[set[int]]:
    # Transfers only part of mwlp

    if Graph.is_complete(g) is False:
        raise ValueError("Passed graph is not complete")

    if Graph.is_undirected(g) is False:
        raise ValueError("Passed graph is not undirected")

    if Graph.is_agent_partition(g, part) is False:
        raise ValueError("Passed partition is invalid")

    # don't want to edit original partition
    m: int = len(part)
    partition: list[set[int]] = [{0} for _ in range(m)]
    for i in range(m):
        partition[i] = partition[i] | part[i]

    # initialize with all possible transfers / swaps
    transfers: set[tuple[int, int]] = {
        (i, j) for i in range(m) for j in range(m) if i != j
    }

    while len(transfers) > 0:
        i, j = transfers.pop()
        g_i, g_j = partition[i], partition[j]

        assert 0 in g_i
        assert 0 in g_j

        sub_i, _, _ = Graph.subgraph(g, list(g_i))
        sub_j, _, _ = Graph.subgraph(g, list(g_j))

        size_i_init = wlp(g, f(sub_i))
        size_j_init = wlp(g, f(sub_j))

        if size_i_init <= size_j_init:
            continue

        size_max = max(size_i_init, size_j_init)
        v_star: int = -1
        for v in g_i:
            if v != 0:
                # transfer v from g_i to g_j
                new_i = {v_i for v_i in g_i if v_i != v}
                new_j = set(g_j)
                new_j.add(v)

                assert 0 in new_i
                assert 0 in new_j

                new_sub_i, _, _ = Graph.subgraph(g, list(new_i))
                new_sub_j, _, _ = Graph.subgraph(g, list(new_j))

                new_size_i = wlp(new_sub_i, f(new_sub_i))
                new_size_j = wlp(new_sub_j, f(new_sub_j))

                if curr_max := max(new_size_i, new_size_j) < size_max:
                    size_max = curr_max
                    v_star = v

        if v_star > 0:
            # print(f"transferring {v_star} from {i} to {j}")
            g_i.remove(v_star)
            g_j.add(v_star)

            partition[i] = g_i
            partition[j] = g_j

            for k in range(m):
                if k not in {i, j}:
                    # j increased so may need to transfer to other nodes
                    # i decreased so may need to accept other nodes
                    transfers.add((j, k))
                    transfers.add((k, i))

    return partition


def transfers_and_swaps_mwlp(
    g: Graph, partition: list[set[int]], f: Callable[..., list[int]]
) -> list[set[int]]:
    # NOTE: THIS IS REALLY SLOW FOR NOW
    # TODO: Fix and optimize the following:
    #   Determine infinite swaps occur
    #   Recheck transfers after swaps?
    #   One single set of unchecked pairs?
    #   Store information
    #   Add and remove edges without making new graphs each time?

    """
    Outline:
        initialize sets to be checked
        attempt transfers:
            transfer node
            run heuristic on subgraph generated by partition
                start at 0, make sure subgraph contains 0
            keep best
        attempt swap
            swap node
            run heuristic on subgraph generated by partition
                start at 0, make sure subgraph contains 0
            keep best
    """

    if Graph.is_complete(g) is False:
        raise ValueError("Passed graph is not complete")

    if Graph.is_undirected(g) is False:
        raise ValueError("Passed graph is not undirected")

    m: int = len(partition)
    # 0 is the universal start, so all sets should contain it
    for part in partition:
        part.add(0)

    # initialize with all possible transfers / swaps
    transfers: set[tuple[int, int]] = {
        (i, j) for i in range(m) for j in range(m) if i != j
    }
    swaps = set(transfers)

    while len(transfers) > 0 or len(swaps) > 0:
        while len(transfers) > 0:
            i, j = transfers.pop()
            g_i, g_j = partition[i], partition[j]

            assert 0 in g_i
            assert 0 in g_j

            sub_i, _, _ = Graph.subgraph(g, list(g_i))
            sub_j, _, _ = Graph.subgraph(g, list(g_j))

            size_i_init = wlp(g, f(sub_i))
            size_j_init = wlp(g, f(sub_j))

            if size_i_init <= size_j_init:
                continue

            size_max = max(size_i_init, size_j_init)
            v_star: int = -1
            for v in g_i:
                if v != 0:
                    # transfer v from g_i to g_j
                    new_i = {v_i for v_i in g_i if v_i != v}
                    new_j = set(g_j)
                    new_j.add(v)

                    assert 0 in new_i
                    assert 0 in new_j

                    new_sub_i, _, _ = Graph.subgraph(g, list(new_i))
                    new_sub_j, _, _ = Graph.subgraph(g, list(new_j))

                    new_size_i = wlp(new_sub_i, f(new_sub_i))
                    new_size_j = wlp(new_sub_j, f(new_sub_j))

                    if curr_max := max(new_size_i, new_size_j) < size_max:
                        size_max = curr_max
                        v_star = v

            if v_star != -1:
                print(f"transferring {v_star} from {i} to {j}")
                g_i.remove(v_star)
                g_j.add(v_star)

                partition[i] = g_i
                partition[j] = g_j

                for k in range(m):
                    if k not in {i, j}:
                        # j increased so may need to transfer to other nodes
                        # j decreased so may need to accept other nodes
                        transfers.add((j, k))
                        transfers.add((k, i))

        while len(swaps) > 0:
            i, j = swaps.pop()
            g_i, g_j = partition[i], partition[j]

            assert 0 in g_i
            assert 0 in g_j

            sub_i, _, _ = Graph.subgraph(g, list(g_i))
            sub_j, _, _ = Graph.subgraph(g, list(g_i))

            size_i_init = wlp(g, f(sub_i))
            size_j_init = wlp(g, f(sub_j))

            # TODO: Determine a condition to not check swap or prove one cannot be found

            size_max = max(size_i_init, size_j_init)
            v_i_star, v_j_star = -1, -1

            for v in g_i:
                for v_prime in g_j:
                    if v != 0 and v_prime != 0:
                        # swap v and v_prime
                        new_i = {v_i for v_i in g_i if v_i != v}
                        new_i.add(v_prime)
                        new_j = {v_j for v_j in g_j if v_j != v_prime}
                        new_j.add(v)

                        assert 0 in new_i
                        assert 0 in new_j

                        new_sub_i, _, _ = Graph.subgraph(g, list(new_i))
                        new_sub_j, _, _ = Graph.subgraph(g, list(new_j))

                        new_size_i = wlp(new_sub_i, f(new_sub_i))
                        new_size_j = wlp(new_sub_j, f(new_sub_j))

                        if curr_max := max(new_size_i, new_size_j) < size_max:
                            size_max = curr_max
                            v_i_star, v_j_star = v, v_prime

            if v_i_star != -1 and v_j_star != -1:
                print(f"swapping {v_i_star} to {i} and {v_j_star} to {j}")

                g_i.remove(v_i_star)
                g_i.add(v_j_star)
                g_j.remove(v_j_star)
                g_j.add(v_i_star)

                partition[i] = g_i
                partition[j] = g_j

                for k in range(m):
                    if k not in {i, j}:
                        # j increased so may need to transfer to other nodes
                        # j decreased so may need to accept other nodes
                        swaps.add((j, k))
                        swaps.add((k, i))

    return partition
