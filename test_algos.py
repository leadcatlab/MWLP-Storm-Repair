"""
Test cases to validate graph algorithms
"""
import math
from itertools import permutations
from typing import Callable

import pytest
from pytest import approx

import algos
from graph import Graph, graph_dict

# Bank of graphs
empty = Graph()
no_edges = Graph(20)
random_complete = Graph.random_complete(20)
random_complete_metric = Graph.random_complete_metric(20)

small_graph_dict: graph_dict = {
    "num_nodes": 4,
    "edges": [(0, 1, 1.0), (1, 2, 3.0), (2, 3, 5.0), (0, 2, 2.0)],
    "node_weight": [10, 2, 6, 20],
}
small_graph = Graph.from_dict(small_graph_dict)

complete_graph_dict: graph_dict = {
    "num_nodes": 4,
    "edges": [
        (0, 1, 1.0),
        (0, 2, 3.0),
        (0, 3, 5.0),
        (1, 0, 6.0),
        (1, 2, 1.0),
        (1, 3, 50.0),
        (2, 0, 2.0),
        (2, 1, 7.0),
        (2, 3, 1.0),
        (3, 0, 8.0),
        (3, 1, 100.0),
        (3, 2, 2.0),
    ],
    "node_weight": [10, 5, 20, 7],
}
complete = Graph.from_dict(complete_graph_dict)

complete_two_graph_dict: graph_dict = {
    "num_nodes": 4,
    "edges": [
        (0, 1, 1.0),
        (0, 2, 3.0),
        (0, 3, 1.0),
        (1, 0, 6.0),
        (1, 2, 1.0),
        (1, 3, 50.0),
        (2, 0, 2.0),
        (2, 1, 7.0),
        (2, 3, 1.0),
        (3, 0, 8.0),
        (3, 1, 100.0),
        (3, 2, 2.0),
    ],
    "node_weight": [10, 5, 20, 7],
}
complete_two = Graph.from_dict(complete_two_graph_dict)

# Missing 1 -> 0
almost_complete_graph_dict: graph_dict = {
    "num_nodes": 4,
    "edges": [
        (0, 1, 1.0),
        (0, 2, 3.0),
        (0, 3, 5.0),
        (1, 2, 1.0),
        (1, 3, 50.0),
        (2, 0, 2.0),
        (2, 1, 7.0),
        (2, 3, 1.0),
        (3, 0, 8.0),
        (3, 1, 100.0),
        (3, 2, 2.0),
    ],
    "node_weight": [10, 5, 20, 7],
}
almost_complete = Graph.from_dict(almost_complete_graph_dict)

undirected_graph_dict: graph_dict = {
    "num_nodes": 4,
    "edges": [
        (0, 1, 1.0),
        (0, 2, 4.0),
        (0, 3, 3.0),
        (1, 0, 1.0),
        (1, 2, 6.0),
        (1, 3, 5.0),
        (2, 0, 4.0),
        (2, 1, 6.0),
        (2, 3, 7.0),
        (3, 0, 3.0),
        (3, 1, 5.0),
        (3, 2, 7.0),
    ],
    "node_weight": [10, 5, 20, 7],
}
undirected = Graph.from_dict(undirected_graph_dict)

### Correctness Tests ###


def test_num_visited_along_path() -> None:
    g = complete
    assert not algos.num_visited_along_path(g, [])
    assert algos.num_visited_along_path(g, [0]) == [10]
    assert algos.num_visited_along_path(g, [0, 1]) == [10, 15]
    assert algos.num_visited_along_path(g, [0, 1, 2]) == [10, 15, 35]
    assert algos.num_visited_along_path(g, [0, 1, 2, 3]) == [10, 15, 35, 42]


def test_length_along_path() -> None:
    g = complete
    assert algos.length_along_path(g, []) == [0.0]
    assert algos.length_along_path(g, [0]) == [0.0]
    assert algos.length_along_path(g, [0, 1]) == [0.0, 1.0]
    assert algos.length_along_path(g, [0, 1, 2]) == [0.0, 1.0, 2.0]
    assert algos.length_along_path(g, [0, 1, 2, 3]) == [0.0, 1.0, 2.0, 3.0]


def test_path_function() -> None:
    g = complete
    path_function: Callable[[float], int] = algos.generate_path_function(
        g, [0, 1, 2, 3]
    )
    assert path_function(0.5) == 10
    assert path_function(1.5) == 15
    assert path_function(2.5) == 35
    assert path_function(3.5) == 42


def test_partition_path_function() -> None:
    g = complete
    partition_function: Callable[[float], int] = algos.generate_partition_path_function(
        g, [[0, 1], [0, 2, 3]]
    )
    assert partition_function(1.0) == 15
    assert partition_function(3.5) == 35
    assert partition_function(5.0) == 42


def test_path_length() -> None:
    g = small_graph
    assert algos.path_length(g, []) == 0.0
    assert algos.path_length(g, [0]) == 0.0
    assert algos.path_length(g, [0, 1]) == 1.0
    assert algos.path_length(g, [0, 1, 2, 3]) == 9.0


def test_fw_empty() -> None:
    dist: list[list[float]] = algos.floyd_warshall(empty)
    assert len(dist) == 0


def test_fw_no_edges() -> None:
    g = no_edges
    dist: list[list[float]] = algos.floyd_warshall(g)

    assert len(dist) == len(dist[0]) == g.num_nodes
    for i in range(g.num_nodes):
        assert dist[i][i] == 0.0
        for j in range(g.num_nodes):
            if i != j:
                assert dist[i][j] == float("inf")


def test_fw_complete() -> None:
    g = random_complete_metric
    dist: list[list[float]] = algos.floyd_warshall(g)

    for i in range(g.num_nodes):
        for j in range(g.num_nodes):
            if i == j:
                assert dist[i][j] == 0.0
            else:
                assert dist[i][j] != float("inf")


def test_fw_metric_complete() -> None:
    g = random_complete_metric
    dist: list[list[float]] = algos.floyd_warshall(g)

    for i in range(g.num_nodes):
        for j in range(g.num_nodes):
            if i == j:
                assert dist[i][j] == 0.0
            else:
                assert dist[i][j] == g.edge_weight[i][j]


def test_create_metric_from_graph_complete() -> None:
    g = random_complete
    metric: Graph = algos.create_metric_from_graph(g)

    assert Graph.is_complete(metric)
    assert Graph.is_metric(metric)


def test_create_metric_from_graph_metric_is_same() -> None:
    g = random_complete_metric
    metric: Graph = algos.create_metric_from_graph(g)

    for i in range(g.num_nodes):
        for j in range(g.num_nodes):
            if i != j:
                assert g.edge_weight[i][j] == metric.edge_weight[i][j]


def test_wlp_small_orders() -> None:
    g = random_complete
    assert algos.wlp(g, []) == 0.0
    assert algos.wlp(g, [0]) == 0.0
    assert algos.wlp(g, [1]) == 0.0


def test_wlp() -> None:
    g = small_graph
    assert algos.wlp(g, [0, 1, 2, 3]) == 206.0
    assert algos.wlp(g, [0, 1]) == 2.0
    assert algos.wlp(g, [0, 2]) == 12.0
    assert algos.wlp(g, [0, 2, 3]) == 152.0
    assert algos.wlp(g, [1, 2, 3]) == 178.0


def test_brute_force_mwlp() -> None:
    g = complete
    assert algos.brute_force_mwlp(g) == [0, 1, 2, 3]
    assert algos.brute_force_mwlp(g, start=[1]) == [1, 2, 0, 3]


def test_brute_force_mwlp_seq_start() -> None:
    g = Graph.random_complete(5)
    order: list[int] = algos.brute_force_mwlp(g)

    for i in range(len(order)):
        assert algos.brute_force_mwlp(g, start=order[: i + 1]) == order


def test_nearest_neighbor() -> None:
    g = complete
    assert algos.nearest_neighbor(g) == [0, 1, 2, 3]
    assert algos.nearest_neighbor(g, start=[1]) == [1, 2, 3, 0]


def test_nearest_neighbor_start() -> None:
    g = random_complete
    order: list[int] = algos.nearest_neighbor(g)

    for i in range(1, len(order) + 1):
        assert algos.nearest_neighbor(g, start=order[:i]) == order


def test_greedy() -> None:
    g = complete
    assert algos.greedy(g) == [0, 2, 3, 1]
    assert algos.greedy(g, start=[1]) == [1, 2, 0, 3]


def test_greedy_start() -> None:
    g = random_complete
    order: list[int] = algos.greedy(g)

    for i in range(1, len(order) + 1):
        assert algos.greedy(g, start=order[:i]) == order


def test_random_order() -> None:
    g = random_complete
    rand: list[int] = algos.random_order(g)
    assert rand[0] == 0
    for i in range(g.num_nodes):
        assert i in rand

    rand_start_at_one: list[int] = algos.random_order(g, start=[1])
    assert rand_start_at_one[0] == 1
    for i in range(g.num_nodes):
        assert i in rand_start_at_one


def test_brute_force_tsp() -> None:
    g = complete
    assert algos.brute_force_tsp(g) == [0, 1, 2, 3]
    assert algos.brute_force_tsp(g, start=1) == [1, 2, 0, 3]


def test_held_karp() -> None:
    g = complete
    assert algos.held_karp(g, start=0) == [0, 1, 2, 3]


def test_tsp_correctness() -> None:
    for _ in range(10):
        g = Graph.random_complete(7)
        assert algos.held_karp(g) == algos.brute_force_tsp(g)


def test_uconn_strats_agent_partition() -> None:
    g = random_complete_metric
    k = 5
    strat_1: list[list[int]] = algos.uconn_strat_1(g, k)
    assert Graph.is_agent_partition(g, [set(subset) for subset in strat_1])

    strat_2_no_rad: list[list[int]] = algos.uconn_strat_2(g, k, 0.0)
    assert Graph.is_agent_partition(g, [set(subset) for subset in strat_2_no_rad])

    strat_2: list[list[int]] = algos.uconn_strat_2(g, k, 5.0)
    assert Graph.is_agent_partition(g, [set(subset) for subset in strat_2])


def test_evaluate_partition_heuristic() -> None:
    g = Graph.random_complete(8)

    res: float = algos.wlp(g, algos.brute_force_mwlp(g))
    assert (
        algos.evaluate_partition_heuristic(g, [set(range(8))], algos.brute_force_mwlp)
        == res
    )

    res = algos.wlp(g, algos.greedy(g))
    assert algos.evaluate_partition_heuristic(g, [set(range(8))], algos.greedy) == res

    res = algos.wlp(g, algos.nearest_neighbor(g))
    assert (
        algos.evaluate_partition_heuristic(g, [set(range(8))], algos.nearest_neighbor)
        == res
    )


def test_all_possible_wlp_orders_avg() -> None:
    n: int = 0
    g = Graph.random_complete(n, directed=False)
    assert algos.all_possible_wlp_orders_avg(g) == 0.0

    n = 4
    g = Graph.random_complete(n, directed=False)
    avg: float = algos.all_possible_wlp_orders_avg(g)
    brute: float = 0.0
    nodes: list[int] = list(range(1, n))
    for order in permutations(nodes):
        brute += algos.wlp(g, [0] + list(order))
    assert avg == approx(brute / math.factorial(n - 1))

    n = 5
    g = Graph.random_complete(n, directed=False)
    avg = algos.all_possible_wlp_orders_avg(g)
    brute = 0.0
    nodes = list(range(1, n))
    for order in permutations(nodes):
        brute += algos.wlp(g, [0] + list(order))
    assert avg == approx(brute / math.factorial(n - 1))

    n = 6
    g = Graph.random_complete(n, directed=False)
    avg = algos.all_possible_wlp_orders_avg(g)
    brute = 0.0
    nodes = list(range(1, n))
    for order in permutations(nodes):
        brute += algos.wlp(g, [0] + list(order))
    assert avg == approx(brute / math.factorial(n - 1))

    n = 7
    g = Graph.random_complete(n, directed=False)
    avg = algos.all_possible_wlp_orders_avg(g)
    brute = 0.0
    nodes = list(range(1, n))
    for order in permutations(nodes):
        brute += algos.wlp(g, [0] + list(order))
    assert avg == approx(brute / math.factorial(n - 1))


def test_evaluate_partition_with_average() -> None:
    n = 4
    g = Graph.random_complete(n, directed=False)
    brute: float = 0.0
    nodes: list[int] = list(range(1, n))
    for order in permutations(nodes):
        brute += algos.wlp(g, [0] + list(order))
    assert algos.evaluate_partition_with_average(g, [{0, 1, 2, 3}]) == approx(
        brute / math.factorial(n - 1)
    )


### Error Tests ###


def test_num_visited_along_path_missing_edges() -> None:
    g = no_edges
    with pytest.raises(ValueError):
        algos.num_visited_along_path(g, [0, 1])


def test_length_along_path_missing_edges() -> None:
    g = no_edges
    with pytest.raises(ValueError):
        algos.length_along_path(g, [0, 1])


def test_path_function_empty() -> None:
    g = complete
    with pytest.raises(ValueError):
        algos.generate_path_function(g, [])


def test_path_function_negative_dist() -> None:
    g = complete
    path_function: Callable[[float], int] = algos.generate_path_function(
        g, [0, 1, 2, 3]
    )
    with pytest.raises(ValueError):
        path_function(-0.5)


def test_partition_function_empty() -> None:
    g = complete
    with pytest.raises(ValueError):
        algos.generate_partition_path_function(g, [[0]])


def test_partition_path_function_negative_dist() -> None:
    g = complete
    partition_function: Callable[[float], int] = algos.generate_partition_path_function(
        g, [[0, 1, 2, 3]]
    )
    with pytest.raises(ValueError):
        partition_function(-0.5)


def test_path_length_missing_edges() -> None:
    g = no_edges
    with pytest.raises(ValueError):
        algos.path_length(g, [0, 1])


def test_wlp_node_not_in_graph() -> None:
    g = random_complete
    n: int = g.num_nodes
    with pytest.raises(ValueError):
        algos.wlp(g, [0, 1, 2, n])


def test_wlp_missing_edge() -> None:
    g = small_graph
    with pytest.raises(ValueError):
        algos.wlp(g, [0, 1, 2, 3, 2])


def test_brute_force_mwlp_incomplete() -> None:
    g = almost_complete
    with pytest.raises(ValueError):
        algos.brute_force_mwlp(g)


def test_brute_force_mwlp_invalid_start() -> None:
    g = random_complete
    with pytest.raises(ValueError):
        algos.brute_force_mwlp(g, start=[g.num_nodes])


def test_nearest_neighbor_incomplete() -> None:
    g = almost_complete
    with pytest.raises(ValueError):
        algos.nearest_neighbor(g)


def test_nearest_neighbor_invalid_start() -> None:
    g = random_complete
    with pytest.raises(ValueError):
        algos.nearest_neighbor(g, start=[g.num_nodes])


def test_greedy_incomplete() -> None:
    g = almost_complete
    with pytest.raises(ValueError):
        algos.greedy(g)


def test_greedy_invalid_start() -> None:
    g = random_complete
    with pytest.raises(ValueError):
        algos.greedy(g, start=[g.num_nodes])


def test_random_order_incomplete() -> None:
    g = almost_complete
    with pytest.raises(ValueError):
        algos.random_order(g)


def test_random_order_invalid_start() -> None:
    g = random_complete
    with pytest.raises(ValueError):
        algos.random_order(g, start=[g.num_nodes])


def test_tsp_incomplete() -> None:
    g = almost_complete
    with pytest.raises(ValueError):
        algos.brute_force_tsp(g)


def test_tsp_invalid_start() -> None:
    g = random_complete
    with pytest.raises(ValueError):
        algos.brute_force_tsp(g, start=g.num_nodes)


def test_hk_incomplete() -> None:
    g = almost_complete
    with pytest.raises(ValueError):
        algos.held_karp(g)


def test_hk_invalid_start() -> None:
    g = random_complete
    with pytest.raises(ValueError):
        algos.held_karp(g, start=g.num_nodes)


def test_uconn_strat_1_incomplete() -> None:
    g: Graph = almost_complete
    with pytest.raises(ValueError):
        algos.uconn_strat_1(g, 2)


def test_uconn_strat_2_incomplete() -> None:
    g: Graph = almost_complete
    with pytest.raises(ValueError):
        algos.uconn_strat_2(g, 2, 1.0)


def test_uconn_strat_1_directed() -> None:
    g: Graph = complete
    with pytest.raises(ValueError):
        algos.uconn_strat_1(g, 2)


def test_uconn_strat_2_directed() -> None:
    g: Graph = complete
    with pytest.raises(ValueError):
        algos.uconn_strat_2(g, 2, 1.0)


def test_all_possible_wlp_orders_avg_incomplete() -> None:
    g = almost_complete
    with pytest.raises(ValueError):
        algos.all_possible_wlp_orders_avg(g)


def test_all_possible_wlp_orders_avg_undirected() -> None:
    g = Graph.random_complete(10, directed=True)
    with pytest.raises(ValueError):
        algos.all_possible_wlp_orders_avg(g)


def test_evaluate_partition_heuristic_incomplete_graph() -> None:
    with pytest.raises(ValueError):
        algos.evaluate_partition_heuristic(
            almost_complete, [{0, 1, 2, 3}], algos.greedy
        )


def test_evaluate_partition_heuristic_directed_graph() -> None:
    with pytest.raises(ValueError):
        algos.evaluate_partition_heuristic(complete, [{0, 1, 2, 3}], algos.greedy)


def test_evaluate_partition_heuristic_invalid_partition() -> None:
    g: Graph = undirected
    with pytest.raises(ValueError):
        algos.evaluate_partition_heuristic(g, [], algos.greedy)


def test_evaluate_partition_average_incomplete_graph() -> None:
    g: Graph = almost_complete
    with pytest.raises(ValueError):
        algos.evaluate_partition_with_average(g, [{0, 1, 2, 3}])


def test_evaluate_partition_average_directed_graph() -> None:
    g: Graph = complete
    with pytest.raises(ValueError):
        algos.evaluate_partition_with_average(g, [{0, 1, 2, 3}])


def test_evaluate_partition_average_invalid_partition() -> None:
    g: Graph = undirected
    with pytest.raises(ValueError):
        algos.evaluate_partition_with_average(g, [])


def test_transfers_and_swaps_mwlp_incomplete() -> None:
    g: Graph = almost_complete
    part: list[set[int]] = Graph.create_agent_partition(g, 2)
    with pytest.raises(ValueError):
        algos.transfers_and_swaps_mwlp(g, part, algos.greedy)


def test_transfers_and_swaps_mwlp_undirected() -> None:
    g: Graph = complete
    part: list[set[int]] = Graph.create_agent_partition(g, 2)
    with pytest.raises(ValueError):
        algos.transfers_and_swaps_mwlp(g, part, algos.greedy)


def test_transfers_and_swaps_mwlp_invalid_partition() -> None:
    g: Graph = undirected
    with pytest.raises(ValueError):
        algos.transfers_and_swaps_mwlp(g, [], algos.greedy)


def test_transfer_outliers_mwlp_incomplete() -> None:
    g: Graph = almost_complete
    part: list[set[int]] = Graph.create_agent_partition(g, 2)
    with pytest.raises(ValueError):
        algos.transfer_outliers_mwlp(g, part, algos.greedy, 0.5)


def test_transfer_outliers_undirected() -> None:
    g: Graph = complete
    part: list[set[int]] = Graph.create_agent_partition(g, 2)
    with pytest.raises(ValueError):
        algos.transfer_outliers_mwlp(g, part, algos.greedy, 0.5)


def test_transfer_outliers_mwlp_invalid_partition() -> None:
    g: Graph = undirected
    with pytest.raises(ValueError):
        algos.transfer_outliers_mwlp(g, [], algos.greedy, 0.5)


def test_transfer_outliers_mwlp_invalid_threshold() -> None:
    g: Graph = undirected
    part: list[set[int]] = Graph.create_agent_partition(g, 2)
    with pytest.raises(ValueError):
        algos.transfer_outliers_mwlp(g, part, algos.greedy, 1.1)


def test_find_partition_heuristic_incomplete() -> None:
    g: Graph = almost_complete
    part: list[set[int]] = Graph.create_agent_partition(g, 2)
    with pytest.raises(ValueError):
        algos.find_partition_with_heuristic(g, part, algos.greedy, 0.5)


def test_find_partition_heuristic_undirected() -> None:
    g: Graph = complete
    part: list[set[int]] = Graph.create_agent_partition(g, 2)
    with pytest.raises(ValueError):
        algos.find_partition_with_heuristic(g, part, algos.greedy, 0.5)


def test_find_partition_heuristic_invalid_partition() -> None:
    g: Graph = undirected
    with pytest.raises(ValueError):
        algos.find_partition_with_heuristic(g, [], algos.greedy, 0.5)


def test_find_partition_heuristic_invalid_alpha() -> None:
    g: Graph = random_complete
    part: list[set[int]] = Graph.create_agent_partition(g, 2)
    with pytest.raises(ValueError):
        algos.find_partition_with_heuristic(g, part, algos.greedy, -0.5)


def test_transfers_and_swaps_mwlp_with_average_incomplete() -> None:
    g: Graph = almost_complete
    part: list[set[int]] = Graph.create_agent_partition(g, 2)
    with pytest.raises(ValueError):
        algos.transfers_and_swaps_mwlp_with_average(g, part)


def test_transfers_and_swaps_mwlp_with_average_undirected() -> None:
    g: Graph = complete
    part: list[set[int]] = Graph.create_agent_partition(g, 2)
    with pytest.raises(ValueError):
        algos.transfers_and_swaps_mwlp_with_average(g, part)


def test_transfers_and_swaps_mwlp_with_average_invalid_partition() -> None:
    g: Graph = undirected
    with pytest.raises(ValueError):
        algos.transfers_and_swaps_mwlp_with_average(g, [])


def test_transfer_outliers_with_average_mwlp_incomplete() -> None:
    g: Graph = almost_complete
    part: list[set[int]] = Graph.create_agent_partition(g, 2)
    with pytest.raises(ValueError):
        algos.transfer_outliers_mwlp_with_average(g, part, 0.5)


def test_transfer_outliers_with_average_undirected() -> None:
    g: Graph = complete
    part: list[set[int]] = Graph.create_agent_partition(g, 2)
    with pytest.raises(ValueError):
        algos.transfer_outliers_mwlp_with_average(g, part, 0.5)


def test_transfer_outliers_mwlp_with_average_invalid_partition() -> None:
    g: Graph = undirected
    with pytest.raises(ValueError):
        algos.transfer_outliers_mwlp_with_average(g, [], 0.5)


def test_transfer_outliers_with_average_invalid_threshold() -> None:
    g: Graph = undirected
    part: list[set[int]] = Graph.create_agent_partition(g, 2)
    with pytest.raises(ValueError):
        algos.transfer_outliers_mwlp_with_average(g, part, 1.1)


def test_find_partition_with_average_incomplete() -> None:
    g: Graph = almost_complete
    part: list[set[int]] = Graph.create_agent_partition(g, 2)
    with pytest.raises(ValueError):
        algos.find_partition_with_average(g, part, 0.5)


def test_find_partition_with_average_undirected() -> None:
    g: Graph = complete
    part: list[set[int]] = Graph.create_agent_partition(g, 2)
    with pytest.raises(ValueError):
        algos.find_partition_with_average(g, part, 0.5)


def test_find_partition_with_average_invalid_partition() -> None:
    g: Graph = undirected
    with pytest.raises(ValueError):
        algos.find_partition_with_average(g, [], 0.5)


def test_find_partition_with_average_invalid_alpha() -> None:
    g: Graph = random_complete
    part: list[set[int]] = Graph.create_agent_partition(g, 2)
    with pytest.raises(ValueError):
        algos.find_partition_with_average(g, part, -0.5)
