"""
Test cases to validate graph algorithms
"""
from typing import Callable

import pytest

import algos
from graph import Graph, graph_dict

# Bank of graphs
empty = Graph()
no_edges = Graph(20)

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
    for _ in range(5):
        g = Graph.random_complete_metric(30)
        dist: list[list[float]] = algos.floyd_warshall(g)

        for i in range(g.num_nodes):
            for j in range(g.num_nodes):
                if i == j:
                    assert dist[i][j] == 0.0
                else:
                    assert dist[i][j] != float("inf")


def test_fw_metric_complete() -> None:
    for _ in range(5):
        g = Graph.random_complete_metric(30)
        dist: list[list[float]] = algos.floyd_warshall(g)

        for i in range(g.num_nodes):
            for j in range(g.num_nodes):
                if i == j:
                    assert dist[i][j] == 0.0
                else:
                    assert dist[i][j] == g.edge_weight[i][j]


def test_create_metric_from_graph_complete() -> None:
    for _ in range(5):
        g = Graph.random_complete(30)
        metric: Graph = algos.create_metric_from_graph(g)

        assert Graph.is_complete(metric)
        assert Graph.is_metric(metric)


def test_create_metric_from_graph_metric_is_same() -> None:
    for _ in range(5):
        g = Graph.random_complete_metric(30)
        metric: Graph = algos.create_metric_from_graph(g)

        for i in range(g.num_nodes):
            for j in range(g.num_nodes):
                if i != j:
                    assert g.edge_weight[i][j] == metric.edge_weight[i][j]


def test_wlp_small_orders() -> None:
    for _ in range(5):
        g = Graph.random_complete(30)
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
    for _ in range(5):
        g = Graph.random_complete(5)
        order: list[int] = algos.brute_force_mwlp(g)

        for i in range(len(order)):
            assert algos.brute_force_mwlp(g, start=order[: i + 1]) == order


def test_nearest_neighbor() -> None:
    g = complete
    assert algos.nearest_neighbor(g) == [0, 1, 2, 3]
    assert algos.nearest_neighbor(g, start=[1]) == [1, 2, 3, 0]


def test_nearest_neighbor_start() -> None:
    for _ in range(5):
        g = Graph.random_complete(30)
        order: list[int] = algos.nearest_neighbor(g)

        for i in range(1, len(order) + 1):
            assert algos.nearest_neighbor(g, start=order[:i]) == order


def test_greedy() -> None:
    g = complete
    assert algos.greedy(g) == [0, 2, 3, 1]
    assert algos.greedy(g, start=[1]) == [1, 2, 0, 3]


def test_greedy_start() -> None:
    for _ in range(5):
        g = Graph.random_complete(30)
        order: list[int] = algos.greedy(g)

        for i in range(1, len(order) + 1):
            assert algos.greedy(g, start=order[:i]) == order


def test_alternate() -> None:
    g = complete
    assert algos.alternate(g) == [0, 2, 3, 1]
    assert algos.alternate(g, start=[1]) == [1, 2, 3, 0]


def test_alternate_start() -> None:
    for _ in range(5):
        g = Graph.random_complete(30)
        order: list[int] = algos.alternate(g)

        for i in range(1, len(order) + 1, 2):
            assert algos.alternate(g, start=order[:i]) == order


def test_random_order() -> None:
    for _ in range(5):
        g = Graph.random_complete(30)
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
    for _ in range(5):
        g = Graph.random_complete_metric(30)
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


def test_transfers_and_swaps_maintains_agent_partition() -> None:
    for _ in range(5):
        g = Graph.random_complete_metric(30)
        k: int = 4
        part: list[set[int]] = Graph.create_agent_partition(g, k)
        res: list[set[int]] = algos.transfers_and_swaps_mwlp(
            g, part, algos.nearest_neighbor
        )
        assert Graph.is_agent_partition(g, res)


def test_transfer_outliers_maintains_agent_partition() -> None:
    for _ in range(5):
        g = Graph.random_complete_metric(30)
        k: int = 4
        part: list[set[int]] = Graph.create_agent_partition(g, k)
        res: list[set[int]] = algos.transfer_outliers_mwlp(
            g, part, algos.nearest_neighbor, 0.5
        )
        assert Graph.is_agent_partition(g, res)


def test_find_partition_maintains_agent_partition() -> None:
    for _ in range(5):
        g = Graph.random_complete_metric(30)
        k: int = 4
        part: list[set[int]] = Graph.create_agent_partition(g, k)
        res: list[set[int]] = algos.find_partition_with_heuristic(
            g, part, algos.nearest_neighbor, 0.5
        )
        assert Graph.is_agent_partition(g, res)


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
    for _ in range(5):
        g = Graph.random_complete(30)
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
    for _ in range(5):
        g = Graph.random_complete(30)
        with pytest.raises(ValueError):
            algos.brute_force_mwlp(g, start=[g.num_nodes])


def test_nearest_neighbor_incomplete() -> None:
    g = almost_complete
    with pytest.raises(ValueError):
        algos.nearest_neighbor(g)


def test_nearest_neighbor_invalid_start() -> None:
    for _ in range(5):
        g = Graph.random_complete(30)
        with pytest.raises(ValueError):
            algos.nearest_neighbor(g, start=[g.num_nodes])


def test_greedy_incomplete() -> None:
    g = almost_complete
    with pytest.raises(ValueError):
        algos.greedy(g)


def test_greedy_invalid_start() -> None:
    for _ in range(5):
        g = Graph.random_complete(30)
        with pytest.raises(ValueError):
            algos.greedy(g, start=[g.num_nodes])


def test_alternate_incomplete() -> None:
    g = almost_complete
    with pytest.raises(ValueError):
        algos.alternate(g)


def test_alternate_invalid_start() -> None:
    for _ in range(5):
        g = Graph.random_complete(30)
        with pytest.raises(ValueError):
            algos.alternate(g, start=[g.num_nodes])


def test_random_order_incomplete() -> None:
    g = almost_complete
    with pytest.raises(ValueError):
        algos.random_order(g)


def test_random_order_invalid_start() -> None:
    for _ in range(5):
        g = Graph.random_complete(30)
        with pytest.raises(ValueError):
            algos.random_order(g, start=[g.num_nodes])


def test_tsp_incomplete() -> None:
    g = almost_complete
    with pytest.raises(ValueError):
        algos.brute_force_tsp(g)


def test_tsp_invalid_start() -> None:
    for _ in range(5):
        g = Graph.random_complete(30)
        with pytest.raises(ValueError):
            algos.brute_force_tsp(g, start=g.num_nodes)


def test_hk_incomplete() -> None:
    g = almost_complete
    with pytest.raises(ValueError):
        algos.held_karp(g)


def test_hk_invalid_start() -> None:
    for _ in range(5):
        g = Graph.random_complete(30)
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
    for _ in range(5):
        g = Graph.random_complete(30)
        part: list[set[int]] = Graph.create_agent_partition(g, 2)
        with pytest.raises(ValueError):
            algos.find_partition_with_heuristic(g, part, algos.greedy, -0.5)
