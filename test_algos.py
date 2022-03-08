import math
from itertools import permutations

import pytest
from pytest import approx

import algos
from graph import Graph, graph_dict

# Bank of graphs
empty = Graph()
no_edges = Graph(6)
random_complete = Graph.random_complete(6)
random_complete_metric = Graph.random_complete_metric(6)

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


def test_cost() -> None:
    g = complete
    assert algos.cost(g, [0, 1, 2]) == 59.0

    rand: list[int] = algos.random_order(g)
    assert algos.cost(g, rand) == algos.wlp(g, rand)

    g = random_complete
    rand = algos.random_order(g)
    assert algos.cost(g, rand) == algos.wlp(g, rand)


def test_cost_small_orders() -> None:
    g = random_complete
    assert algos.cost(g, []) == 0.0
    assert algos.cost(g, [0]) == 0.0


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


def test_partition_heuristic_mwlp_2_agents() -> None:
    g = complete_two
    result, partition = algos.partition_heuristic(g, algos.brute_force_mwlp, 2)

    assert result == 52.0
    assert [0, 1, 2] in partition and [0, 3] in partition


def test_optimal_number_of_agents_mwlp() -> None:
    g = complete_two
    optimalmwlp, optimal_order = algos.optimal_number_of_agents(
        g, algos.brute_force_mwlp, 1, 3
    )

    assert optimalmwlp == 52.0
    assert len(optimal_order) == 2
    assert [0, 1, 2] in optimal_order and [0, 3] in optimal_order


def test_choose2() -> None:
    pairs: list[tuple[int, int]] = algos.choose2(3)
    assert (0, 1) in pairs
    assert (0, 2) in pairs
    assert (1, 2) in pairs

    pairs = algos.choose2(10)
    assert len(pairs) == 45

    pairs = algos.choose2(74)
    assert len(pairs) == 2701

    pairs = algos.choose2(18)
    assert len(pairs) == 153

    pairs = algos.choose2(16)
    assert len(pairs) == 120


def test_all_possible_wlp_orders_avg() -> None:
    n: int = 4
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


### Error Tests ###


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


def test_cost_invalid_nodes() -> None:
    g = random_complete
    with pytest.raises(ValueError):
        algos.cost(g, [0, g.num_nodes])


def test_cost_invalid_edges() -> None:
    g = almost_complete
    with pytest.raises(ValueError):
        algos.cost(g, [2, 1, 0])


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


def test_partition_incomplete() -> None:
    g = almost_complete
    with pytest.raises(ValueError):
        algos.partition_heuristic(g, algos.brute_force_tsp, 2)


def test_partition_too_few_agents() -> None:
    g = random_complete
    with pytest.raises(ValueError):
        algos.partition_heuristic(g, algos.brute_force_tsp, 0)


def test_partition_too_many_agents() -> None:
    g = random_complete
    with pytest.raises(ValueError):
        algos.partition_heuristic(g, algos.brute_force_tsp, g.num_nodes + 1)


def test_optimal_number_of_agents_incomplete() -> None:
    g = almost_complete
    with pytest.raises(ValueError):
        algos.optimal_number_of_agents(g, algos.brute_force_tsp, 1, 3)


def test_optimal_number_of_agents_invalid_order() -> None:
    g = random_complete
    with pytest.raises(ValueError):
        algos.optimal_number_of_agents(g, algos.brute_force_tsp, 3, 1)


def test_optimal_number_of_agents_too_few() -> None:
    g = random_complete
    with pytest.raises(ValueError):
        algos.optimal_number_of_agents(g, algos.brute_force_tsp, -1, 1)


def test_optimal_number_of_agents_too_many() -> None:
    g = random_complete
    with pytest.raises(ValueError):
        algos.optimal_number_of_agents(g, algos.brute_force_tsp, 1, g.num_nodes)


def test_choose2_small_n() -> None:
    with pytest.raises(ValueError):
        algos.choose2(1)


def test_all_possible_wlp_orders_avg_incomplete() -> None:
    g = almost_complete
    with pytest.raises(ValueError):
        algos.all_possible_wlp_orders_avg(g)


def test_all_possible_wlp_orders_avg_undirected() -> None:
    g = Graph.random_complete(10)
    with pytest.raises(ValueError):
        algos.all_possible_wlp_orders_avg(g)
