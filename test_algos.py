from typing_extensions import TypedDict
import pytest
import algos
from graph import Graph

# TODO: Needs major refactor. So much duplicate code

graph_dict = TypedDict(
    "graph_dict",
    {
        "num_nodes": int,
        "node_weight": list[int],
        "edges": list[tuple[int, int, float]],
    },
)

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


### Correctness Tests ###


def test_FW_empty() -> None:
    dist: list[list[float]] = algos.floyd_warshall(empty)

    assert len(dist) == 0


def test_FW_no_edges() -> None:
    g = no_edges
    dist: list[list[float]] = algos.floyd_warshall(g)

    assert len(dist) == len(dist[0]) == g.num_nodes
    for i in range(g.num_nodes):
        assert dist[i][i] == 0.0
        for j in range(g.num_nodes):
            if i != j:
                assert dist[i][j] == float("inf")


def test_FW_complete() -> None:
    g = random_complete_metric
    dist: list[list[float]] = algos.floyd_warshall(g)

    for i in range(g.num_nodes):
        for j in range(g.num_nodes):
            if i == j:
                assert dist[i][j] == 0.0
            else:
                assert dist[i][j] != float("inf")


def test_FW_metric_complete() -> None:
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


def test_WLP_small_orders() -> None:
    g = random_complete
    assert algos.WLP(g, []) == 0.0
    assert algos.WLP(g, [0]) == 0.0
    assert algos.WLP(g, [1]) == 0.0


def test_WLP() -> None:
    g = small_graph
    assert algos.WLP(g, [0, 1, 2, 3]) == 206.0
    assert algos.WLP(g, [0, 1]) == 2.0
    assert algos.WLP(g, [0, 2]) == 12.0
    assert algos.WLP(g, [0, 2, 3]) == 152.0
    assert algos.WLP(g, [1, 2, 3]) == 178.0


def test_brute_force_MWLP() -> None:
    g = complete
    assert algos.brute_force_MWLP(g) == [0, 1, 2, 3]
    assert algos.brute_force_MWLP(g, start=1) == [1, 2, 0, 3]


def test_brute_force_MWLP_seq_start() -> None:
    g = Graph.random_complete(5)
    order: list[int] = algos.brute_force_MWLP(g)

    for i in range(len(order)):
        assert algos.brute_force_MWLP(g, start=order[i], seq_start=order[:i]) == order


def test_cost() -> None:
    g = complete
    assert algos.cost(g, [0, 1, 2]) == 59.0

    rand: list[int] = algos.random_order(g)
    assert algos.cost(g, rand) == algos.WLP(g, rand)

    g = random_complete
    rand = algos.random_order(g)
    assert algos.cost(g, rand) == algos.WLP(g, rand)


def test_cost_small_orders() -> None:
    g = random_complete
    assert algos.cost(g, []) == 0.0
    assert algos.cost(g, [0]) == 0.0


def test_nearest_neighbor() -> None:
    g = complete
    assert algos.nearest_neighbor(g) == [0, 1, 2, 3]
    assert algos.nearest_neighbor(g, start=1) == [1, 2, 3, 0]


def test_nearest_neighbor_seq_start() -> None:
    g = random_complete
    order: list[int] = algos.nearest_neighbor(g)

    for i in range(len(order)):
        assert algos.nearest_neighbor(g, start=order[i], seq_start=order[:i]) == order


def test_greedy() -> None:
    g = complete
    assert algos.greedy(g) == [0, 2, 3, 1]
    assert algos.greedy(g, start=1) == [1, 2, 0, 3]


def test_greedy_seq_start() -> None:
    g = random_complete
    order: list[int] = algos.greedy(g)

    for i in range(len(order)):
        assert algos.greedy(g, start=order[i], seq_start=order[:i]) == order


def test_random_order() -> None:
    g = random_complete
    rand: list[int] = algos.random_order(g)
    assert rand[0] == 0
    for i in range(g.num_nodes):
        assert i in rand

    rand_start_at_one: list[int] = algos.random_order(g, start=1)
    assert rand_start_at_one[0] == 1
    for i in range(g.num_nodes):
        assert i in rand_start_at_one


def test_random_order_seq_start() -> None:
    g = random_complete
    rand: list[int] = algos.random_order(g, start=2, seq_start=[0])
    assert rand[0:2] == [0, 2]
    for i in range(g.num_nodes):
        assert i in rand


def test_TSP() -> None:
    g = complete
    assert algos.TSP(g) == [0, 1, 2, 3]
    assert algos.TSP(g, start=1) == [1, 2, 0, 3]


def test_held_karp() -> None:
    g = complete
    assert algos.held_karp(g, start=0) == [0, 1, 2, 3]


def test_TSP_correctness() -> None:
    for _ in range(10):
        g = Graph.random_complete(7)
        assert algos.held_karp(g) == algos.TSP(g)


def test_partition_heuristic_MWLP_2_agents() -> None:
    g = complete_two
    result, partition = algos.partition_heuristic(g, algos.brute_force_MWLP, 2)

    assert result == 52.0
    assert [0, 1, 2] in partition and [0, 3] in partition


def test_optimal_number_of_agents_MWLP() -> None:
    g = complete_two
    optimalMWLP, optimal_order = algos.optimal_number_of_agents(
        g, algos.brute_force_MWLP, 1, 3
    )

    assert optimalMWLP == 52.0
    assert len(optimal_order) == 2
    assert [0, 1, 2] in optimal_order and [0, 3] in optimal_order


def test_total_edge_weight() -> None:
    g = complete
    all_nodes = set(range(g.num_nodes))
    assert algos.total_edge_weight(g, all_nodes) == 312.0
    assert algos.total_edge_weight(g, {0, 1}) == 22.0


def test_marginal_edge_weight() -> None:
    g = complete
    all_nodes = set(range(g.num_nodes))
    assert algos.marginal_edge_weight(g, all_nodes, 0) == 40.0
    assert algos.marginal_edge_weight(g, all_nodes, 1) == 83.0


### Error Tests ###


def test_WLP_node_not_in_graph() -> None:
    g = random_complete
    n: int = g.num_nodes
    with pytest.raises(ValueError):
        algos.WLP(g, [0, 1, 2, n])


def test_WLP_missing_edge() -> None:
    g = small_graph
    with pytest.raises(ValueError):
        algos.WLP(g, [0, 1, 2, 3, 2])


def test_brute_force_MWLP_incomplete() -> None:
    g = almost_complete
    with pytest.raises(ValueError):
        algos.brute_force_MWLP(g)


def test_brute_force_MWLP_invalid_start() -> None:
    g = random_complete
    with pytest.raises(ValueError):
        algos.brute_force_MWLP(g, start=g.num_nodes)


def test_brute_force_MWLP_start_already_visited() -> None:
    g = random_complete
    with pytest.raises(ValueError):
        algos.brute_force_MWLP(g, start=0, seq_start=[0, 1])


def test_brute_force_MWLP_invalid_seq_start() -> None:
    g = random_complete
    with pytest.raises(ValueError):
        algos.brute_force_MWLP(g, start=0, seq_start=[g.num_nodes])


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
        algos.nearest_neighbor(g, start=g.num_nodes)


def test_nearest_neighbor_start_already_visited() -> None:
    g = random_complete
    with pytest.raises(ValueError):
        algos.nearest_neighbor(g, start=0, seq_start=[0, 1])


def test_nearest_neighbor_invalid_seq_start() -> None:
    g = random_complete
    with pytest.raises(ValueError):
        algos.nearest_neighbor(g, start=0, seq_start=[g.num_nodes])


def test_greedy_incomplete() -> None:
    g = almost_complete
    with pytest.raises(ValueError):
        algos.greedy(g)


def test_greedy_invalid_start() -> None:
    g = random_complete
    with pytest.raises(ValueError):
        algos.greedy(g, start=g.num_nodes)


def test_greedy_start_already_visited() -> None:
    g = random_complete
    with pytest.raises(ValueError):
        algos.greedy(g, start=0, seq_start=[0, 1])


def test_greedy_invalid_seq_start() -> None:
    g = random_complete
    with pytest.raises(ValueError):
        algos.greedy(g, start=0, seq_start=[g.num_nodes])


def test_random_order_incomplete() -> None:
    g = almost_complete
    with pytest.raises(ValueError):
        algos.random_order(g)


def test_random_order_invalid_start() -> None:
    g = random_complete
    with pytest.raises(ValueError):
        algos.random_order(g, start=g.num_nodes)


def test_random_order_start_already_visited() -> None:
    g = random_complete
    with pytest.raises(ValueError):
        algos.random_order(g, start=0, seq_start=[0, 1])


def test_random_order_invalid_seq_start() -> None:
    g = random_complete
    with pytest.raises(ValueError):
        algos.random_order(g, start=0, seq_start=[g.num_nodes])


def test_TSP_incomplete() -> None:
    g = almost_complete
    with pytest.raises(ValueError):
        algos.TSP(g)


def test_TSP_invalid_start() -> None:
    g = random_complete
    with pytest.raises(ValueError):
        algos.TSP(g, start=g.num_nodes)


def test_HK_incomplete() -> None:
    g = almost_complete
    with pytest.raises(ValueError):
        algos.held_karp(g)


def test_HK_invalid_start() -> None:
    g = random_complete
    with pytest.raises(ValueError):
        algos.held_karp(g, start=g.num_nodes)


def test_partition_incomplete() -> None:
    g = almost_complete
    with pytest.raises(ValueError):
        algos.partition_heuristic(g, algos.TSP, 2)


def test_partition_too_few_agents() -> None:
    g = random_complete
    with pytest.raises(ValueError):
        algos.partition_heuristic(g, algos.TSP, 0)


def test_partition_too_many_agents() -> None:
    g = random_complete
    with pytest.raises(ValueError):
        algos.partition_heuristic(g, algos.TSP, g.num_nodes + 1)


def test_optimal_number_of_agents_incomplete() -> None:
    g = almost_complete
    with pytest.raises(ValueError):
        algos.optimal_number_of_agents(g, algos.TSP, 1, 3)


def test_optimal_number_of_agents_invalid_order() -> None:
    g = random_complete
    with pytest.raises(ValueError):
        algos.optimal_number_of_agents(g, algos.TSP, 3, 1)


def test_optimal_number_of_agents_too_few() -> None:
    g = random_complete
    with pytest.raises(ValueError):
        algos.optimal_number_of_agents(g, algos.TSP, -1, 1)


def test_optimal_number_of_agents_too_many() -> None:
    g = random_complete
    with pytest.raises(ValueError):
        algos.optimal_number_of_agents(g, algos.TSP, 1, g.num_nodes)


def test_total_edge_weight_incomplete() -> None:
    g = almost_complete
    with pytest.raises(ValueError):
        nodes = set(range(1, g.num_nodes))
        algos.total_edge_weight(g, nodes)


def test_total_edge_weight_invalid_nodes() -> None:
    g = random_complete
    with pytest.raises(ValueError):
        nodes = set(range(1, g.num_nodes + 1))
        algos.total_edge_weight(g, nodes)


def test_marginal_edge_weight_incomplete() -> None:
    g = almost_complete
    with pytest.raises(ValueError):
        nodes = set(range(1, g.num_nodes))
        algos.marginal_edge_weight(g, nodes, 0)


def test_marginal_edge_weight_invalid_margin_node() -> None:
    g = random_complete
    with pytest.raises(ValueError):
        nodes = set(range(1, g.num_nodes))
        algos.marginal_edge_weight(g, nodes, g.num_nodes + 1)


def test_marginal_edge_weight_invalid_nodes() -> None:
    g = random_complete
    with pytest.raises(ValueError):
        nodes = set(range(1, g.num_nodes + 1))
        algos.marginal_edge_weight(g, nodes, 0)


def test_improve_partition_incomplete() -> None:
    g = almost_complete
    with pytest.raises(ValueError):
        evens = set(i for i in range(g.num_nodes) if i % 2 == 0)
        odds = set(i for i in range(g.num_nodes) if i % 2 == 1)
        algos.improve_partition(g, [evens, odds])


def test_improve_partition_empty_set() -> None:
    g = random_complete
    with pytest.raises(ValueError):
        odds = set(i for i in range(g.num_nodes) if i % 2 == 1)
        algos.improve_partition(g, [set(), odds])


def test_improve_partition_invalid_node() -> None:
    g = random_complete
    with pytest.raises(ValueError):
        evens = set(i for i in range(g.num_nodes) if i % 2 == 0)
        odds = set(i for i in range(g.num_nodes) if i % 2 == 1)
        odds.add(g.num_nodes)
        algos.improve_partition(g, [evens, odds])


def test_improve_partition_node_in_multiple_sets() -> None:
    g = random_complete
    with pytest.raises(ValueError):
        evens = set(i for i in range(g.num_nodes) if i % 2 == 0)
        odds = set(i for i in range(g.num_nodes) if i % 2 == 1)
        odds.add(0)
        algos.improve_partition(g, [evens, odds])


def test_improve_partition_missing_node() -> None:
    g = random_complete
    with pytest.raises(ValueError):
        evens = set(i for i in range(1, g.num_nodes) if i % 2 == 0)
        odds = set(i for i in range(g.num_nodes) if i % 2 == 1)
        algos.improve_partition(g, [evens, odds])


### DEPRECIATED MWLP_DP TESTS

# def test_MWLP_failure() -> None:
#     gd: graph_dict = {
#         "num_nodes": 5,
#         "edges": [
#             (0, 1, 0.08460),
#             (0, 2, 0.18341),
#             (0, 3, 0.88504),
#             (0, 4, 0.14662),
#             (1, 0, 0.08460),
#             (1, 2, 0.12179),
#             (1, 3, 0.69681),
#             (1, 4, 0.12064),
#             (2, 0, 0.18341),
#             (2, 1, 0.12179),
#             (2, 3, 0.89715),
#             (2, 4, 0.45557),
#             (3, 0, 0.88504),
#             (3, 1, 0.69681),
#             (3, 2, 0.89715),
#             (3, 4, 0.60908),
#             (4, 0, 0.14662),
#             (4, 1, 0.12064),
#             (4, 2, 0.45557),
#             (4, 3, 0.60908),
#         ],
#         "node_weight": [86, 62, 27, 63, 11],
#     }
#
#     g = Graph.from_dict(gd)
#     brute: list[int] = algos.brute_force_MWLP(g)
#     DP: list[int] = algos.MWLP_DP(g)
#     assert algos.WLP(g, brute) == algos.WLP(g, DP)
#
#
# def test_MWLP_correctness() -> None:
#     for _ in range(100):
#         g = Graph.random_complete(5)
#         brute: list[int] = algos.brute_force_MWLP(g)
#         DP: list[int] = algos.MWLP_DP(g)
#         assert algos.WLP(g, brute) == algos.WLP(g, DP)
#
# def test_MWLP_DP_incomplete() -> None:
#     gd: graph_dict = {
#         "num_nodes": 4,
#         "edges": [
#             (0, 1, 1.0),
#             (0, 2, 3.0),
#             (0, 3, 5.0),
#             (1, 2, 1.0),
#             (1, 3, 50.0),
#             (2, 0, 2.0),
#             (2, 1, 7.0),
#             (2, 3, 1.0),
#             (3, 0, 8.0),
#             (3, 1, 100.0),
#             (3, 2, 2.0),
#         ],
#         "node_weight": [10, 5, 20, 7],
#     }
#     g = Graph.from_dict(gd)
#
#     with pytest.raises(ValueError):
#         algos.MWLP_DP(g)
#
#
# def test_MWLP_DP_invalid_start() -> None:
#     g = Graph.random_complete(4)
#
#     with pytest.raises(ValueError):
#         algos.MWLP_DP(g, start=4)
