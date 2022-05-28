"""
Test cases to validate Graph class
"""
from itertools import product

import pytest

from graph import Graph, graph_dict


### Correctness Tests ###
def test_default_graph() -> None:
    g = Graph()
    assert g.num_nodes == 0
    assert len(g.adjacen_list) == 0
    assert len(g.edge_weight) == 0
    assert len(g.node_weight) == 0


def test_add_node() -> None:
    g = Graph(5)
    assert g.num_nodes == 5

    g.add_node()
    assert g.num_nodes == 6
    assert g.node_weight == [0, 0, 0, 0, 0, 0]

    g.add_node(5)
    assert g.num_nodes == 7
    g.set_node_weight(5, 6)
    assert g.node_weight == [0, 0, 0, 0, 0, 6, 5]

    assert len(g.adjacen_list) == g.num_nodes


def test_add_edge() -> None:
    g = Graph(2)

    g.add_edge(0, 1, 1.0)
    g.add_edge(1, 0, 2.0)
    assert 1 in g.adjacen_list[0]
    assert 0 in g.adjacen_list[1]
    assert g.edge_weight[0][1] == 1.0
    assert g.edge_weight[1][0] == 2.0

    g.add_node()

    g.add_edge(0, 2, 3.0)
    assert 2 in g.adjacen_list[0]
    g.add_edge(2, 0, 0.0)
    g.set_edge_weight(2, 0, 4.0)
    assert g.edge_weight[0][2] == 3.0
    assert g.edge_weight[2][0] == 4.0


def test_from_dict() -> None:
    gd: graph_dict = {
        "num_nodes": 3,
        "edges": [(0, 1, 1.0), (1, 0, 2.0), (0, 2, 3.0), (2, 0, 4.0)],
        "node_weight": [0, 0, 0],
    }
    g = Graph.from_dict(gd)

    assert g.num_nodes == 3
    assert 1 in g.adjacen_list[0]
    assert 0 in g.adjacen_list[1]
    assert g.edge_weight[0][1] == 1.0
    assert g.edge_weight[1][0] == 2.0
    assert 2 in g.adjacen_list[0]
    assert g.edge_weight[0][2] == 3.0
    assert g.edge_weight[2][0] == 4.0
    assert g.node_weight == [0, 0, 0]


def test_dict_from_graph() -> None:
    g = Graph.random_complete(5)
    gd = Graph.dict_from_graph(g)
    assert gd["num_nodes"] == 5
    assert len(gd["edges"]) == 20
    assert len(gd["node_weight"]) == 5

    g_again = Graph.from_dict(gd)
    assert g.num_nodes == g_again.num_nodes
    assert g.node_weight == g_again.node_weight
    assert g.adjacen_list == g_again.adjacen_list
    assert g.edge_weight == g_again.edge_weight


def test_json_file() -> None:
    gd: graph_dict = {
        "num_nodes": 3,
        "edges": [(0, 1, 1.0), (1, 0, 2.0), (0, 2, 3.0), (2, 0, 4.0)],
        "node_weight": [0, 0, 0],
    }
    g = Graph.from_dict(gd)

    Graph.to_file(g, "json_testcase.json")
    new_g = Graph.from_file("json_testcase.json")
    new_gd: graph_dict = Graph.dict_from_graph(new_g)

    assert gd["num_nodes"] == new_gd["num_nodes"]
    assert gd["node_weight"] == new_gd["node_weight"]
    for edge in gd["edges"]:
        assert edge in new_gd["edges"]


def test_to_networkx() -> None:
    gd: graph_dict = {
        "num_nodes": 3,
        "edges": [(0, 1, 1.0), (1, 0, 2.0), (0, 2, 3.0), (2, 0, 4.0)],
        "node_weight": [0, 0, 0],
    }
    g = Graph.from_dict(gd)
    nx_g = Graph.to_networkx(g)

    assert 0 in nx_g.nodes
    assert 1 in nx_g.nodes
    assert 2 in nx_g.nodes
    assert len(nx_g.nodes) == 3
    assert (0, 1) in nx_g.edges
    assert (1, 0) in nx_g.edges
    assert nx_g[0][1]["weight"] == 1.0
    assert nx_g[1][0]["weight"] == 2.0
    assert (0, 2) in nx_g.edges
    assert nx_g[0][2]["weight"] == 3.0
    assert nx_g[2][0]["weight"] == 4.0
    assert nx_g.nodes[0]["weight"] == 0
    assert nx_g.nodes[1]["weight"] == 0
    assert nx_g.nodes[2]["weight"] == 0


def test_is_complete() -> None:
    g = Graph(4)
    for i in range(4):
        for j in range(i + 1, 4):
            g.add_edge(i, j, 1.0)
            g.add_edge(j, i, 1.0)

    assert Graph.is_complete(g)


def test_random_complete_is_complete() -> None:
    assert Graph.is_complete(Graph.random_complete(10))


def test_is_metric() -> None:
    gd: graph_dict = {
        "num_nodes": 4,
        "edges": [
            (0, 1, 2.0),
            (0, 2, 2.0),
            (0, 3, 2.0),
            (1, 0, 2.0),
            (1, 2, 2.0),
            (1, 3, 2.0),
            (2, 0, 2.0),
            (2, 1, 2.0),
            (2, 3, 2.0),
            (3, 0, 2.0),
            (3, 1, 2.0),
            (3, 2, 2.0),
        ],
        "node_weight": [0, 0, 0, 0],
    }
    g = Graph.from_dict(gd)

    assert Graph.is_metric(g)


def test_random_complete_metric_is_both() -> None:
    g = Graph.random_complete_metric(10)
    assert Graph.is_complete(g) and Graph.is_metric(g)


def test_is_agent_partition() -> None:
    n: int = 10
    k: int = 4
    g = Graph.random_complete(n)
    partition: list[set[int]] = [{0} for _ in range(k)]
    partition[0].update({1, 2})
    partition[1].update({3})
    partition[2].update({4, 5, 6})
    partition[3].update({7, 8, 9})
    assert Graph.is_agent_partition(g, partition)


def test_create_agent_partition() -> None:
    n: int = 20
    k_max: int = 4
    g = Graph.random_complete(n)
    for k in range(1, k_max):
        part: list[set[int]] = Graph.create_agent_partition(g, k)
        assert Graph.is_agent_partition(g, part)


def test_is_undirected() -> None:
    assert Graph.is_undirected(Graph()) is True
    assert Graph.is_undirected(Graph(10)) is True
    assert Graph.is_undirected(Graph.random_complete(10, directed=False)) is True
    assert Graph.is_undirected(Graph.random_complete(10, directed=True)) is False

    gd: graph_dict = {
        "num_nodes": 4,
        "edges": [
            (0, 1, 2.0),
            (0, 2, 2.0),
            (0, 3, 2.0),
            (1, 0, 2.0),
            (1, 2, 2.0),
            (1, 3, 2.0),
            (2, 0, 2.0),
            (2, 1, 2.0),
            (2, 3, 2.0),
            (3, 0, 2.0),
            (3, 1, 2.0),
            (3, 2, 2.0),
        ],
        "node_weight": [0, 0, 0, 0],
    }
    g = Graph.from_dict(gd)
    assert Graph.is_undirected(g) is True

    # test if unequal edges are found
    g.edge_weight[0][3] = 3.0
    assert Graph.is_undirected(g) is False

    # check if missing edges are found in both directions
    gd = {
        "num_nodes": 4,
        "edges": [
            (0, 1, 2.0),
            (0, 2, 2.0),
            (0, 3, 2.0),
            (1, 0, 2.0),
            (1, 3, 2.0),
            (2, 0, 2.0),
            (2, 1, 2.0),
            (2, 3, 2.0),
            (3, 0, 2.0),
            (3, 2, 2.0),
        ],
        "node_weight": [0, 0, 0, 0],
    }
    g = Graph.from_dict(gd)
    assert Graph.is_undirected(g) is False

    gd = {
        "num_nodes": 4,
        "edges": [
            (0, 1, 2.0),
            (0, 2, 2.0),
            (0, 3, 2.0),
            (1, 2, 2.0),
            (1, 3, 2.0),
            (2, 0, 2.0),
            (2, 1, 2.0),
            (2, 3, 2.0),
            (3, 0, 2.0),
            (3, 1, 2.0),
            (3, 2, 2.0),
        ],
        "node_weight": [0, 0, 0, 0],
    }
    g = Graph.from_dict(gd)
    assert Graph.is_undirected(g) is False


def test_repair_time() -> None:
    g = Graph.random_complete_metric(10)
    before: list[list[float]] = [
        [g.edge_weight[u][v] for v in range(10)] for u in range(10)
    ]

    amt: float = 3.14
    g.add_repair_time(amt)

    for u, v in product(range(10), range(10)):
        if u != v:
            assert g.edge_weight[u][v] == before[u][v] + amt


def test_subgraph_one_to_one() -> None:
    gd: graph_dict = {
        "num_nodes": 4,
        "edges": [(0, 1, 1.0), (1, 2, 3.0), (2, 3, 5.0), (0, 2, 2.0)],
        "node_weight": [10, 2, 6, 20],
    }
    g = Graph.from_dict(gd)

    sg, sto, ots = Graph.subgraph(g, [0, 1, 2])

    assert sg.num_nodes == 3

    for i in range(3):
        assert sto[i] == i
        assert ots[i] == i
        assert g.node_weight[i] == sg.node_weight[i]

    for i in range(3):
        for j in range(3):
            if i != j and j in g.adjacen_list[i]:
                assert j in sg.adjacen_list[i]
                assert g.edge_weight[i][j] == sg.edge_weight[i][j]


def test_subgraph_empty() -> None:
    g = Graph.random_complete(4)
    sg, sto, ots = Graph.subgraph(g, [])

    assert sg.num_nodes == 0
    assert sg.adjacen_list == []
    assert sg.node_weight == []
    assert sg.edge_weight == []

    assert len(sto) == len(ots) == 0


def test_subgraph() -> None:
    gd: graph_dict = {
        "num_nodes": 4,
        "edges": [(0, 1, 1.0), (1, 2, 3.0), (2, 3, 5.0), (0, 2, 2.0)],
        "node_weight": [10, 2, 6, 20],
    }
    g = Graph.from_dict(gd)

    sg, sto, ots = Graph.subgraph(g, [0, 1, 3])

    assert sg.num_nodes == 3

    assert ots[0] == 0
    assert ots[1] == 1
    assert ots[3] == 2
    assert sto[0] == 0
    assert sto[1] == 1
    assert sto[2] == 3

    for i in range(3):
        assert sg.node_weight[i] == g.node_weight[sto[i]]

    for i in range(3):
        for j in range(3):
            if i != j and j in sg.adjacen_list[i]:
                assert sto[j] in g.adjacen_list[sto[i]]
                assert sg.edge_weight[i][j] == g.edge_weight[sto[i]][sto[j]]


def test_subgraph_maintains_properties() -> None:
    g = Graph.random_complete_metric(6)

    sg, _, _ = Graph.subgraph(g, [0, 2, 4, 5])

    assert Graph.is_complete(sg)
    assert Graph.is_metric(sg)


### Failure Tests ###
def test_is_complete_failure() -> None:
    gd: graph_dict = {
        "num_nodes": 4,
        "edges": [
            (0, 1, 2.0),
            (0, 2, 2.0),
            (1, 3, 2.0),
            (2, 1, 2.0),
            (2, 3, 2.0),
            (3, 1, 2.0),
            (3, 2, 2.0),
        ],
        "node_weight": [0, 0, 0, 0],
    }
    g = Graph.from_dict(gd)
    assert Graph.is_complete(g) is False

    rand_g = Graph.random_complete_metric(5)
    rand_g.edge_weight[0][1] = -1.0
    assert Graph.is_complete(rand_g) is False


def test_is_metric_failure() -> None:
    gd: graph_dict = {
        "num_nodes": 4,
        "edges": [
            (0, 1, 2.0),
            (0, 2, 2.0),
            (0, 3, 2.0),
            (1, 0, 2.0),
            (1, 2, 5.0),
            (1, 3, 100.0),
            (2, 0, 2.0),
            (2, 1, 2.0),
            (2, 3, 1.0),
            (3, 0, 2.0),
            (3, 1, 2.0),
            (3, 2, 2.0),
        ],
        "node_weight": [0, 0, 0, 0],
    }
    g = Graph.from_dict(gd)

    assert Graph.is_metric(g) is False


def test_is_agent_partition_failure() -> None:
    n: int = 10
    g = Graph(n)

    # Check that empty partitions are caught
    empty_part: list[set[int]] = []
    assert Graph.is_agent_partition(g, empty_part) is False

    # Check if 0 is missing
    missing_zero: list[set[int]] = [{1}, {0}]
    missing_zero[1].update(set(range(2, 20)))
    assert Graph.is_agent_partition(g, missing_zero) is False

    # Check subset only contain valid nodes
    has_negative: list[set[int]] = [{0, -1}, set(range(n))]
    assert Graph.is_agent_partition(g, has_negative) is False
    out_of_range: list[set[int]] = [{0, n}, set(range(n))]
    assert Graph.is_agent_partition(g, out_of_range) is False

    # Catch duplicates that aren't 0
    has_dupes: list[set[int]] = [{0, 1}, set(range(n))]
    assert Graph.is_agent_partition(g, has_dupes) is False

    # Catch missing nodes
    missing_nodes: list[set[int]] = [{0, 1}, {0}]
    missing_nodes[1].update(set(range(3, n)))
    assert Graph.is_agent_partition(g, missing_nodes) is False


### Error Tests ###
def test_negative_num_nodes() -> None:
    with pytest.raises(ValueError):
        Graph(-1)


def test_from_dict_negative_num_nodes() -> None:
    with pytest.raises(ValueError):
        gd: graph_dict = {
            "num_nodes": -1,
            "edges": [(0, 1, 1.0), (1, 0, 2.0), (0, 2, 3.0), (2, 0, 4.0)],
            "node_weight": [0, 0, 0, 0],
        }
        Graph.from_dict(gd)


def test_from_dict_negative_nodeweight() -> None:
    with pytest.raises(ValueError):
        gd: graph_dict = {
            "num_nodes": 3,
            "edges": [(0, 1, 1.0), (1, 0, 2.0), (0, 2, 3.0), (2, 0, 4.0)],
            "node_weight": [0, -2, 0],
        }
        Graph.from_dict(gd)


def test_from_dict_nonexistant_start_node() -> None:
    with pytest.raises(ValueError):
        gd: graph_dict = {
            "num_nodes": 3,
            "edges": [(3, 1, 1.0), (1, 0, 2.0), (0, 2, 3.0), (2, 0, 4.0)],
            "node_weight": [0, 0, 0],
        }
        Graph.from_dict(gd)


def test_from_dict_nonexistant_end_node() -> None:
    with pytest.raises(ValueError):
        gd: graph_dict = {
            "num_nodes": 3,
            "edges": [(1, 3, 1.0), (1, 0, 2.0), (0, 2, 3.0), (2, 0, 4.0)],
            "node_weight": [0, 0, 0],
        }
        Graph.from_dict(gd)


def test_from_dict_wrong_node_weight_len() -> None:
    with pytest.raises(ValueError):
        gd: graph_dict = {
            "num_nodes": 3,
            "edges": [(0, 1, 1.0), (1, 0, 2.0), (0, 2, 3.0), (2, 0, 4.0)],
            "node_weight": [0, 0, 0, 0],
        }
        Graph.from_dict(gd)


def test_random_complete_negative_num_nodes() -> None:
    with pytest.raises(ValueError):
        Graph.random_complete(-1)


def test_random_complete_negative_edge_weight_range() -> None:
    with pytest.raises(ValueError):
        Graph.random_complete(10, (-1.0, 3.0))


def test_random_complete_wrong_edge_weight_order() -> None:
    with pytest.raises(ValueError):
        Graph.random_complete(10, (6.0, 3.0))


def test_random_complete_wrong_node_weight_order() -> None:
    with pytest.raises(ValueError):
        Graph.random_complete(10, (3.0, 6.0), (4, 1))


def test_random_complete_negative_node_weight_range() -> None:
    with pytest.raises(ValueError):
        Graph.random_complete(10, (3.0, 6.0), (-1, 4))


def test_random_metric_complete_negative_num_nodes() -> None:
    with pytest.raises(ValueError):
        Graph.random_complete_metric(-1)


def test_random_metric_complete_negative_node_weight_range() -> None:
    with pytest.raises(ValueError):
        Graph.random_complete_metric(10, 6.0, (-1, 4))


def test_random_metric_complete_wrong_node_weight_order() -> None:
    with pytest.raises(ValueError):
        Graph.random_complete_metric(10, 6.0, (4, 1))


def test_random_metric_complete_negative_upper_edge_weight() -> None:
    with pytest.raises(ValueError):
        Graph.random_complete_metric(10, -1.0)


def test_add_node_negative_weight() -> None:
    with pytest.raises(ValueError):
        g = Graph(2)
        g.add_node(-1)


def test_add_edge_nonexistant_start() -> None:
    with pytest.raises(ValueError):
        g = Graph(2)
        g.add_edge(2, 0, 5.0)


def test_add_edge_nonexistant_end() -> None:
    with pytest.raises(ValueError):
        g = Graph(2)
        g.add_edge(0, 2, 5.0)


def test_add_edge_again() -> None:
    with pytest.raises(ValueError):
        g = Graph(2)
        g.add_edge(0, 1, 5.0)
        g.add_edge(0, 1, 4.0)


def test_set_node_weight_nonexistant() -> None:
    with pytest.raises(ValueError):
        g = Graph(2)
        g.set_node_weight(2, 3)


def test_set_node_weight_negative() -> None:
    with pytest.raises(ValueError):
        g = Graph(2)
        g.set_node_weight(1, -3)


def test_set_edge_weight_nonexistant_start() -> None:
    with pytest.raises(ValueError):
        g = Graph(2)
        g.add_edge(0, 1, 1.0)
        g.set_edge_weight(2, 0, 2.0)


def test_set_edge_weight_nonexistant_end() -> None:
    with pytest.raises(ValueError):
        g = Graph(2)
        g.add_edge(0, 1, 1.0)
        g.set_edge_weight(0, 2, 2.0)


def test_set_edge_weight_negative_weight() -> None:
    with pytest.raises(ValueError):
        g = Graph(2)
        g.add_edge(0, 1, 1.0)
        g.set_edge_weight(0, 1, -2.0)


def test_set_edge_weight_nonexistant_edge() -> None:
    with pytest.raises(ValueError):
        g = Graph(2)
        g.add_edge(0, 1, 1.0)
        g.set_edge_weight(1, 0, 2.0)


def test_subgraph_nonexistant_node() -> None:
    with pytest.raises(ValueError):
        g = Graph.random_complete(4)
        Graph.subgraph(g, [0, 2, 4])


def test_add_repair_time_negative_time() -> None:
    with pytest.raises(ValueError):
        g = Graph.random_complete(5)
        g.add_repair_time(-1.0)
