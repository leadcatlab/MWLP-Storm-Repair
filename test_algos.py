from graph import Graph
import algos
from typing_extensions import TypedDict
import pytest

graphDict = TypedDict(
    "graphDict",
    {
        "numNodes": int,
        "nodeWeight": list[int],
        "edges": list[tuple[int, int, float]],
    },
)

### Correctness Tests ###


def test_fW_empty() -> None:
    g = Graph()
    dist: list[list[float]] = algos.FloydWarshall(g)

    assert len(dist) == 0


def test_fW_no_edges() -> None:
    g = Graph(5)
    dist: list[list[float]] = algos.FloydWarshall(g)

    assert len(dist) == len(dist[0]) == 5
    for i in range(5):
        assert dist[i][i] == 0.0
        for j in range(5):
            if i != j:
                assert dist[i][j] == float("inf")


def test_fW_complete() -> None:
    g = Graph.randomCompleteMetric(5)
    dist: list[list[float]] = algos.FloydWarshall(g)

    for i in range(g.numNodes):
        for j in range(g.numNodes):
            if i == j:
                assert dist[i][j] == 0.0
            else:
                assert dist[i][j] != float("inf")


def test_fW_metric_complete() -> None:
    g = Graph.randomCompleteMetric(5)
    dist: list[list[float]] = algos.FloydWarshall(g)

    for i in range(g.numNodes):
        for j in range(g.numNodes):
            if i == j:
                assert dist[i][j] == 0.0
            else:
                assert dist[i][j] == g.edgeWeight[i][j]


def test_WLP_small_orders() -> None:
    g = Graph.randomComplete(4)
    assert algos.WLP(g, []) == 0.0
    assert algos.WLP(g, [0]) == 0.0
    assert algos.WLP(g, [1]) == 0.0


def test_WLP() -> None:
    gd: graphDict = {
        "numNodes": 4,
        "edges": [(0, 1, 1.0), (1, 2, 3.0), (2, 3, 5.0), (0, 2, 2.0)],
        "nodeWeight": [10, 2, 6, 20],
    }
    g = Graph.fromDict(gd)

    assert algos.WLP(g, [0, 1, 2, 3]) == 206.0
    assert algos.WLP(g, [0, 1]) == 2.0
    assert algos.WLP(g, [0, 2]) == 12.0
    assert algos.WLP(g, [0, 2, 3]) == 152.0
    assert algos.WLP(g, [1, 2, 3]) == 178.0


def test_bruteForceMWLP() -> None:
    gd: graphDict = {
        "numNodes": 4,
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
        "nodeWeight": [10, 5, 20, 7],
    }
    g = Graph.fromDict(gd)

    assert algos.bruteForceMWLP(g) == [0, 1, 2, 3]
    assert algos.bruteForceMWLP(g, start=1) == [1, 2, 0, 3]


def test_MWLP_failure() -> None:
    gd: graphDict = {
        "numNodes": 5,
        "edges": [
            (0, 1, 0.08460),
            (0, 2, 0.18341),
            (0, 3, 0.88504),
            (0, 4, 0.14662),
            (1, 0, 0.08460),
            (1, 2, 0.12179),
            (1, 3, 0.69681),
            (1, 4, 0.12064),
            (2, 0, 0.18341),
            (2, 1, 0.12179),
            (2, 3, 0.89715),
            (2, 4, 0.45557),
            (3, 0, 0.88504),
            (3, 1, 0.69681),
            (3, 2, 0.89715),
            (3, 4, 0.60908),
            (4, 0, 0.14662),
            (4, 1, 0.12064),
            (4, 2, 0.45557),
            (4, 3, 0.60908),
        ],
        "nodeWeight": [86, 62, 27, 63, 11],
    }

    g = Graph.fromDict(gd)
    brute: list[int] = algos.bruteForceMWLP(g)
    DP: list[int] = algos.MWLP_DP(g)
    assert algos.WLP(g, brute) == algos.WLP(g, DP)


def test_MWLP_correctness() -> None:
    # TODO: This does not work 100% of the time and I have no idea why
    for _ in range(100):
        g = Graph.randomComplete(5)
        brute: list[int] = algos.bruteForceMWLP(g)
        DP: list[int] = algos.MWLP_DP(g)
        assert algos.WLP(g, brute) == algos.WLP(g, DP)


def test_nearestNeighbor() -> None:
    gd: graphDict = {
        "numNodes": 4,
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
        "nodeWeight": [10, 5, 20, 7],
    }
    g = Graph.fromDict(gd)

    assert algos.nearestNeighbor(g) == [0, 1, 2, 3]
    assert algos.nearestNeighbor(g, start=1) == [1, 2, 3, 0]


def test_greedy() -> None:
    gd: graphDict = {
        "numNodes": 4,
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
        "nodeWeight": [10, 5, 20, 7],
    }
    g = Graph.fromDict(gd)

    assert algos.greedy(g) == [0, 2, 3, 1]
    assert algos.greedy(g, start=1) == [1, 2, 0, 3]


def test_TSP() -> None:
    gd: graphDict = {
        "numNodes": 4,
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
        "nodeWeight": [10, 5, 20, 7],
    }
    g = Graph.fromDict(gd)

    assert algos.TSP(g) == [0, 1, 2, 3]
    assert algos.TSP(g, start=1) == [1, 2, 0, 3]


def test_HeldKarp() -> None:
    gd: graphDict = {
        "numNodes": 4,
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
        "nodeWeight": [10, 5, 20, 7],
    }
    g = Graph.fromDict(gd)

    assert algos.HeldKarp(g, start=0) == [0, 1, 2, 3]


def test_TSP_correctness() -> None:
    for _ in range(10):
        g = Graph.randomComplete(7)
        assert algos.HeldKarp(g) == algos.TSP(g)


def test_partitionHeuristic_MWLP_2_agents() -> None:
    gd: graphDict = {
        "numNodes": 4,
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
        "nodeWeight": [10, 5, 20, 7],
    }
    g = Graph.fromDict(gd)

    result, partition = algos.partitionHeuristic(g, algos.bruteForceMWLP, 2)

    assert result == 52.0
    assert [0, 1, 2] in partition and [0, 3] in partition


def test_optimalNumberOfAgentsMWLP() -> None:
    gd: graphDict = {
        "numNodes": 4,
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
        "nodeWeight": [10, 5, 20, 7],
    }
    g = Graph.fromDict(gd)

    optimalMWLP, optimalOrder = algos.optimalNumberOfAgents(
        g, algos.bruteForceMWLP, 1, 3
    )

    assert optimalMWLP == 52.0
    assert len(optimalOrder) == 2
    assert [0, 1, 2] in optimalOrder and [0, 3] in optimalOrder


### Error Tests ###


def test_WLP_node_not_in_graph() -> None:
    g = Graph.randomComplete(4)
    with pytest.raises(ValueError):
        algos.WLP(g, [0, 1, 2, 4])


def test_WLP_missing_edge() -> None:
    gd: graphDict = {
        "numNodes": 4,
        "edges": [(0, 1, 1.0), (1, 2, 3.0), (2, 3, 5.0), (0, 2, 2.0)],
        "nodeWeight": [10, 2, 6, 20],
    }
    g = Graph.fromDict(gd)

    with pytest.raises(ValueError):
        algos.WLP(g, [0, 1, 2, 3, 2])


def test_bruteForceMWLP_incomplete() -> None:
    gd: graphDict = {
        "numNodes": 4,
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
        "nodeWeight": [10, 5, 20, 7],
    }
    g = Graph.fromDict(gd)

    with pytest.raises(ValueError):
        algos.bruteForceMWLP(g)


def test_bruteForceMWLP_invalid_start() -> None:
    g = Graph.randomComplete(4)

    with pytest.raises(ValueError):
        algos.bruteForceMWLP(g, start=4)


def test_nearestNeighbor_incomplete() -> None:
    gd: graphDict = {
        "numNodes": 4,
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
        "nodeWeight": [10, 5, 20, 7],
    }
    g = Graph.fromDict(gd)

    with pytest.raises(ValueError):
        algos.nearestNeighbor(g)


def test_nearestNeighbor_invalid_start() -> None:
    g = Graph.randomComplete(4)

    with pytest.raises(ValueError):
        algos.nearestNeighbor(g, start=4)


def test_greedy_incomplete() -> None:
    gd: graphDict = {
        "numNodes": 4,
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
        "nodeWeight": [10, 5, 20, 7],
    }
    g = Graph.fromDict(gd)

    with pytest.raises(ValueError):
        algos.greedy(g)


def test_greedy_invalid_start() -> None:
    g = Graph.randomComplete(4)

    with pytest.raises(ValueError):
        algos.greedy(g, start=4)


def test_TSP_incomplete() -> None:
    gd: graphDict = {
        "numNodes": 4,
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
        "nodeWeight": [10, 5, 20, 7],
    }
    g = Graph.fromDict(gd)

    with pytest.raises(ValueError):
        algos.TSP(g)


def test_TSP_invalid_start() -> None:
    g = Graph.randomComplete(4)

    with pytest.raises(ValueError):
        algos.TSP(g, start=4)


def test_HK_incomplete() -> None:
    gd: graphDict = {
        "numNodes": 4,
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
        "nodeWeight": [10, 5, 20, 7],
    }
    g = Graph.fromDict(gd)

    with pytest.raises(ValueError):
        algos.HeldKarp(g)


def test_HK_invalid_start() -> None:
    g = Graph.randomComplete(4)

    with pytest.raises(ValueError):
        algos.HeldKarp(g, start=4)


def test_MWLP_DP_incomplete() -> None:
    gd: graphDict = {
        "numNodes": 4,
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
        "nodeWeight": [10, 5, 20, 7],
    }
    g = Graph.fromDict(gd)

    with pytest.raises(ValueError):
        algos.MWLP_DP(g)


def test_MWLP_DP_invalid_start() -> None:
    g = Graph.randomComplete(4)

    with pytest.raises(ValueError):
        algos.MWLP_DP(g, start=4)


def test_partition_incomplete() -> None:
    gd: graphDict = {
        "numNodes": 4,
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
        "nodeWeight": [10, 5, 20, 7],
    }
    g = Graph.fromDict(gd)

    with pytest.raises(ValueError):
        algos.partitionHeuristic(g, algos.TSP, 2)


def test_partition_too_few_agents() -> None:
    g = Graph.randomComplete(4)
    with pytest.raises(ValueError):
        algos.partitionHeuristic(g, algos.TSP, 0)


def test_partition_too_many_agents() -> None:
    g = Graph.randomComplete(4)
    with pytest.raises(ValueError):
        algos.partitionHeuristic(g, algos.TSP, 5)


def test_optimalNumberOfAgents_incomplete() -> None:
    gd: graphDict = {
        "numNodes": 4,
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
        "nodeWeight": [10, 5, 20, 7],
    }
    g = Graph.fromDict(gd)

    with pytest.raises(ValueError):
        algos.optimalNumberOfAgents(g, algos.TSP, 1, 3)


def test_optimalNumberOfAgents_invalid_order() -> None:
    g = Graph.randomComplete(6)
    with pytest.raises(ValueError):
        algos.optimalNumberOfAgents(g, algos.TSP, 3, 1)


def test_optimalNumberOfAgents_too_few() -> None:
    g = Graph.randomComplete(6)
    with pytest.raises(ValueError):
        algos.optimalNumberOfAgents(g, algos.TSP, -1, 1)


def test_optimalNumberOfAgents_too_many() -> None:
    g = Graph.randomComplete(6)
    with pytest.raises(ValueError):
        algos.optimalNumberOfAgents(g, algos.TSP, 1, 6)
