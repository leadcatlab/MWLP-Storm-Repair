from graph import Graph
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
def test_default_graph() -> None:
    g = Graph()
    assert g.numNodes == 0
    assert len(g.adjacenList) == 0
    assert len(g.edgeWeight) == 0
    assert len(g.nodeWeight) == 0


def test_addNode() -> None:
    g = Graph(5)
    assert g.numNodes == 5

    g.addNode()
    assert g.numNodes == 6
    assert g.nodeWeight == [0, 0, 0, 0, 0, 0]

    g.addNode(5)
    assert g.numNodes == 7
    g.setNodeWeight(5, 6)
    assert g.nodeWeight == [0, 0, 0, 0, 0, 6, 5]

    assert len(g.adjacenList) == g.numNodes


def test_addEdge() -> None:
    g = Graph(2)

    g.addEdge(0, 1, 1.0)
    g.addEdge(1, 0, 2.0)
    assert 1 in g.adjacenList[0]
    assert 0 in g.adjacenList[1]
    assert g.edgeWeight[0][1] == 1.0
    assert g.edgeWeight[1][0] == 2.0

    g.addNode()

    g.addEdge(0, 2, 3.0)
    assert 2 in g.adjacenList[0]
    g.addEdge(2, 0, 0.0)
    g.setEdgeWeight(2, 0, 4.0)
    assert g.edgeWeight[0][2] == 3.0
    assert g.edgeWeight[2][0] == 4.0


def test_fromDict() -> None:
    gd: graphDict = {
        "numNodes": 3,
        "edges": [(0, 1, 1.0), (1, 0, 2.0), (0, 2, 3.0), (2, 0, 4.0)],
        "nodeWeight": [0, 0, 0],
    }
    g = Graph.fromDict(gd)

    assert g.numNodes == 3
    assert 1 in g.adjacenList[0]
    assert 0 in g.adjacenList[1]
    assert g.edgeWeight[0][1] == 1.0
    assert g.edgeWeight[1][0] == 2.0
    assert 2 in g.adjacenList[0]
    assert g.edgeWeight[0][2] == 3.0
    assert g.edgeWeight[2][0] == 4.0
    assert g.nodeWeight == [0, 0, 0]


def test_isComplete() -> None:
    g = Graph(4)
    for i in range(4):
        for j in range(i + 1, 4):
            g.addEdge(i, j)
            g.addEdge(j, i)

    assert Graph.isComplete(g)


def test_randomComplete_isComplete() -> None:
    assert Graph.isComplete(Graph.randomComplete(10))


def test_isMetric() -> None:
    gd: graphDict = {
        "numNodes": 4,
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
        "nodeWeight": [0, 0, 0, 0],
    }
    g = Graph.fromDict(gd)

    assert Graph.isMetric(g)


def test_randomCompleteMetric_is_both() -> None:
    g = Graph.randomCompleteMetric(10)
    assert Graph.isComplete(g) and Graph.isMetric(g)


### Failure Tests ###
def test_isComplete_failure() -> None:
    gd: graphDict = {
        "numNodes": 4,
        "edges": [
            (0, 1, 2.0),
            (0, 2, 2.0),
            (1, 3, 2.0),
            (2, 1, 2.0),
            (2, 3, 2.0),
            (3, 1, 2.0),
            (3, 2, 2.0),
        ],
        "nodeWeight": [0, 0, 0, 0],
    }
    g = Graph.fromDict(gd)

    assert Graph.isComplete(g) is False


def test_isMetric_failure() -> None:
    gd: graphDict = {
        "numNodes": 4,
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
        "nodeWeight": [0, 0, 0, 0],
    }
    g = Graph.fromDict(gd)

    assert Graph.isMetric(g) is False


### Error Tests ###
def test_negative_num_nodes() -> None:
    with pytest.raises(ValueError):
        Graph(-1)


def test_fromDict_negative_num_nodes() -> None:
    with pytest.raises(ValueError):
        gd: graphDict = {
            "numNodes": -1,
            "edges": [(0, 1, 1.0), (1, 0, 2.0), (0, 2, 3.0), (2, 0, 4.0)],
            "nodeWeight": [0, 0, 0, 0],
        }
        Graph.fromDict(gd)


def test_fromDict_negative_nodeweight() -> None:
    with pytest.raises(ValueError):
        gd: graphDict = {
            "numNodes": 3,
            "edges": [(0, 1, 1.0), (1, 0, 2.0), (0, 2, 3.0), (2, 0, 4.0)],
            "nodeWeight": [0, -2, 0],
        }
        Graph.fromDict(gd)


def test_fromDict_nonexistant_start_node() -> None:
    with pytest.raises(ValueError):
        gd: graphDict = {
            "numNodes": 3,
            "edges": [(3, 1, 1.0), (1, 0, 2.0), (0, 2, 3.0), (2, 0, 4.0)],
            "nodeWeight": [0, 0, 0],
        }
        Graph.fromDict(gd)


def test_fromDict_nonexistant_end_node() -> None:
    with pytest.raises(ValueError):
        gd: graphDict = {
            "numNodes": 3,
            "edges": [(1, 3, 1.0), (1, 0, 2.0), (0, 2, 3.0), (2, 0, 4.0)],
            "nodeWeight": [0, 0, 0],
        }
        Graph.fromDict(gd)


def test_fromDict_wrong_nodeWeight_len() -> None:
    with pytest.raises(ValueError):
        gd: graphDict = {
            "numNodes": 3,
            "edges": [(0, 1, 1.0), (1, 0, 2.0), (0, 2, 3.0), (2, 0, 4.0)],
            "nodeWeight": [0, 0, 0, 0],
        }
        Graph.fromDict(gd)


def test_randomComplete_negative_edgeWeight_range() -> None:
    with pytest.raises(ValueError):
        Graph.randomComplete(10, (-1.0, 3.0))


def test_randomComplete_wrong_edge_weight_order() -> None:
    with pytest.raises(ValueError):
        Graph.randomComplete(10, (6.0, 3.0))


def test_randomComplete_wrong_node_weight_order() -> None:
    with pytest.raises(ValueError):
        Graph.randomComplete(10, (3.0, 6.0), (4, 1))


def test_randomComplete_negative_node_weight_range() -> None:
    with pytest.raises(ValueError):
        Graph.randomComplete(10, (3.0, 6.0), (-1, 4))


def test_addNode_negative_weight() -> None:
    with pytest.raises(ValueError):
        g = Graph(2)
        g.addNode(-1)


def test_addEdge_nonexistant_start() -> None:
    with pytest.raises(ValueError):
        g = Graph(2)
        g.addEdge(2, 0, 5.0)


def test_addEdge_nonexistant_end() -> None:
    with pytest.raises(ValueError):
        g = Graph(2)
        g.addEdge(0, 2, 5.0)


def test_addEdge_again() -> None:
    with pytest.raises(ValueError):
        g = Graph(2)
        g.addEdge(0, 1, 5.0)
        g.addEdge(0, 1, 4.0)


def test_setNodeWeight_nonexistant() -> None:
    with pytest.raises(ValueError):
        g = Graph(2)
        g.setNodeWeight(2, 3)


def test_setNodeWeight_negative() -> None:
    with pytest.raises(ValueError):
        g = Graph(2)
        g.setNodeWeight(1, -3)


def test_setEdgeWeight_nonexistant_start() -> None:
    with pytest.raises(ValueError):
        g = Graph(2)
        g.addEdge(0, 1, 1.0)
        g.setEdgeWeight(2, 0, 2.0)


def test_setEdgeWeight_nonexistant_end() -> None:
    with pytest.raises(ValueError):
        g = Graph(2)
        g.addEdge(0, 1, 1.0)
        g.setEdgeWeight(0, 2, 2.0)


def test_setEdgeWeight_negative_weight() -> None:
    with pytest.raises(ValueError):
        g = Graph(2)
        g.addEdge(0, 1, 1.0)
        g.setEdgeWeight(0, 1, -2.0)


def test_setEdgeWeight_nonexistant_edge() -> None:
    with pytest.raises(ValueError):
        g = Graph(2)
        g.addEdge(0, 1, 1.0)
        g.setEdgeWeight(1, 0, 2.0)
