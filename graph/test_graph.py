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


def test_add_nodes() -> None:
    g = Graph(5)
    assert g.numNodes == 5

    g.addNode()
    assert g.numNodes == 6
    assert g.nodeWeight == [0, 0, 0, 0, 0, 0]

    g.addNode(5)
    assert g.numNodes == 7
    g.setNodeWeight(5, 6)
    assert g.nodeWeight == [0, 0, 0, 0, 0, 6, 5]


def test_add_edges() -> None:
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


def test_dict() -> None:
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


def test_complete() -> None:
    g = Graph(4)
    for i in range(4):
        for j in range(i + 1, 4):
            g.addEdge(i, j)
            g.addEdge(j, i)

    assert Graph.isComplete(g)


def test_metric() -> None:
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


### Failure Tests ###

### Error Tests ###
def test_wrong_nodeWeight_len() -> None:
    with pytest.raises(ValueError):
        gd: graphDict = {
            "numNodes": 3,
            "edges": [(0, 1, 1.0), (1, 0, 2.0), (0, 2, 3.0), (2, 0, 4.0)],
            "nodeWeight": [0, 0, 0, 0],
        }
        Graph.fromDict(gd)
