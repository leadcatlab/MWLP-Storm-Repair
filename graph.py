from __future__ import annotations
from typing_extensions import TypedDict
import random

# TODO: Comments


class graph:
    def __init__(self, numNodes: int = 0):
        self.numNodes = numNodes
        self.adjacenList: list[list[int]] = [[] for _ in range(numNodes)]
        self.edgeWeight: list[list[float]] = [
            [0.0 for _ in range(numNodes)] for _ in range(numNodes)
        ]
        self.nodeWeight: list[int] = [0 for _ in range(numNodes)]

    """
        Graph dict structure
        dict[numNodes] = number of nodes
        dict[nodeWeight] = list of correct length of nodeWeights
        dict[edges] = list of tuples of form (startingNode, endingNode, nodeWeight)
    """
    graphDict = TypedDict(
        "graphDict",
        {
            "numNodes": int,
            "nodeWeight": list[int],
            "edges": list[tuple[int, int, float]],
        },
    )

    @staticmethod
    def FromDict(dictionary: graphDict) -> graph:
        g = graph(dictionary["numNodes"])

        assert len(dictionary["nodeWeight"]) == g.numNodes
        g.nodeWeight = dictionary["nodeWeight"]

        for (startingNode, endingNode, nodeWeight) in dictionary["edges"]:
            g.addEdge(startingNode, endingNode, nodeWeight)

        return g

    @staticmethod
    def randomComplete(n: int, interval: tuple[float, float] = (0, 1)) -> graph:
        g = graph(n)

        g.nodeWeight = [random.randint(1, 100) for _ in range(n)]

        for i in range(n):
            for j in range(i + 1, n):
                weight: float = random.uniform(interval[0], interval[1])
                g.addEdge(i, j, weight)
                g.addEdge(j, i, weight)

        assert graph.isComplete(g)
        return g

    @staticmethod
    def randomCompleteMetric(n: int, upper: float = 1) -> graph:
        g = graph.randomComplete(n, (upper / 2, upper))
        assert graph.isMetric(g)
        return g

    def addNode(self, nodeWeight: int = 0) -> None:
        self.numNodes += 1
        self.adjacenList.append([])
        self.nodeWeight.append(nodeWeight)

        # need to add slot for new node to edge weight matrix
        for weightList in self.edgeWeight:
            weightList.append(0.0)

        # add new row to edge weight matrix
        self.edgeWeight.append([0.0 for _ in range(self.numNodes)])

    def addEdge(self, startingNode: int, endingNode: int, weight: float = 0.0) -> None:
        assert startingNode < self.numNodes
        assert endingNode < self.numNodes
        # add edge only once
        assert endingNode not in self.adjacenList[startingNode]

        self.adjacenList[startingNode].append(endingNode)
        self.edgeWeight[startingNode][endingNode] = weight

    def setNodeWeight(self, node: int, weight: int) -> None:
        assert node < self.numNodes
        self.nodeWeight[node] = weight

    def setEdgeWeight(
        self, startingNode: int, endingNode: int, edgeWeight: float
    ) -> None:
        assert startingNode < self.numNodes
        assert endingNode < self.numNodes
        assert edgeWeight > 0.0
        # add weight only if edge exists
        assert endingNode in self.adjacenList[startingNode]

        self.edgeWeight[startingNode][endingNode] = edgeWeight

    @staticmethod
    def isComplete(cls: graph) -> bool:
        for u in range(cls.numNodes):
            for v in range(cls.numNodes):
                if u != v and v not in cls.adjacenList[u]:
                    return False

        return True

    @staticmethod
    def isMetric(cls: graph) -> bool:
        for u in range(cls.numNodes):
            for v in range(cls.numNodes):
                if u != v and v in cls.adjacenList[u]:
                    for w in range(cls.numNodes):
                        if w in cls.adjacenList[u] and v in cls.adjacenList[w]:
                            if (
                                cls.edgeWeight[u][w] + cls.edgeWeight[w][v]
                                < cls.edgeWeight[u][v]
                            ):
                                return False

        return True

    def __str__(self) -> str:
        toPrint: str = ""
        for i in range(len(self.adjacenList)):
            currentList = self.adjacenList[i]

            nodes: str = ""

            for j in currentList:
                nodes += str(j) + " with distance " + str(self.edgeWeight[i][j]) + " "

            toPrint += "Node " + str(i) + " is connected to " + nodes + "\n"

        toPrint += "\n"

        for i in range(len(self.nodeWeight)):
            toPrint += (
                "The weight of node " + str(i) + " is " + str(self.nodeWeight[i]) + "\n"
            )

        return toPrint
