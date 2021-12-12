from __future__ import annotations
from typing_extensions import TypedDict
import random

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


class graph:
    """Directed graph class with weighted edges and nodes

    Attributes:
        numNodes: number of nodes from [0, n)
        adjacenList: directed adjacency list
        edgeWeight: 2D matrix representing l(i, j)
        nodeWeight: list of all weights w(i)

    """

    def __init__(self, n: int = 0):
        """Inits graph with n nodes

        Args:
            n: number of nodes (by default graph is empty)

        """

        self.numNodes = n
        self.adjacenList: list[list[int]] = [[] for _ in range(n)]
        self.edgeWeight: list[list[float]] = [[0.0 for _ in range(n)] for _ in range(n)]
        self.nodeWeight: list[int] = [0 for _ in range(n)]

    @staticmethod
    def FromDict(dictionary: graphDict) -> graph:
        """Alternate constructor using dictionary

        Args:
            graphDict: Dictionary containing the needed information for a graph

        Returns:
            graph
        """

        g = graph(dictionary["numNodes"])

        assert len(dictionary["nodeWeight"]) == g.numNodes
        g.nodeWeight = dictionary["nodeWeight"]

        for (startingNode, endingNode, nodeWeight) in dictionary["edges"]:
            g.addEdge(startingNode, endingNode, nodeWeight)

        return g

    @staticmethod
    def randomComplete(
        n: int, edgeW: tuple[float, float] = (0, 1), nodeW: tuple[int, int] = (1, 100)
    ) -> graph:
        """Create a randomly generated complete weighted undirected graph

        Creates a complete undirected graph with randomly weighted edges and nodes

        Args:
            n: number of nodes
            edgeW: the interval that edge weights can be in
            nodeW: the interval that node weights can be in

        Returns:
            graph
        """

        g = graph(n)

        g.nodeWeight = [random.randint(nodeW[0], nodeW[1]) for _ in range(n)]

        for i in range(n):
            for j in range(i + 1, n):
                weight: float = random.uniform(edgeW[0], edgeW[1])
                g.addEdge(i, j, weight)
                g.addEdge(j, i, weight)

        assert graph.isComplete(g)
        return g

    @staticmethod
    def randomCompleteMetric(
        n: int, upper: float = 1, nodeW: tuple[int, int] = (1, 100)
    ) -> graph:
        """Create a randomly generated complete weighted undirected metric graph

        Creates a complete undirected graph with randomly weighted edges and nodes
        Edges satisfy the inequality l(u, w) + l(w, v) >= l(u, v) for all nodes u, v, w

        Args:
            n: number of nodes
            upper: the upper bound of edge weights used to create the interval (upper / 2, upper)
            nodeW: the interval that node weights can be in

        Returns:
            graph
        """

        g = graph.randomComplete(n, edgeW=(upper / 2, upper))
        assert graph.isMetric(g)
        return g

    def addNode(self, nodeWeight: int = 0) -> None:
        """Adds a node to the graph

        Args:
            nodeWeight: weight of the new node
        """

        self.numNodes += 1
        self.adjacenList.append([])
        self.nodeWeight.append(nodeWeight)

        # need to add slot for new node to edge weight matrix
        for weightList in self.edgeWeight:
            weightList.append(0.0)

        # add new row to edge weight matrix
        self.edgeWeight.append([0.0 for _ in range(self.numNodes)])

    def addEdge(self, startingNode: int, endingNode: int, weight: float = 0.0) -> None:
        """Adds a directed edge to the graph

        Args:
            startingNode: u
            endingNode: v
            weight: l(u, v) = weight
        """

        assert startingNode < self.numNodes
        assert endingNode < self.numNodes
        # add edge only once
        assert endingNode not in self.adjacenList[startingNode]

        self.adjacenList[startingNode].append(endingNode)
        self.edgeWeight[startingNode][endingNode] = weight

    def setNodeWeight(self, node: int, weight: int) -> None:
        """Set the weight of a node in a graph

        Args:
            node: the node whose weight is to be set
            weight: the new weight
        """

        assert node < self.numNodes
        self.nodeWeight[node] = weight

    def setEdgeWeight(
        self, startingNode: int, endingNode: int, edgeWeight: float
    ) -> None:
        """Set the weight of a directed edge in the graph

        Args:
            startingNode: u
            endingNode: v
            edgeWeight: l(u, v) = weight
        """

        assert startingNode < self.numNodes
        assert endingNode < self.numNodes
        assert edgeWeight > 0.0
        # add weight only if edge exists
        assert endingNode in self.adjacenList[startingNode]

        self.edgeWeight[startingNode][endingNode] = edgeWeight

    @staticmethod
    def isComplete(cls: graph) -> bool:
        """Checks if a graph is complete

        Args:
            cls: the graph to check

        Returns:
            bool: True if complete, false O.W.
        """

        for u in range(cls.numNodes):
            for v in range(cls.numNodes):
                if u != v and v not in cls.adjacenList[u]:
                    return False

        return True

    @staticmethod
    def isMetric(cls: graph) -> bool:
        """Checks if a graph is complete

        Checks if for all u, v, w in the graph that the following inequality is maintained
        l(u, w) + l(w, v) >= l(u, v)

        Args:
            cls: the graph to check

        Returns:
            bool: True if metric, false O.W.
        """

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
        """String representation of the graph"""

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
