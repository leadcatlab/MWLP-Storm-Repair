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


class Graph:
    """Directed graph class with weighted edges and nodes

    Attributes:
        numNodes: number of nodes from [0, n)
        adjacenList: directed adjacency list
        edgeWeight: 2D matrix representing l(i, j)
        nodeWeight: list of all weights w(i)

    """

    def __init__(self, n: int = 0):
        """Inits Graph with n nodes

        Args:
            n: number of nodes (by default graph is empty)

        """

        if n < 0:
            raise ValueError(f"Number of nodes passed in is negative: {n}")

        self.numNodes = n
        self.adjacenList: list[list[int]] = [[] for _ in range(n)]
        self.edgeWeight: list[list[float]] = [
            [-1.0 for _ in range(n)] for _ in range(n)
        ]
        self.nodeWeight: list[int] = [0 for _ in range(n)]

    @staticmethod
    def fromDict(dictionary: graphDict) -> Graph:
        """Alternate constructor using dictionary

        Args:
            graphDict: Dictionary containing the needed information for a graph

        Returns:
            Graph
        """

        if dictionary["numNodes"] < 0:
            raise ValueError(f"Number of nodes is negative: {dictionary['numNodes']}")
        g = Graph(dictionary["numNodes"])

        if len(dictionary["nodeWeight"]) != g.numNodes:
            raise ValueError(
                f"nodeWeight list is incorrect length ({dictionary['nodeWeight']} vs {g.numNodes})"
            )

        if min(dictionary["nodeWeight"]) < 0:
            raise ValueError(
                f"nodeWeight list has nodes of a negative weight: {dictionary['nodeWeight']}"
            )

        g.nodeWeight = dictionary["nodeWeight"]

        for (startingNode, endingNode, nodeWeight) in dictionary["edges"]:
            if startingNode >= g.numNodes:
                raise ValueError(
                    f"Starting node {startingNode} is out of range [0, {g.numNodes - 1}]"
                )
            if endingNode >= g.numNodes:
                raise ValueError(
                    f"Ending node {endingNode} is out of range [0, {g.numNodes - 1}]"
                )

            g.addEdge(startingNode, endingNode, nodeWeight)

        return g

    @staticmethod
    def dictFromGraph(g: Graph) -> graphDict:
        numNodes: int = g.numNodes
        nodeWeight: list[int] = g.nodeWeight

        edges: list[tuple[int, int, float]] = []
        for i in range(numNodes):
            for j in range(numNodes):
                if i != j and j in g.adjacenList[i]:
                    edges.append((i, j, g.edgeWeight[i][j]))

        gd: graphDict = {
            "numNodes": numNodes,
            "edges": edges,
            "nodeWeight": nodeWeight,
        }

        return gd

    @staticmethod
    def randomComplete(
        n: int, edgeW: tuple[float, float] = (0, 1), nodeW: tuple[int, int] = (0, 100)
    ) -> Graph:
        """Create a randomly generated complete weighted undirected graph

        Creates a complete undirected graph with randomly weighted edges and nodes

        Args:
            n: number of nodes
            edgeW: the interval that edge weights can be in
            nodeW: the interval that node weights can be in

        Returns:
            Graph
        """

        g = Graph(n)

        if nodeW[0] < 0 or nodeW[1] < 0:
            raise ValueError(
                f"Passed node weight range contains negative values: {nodeW}"
            )
        if nodeW[1] < nodeW[0]:
            raise ValueError(f"Passed node weight range is in wrong order: {nodeW}")
        if edgeW[0] < 0.0 or nodeW[1] < 0.0:
            raise ValueError(
                f"Passed edge weight range contains negative values: {edgeW}"
            )
        if edgeW[1] < edgeW[0]:
            raise ValueError(f"Passed edge weight range is in wrong order: {edgeW}")

        g.nodeWeight = [random.randint(nodeW[0], nodeW[1]) for _ in range(n)]

        for i in range(n):
            for j in range(i + 1, n):
                weight: float = random.uniform(edgeW[0], edgeW[1])
                g.addEdge(i, j, weight)
                g.addEdge(j, i, weight)

        assert Graph.isComplete(g)
        return g

    @staticmethod
    def randomCompleteMetric(
        n: int, upper: float = 1, nodeW: tuple[int, int] = (1, 100)
    ) -> Graph:
        """Create a randomly generated complete weighted undirected metric graph

        Creates a complete undirected graph with randomly weighted edges and nodes
        Edges satisfy the inequality l(u, w) + l(w, v) >= l(u, v) for all nodes u, v, w

        Args:
            n: number of nodes
            upper: the upper bound of edge weights used to create the interval (upper / 2, upper)
            nodeW: the interval that node weights can be in

        Returns:
            Graph
        """

        g = Graph.randomComplete(n, edgeW=(upper / 2, upper), nodeW=nodeW)
        assert Graph.isMetric(g)
        return g

    @staticmethod
    def subgraph(
        g: Graph, nodes: list[int]
    ) -> tuple[Graph, dict[int, int], dict[int, int]]:
        """Create a subgraph based on a graph and passed list of nodes

        Creates a subgraph of the original graph plus a set of dictionaries
        which relates the nodes from one graph with another

        Args:
            g: passed in graph to based subgraph off of
            nodes: list of nodes to use in original graph for subgraph

        Returns:
            subgraph and two dictionaries:
            sto is a dictionary mapping nodes from subgraph to original
            ots is a dictionary mapping nodes from original to subgraph
        """

        for n in nodes:
            if n >= g.numNodes:
                raise ValueError(
                    f"Passed in {nodes = } contains nodes not in passed in graph"
                )

        newNodeList = list(range(len(nodes)))

        # mapping for original -> subgraph
        ots: dict[int, int] = {o: s for (o, s) in zip(nodes, newNodeList)}
        # mapping for subgraph -> original
        sto: dict[int, int] = {s: o for (o, s) in zip(nodes, newNodeList)}

        subg = Graph(len(nodes))
        for n in newNodeList:
            subg.setNodeWeight(n, g.nodeWeight[sto[n]])

        for i in nodes:
            for j in nodes:
                if i != j and j in g.adjacenList[i]:
                    subg.addEdge(ots[i], ots[j], g.edgeWeight[i][j])

        return subg, sto, ots

    def addNode(self, nodeWeight: int = 0) -> None:
        """Adds a node to the graph

        Args:
            nodeWeight: weight of the new node
        """

        if nodeWeight < 0:
            raise ValueError(f"Passed nodeWeight is negative: {nodeWeight}")

        self.numNodes += 1
        self.adjacenList.append([])
        self.nodeWeight.append(nodeWeight)

        # need to add slot for new node to edge weight matrix
        for weightList in self.edgeWeight:
            weightList.append(-1.0)

        # add new row to edge weight matrix
        self.edgeWeight.append([-1.0 for _ in range(self.numNodes)])

    def addEdge(self, start: int, end: int, weight: float = 0.0) -> None:
        """Adds a directed edge to the graph

        Args:
            start
            end
            weight: l(start, end) = weight
        """

        # ensure nodes exist
        if start >= self.numNodes:
            raise ValueError(
                f"Starting node {start} is out of range [0, {self.numNodes - 1}]"
            )
        if end >= self.numNodes:
            raise ValueError(
                f"Ending node {end} is out of range [0, {self.numNodes - 1}]"
            )

        # add edge only once
        if end in self.adjacenList[start]:
            raise ValueError(
                f"Edge from {start} to {end} already exists with weight {self.edgeWeight[start][end]}"
            )

        self.adjacenList[start].append(end)
        self.edgeWeight[start][end] = weight

    def setNodeWeight(self, node: int, nodeWeight: int) -> None:
        """Set the weight of a node in a graph

        Args:
            node: the node whose weight is to be set
            weight: the new weight
        """

        if node >= self.numNodes:
            raise ValueError(f"Node {node} is out of range [0, {self.numNodes - 1}]")

        if nodeWeight < 0:
            raise ValueError(f"Passed nodeWeight is negative: {nodeWeight}")

        self.nodeWeight[node] = nodeWeight

    def setEdgeWeight(
        self, startingNode: int, endingNode: int, edgeWeight: float
    ) -> None:
        """Set the weight of a directed edge in the graph

        Args:
            startingNode: u
            endingNode: v
            edgeWeight: l(u, v) = weight
        """

        # ensure nodes exist
        if startingNode >= self.numNodes:
            raise ValueError(
                f"Starting node {startingNode} is out of range [0, {self.numNodes - 1}]"
            )
        if endingNode >= self.numNodes:
            raise ValueError(
                f"Ending node {endingNode} is out of range [0, {self.numNodes - 1}]"
            )

        if edgeWeight < 0.0:
            raise ValueError(f"Edge has negative weight {edgeWeight}")
        # add weight only if edge exists
        if endingNode not in self.adjacenList[startingNode]:
            raise ValueError(f"{endingNode} is not a neighbor of {startingNode}")

        self.edgeWeight[startingNode][endingNode] = edgeWeight

    @staticmethod
    def isComplete(cls: Graph) -> bool:
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
    def isMetric(cls: Graph) -> bool:
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

    def __str__(self) -> str:  # pragma: no cover
        """String representation of the graph"""

        toPrint: str = ""
        for i in range(len(self.adjacenList)):
            currentList = self.adjacenList[i]

            nodes: str = ""

            for j in currentList:
                nodes += (
                    " " * 4
                    + str(j)
                    + " with distance "
                    + str(self.edgeWeight[i][j])
                    + "\n"
                )

            toPrint += "Node " + str(i) + " is connected to: \n" + nodes + "\n"

        toPrint += "\n"

        for i in range(len(self.nodeWeight)):
            toPrint += (
                "The weight of node " + str(i) + " is " + str(self.nodeWeight[i]) + "\n"
            )

        return toPrint
