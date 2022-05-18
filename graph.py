"""
Graph class with integer nodes, integer node weights, float edge weights
"""
from __future__ import annotations

import json
import random
from itertools import product
from typing import Collection, no_type_check

import networkx as nx  # type: ignore
from typing_extensions import TypedDict

graph_dict = TypedDict(
    "graph_dict",
    {
        # number of nodes
        "num_nodes": int,
        # list of correct length of node_weights
        "node_weight": list[int],
        # list of tuples of form (start_node, end_node, node_weight)
        "edges": list[tuple[int, int, float]],
    },
)


class Graph:
    """Directed graph class with weighted edges and nodes

    Attributes:
        num_nodes: number of nodes from [0, n)
        adjacen_list: directed adjacency list
        edge_weight: 2D matrix representing l(i, j)
        node_weight: list of all weights w(i)

    """

    def __init__(self, n: int = 0):
        """
        Creates a graph with n nodes

        Parameters
        ----------
        n: int
            Number of nodes in the graph
            Default: 0
            Assertions:
                Assertion 1
                Assertion 2

        Returns
        -------
        Graph
            Graph with n nodes and no edges or node weights

        """

        if n < 0:
            raise ValueError(f"Number of nodes passed in is negative: {n}")

        self.num_nodes = n
        self.adjacen_list: list[list[int]] = [[] for _ in range(n)]
        self.edge_weight: list[list[float]] = [
            [-1.0 for _ in range(n)] for _ in range(n)
        ]
        self.node_weight: list[int] = [0 for _ in range(n)]

    @staticmethod
    def from_dict(gd: graph_dict) -> Graph:
        """
        Alternate constructor using dictionary

        Parameters
        ----------
        gd: graph_dict
            Contains the needed information for a graph
            Assertions:
                Number of nodes must be >= 0
                List of node weights must be correct length
                Node weights must be all >= 0
                Edges must be between nodes that exist

        Returns
        -------
        Graph
            Graph with nodes and edges as described by the dictionary

        """

        if gd["num_nodes"] < 0:
            raise ValueError(f"Number of nodes is negative: {gd['num_nodes']}")
        g = Graph(gd["num_nodes"])

        if len(gd["node_weight"]) != g.num_nodes:
            raise ValueError(
                "node_weight list is incorrect length: "
                + f"{gd['node_weight']} vs {g.num_nodes}"
            )

        if min(gd["node_weight"]) < 0:
            raise ValueError(
                f"node_weight list contains negative values: {gd['node_weight']}"
            )
        g.node_weight = gd["node_weight"]

        for start_node, end_node, node_weight in gd["edges"]:
            if start_node >= g.num_nodes or g.num_nodes < 0:
                raise ValueError(
                    f"Starting node {start_node} is out of range [0, {g.num_nodes - 1}]"
                )
            if end_node >= g.num_nodes or g.num_nodes < 0:
                raise ValueError(
                    f"Ending node {end_node} is out of range [0, {g.num_nodes - 1}]"
                )
            g.add_edge(start_node, end_node, node_weight)

        return g

    @staticmethod
    def dict_from_graph(g: Graph) -> graph_dict:
        """
        Create graph_dict from passed Graph

        Parameters
        ----------
        g: Graph
            Graph to be encoded into a dictionary

        Returns
        -------
        graph_dict
            Dictionary that can be transformed back into passed graph

        """

        num_nodes: int = g.num_nodes
        node_weight: list[int] = g.node_weight

        edges: list[tuple[int, int, float]] = []
        for (i, j) in product(range(num_nodes), range(num_nodes)):
            if i != j and j in g.adjacen_list[i]:
                edges.append((i, j, g.edge_weight[i][j]))

        gd: graph_dict = {
            "num_nodes": num_nodes,
            "edges": edges,
            "node_weight": node_weight,
        }

        return gd

    @staticmethod
    def from_file(loc: str) -> Graph:
        """
        Alternate constructor using graph_dict from json

        Parameters
        ----------
        loc: str
            file location of dictionary json

        Returns
        -------
        Graph
            Graph with nodes and edges as described by the json file

        """

        with open(loc, encoding="utf-8") as gd_json:
            gd: graph_dict = json.load(gd_json)
            return Graph.from_dict(gd)

    @staticmethod
    def to_file(g: Graph, loc: str) -> None:
        """
        Write graph to a file

        Parameters
        ----------
        g: Graph
            input graph

        loc: str
            file location of dictionary json

        """

        with open(loc, "w", encoding="utf-8") as outfile:
            gd: graph_dict = Graph.dict_from_graph(g)
            json.dump(gd, outfile)

    @staticmethod
    @no_type_check
    def to_networkx(g: Graph):
        """
        Creates a complete graph with randomly weighted edges and nodes

        Runtime: (if applicable)

        Parameters
        ----------
        g: Graph
            Input graph

        Returns
        -------
        nx.DiGraph()
            networkx graph equivalent to passed input graph

        """
        n: int = g.num_nodes
        nx_g = nx.DiGraph()
        for i in range(n):
            nx_g.add_node(i)
        for i, w in enumerate(g.node_weight):
            nx_g.nodes[i]["weight"] = w
        for u, v in product(range(n), range(n)):
            if u != v:
                nx_g.add_edge(u, v, weight=g.edge_weight[u][v])

        return nx_g

    @staticmethod
    def random_complete(
        n: int,
        edge_w: tuple[float, float] = (0.0, 1.0),
        node_w: tuple[int, int] = (0, 100),
        directed: bool = False,
    ) -> Graph:
        """
        Creates a complete graph with randomly weighted edges and nodes

        Runtime: (if applicable)

        Parameters
        ----------
        n: int
            Number of nodes
            Assertions:
                Number of nodes must be  >= 0

        edge_w: tuple[float, float]
            Range of edge weights
            Default: (0.0, 1.0)
            Assertions:
                Edge weights must be non-negative
                Edge weights must be passed in the correct order

        node_w: tuple[int, int]
            Range of node weights
            Default: (0, 100)
            Assertions:
                Int weights must be non-negative
                Int weights must be passed in the correct order

        directed: bool
            Set if created graph is directed or undirected
            Default: False

        Returns
        -------
        Graph
            Randomly created graph based on the passed parameters

        """

        if n < 0:
            raise ValueError(f"Number of nodes passed in is negative: {n}")

        if node_w[0] < 0 or node_w[1] < 0:
            raise ValueError(
                f"Passed node weight range contains negative values: {node_w}"
            )
        if node_w[1] < node_w[0]:
            raise ValueError(f"Passed node weight range is in wrong order: {node_w}")

        if edge_w[0] < 0.0 or node_w[1] < 0.0:
            raise ValueError(
                f"Passed edge weight range contains negative values: {edge_w}"
            )
        if edge_w[1] < edge_w[0]:
            raise ValueError(f"Passed edge weight range is in wrong order: {edge_w}")

        g = Graph(n)

        g.node_weight = [random.randint(node_w[0], node_w[1]) for _ in range(n)]

        if directed:
            for i, j in product(range(n), range(n)):
                if i != j:
                    weight: float = random.uniform(edge_w[0], edge_w[1])
                    g.add_edge(i, j, weight)
        else:
            for i in range(n):
                for j in range(i + 1, n):
                    weight = random.uniform(edge_w[0], edge_w[1])
                    g.add_edge(i, j, weight)
                    g.add_edge(j, i, weight)

        return g

    @staticmethod
    def random_complete_metric(
        n: int,
        upper: float = 1.0,
        node_w: tuple[int, int] = (0, 100),
        directed: bool = False,
    ) -> Graph:
        """
        Creates a complete metric graph with randomly weighted edges and nodes
        The metric property is guarenteed by the following:
            Suppose the edge weights range between (x, 2x)
            We have that x + x = 2x <= 2x for all x
            Thus the triangle inequality is always satisfied

        Parameters
        ----------
        n: int
            Number of nodes
            Assertions:
                Number of nodes must be  >= 0

        upper: float
            Upper bound of edge weight
            Default: 1.0
            Assertions:
                Must be non-negative

        node_w: tuple[int, int]
            Range of node weights
            Default: (0, 100)
            Assertions:
                Int weights must be non-negative
                Int weights must be passed in the correct order

        directed: bool
            Set if created graph is directed or undirected
            Default: False

        Returns
        -------
        Graph
            Randomly created metric graph based on the passed parameters

        """

        if n < 0:
            raise ValueError(f"Number of nodes passed in is negative: {n}")

        if node_w[0] < 0 or node_w[1] < 0:
            raise ValueError(
                f"Passed node weight range contains negative values: {node_w}"
            )
        if node_w[1] < node_w[0]:
            raise ValueError(f"Passed node weight range is in wrong order: {node_w}")

        if upper < 0.0:
            raise ValueError(f"Passed edge weight upper bound is negative: {upper}")

        return Graph.random_complete(
            n, edge_w=(upper / 2, upper), node_w=node_w, directed=directed
        )

    def add_repair_time(self, amt: float) -> None:
        """
        Adds a desired fixed repair time to all edges

        Parameters
        ----------
        amt: float
            Repair time to be added
            Assertions:
                Must be non-negative

        """

        if amt < 0.0:
            raise ValueError(f"Passed repair time is negative: {amt}")

        n: int = self.num_nodes
        for u, v in product(range(n), range(n)):
            if u != v:
                self.edge_weight[u][v] += amt

    @staticmethod
    def subgraph(
        g: Graph, nodes: Collection[int]
    ) -> tuple[Graph, dict[int, int], dict[int, int]]:
        """
        Creates a subgraph of the original graph plus a set of dictionaries
        which relates the nodes between the original graph and subgraph

        Parameters
        ----------
        g: Graph
            Original graph

        nodes: Collection[int]
            Nodes to create a subgraph out of
            Assertions:
                Nodes in collection must be in the graph

        Returns
        -------
        Graph
            Subgraph
        dict[int, int]
            Dictionary mapping nodes from subgraph to original
        dict[int, int]
            ictionary mapping nodes from original to subgraph

        """

        for n in nodes:
            if n >= g.num_nodes or n < 0:
                raise ValueError(
                    f"Passed in {nodes = } contains nodes not in passed g ({n = })"
                )

        new_node_list = list(range(len(nodes)))

        # mapping for original -> subgraph
        ots: dict[int, int] = dict(zip(nodes, new_node_list))
        # mapping for subgraph -> original
        sto: dict[int, int] = {v: k for k, v in ots.items()}

        sub_g = Graph(len(nodes))
        for n in new_node_list:
            sub_g.set_node_weight(n, g.node_weight[sto[n]])

        for i, j in product(nodes, nodes):
            if i != j and j in g.adjacen_list[i]:
                sub_g.add_edge(ots[i], ots[j], g.edge_weight[i][j])

        return sub_g, sto, ots

    def add_node(self, node_weight: int = 0) -> None:
        """
        Add a node to a graph

        Parameters
        ----------
        node_weight: int
            Node weight of new node
            Default: 0
            Assertions:
                Node weight must be >= 0

        """

        if node_weight < 0:
            raise ValueError(f"Passed node_weight is negative: {node_weight}")

        self.num_nodes += 1
        self.adjacen_list.append([])
        self.node_weight.append(node_weight)

        # need to add slot for new node to edge weight matrix
        for weight_list in self.edge_weight:
            weight_list.append(-1.0)

        # add new row to edge weight matrix
        self.edge_weight.append([-1.0 for _ in range(self.num_nodes)])

    def add_edge(self, start: int, end: int, weight: float = 0.0) -> None:
        """
        Add an edge to a graph

        Parameters
        ----------
        start: int
            Start node of edge
            Assertions:
                Node must be in the graph

        end: int
            End node of edge
            Assertions:
                Node must be in the graph

        Assertion:
            Edge cannot already exist

        weight: float
            Weight of edge start -> end
            Default: 0.0

        """

        # ensure nodes exist
        if start >= self.num_nodes or start < 0:
            raise ValueError(
                f"Starting node {start} is out of range [0, {self.num_nodes - 1}]"
            )
        if end >= self.num_nodes or end < 0:
            raise ValueError(
                f"Ending node {end} is out of range [0, {self.num_nodes - 1}]"
            )

        # add edge only once
        if end in self.adjacen_list[start]:
            raise ValueError(
                f"Edge from {start} to {end} already exists "
                + "with weight {self.edge_weight[start][end]}"
            )

        self.adjacen_list[start].append(end)
        self.edge_weight[start][end] = weight

    def set_node_weight(self, node: int, node_weight: int) -> None:
        """
        Change node weight of a desired node

        Parameters
        ----------
        node: int
            Node whose weight is being changed
            Assertions:
                Node must be in the graph

        node_weight: int
            New node weight of node
            Assertions:
                Node weight must be >= 0

        """

        if node >= self.num_nodes or node < 0:
            raise ValueError(f"Node {node} is out of range [0, {self.num_nodes - 1}]")

        if node_weight < 0:
            raise ValueError(f"Passed node_weight is negative: {node_weight}")

        self.node_weight[node] = node_weight

    def set_edge_weight(
        self, start_node: int, end_node: int, edge_weight: float
    ) -> None:
        """
        Change edge weight of an edge in the graph

        Parameters
        ----------
        start_node: int
            Start node of edge
            Assertions:
                Node must be in the graph

        end_node: int
            End node of edge
            Assertions:
                Node must be in the graph

        Assertion:
            Edge must exist

        edge_weight: float
            New weight of edge start -> end

        """

        # ensure nodes exist
        if start_node >= self.num_nodes:
            raise ValueError(
                f"Starting node {start_node} is out of range [0, {self.num_nodes - 1}]"
            )
        if end_node >= self.num_nodes:
            raise ValueError(
                f"Ending node {end_node} is out of range [0, {self.num_nodes - 1}]"
            )

        if edge_weight < 0.0:
            raise ValueError(f"Edge has negative weight {edge_weight}")
        # add weight only if edge exists
        if end_node not in self.adjacen_list[start_node]:
            raise ValueError(f"{end_node} is not a neighbor of {start_node}")

        self.edge_weight[start_node][end_node] = edge_weight

    @staticmethod
    def is_undirected(g: Graph) -> bool:
        """
        Determine if a graph is undirected

        Parameters
        ----------
        g: Graph
            Input graph

        Returns
        -------
        bool
            True if undirected, False if directed

        """

        for u in range(g.num_nodes):
            for v in range(u + 1, g.num_nodes):
                if v in g.adjacen_list[u] and u in g.adjacen_list[v]:
                    if g.edge_weight[u][v] != g.edge_weight[v][u]:
                        return False
                elif v in g.adjacen_list[u] and u not in g.adjacen_list[v]:
                    return False
                elif v not in g.adjacen_list[u] and u in g.adjacen_list[v]:
                    return False
        return True

    @staticmethod
    def is_complete(g: Graph) -> bool:
        """
        Determine if a graph is complete

        Parameters
        ----------
        g: Graph
            Input graph

        Returns
        -------
        bool
            True if complete, False otherwise

        """

        for u, v in product(range(g.num_nodes), range(g.num_nodes)):
            if u != v and v not in g.adjacen_list[u]:
                return False

        return True

    @staticmethod
    def is_metric(g: Graph) -> bool:
        """
        Determine if a graph maintains the triangle inequality

        Parameters
        ----------
        g: Graph
            Input graph

        Returns
        -------
        bool
            True if metric, False otherwise

        """

        for u, v in product(range(g.num_nodes), range(g.num_nodes)):
            if u != v and v in g.adjacen_list[u]:
                for w in range(g.num_nodes):
                    if w in g.adjacen_list[u] and v in g.adjacen_list[w]:
                        if (
                            g.edge_weight[u][w] + g.edge_weight[w][v]
                            < g.edge_weight[u][v]
                        ):
                            return False

        return True

    @staticmethod
    def create_agent_partition(g: Graph, k: int) -> list[set[int]]:
        """
        Creates a random agent partition for k agents over a graph

        Parameters
        ----------
        g: Graph
            Input graph

        k: int
            Number of agents

        Returns
        -------
        list[set[int]]
            Random agent partition

        """

        n: int = g.num_nodes
        barriers: list[bool] = [False] * (n - 1)
        nodes = list(range(1, n))
        random.shuffle(nodes)

        for _ in range(k - 1):
            found: bool = False
            while not found:
                pos: int = random.randint(1, n - 2)
                if barriers[pos] is False:
                    barriers[pos] = True
                    found = True

        # all agent contain 0 as start
        partition: list[set[int]] = [{0} for _ in range(k)]
        start, end = 0, 1
        part: int = 0
        while start < n - 1:
            while end < n - 1 and barriers[end] is not True:
                end += 1
            if end >= n - 1:
                partition[part].update(nodes[start:])
            else:
                partition[part].update(nodes[start:end])
            start, end = end, end + 1
            part += 1

        return partition

    @staticmethod
    def is_agent_partition(g: Graph, partition: list[set[int]]) -> bool:
        """
        Determines if a partition of graph nodes is a valid agent partition

        Agent partition:
            k agents, each starting from start note 0
            Distribute notes [1, n] to the k agents
            Each node other than 0 is used only one time

        Parameters
        ----------
        g: Graph
            Input graph

        partition: list[set[int]]
            Input partition to check

        Returns
        -------
        bool
            True if valid, False otherwise

        """

        if len(partition) == 0:
            return False

        nodes: list[bool] = [False] * g.num_nodes
        for subset in partition:
            if 0 not in subset:
                return False
            for n in subset:
                if n >= g.num_nodes or n < 0:
                    return False
                if n != 0 and nodes[n] is True:
                    return False
                nodes[n] = True

        if False in nodes:
            return False

        return True

    def __str__(self) -> str:  # pragma: no cover
        """
        String representation of the graph
        """

        to_print: str = ""
        for start, current_list in enumerate(self.adjacen_list):
            nodes: str = ""

            for end in current_list:
                nodes += (
                    " " * 4
                    + str(end)
                    + " with distance "
                    + str(self.edge_weight[start][end])
                    + "\n"
                )

            to_print += "Node " + str(start) + " is connected to: \n" + nodes + "\n"

        to_print += "\n"

        for node in range(self.num_nodes):
            to_print += (
                "The weight of node "
                + str(node)
                + " is "
                + str(self.node_weight[node])
                + "\n"
            )

        return to_print
