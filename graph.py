from __future__ import annotations

import random

from typing_extensions import TypedDict

# TODO: docstring
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
        """Inits Graph with n nodes

        Args:
            n: number of nodes (by default graph is empty)

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
        """Alternate constructor using dictionary

        Args:
            gd: graph_dict containing the needed information for a graph

        Returns:
            Graph with the same information as gd
        """

        if gd["num_nodes"] < 0:
            raise ValueError(f"Number of nodes is negative: {gd['num_nodes']}")
        g = Graph(gd["num_nodes"])

        if len(gd["node_weight"]) != g.num_nodes:
            raise ValueError(
                "node_weight list is incorrect length: "
                + f"({gd['node_weight']} vs {g.num_nodes})"
            )

        if min(gd["node_weight"]) < 0:
            raise ValueError(
                f"node_weight list contains negative values: {gd['node_weight']}"
            )

        g.node_weight = gd["node_weight"]

        for (start_node, end_node, node_weight) in gd["edges"]:
            if start_node >= g.num_nodes:
                raise ValueError(
                    f"Starting node {start_node} is out of range [0, {g.num_nodes - 1}]"
                )
            if end_node >= g.num_nodes:
                raise ValueError(
                    f"Ending node {end_node} is out of range [0, {g.num_nodes - 1}]"
                )

            g.add_edge(start_node, end_node, node_weight)

        return g

    @staticmethod
    def dict_from_graph(g: Graph) -> graph_dict:
        """create graph_dict from Graph

        Args:
            g: input graph

        Returns:
            graph_dict gd such that Graph.from_dict(gd) == g
        """

        num_nodes: int = g.num_nodes
        node_weight: list[int] = g.node_weight

        edges: list[tuple[int, int, float]] = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j and j in g.adjacen_list[i]:
                    edges.append((i, j, g.edge_weight[i][j]))

        gd: graph_dict = {
            "num_nodes": num_nodes,
            "edges": edges,
            "node_weight": node_weight,
        }

        return gd

    @staticmethod
    def random_complete(
        n: int,
        edge_w: tuple[float, float] = (0, 1),
        node_w: tuple[int, int] = (0, 100),
        directed: bool = True,
    ) -> Graph:
        """Create a randomly generated complete weighted undirected graph

        Creates a complete undirected graph with randomly weighted edges and nodes

        Args:
            n: number of nodes
            edge_w: the interval that edge weights can be in
            node_w: the interval that node weights can be in

        Returns:
            Graph
        """

        g = Graph(n)

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

        g.node_weight = [random.randint(node_w[0], node_w[1]) for _ in range(n)]

        if directed:
            for i in range(n):
                for j in range(n):
                    if i != j:
                        weight: float = random.uniform(edge_w[0], edge_w[1])
                        g.add_edge(i, j, weight)
        else:
            for i in range(n):
                for j in range(i + 1, n):
                    weight = random.uniform(edge_w[0], edge_w[1])
                    g.add_edge(i, j, weight)
                    g.add_edge(j, i, weight)

        assert Graph.is_complete(g)
        return g

    @staticmethod
    def random_complete_metric(
        n: int,
        upper: float = 1,
        node_w: tuple[int, int] = (1, 100),
        directed: bool = True,
    ) -> Graph:
        """Create a randomly generated complete weighted undirected metric graph

        Creates a complete undirected graph with randomly weighted edges and nodes
        Edges satisfy the inequality l(u, w) + l(w, v) >= l(u, v) for all nodes u, v, w

        Args:
            n: number of nodes
            upper: used to create the edge weight interval (upper / 2, upper)
            node_w: the interval that node weights can be in

        Returns:
            Graph
        """

        g = Graph.random_complete(
            n, edge_w=(upper / 2, upper), node_w=node_w, directed=directed
        )
        assert Graph.is_metric(g)
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
            if n >= g.num_nodes:
                raise ValueError(
                    f"Passed in {nodes = } contains nodes not in passed in graph"
                )

        new_node_list = list(range(len(nodes)))

        # mapping for original -> subgraph
        ots: dict[int, int] = dict(zip(nodes, new_node_list))
        # mapping for subgraph -> original
        sto: dict[int, int] = dict(zip(new_node_list, nodes))

        subg = Graph(len(nodes))
        for n in new_node_list:
            subg.set_node_weight(n, g.node_weight[sto[n]])

        for i in nodes:
            for j in nodes:
                if i != j and j in g.adjacen_list[i]:
                    subg.add_edge(ots[i], ots[j], g.edge_weight[i][j])

        return subg, sto, ots

    def add_node(self, node_weight: int = 0) -> None:
        """Adds a node to the graph

        Args:
            node_weight: weight of the new node
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
        """Adds a directed edge to the graph

        Args:
            start
            end
            weight: l(start, end) = weight
        """

        # ensure nodes exist
        if start >= self.num_nodes:
            raise ValueError(
                f"Starting node {start} is out of range [0, {self.num_nodes - 1}]"
            )
        if end >= self.num_nodes:
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
        """Set the weight of a node in a graph

        Args:
            node: the node whose weight is to be set
            weight: the new weight
        """

        if node >= self.num_nodes:
            raise ValueError(f"Node {node} is out of range [0, {self.num_nodes - 1}]")

        if node_weight < 0:
            raise ValueError(f"Passed node_weight is negative: {node_weight}")

        self.node_weight[node] = node_weight

    def set_edge_weight(
        self, start_node: int, end_node: int, edge_weight: float
    ) -> None:
        """Set the weight of a directed edge in the graph

        Args:
            start_node: u
            end_node: v
            edge_weight: l(u, v) = weight
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
        """Checks if a graph is undirected

        Args:
            cls: the graph to check

        Returns:
            bool: True if undirected, false O.W.
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
        """Checks if a graph is complete

        Args:
            cls: the graph to check

        Returns:
            bool: True if complete, false O.W.
        """

        for u in range(g.num_nodes):
            for v in range(g.num_nodes):
                if u != v and v not in g.adjacen_list[u]:
                    return False

        return True

    @staticmethod
    def is_metric(g: Graph) -> bool:
        """Checks if a graph is metric

        Checks triangle inequality for all nodes  u, v, w in the graph
        l(u, w) + l(w, v) >= l(u, v)

        Args:
            cls: the graph to check

        Returns:
            bool: True if metric, false O.W.
        """

        for u in range(g.num_nodes):
            for v in range(g.num_nodes):
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
    def is_partition(g: Graph, partition: list[set[int]]) -> bool:
        """Determines if a partition of graph nodes is valid"""

        nodes: list[bool] = [False] * g.num_nodes
        for subset in partition:
            if len(subset) == 0:
                return False
            for n in subset:
                if n >= g.num_nodes or n < 0:
                    return False
                if nodes[n] is True:
                    return False
                nodes[n] = True

        for n in range(g.num_nodes):
            if nodes[n] is False:
                return False

        return True

    @staticmethod
    def create_agent_partition(g: Graph, k: int) -> list[set[int]]:
        """Creates agent partition for algorithms

        Agent partition:
        ---------------
        k agents, each starting from start note 0
        Distribute notes [1, n] to the k agents
        Each node other than 0 is used only one
        Each agent gets partitions that contain more nodes than just {0}
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
        """Determines if a partition of graph nodes is a valid agent partition"""

        if len(partition) == 0:
            return False

        nodes: list[bool] = [False] * g.num_nodes
        for subset in partition:
            if len(subset) == 0 or subset == {0}:
                return False
            if 0 not in subset:
                return False
            for n in subset:
                if n >= g.num_nodes or n < 0:
                    return False
                if n != 0 and nodes[n] is True:
                    return False
                nodes[n] = True

        for n in range(g.num_nodes):
            if nodes[n] is False:
                return False

        return True

    def __str__(self) -> str:  # pragma: no cover
        """String representation of the graph"""

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
