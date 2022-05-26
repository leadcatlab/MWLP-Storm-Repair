"""
Driver code for testing functions
"""
import json
import math
import random
from collections import defaultdict
from itertools import product
from typing import Any, DefaultDict

import matplotlib.pyplot as plt  # type: ignore
import networkx as nx  # type: ignore
import numpy as np
import osmnx as ox  # type: ignore
import pandas as pd  # type: ignore

import algos
import benchmark
from graph import Graph, graph_dict


class Bcolors:
    """
    Helper class for adding colors to prints
    https://svn.blender.org/svnroot/bf-blender/trunk/blender/build_files/scons/tools/Bcolors.py
    """

    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    CLEAR_LAST_LINE = (
        "\033[A                                                             \033[A"
    )


def main() -> None:
    ############################################################################
    ########################### Champaign Testing ##############################
    ############################################################################

    # Thanks Pranay
    ox.config(log_console=True, use_cache=True)
    place = "Champaign, Illinois, USA"
    # gdf = ox.geocode_to_gdf(place)
    # area = ox.projection.project_gdf(gdf).unary_union.area
    # 'drive_service' := drivable and service roads both
    G = ox.graph_from_place(place, network_type="drive_service", simplify=True)
    G = ox.distance.add_edge_lengths(G, precision=5)

    # From "Predicting Outage Restoration..."
    # Agent speed was 25 mph
    kph: float = 25.0 * 1.609344
    print(f"Setting all travel speeds to {kph} kph")
    for u, v, key in G.edges(keys=True):
        G[u][v][key]["speed_kph"] = kph
    G = ox.add_edge_travel_times(G, precision=5)

    # Remove unreachable /  empty nodes
    print("Removing unreachable nodes")
    components = list(nx.strongly_connected_components(G))
    for item in components:
        if len(item) == 0 or len(item) == 1:
            G.remove_node(item.pop())

    order: int = G.order()
    print(f"{order} nodes")

    # During repairs we do not care about one way roads
    print("Turning G into undirected graph")
    G = ox.utils_graph.get_undirected(G)

    # Add population to the nearest points
    pop_data = pd.read_csv("cus_blockdata.csv", index_col=0)
    pop_points = list(pop_data.to_records(index=False))

    def dist(x1: float, y1: float, x2: float, y2: float) -> float:
        x_diff: float = x1 - x2
        y_diff: float = y1 - y2
        return math.sqrt(x_diff**2 + y_diff**2)

    print("Initializing population of each node to 0")
    for i in G.nodes():
        G.nodes[i]["pop"] = 0

    print("Adding populations")
    for population, lat, long in pop_points:
        if population > 0:
            # Find the closest node in G to lat, long
            closest: int = min(
                G.nodes(),
                key=lambda i: dist(long, lat, G.nodes[i]["x"], G.nodes[i]["y"]),
            )

            # print(f"Adding {population} to node G.nodes[{closest}]['pop']")
            G.nodes[closest]["pop"] += population

    print("Writing graphML")
    ox.save_graphml(G, "champaign.graphml")

    print("Loading graphml")
    G = ox.load_graphml("champaign.graphml")

    print("Fixing population numbers")
    for node in G.nodes():
        G.nodes[node]["pop"] = int(G.nodes[node]["pop"])

    # Find populated nodes
    node_list: list[int] = [int(node) for node in G.nodes()]
    populated: list[int] = list(
        filter(lambda node: G.nodes[node]["pop"] > 0, node_list)
    )

    # Choose random nodes to be damaged
    num_nodes: int = 25

    # Construct smaller graph of just damaged nodes out of G
    count: int = 10
    print(f"Constructing graph bank of {count} graphs")
    graph_bank: list[Graph] = []

    # Saving travel times to prevent recalculating
    travel_times: dict[tuple[int, int], float] = {}

    for i in range(count):
        print(f"Creating graph {i}")
        g = Graph(num_nodes)

        # Initializing a bunch of empty nodes and edges is faster than calling add_edge
        print("\tInitializing adjacency lists")
        for i in range(num_nodes):  # make complete
            g.adjacen_list[i] = list(range(num_nodes))
        print("\tInitializing edge weights")
        for i in range(num_nodes):
            g.edge_weight[i] = [-1.0 for _ in range(num_nodes)]

        print(f"\tChoosing {num_nodes} damaged nodes")
        damaged: list[int] = list(populated)
        random.shuffle(damaged)
        damaged = damaged[:num_nodes]
        print("\tAdding node weights to g")
        for i in range(num_nodes):
            g.node_weight[i] = G.nodes[damaged[i]]["pop"]

        print("\tFinding shortest path travel times in hours")
        for u, v in product(range(num_nodes), range(num_nodes)):
            if u != v:
                if (u, v) in travel_times:
                    time = travel_times[(u, v)]
                    g.edge_weight[u][v] = time
                else:
                    u_prime, v_prime = damaged[u], damaged[v]
                    time = nx.shortest_path_length(
                        G, u_prime, v_prime, weight="travel_time"
                    )
                    g.edge_weight[u][v] = time / (60 * 60)
                    travel_times[(u, v)] = time / (60 * 60)

        print("\tAdding repair times")
        # Ranges from "Predicting Outage Restoration ..."
        for u, v in product(range(num_nodes), range(num_nodes)):
            if u != v:
                pop: int = g.node_weight[v]
                if pop <= 10:
                    g.edge_weight[u][v] += random.uniform(2, 4)
                elif pop <= 100:
                    g.edge_weight[u][v] += random.uniform(2, 6)
                elif pop <= 1000:
                    g.edge_weight[u][v] += random.uniform(3, 8)
                else:
                    g.edge_weight[u][v] += random.uniform(5, 10)
        graph_bank.append(g)

        print(Bcolors.CLEAR_LAST_LINE)
        print(Bcolors.CLEAR_LAST_LINE)
        print(Bcolors.CLEAR_LAST_LINE)
        print(Bcolors.CLEAR_LAST_LINE)
        print(Bcolors.CLEAR_LAST_LINE)
        print(Bcolors.CLEAR_LAST_LINE)
        print(Bcolors.CLEAR_LAST_LINE)

    num_agents: int = 5
    print(f"Creating partitions for {num_agents} agents")
    partition_bank: list[list[set[int]]] = benchmark.generate_agent_partitions(
        graph_bank, num_agents
    )

    print("Serializing graphs and partitions")
    graphs: dict[int, graph_dict] = {}
    parts: dict[int, list[list[int]]] = {}
    graph_dict_bank: list[graph_dict] = [Graph.dict_from_graph(g) for g in graph_bank]

    # sets and frozensets are both unserializable
    serializable_partition_bank: list[list[list[int]]] = [
        [list(s) for s in part] for part in partition_bank
    ]

    for i in range(count):
        graphs[i] = graph_dict_bank[i]
        parts[i] = serializable_partition_bank[i]

    # Saving Graphs and Partitions to files
    loc: str = "results/champaign/champaign_graphs.json"
    with open(loc, "w", encoding="utf-8") as outfile:
        json.dump(graphs, outfile)
    loc = "results/champaign/champaign_parts.json"
    with open(loc, "w", encoding="utf-8") as outfile:
        json.dump(parts, outfile)

    # Reading graphs and partitons from files
    print("Reading graphs and partitions")
    graphs_from_file: list[Graph] = benchmark.graph_bank_from_file(
        "results/champaign/champaign_graphs.json"
    )
    parts_from_file: list[list[set[int]]] = benchmark.agent_partitions_from_file(
        "results/champaign/champaign_parts.json"
    )
    assert len(graphs_from_file) == len(parts_from_file)
    for g, p in zip(graphs_from_file, parts_from_file):
        assert Graph.is_agent_partition(g, p)

    n: int = graphs_from_file[0].num_nodes
    k: int = len(parts_from_file[0])
    print(f"Number of graphs: {len(graphs_from_file)}")
    print(f"Number of nodes: {n}")
    print(f"Number of agents: {k}")
    print()

    g: Graph = graphs_from_file[0]
    print("Representative information from one graph")
    min_time = min(
        g.edge_weight[u][v] for u, v in product(range(n), range(n)) if u != v
    )
    max_time = max(
        g.edge_weight[u][v] for u, v in product(range(n), range(n)) if u != v
    )
    kph: float = 25.0 * 1.609344
    min_dist, max_dist = min_time * kph, max_time * kph
    print(f"Minimum Distance (km) = {min_dist}")
    print(f"Maximum Distance (km) = {max_dist}")
    print(f"Minimum Travel Time (hours) = {min_time}")
    print(f"Maximum Travel Time (hours) = {max_time}")
    print(f"Maximum Population = {max(g.node_weight)}")
    print(f"Minimum Population = {min(g.node_weight)}")
    print()

    print("Beginning mass benchmark")
    benchmark.mass_benchmark(graphs_from_file, parts_from_file, (min_time, max_time))


if __name__ == "__main__":
    main()
