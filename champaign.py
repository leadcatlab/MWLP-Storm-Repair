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

    # # Thanks Pranay
    # ox.config(log_console=True, use_cache=True)
    # place = "Champaign, Illinois, USA"
    # # gdf = ox.geocode_to_gdf(place)
    # # area = ox.projection.project_gdf(gdf).unary_union.area
    # # 'drive_service' := drivable and service roads both
    # G = ox.graph_from_place(place, network_type="drive_service", simplify=True)
    # G = ox.distance.add_edge_lengths(G, precision=5)

    # # From "Predicting Outage Restoration..."
    # # Agent speed was 25 mph
    # kph: float = 25.0 * 1.609344
    # print(f"Setting all travel speeds to {kph} kph")
    # for u, v, key in G.edges(keys=True):
    #     G[u][v][key]["speed_kph"] = kph
    # G = ox.add_edge_travel_times(G, precision=5)

    # # Remove unreachable /  empty nodes
    # print("Removing unreachable nodes")
    # components = list(nx.strongly_connected_components(G))
    # for item in components:
    #     if len(item) == 0 or len(item) == 1:
    #         G.remove_node(item.pop())

    # order: int = G.order()
    # print(f"{order} nodes")

    # # During repairs we do not care about one way roads
    # print("Turning G into undirected graph")
    # G = ox.utils_graph.get_undirected(G)

    # # Add population to the nearest points
    # pop_data = pd.read_csv("cus_blockdata.csv", index_col=0)
    # pop_points = list(pop_data.to_records(index=False))

    # def dist(x1: float, y1: float, x2: float, y2: float) -> float:
    #     x_diff: float = x1 - x2
    #     y_diff: float = y1 - y2
    #     return math.sqrt(x_diff**2 + y_diff**2)

    # print("Initializing population of each node to 0")
    # for i in G.nodes():
    #     G.nodes[i]["pop"] = 0

    # print("Adding populations")
    # for population, lat, long in pop_points:
    #     if population > 0:
    #         # Find the closest node in G to lat, long
    #         closest: int = min(
    #             G.nodes(),
    #             key=lambda i: dist(long, lat, G.nodes[i]["x"], G.nodes[i]["y"]),
    #         )

    #         # print(f"Adding {population} to node G.nodes[{closest}]['pop']")
    #         G.nodes[closest]["pop"] += population

    # print("Writing graphML")
    # ox.save_graphml(G, "champaign.graphml")

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
    num_nodes: int = 251
    g = Graph(num_nodes)

    # Initializing a bunch of empty nodes and edges is faster than calling add_edge
    print("Initializing adjacency lists")
    for i in range(num_nodes):  # make complete
        g.adjacen_list[i] = list(range(num_nodes))
    print("Initializing edge weights")
    for i in range(num_nodes):
        g.edge_weight[i] = [-1.0 for _ in range(num_nodes)]

    print(f"Choosing {num_nodes} damaged nodes")
    damaged: list[int] = list(populated)
    random.shuffle(damaged)
    damaged = damaged[:num_nodes]
    print("Adding node weights to g")
    for i in range(num_nodes):
        g.node_weight[i] = G.nodes[damaged[i]]["pop"]
    g.node_weight[0] = 0

    print("Finding shortest path travel times in hours")
    for u in range(num_nodes):
        for v in range(u + 1, num_nodes)
            u_prime, v_prime = damaged[u], damaged[v]
            time = nx.shortest_path_length(
                G, u_prime, v_prime, weight="travel_time"
            )
            g.edge_weight[u][v] = g.edge_weight[v][u] = time / (60 * 60)

    print("Adding repair times")
    # Ranges from "Predicting Outage Restoration ..."
    for v in range(num_nodes):
        pop: int = g.node_weight[v]
        if pop <= 10:
            repair_time: float = random.uniform(2, 4)
        elif pop <= 100:
            repair_time = random.uniform(2, 6)
        elif pop <= 1000:
            repair_time = random.uniform(3, 8)
        else:
            repair_time = random.uniform(5, 10)
        for u in range(num_nodes):
            if u != v:
                g.edge_weight[u][v] += repair_time

    num_agents: int = 25
    print(f"Creating partitions for {num_agents} agents")
    partition: list[set[int]] = Graph.create_agent_partition(g, num_agents)

    assignments: list[list[list[int]]] = []
    names: list[str] = []
    colors: list[str] = []

    paths = algos.greedy_assignment(g, num_agents)
    assignments.append(paths)
    names.append("Greedy")
    colors.append("royalblue")

    paths = algos.nearest_neighbor_assignment(g, num_agents)
    assignments.append(paths)
    names.append("Nearest Neighbor")
    colors.append("aqua")

    dist_range: float = 1.0 - 0.5
    paths = algos.greedy_random_assignment(g, num_agents, 0.5 + (dist_range * 0.25))
    assignments.append(paths)
    names.append("Greedy & Random")
    colors.append("blue")

    part = algos.find_partition_with_heuristic(g, partition, algos.greedy, 0.13)
    paths = benchmark.solve_partition(g, part, algos.greedy)
    assignments.append(paths)
    names.append("T&S Greedy")
    colors.append("limegreen")

    part = algos.find_partition_with_heuristic(
        g, partition, algos.nearest_neighbor, 0.13
    )
    paths = benchmark.solve_partition(g, part, algos.nearest_neighbor)
    assignments.append(paths)
    names.append("T&S Nearest Neighbor")
    colors.append("darkgreen")

    benchmark.line_plot(g, assignments, names, colors, x_range=(0, 100), loc='champaign_unvisited.png')


if __name__ == "__main__":
    main()
