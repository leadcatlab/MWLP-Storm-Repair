"""
Driver code for mass benchmark
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
    ########################### Mass Benchmarking ##############################
    ############################################################################

    # # Parameters for Graphs and Partitions
    # num_graphs: int = 100
    # num_nodes: int = 201  # 20 agents * 10 nodes per agent + start
    # metric = True
    # upper: float = 1.0  # Travel time between 0.5-1 hour
    # node_w: tuple[int, int] = (1, 1500)
    # num_agents: int = 20

    # print("Generating graphs")
    # graph_bank: list[Graph] = benchmark.generate_graph_bank(
    #     count=num_graphs, n=num_nodes, metric=metric, upper=upper, node_w=node_w
    # )

    # print("Adding repair times")
    # for g in graph_bank:
    #     # Ranges from "Predicting Outage Restoration ..."
    #     for u, v in product(range(num_nodes), range(num_nodes)):
    #         if u != v:
    #             pop: int = g.node_weight[v]
    #             if pop <= 10:
    #                 g.edge_weight[u][v] += random.uniform(2, 4)
    #             elif pop <= 100:
    #                 g.edge_weight[u][v] += random.uniform(2, 6)
    #             elif pop <= 1000:
    #                 g.edge_weight[u][v] += random.uniform(3, 8)
    #             else:
    #                 g.edge_weight[u][v] += random.uniform(5, 10)

    # print("Generating initial partitions")
    # partition_bank: list[list[set[int]]] = benchmark.generate_agent_partitions(
    #     graph_bank, num_agents
    # )

    # # Mass benchmark of graphs given bank
    # # Need to edit the ranges
    # #   If metric: do (upper / 2, upper)
    # results: list[DefaultDict[Any, Any]] = benchmark.mass_benchmark(
    #     graphs_from_file, parts_from_file, (0.5, 1.0)
    # )

    # # Write to files
    # names: list[str] = [
    #     "maximums",
    #     "wait_times",
    #     "times",
    #     "minimums",
    #     "sums",
    #     "ranges",
    #     "averages",
    #     "bests",
    # ]
    # for res, name in zip(results, names):
    #     with open(
    #         f"results/mass_benchmark/{name}.json", "w", encoding="utf-8"
    #     ) as outfile:
    #         json.dump(res, outfile)

    ############################################################################
    ######################## Plotting Visited Targets ##########################
    ############################################################################

    # Parameters for Graphs and Partitions
    num_agents: int = 5
    num_nodes: int = 51  # 20 agents * 10 nodes per agent + start
    upper: float = 1.0  # Travel time between 0.5-1 hour
    node_w: tuple[int, int] = (1, 1500)

    g = Graph.random_complete_metric(n=num_nodes, upper=upper, node_w=node_w)

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

    partition: list[set[int]] = Graph.create_agent_partition(g, num_agents)

    assignments: list[list[list[int]]] = []
    names: list[str] = []
    colors: list[str] = []

    part: list[set[int]] = [set(s) for s in partition]
    paths: list[list[int]] = benchmark.solve_partition(g, part, algos.random_order)
    assignments.append(paths)
    names.append("Random")
    colors.append("darkviolet")

    paths = algos.greedy_assignment(g, num_agents)
    assignments.append(paths)
    names.append("Greedy Assignment")
    colors.append("lightsteelblue")

    paths = algos.nearest_neighbor_assignment(g, num_agents)
    assignments.append(paths)
    names.append("Nearest Neighbor Assignment")
    colors.append("aqua")

    dist_range: float = 1.0 - 0.5
    paths = algos.greedy_random_assignment(g, num_agents, 0.5 + (dist_range * 0.25))
    assignments.append(paths)
    names.append("Greedy + Random (25%) Assignment")
    colors.append("royalblue")
    
    paths = algos.greedy_random_assignment(g, num_agents, 0.5 + (dist_range * 0.50))
    assignments.append(paths)
    names.append("Greedy + Random (50%) Assignment")
    colors.append("blue")
    
    paths = algos.greedy_random_assignment(g, num_agents, 0.5 + (dist_range * 0.75))
    assignments.append(paths)
    names.append("Greedy + Random (75%) Assignment")
    colors.append("mediumslateblue")

    part = algos.find_partition_with_heuristic(g, partition, algos.greedy, 0.13)
    paths = benchmark.solve_partition(g, part, algos.greedy)
    assignments.append(paths)
    names.append("Transfers and Swaps Greedy")
    colors.append("limegreen")

    part = algos.find_partition_with_heuristic(g, partition, algos.nearest_neighbor, 0.13)
    paths = benchmark.solve_partition(g, part, algos.nearest_neighbor)
    assignments.append(paths)
    names.append("Transfers and Swaps Nearest Neighbor")
    colors.append("darkgreen")
    
    benchmark.line_plot(g, assignments, names, colors, x_range=(0, 100))

if __name__ == "__main__":
    main()
