"""
Driver code for mass benchmark
"""
import json
import math
import random
from collections import defaultdict
from itertools import product
from typing import Any, DefaultDict

from matplotlib.patches import Patch
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

    # Parameters for Graphs and Partitions
    num_graphs: int = 100
    num_nodes: int = 201  # 20 agents * 10 nodes per agent + start
    metric = True
    upper: float = 1.0  # Travel time between 0.5-1 hour
    node_w: tuple[int, int] = (1, 1500)
    num_agents: int = 20

    print("Generating graphs")
    graph_bank: list[Graph] = benchmark.generate_graph_bank(
        count=num_graphs, n=num_nodes, metric=metric, upper=upper, node_w=node_w
    )

    print("Seeing start location weight to 0")
    for g in graph_bank:
        g.node_weight[0] = 0

    print("Adding repair times")
    for g in graph_bank:
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

    print("Generating initial partitions")
    partition_bank: list[list[set[int]]] = benchmark.generate_agent_partitions(
        graph_bank, num_agents
    )

    # Mass benchmark of graphs given bank
    # Need to edit the ranges
    #   If metric: do (upper / 2, upper)
    results: list[DefaultDict[Any, Any]] = benchmark.mass_benchmark(
        graph_bank, partition_bank, (0.5, 1.0)
    )

    # Write to files
    names: list[str] = [
        "maximums",
        "wait_times",
        "times",
        "minimums",
        "sums",
        "ranges",
        "averages",
        "bests",
    ]
    for res, name in zip(results, names):
        with open(
            f"results/mass_benchmark/{name}.json", "w", encoding="utf-8"
        ) as outfile:
            json.dump(res, outfile)

    # Bar Plot for sum of weighted latencies
    with open("results/mass_benchmark/sums.json", encoding="utf-8") as file:
        sums: dict[str, list[float]] = json.load(file)

    count: int = len(sums["Greedy Assignment"])
    results: list[str] = [
        "Greedy Assignment",
        "Nearest Neighbor Assignment",
        "Greedy + Random (25%) Assignment",
        "Transfers and Swaps Greedy",
        "Transfers and Swaps Nearest Neighbor",
    ]
    names: list[str] = [
        "Greedy",
        "Nearest Neighbor",
        "Greedy & Random",
        "T&S Greedy",
        "T&S Nearest Neighbor",
    ]
    values: list[float] = [sum(sums[name]) / count for name in results]
    colors: list[str] = ["royalblue", "aqua", "blue", "limegreen", "darkgreen"]

    fig, ax = plt.subplots(figsize=(6, 6))
    patches = [Patch(color=c, label=k) for c, k in zip(colors, names)]
    plt.legend(
        title="Key",
        labels=names,
        handles=patches,
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
    )
    frame1 = plt.gca()
    frame1.axes.xaxis.set_ticklabels([])
    plt.title("Sum of Weighted Latencies")
    bars = plt.bar(names, values, color=colors)
    plt.gcf().axes[0].yaxis.get_major_formatter().set_scientific(False)
    ax.bar_label(bars, padding=3, fmt="%d")
    fig.savefig("results/mass_benchmark/total_work", bbox_inches="tight")

    # Bar Plot for average wait times
    with open("results/mass_benchmark/wait_times.json", encoding="utf-8") as file:
        wait: dict[str, list[float]] = json.load(file)

    count = len(wait["Greedy Assignment"])
    results = [
        "Greedy Assignment",
        "Nearest Neighbor Assignment",
        "Greedy + Random (25%) Assignment",
        "Transfers and Swaps Greedy",
        "Transfers and Swaps Nearest Neighbor",
    ]
    names = [
        "Greedy",
        "Nearest Neighbor",
        "Greedy & Random",
        "T&S Greedy",
        "T&S Nearest Neighbor",
    ]
    values = [sum(wait[name]) / count for name in results]
    colors = ["lightsteelblue", "aqua", "blue", "limegreen", "darkgreen"]

    fig, ax = plt.subplots(figsize=(6, 6))
    patches = [Patch(color=c, label=k) for c, k in zip(colors, names)]
    plt.legend(
        title="Key",
        labels=names,
        handles=patches,
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
    )
    frame1 = plt.gca()
    frame1.axes.xaxis.set_ticklabels([])
    plt.title("Average Wait Time")
    bars = plt.bar(names, values, color=colors)
    plt.gcf().axes[0].yaxis.get_major_formatter().set_scientific(False)
    ax.bar_label(bars, padding=3, fmt="%d")
    fig.savefig("results/mass_benchmark/wait_time", bbox_inches="tight")

    # Bar Plot for range of work
    with open("results/mass_benchmark/ranges.json", encoding="utf-8") as file:
        ranges: dict[str, list[float]] = json.load(file)

    count = len(ranges["Greedy Assignment"])
    results = [
        "Greedy Assignment",
        "Nearest Neighbor Assignment",
        "Greedy + Random (25%) Assignment",
        "Transfers and Swaps Greedy",
        "Transfers and Swaps Nearest Neighbor",
    ]
    names = [
        "Greedy",
        "Nearest Neighbor",
        "Greedy & Random",
        "T&S Greedy",
        "T&S Nearest Neighbor",
    ]
    values = [sum(ranges[name]) / count for name in results]
    colors = ["lightsteelblue", "aqua", "blue", "limegreen", "darkgreen"]

    fig, ax = plt.subplots(figsize=(6, 6))
    patches = [Patch(color=c, label=k) for c, k in zip(colors, names)]
    plt.legend(
        title="Key",
        labels=names,
        handles=patches,
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
    )
    frame1 = plt.gca()
    frame1.axes.xaxis.set_ticklabels([])
    plt.title("Range of Weighted Latencies")
    bars = plt.bar(names, values, color=colors)
    plt.gcf().axes[0].yaxis.get_major_formatter().set_scientific(False)
    ax.bar_label(bars, padding=3, fmt="%d")
    fig.savefig("results/mass_benchmark/ranges", bbox_inches="tight")

    ############################################################################
    ######################## Plotting Visited Targets ##########################
    ############################################################################

    # Parameters for Graph and Partition
    num_agents: int = 20
    num_nodes: int = 201  # 20 agents * 10 nodes per agent + start
    upper: float = 1.0  # Travel time between 0.5-1 hour
    node_w: tuple[int, int] = (1, 1500)

    g = Graph.random_complete_metric(n=num_nodes, upper=upper, node_w=node_w)

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

    benchmark.line_plot(g, assignments, names, colors, x_range=(0, 100))


if __name__ == "__main__":
    main()
