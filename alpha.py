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
    ############### Code used for finalizing Alpha benchmarking ################
    ############################################################################

    # Parameters for Graphs and Partitions
    num_graphs: int = 20
    num_nodes: int = 71
    metric = True
    upper: float = 10.0
    node_w: tuple[int, int] = (10, 20)
    num_agents: int = 10

    # Generating Graphs
    graph_bank: list[Graph] = benchmark.generate_graph_bank(
        count=num_graphs, n=num_nodes, metric=metric, upper=upper, node_w=node_w
    )

    # Generating Partitions
    partition_bank: list[list[set[int]]] = benchmark.generate_agent_partitions(
        graph_bank, num_agents
    )

    # Serializing Graphs and Partitions
    graphs: dict[int, graph_dict] = {}
    parts: dict[int, list[list[int]]] = {}
    graph_dict_bank: list[graph_dict] = [Graph.dict_from_graph(g) for g in graph_bank]
    # sets and frozensets are both unserializable
    serializable_partition_bank: list[list[list[int]]] = [
        [list(s) for s in part] for part in partition_bank
    ]
    for i in range(num_graphs):
        graphs[i] = graph_dict_bank[i]
        parts[i] = serializable_partition_bank[i]
    # Saving Graphs and Partitions to files
    loc: str = "results/alpha/final_graph.json"
    with open(loc, "w", encoding="utf-8") as outfile:
        json.dump(graphs, outfile)
    loc = "results/alpha/final_part.json"
    with open(loc, "w", encoding="utf-8") as outfile:
        json.dump(parts, outfile)

    # Read from generated bank files
    graphs_from_file: list[Graph] = benchmark.graph_bank_from_file(
        "results/alpha/final_graph.json"
    )
    parts_from_file: list[list[set[int]]] = benchmark.agent_partitions_from_file(
        "results/alpha/final_part.json"
    )
    assert len(graphs_from_file) == len(parts_from_file)
    for g, p in zip(graphs_from_file, parts_from_file):
        assert Graph.is_agent_partition(g, p)

    # Run Alpha Heuristic Benchmark
    greedy_alpha_dict: dict[float, float] = benchmark.alpha_heuristic_given(
        algos.greedy, graphs_from_file, parts_from_file
    )
    for alpha, val in greedy_alpha_dict.items():
        print(f"{alpha:.2f}: {val}")
    print()
    with open(
        "results/alpha/alpha_greedy_final_results.json", "w", encoding="utf-8"
    ) as outfile:
        json.dump(greedy_alpha_dict, outfile)

    nn_alpha_dict: dict[float, float] = benchmark.alpha_heuristic_given(
        algos.nearest_neighbor, graphs_from_file, parts_from_file
    )
    for alpha, val in nn_alpha_dict.items():
        print(f"{alpha:.2f}: {val}")
    print()
    with open(
        "results/alpha/alpha_nn_final_results.json", "w", encoding="utf-8"
    ) as outfile:
        json.dump(nn_alpha_dict, outfile)

    # Plot results
    plt.plot(
        list(greedy_alpha_dict.keys()), list(greedy_alpha_dict.values()), label="greedy"
    )
    plt.plot(list(nn_alpha_dict.keys()), list(nn_alpha_dict.values()), label="nn")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
