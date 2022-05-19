"""
Driver code for testing functions
"""
import json

import matplotlib.pyplot as plt  # type: ignore

import algos
import benchmark
from graph import Graph, graph_dict


def main() -> None:
    # Generate and save Graphs and Partitions
    num_graphs: int = 10
    num_agents: int = 10
    num_nodes: int = 30
    upper_bound: float = 10.0

    graph_bank: list[Graph] = benchmark.generate_graph_bank(
        num_graphs, num_nodes, upper=upper_bound
    )

    partition_bank: list[list[set[int]]] = benchmark.generate_agent_partitions(
        graph_bank, num_agents
    )

    # Save banks to files
    # Use JSON files with parallel key-value pairs of int -> graph or partitions
    # Using seperate JSON files to make it easier to parse

    # graphs: dict[int, graph_dict] = {}
    # parts: dict[int, list[list[int]]] = {}
    # graph_dict_bank: list[graph_dict] = [Graph.dict_from_graph(g) for g in graph_bank]
    # # sets and frozensets are both unserializable
    # serializable_partition_bank: list[list[list[int]]] = [
    #     [list(s) for s in part] for part in partition_bank
    # ]
    # for i in range(num_graphs):
    #     graphs[i] = graph_dict_bank[i]
    #     parts[i] = serializable_partition_bank[i]

    # loc: str = "results/test_graph.json"
    # with open(loc, "w", encoding="utf-8") as outfile:
    #     json.dump(graphs, outfile)
    # loc = "results/test_part.json"
    # with open(loc, "w", encoding="utf-8") as outfile:
    #     json.dump(parts, outfile)

    # Read from generated bank files
    # graphs_from_file: list[Graph] = benchmark.graph_bank_from_file(
    #     "results/test_graph.json"
    # )
    # parts_from_file: list[list[set[int]]] = benchmark.agent_partitions_from_file(
    #     "results/test_part.json"
    # )

    # assert len(graphs_from_file) == len(parts_from_file)
    # for g, p in zip(graphs_from_file, parts_from_file):
    #     assert Graph.is_agent_partition(g, p)

    # Mass benchmark of graphs given bank
    benchmark.mass_benchmark(graph_bank, partition_bank)

    # Alpha Heuristic Benchmarking
    # greedy_alpha_dict: dict[float, float] = benchmark.alpha_heuristic_data(
    #     f=algos.greedy, graph_bank, partition_bank
    # )
    # for alpha, val in greedy_alpha_dict.items():
    #     print(f"{alpha:.2f}: {val}")
    # print()
    # plt.plot(
    #     list(greedy_alpha_dict.keys()), list(greedy_alpha_dict.values()),
    #     label="greedy"
    # )

    # nn_alpha_dict: dict[float, float] = benchmark.alpha_heuristic_data(
    #     f=algos.nearest_neighbor, graph_bank, partition_bank
    # )
    # for alpha, val in nn_alpha_dict.items():
    #     print(f"{alpha:.2f}: {val}")
    # print()
    # plt.plot(list(nn_alpha_dict.keys()), list(nn_alpha_dict.values()), label="nn")
    # plt.legend()
    # plt.show()

    # Messing with plotting
    # n = 100
    # k = 8
    # g: Graph = Graph.random_complete_metric(n)
    # nx_g = Graph.to_networkx(g)

    # part: list[set[int]] = Graph.create_agent_partition(g, k)
    # initial_assignment: list[list[int]] = benchmark.solve_partition(
    #     g, part, algos.nearest_neighbor
    # )
    # benchmark.draw_graph_with_partitions(nx_g, initial_assignment, "Initial")

    # output = algos.find_partition_with_heuristic(
    #         g,
    #         part,
    #         algos.nearest_neighbor,
    #         0.18
    #     )
    # final_assignment = benchmark.solve_partition(g, output, algos.nearest_neighbor)
    # benchmark.draw_graph_with_partitions(nx_g, final_assignment, "Nearest Neighbor")
    # plt.show()


if __name__ == "__main__":
    main()
