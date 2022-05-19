"""
Driver code for testing functions
"""
import json

import matplotlib.pyplot as plt  # type: ignore

import algos
import benchmark
from graph import Graph, graph_dict


def main() -> None:
    # Generate Graphs and Partitions
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

    ############################################################################
    ############### Code used for finalizing Alpha benchmarking ################
    ############################################################################
    # # Parameters for Graphs and Partitions
    # num_graphs: int = 20
    # num_nodes: int = 80
    # edge_w: tuple[float, float] = (5.0, 10.0)
    # metric = False
    # node_w: tuple[int, int] = (10, 100)
    # num_agents: int = 10
    # repair_time: float = 2.0

    # # Generating Graphs
    # graph_bank: list[Graph] = benchmark.generate_graph_bank(
    #     count=num_graphs,
    #     n=num_nodes,
    #     edge_w=edge_w,
    #     metric=metric,
    #     node_w=node_w
    # )

    # # Adding Repair Times
    # for i, g in enumerate(graph_bank):
    #     g.add_repair_time(repair_time)
    #     graph_bank[i] = g
    # # Generating Partitions
    # partition_bank: list[list[set[int]]] = benchmark.generate_agent_partitions(
    #     graph_bank, num_agents
    # )

    # # Serializing Graphs and Partitions
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
    # # Saving Graphs and Partitions to files
    # loc: str = "results/alpha/final_graph.json"
    # with open(loc, "w", encoding="utf-8") as outfile:
    #     json.dump(graphs, outfile)
    # loc = "results/alpha/final_part.json"
    # with open(loc, "w", encoding="utf-8") as outfile:
    #     json.dump(parts, outfile)

    # # Read from generated bank files
    # graphs_from_file: list[Graph] = benchmark.graph_bank_from_file(
    #     "results/alpha/final_graph.json"
    # )
    # parts_from_file: list[list[set[int]]] = benchmark.agent_partitions_from_file(
    #     "results/alpha/final_part.json"
    # )
    # assert len(graphs_from_file) == len(parts_from_file)
    # for g, p in zip(graphs_from_file, parts_from_file):
    #     assert Graph.is_agent_partition(g, p)

    # # Run Alpha Heuristic Benchmark
    # greedy_alpha_dict: dict[float, float] = benchmark.alpha_heuristic_given(
    #     algos.greedy, graphs_from_file, parts_from_file
    # )
    # for alpha, val in greedy_alpha_dict.items():
    #     print(f"{alpha:.2f}: {val}")
    # print()
    # with open(
    #     "results/alpha/alpha_greedy_final_results.json", "w", encoding="utf-8"
    # ) as outfile:
    #     json.dump(greedy_alpha_dict, outfile)

    # nn_alpha_dict: dict[float, float] = benchmark.alpha_heuristic_given(
    #     algos.nearest_neighbor, graphs_from_file, parts_from_file
    # )
    # for alpha, val in nn_alpha_dict.items():
    #     print(f"{alpha:.2f}: {val}")
    # print()
    # with open(
    #     "results/alpha/alpha_nn_final_results.json", "w", encoding="utf-8"
    # ) as outfile:
    #     json.dump(nn_alpha_dict, outfile)

    # # Finish plotting
    # plt.plot(
    #     list(greedy_alpha_dict.keys()), list(greedy_alpha_dict.values()),
    #     label="greedy"
    # )
    # plt.plot(list(nn_alpha_dict.keys()), list(nn_alpha_dict.values()), label="nn")
    # plt.legend()
    # plt.show()


if __name__ == "__main__":
    main()
