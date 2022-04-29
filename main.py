"""
Driver code for testing functions
"""
import matplotlib.pyplot as plt  # type: ignore
from typing_extensions import TypedDict

import algos
import benchmark
from graph import Graph

graph_dict = TypedDict(
    "graph_dict",
    {
        "numNodes": int,
        "node_weight": list[int],
        "edges": list[tuple[int, int, float]],
    },
)


def main() -> None:
    # Mass benchmark of graphs given parameters
    # num_graphs: int = 1
    # num_agents: int = 50
    # num_nodes: int = 100
    # upper_bound: float = 10.0

    # benchmark.mass_benchmark(
    #     count=num_graphs, k=num_agents, n=num_nodes, metric=True, upper=upper_bound
    # )

    # Alpha threshold benchmarking code
    # best_for_greedy: float = benchmark.alpha_heuristic_search(
    #     f=algos.greedy, count=num_graphs, k=num_agents, n=num_nodes, upper=upper_bound
    # )
    # print(f"{best_for_greedy = }")

    # best_for_nn: float = benchmark.alpha_heuristic_search(
    #     f=algos.nearest_neighbor,
    #     count=num_graphs,
    #     k=num_agents,
    #     n=num_nodes,
    #     upper=upper_bound,
    # )
    # print(f"{best_for_nn = }")

    # Messing with plotting
    n = 100
    k = 8
    g: Graph = Graph.random_complete_metric(n)
    nx_g = Graph.to_networkx(g)

    part: list[set[int]] = Graph.create_agent_partition(g, k)

    initial_assignment: list[list[int]] = benchmark.solve_partition(
        g, part, algos.nearest_neighbor
    )
    benchmark.draw_graph_with_partitions(nx_g, initial_assignment, "Initial")

    output = algos.find_partition_with_heuristic(g, part, algos.nearest_neighbor, 0.18)
    final_assignment = benchmark.solve_partition(g, output, algos.nearest_neighbor)
    benchmark.draw_graph_with_partitions(nx_g, final_assignment, "Nearest Neighbor")
    plt.show()


if __name__ == "__main__":
    main()
