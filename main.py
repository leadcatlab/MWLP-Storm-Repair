"""
Driver code for testing functions
"""
from typing_extensions import TypedDict

from graph import Graph
import matplotlib.pyplot as plt
import algos
import benchmark

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
    # num_graphs: int = 100
    # num_agents: int = 20
    # num_nodes: int = 200
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

    # best_for_avg_with_greedy: float = benchmark.avg_alpha_heuristic_search(
    #     f=algos.greedy, count=num_graphs, k=num_agents, n=num_nodes, upper=upper_bound
    # )
    # print(f"{best_for_avg_with_greedy = }")

    # best_for_avg_with_nn: float = benchmark.avg_alpha_heuristic_search(
    #     f=algos.nearest_neighbor,
    #     count=num_graphs,
    #     k=num_agents,
    #     n=num_nodes,
    #     upper=upper_bound,
    # )
    # print(f"{best_for_avg_with_nn = }")

    # Messing with plotting
    g = Graph.random_complete_metric(50)
    part: list[set[int]] = Graph.create_agent_partition(g, 5)
    paths: list[list[int]] = benchmark.solve_partition(g, part, algos.nearest_neighbor)
    f = algos.generate_partition_path_function(g, paths)
    x = [0.05 * i for i in range(1000)]
    y = [f(i) for i in x]
    plt.plot(x, y)
    plt.show()

if __name__ == "__main__":
    main()
