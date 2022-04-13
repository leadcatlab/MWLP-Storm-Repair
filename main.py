"""
Driver code for testing functions
"""
from typing_extensions import TypedDict

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
    num_graphs: int = 100
    num_agents: int = 20
    num_nodes: int = 200
    upper_bound: float = 10.0

    benchmark.mass_benchmark(
        count=num_graphs, k=num_agents, n=num_nodes, metric=True, upper=upper_bound
    )

    # best_for_greedy: float = benchmark.alpha_heuristic_search(
    #    f=algos.greedy, count=num_graphs, k=num_agents, n=num_nodes, upper=upper_bound
    # )
    # print(f"{best_for_greedy = }")
    #
    # best_for_nn: float = benchmark.alpha_heuristic_search(
    #    f=algos.nearest_neighbor, count=num_graphs, k=num_agents, n=num_nodes, upper=upper_bound
    # )
    # print(f"{best_for_nn = }")
    #
    # best_for_avg_with_greedy: float = benchmark.avg_alpha_heuristic_search(
    #    f=algos.greedy, count=num_graphs, k=num_agents, n=num_nodes, upper=upper_bound
    # )
    # print(f"{best_for_avg_with_greedy = }")

    # best_for_avg_with_nn: float = benchmark.avg_alpha_heuristic_search(
    #    f=algos.nearest_neighbor, count=num_graphs, k=num_agents, n=num_nodes, upper=upper_bound
    # )
    # print(f"{best_for_avg_with_nn = }")


if __name__ == "__main__":
    main()
