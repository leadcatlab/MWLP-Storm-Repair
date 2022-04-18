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
    # Mass benchmark of graphs given parameters
    num_graphs: int = 10
    num_agents: int = 5
    num_nodes: int = 40
    upper_bound: float = 10.0

    benchmark.mass_benchmark(
        count=num_graphs, k=num_agents, n=num_nodes, metric=True, upper=upper_bound
    )

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
    # n = 100
    # k = 10
    # benchmark.line_plot(k, n)


if __name__ == "__main__":
    main()
