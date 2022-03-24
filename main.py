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
    benchmark.mass_benchmark(count=50, k=5, n=30, metric=True, upper=10.0)

    # best_for_greedy: float = benchmark.alpha_heuristic_search(
    #    f=algos.greedy, count=10, k=2, n=10, metric=True, upper=10.0
    # )
    # print(best_for_greedy)
    #
    # best_for_nn: float = benchmark.alpha_heuristic_search(
    #    f=algos.nearest_neighbor, count=10, k=2, n=10, metric=True, upper=10.0
    # )
    # print(best_for_nn)
    #
    # best_for_avg: float = benchmark.avg_alpha_heuristic_search(
    #    count=10, k=2, n=10, metric=True, upper=10.0
    # )
    # print(best_for_avg)


if __name__ == "__main__":
    main()
