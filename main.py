from graph import Graph
import algos
import benchmark
from typing import Callable
from typing_extensions import TypedDict
import random

graph_dict = TypedDict(
    "graph_dict",
    {
        "numNodes": int,
        "node_weight": list[int],
        "edges": list[tuple[int, int, float]],
    },
)


def main() -> None:
    n: int = 40
    k: int = 10
    g = Graph.random_complete_metric(n, directed=False)
    partition: list[set[int]] = Graph.create_agent_partition(g, k)

    f: Callable[..., list[int]] = algos.greedy
    benchmark.print_heuristic_benchmark(g, partition, f)

    f = algos.nearest_neighbor
    benchmark.print_heuristic_benchmark(g, partition, f)

    f = algos.held_karp
    benchmark.print_heuristic_benchmark(g, partition, f)

    f = algos.brute_force_mwlp
    benchmark.print_heuristic_benchmark(g, partition, f)


if __name__ == "__main__":
    main()
