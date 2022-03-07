import timeit
from typing import Callable

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
    n: int = 40
    k: int = 8
    g = Graph.random_complete_metric(n, directed=False)
    partition: list[set[int]] = Graph.create_agent_partition(g, k)

    f: Callable[..., list[int]] = algos.greedy
    benchmark.print_heuristic_benchmark(g, partition, f)

    f = algos.nearest_neighbor
    benchmark.print_heuristic_benchmark(g, partition, f, print_before=False)

    f = algos.held_karp
    benchmark.print_heuristic_benchmark(g, partition, f, print_before=False)

    f = algos.brute_force_mwlp
    benchmark.print_heuristic_benchmark(g, partition, f, print_before=False)

    # start: float = timeit.default_timer()
    # _, parts = algos.partition_heuristic(g, f, k)
    # end: float = timeit.default_timer()
    # print("After full brute force solution (with optimal order)")
    # maximum = float("-inf")
    # for i in range(len(parts)):
    #     curr: float = algos.wlp(g, parts[i])
    #     maximum = max(maximum, curr)
    #     print(f"    Agent {i} = {curr}: {parts[i]}")
    # print(f"Maximum: {maximum}")
    # print(f"Time elapsed = {end - start}")


if __name__ == "__main__":
    main()
