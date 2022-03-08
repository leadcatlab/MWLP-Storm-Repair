import math
import timeit
from itertools import permutations
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
    n: int = 32
    k: int = 4
    g = Graph.random_complete_metric(n, directed=False)
    partition: list[set[int]] = Graph.create_agent_partition(g, k)

    f: Callable[..., list[int]] = algos.greedy
    benchmark.print_heuristic_benchmark(g, partition, f, print_before=True)

    f = algos.nearest_neighbor
    benchmark.print_heuristic_benchmark(g, partition, f, print_before=False)

    # f = algos.held_karp
    # benchmark.print_heuristic_benchmark(g, partition, f, print_before=False)

    # f = algos.brute_force_mwlp
    # benchmark.print_heuristic_benchmark(g, partition, f, print_before=False)

    benchmark.print_avg_benchmark(g, partition, print_before=False)

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

    # n: int = 10
    # g = Graph.random_complete(n, edge_w=(1.0, 1.0), node_w=(0, 10), directed=False)

    # # shortcut???
    # pairs = algos.choose2(n)
    # shortcut = 0.0
    # for node in range(1, n):
    #     for i, j in pairs:
    #         shortcut += g.node_weight[node] * g.edge_weight[i][j] * (n - 2)
    # print(shortcut * math.factorial(n - 3))

    # # Brute Force
    # nodes: list[int] = list(range(1, n))
    # brute: float = 0.0
    # for order in permutations(nodes):
    #     brute += algos.wlp(g, [0] + list(order))
    # print(brute)


if __name__ == "__main__":
    main()
