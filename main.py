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
    # n: int = 30
    # k: int = 5
    # g = Graph.random_complete_metric(n, directed=False)
    # partition: list[set[int]] = Graph.create_agent_partition(g, k)

    # f: Callable[..., list[int]] = algos.greedy
    # benchmark.print_heuristic_benchmark(g, partition, f, print_before=False)

    # f = algos.nearest_neighbor
    # benchmark.print_heuristic_benchmark(g, partition, f, print_before=False)

    # f = algos.held_karp
    # benchmark.print_heuristic_benchmark(g, partition, f, print_before=False)

    # f = algos.brute_force_mwlp
    # benchmark.print_heuristic_benchmark(g, partition, f, print_before=False)

    # benchmark.print_avg_benchmark(g, partition, print_before=False)

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

    # n: int = 8
    # g = Graph.random_complete(n, edge_w=(1.0, 1.0), node_w=(0, 10), directed=False)

    # # shortcut???
    # pairs: list[tuple[int, int]] = algos.choose2(n)
    # shortcut: float = 0.0
    # for node in range(1, n):
    #     for i, j in pairs:
    #         shortcut += (
    #             g.node_weight[node] * g.edge_weight[i][j] * math.factorial(n - 2)
    #         )
    # print(shortcut)

    # # Brute Force
    # nodes: list[int] = list(range(1, n))
    # brute: float = 0.0
    # for order in permutations(nodes):
    #     brute += algos.wlp(g, [0] + list(order))
    # print(brute)

    # n: int = 100
    # k: int = 10
    # g = Graph.random_complete_metric(n, directed=False)
    # partition: list[set[int]] = Graph.create_agent_partition(g, k)
    # res = algos.transfers_and_swaps_mwlp_with_average(g, partition)
    # for i in range(len(res)):
    #     print(f"Agent {i}: {res[i]}")

    n: int = 15
    k: int = 2
    g = Graph.random_complete(n, edge_w=(10.0, 25.0), directed=False)
    res = algos.uconn_strat_1(g, k)
    for i, part in enumerate(res):
        print(f"Agent {i}: {part}:")
        sub, _, _ = Graph.subgraph(g, part)
        print(f"    {algos.wlp(g, algos.brute_force_mwlp(sub))}")
    print()
    res = algos.uconn_strat_2(g, k, 15.0)
    for i, part in enumerate(res):
        print(f"Agent {i}: {part}:")
        sub, _, _ = Graph.subgraph(g, part)
        print(f"    {algos.wlp(g, algos.brute_force_mwlp(sub))}")
    print()

if __name__ == "__main__":
    main()
