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
    n: int = 60
    k: int = 10
    g = Graph.random_complete_metric(n, upper=10.0, directed=False)
    partition: list[set[int]] = Graph.create_agent_partition(g, k)

    # print("Initial:")
    # res = benchmark.solve_partition(g, partition)
    # benchmark.benchmark_partition(g, res)

    print("UConn Greedy:")
    start = timeit.default_timer()
    greedy_res = algos.uconn_strat_1(g, k)
    end = timeit.default_timer()
    benchmark.benchmark_partition(g, greedy_res)
    print(f"Time elapsed = {end - start}\n")

    print("UConn Greedy + Random (dist: 2.5):")
    start = timeit.default_timer()
    greedy_and_rand_res = algos.uconn_strat_2(g, k, 2.5)
    end = timeit.default_timer()
    benchmark.benchmark_partition(g, greedy_and_rand_res)
    print(f"Time elapsed = {end - start}\n")

    print("UConn Greedy + Random (dist: 5.0):")
    start = timeit.default_timer()
    greedy_and_rand_res = algos.uconn_strat_2(g, k, 5.0)
    end = timeit.default_timer()
    benchmark.benchmark_partition(g, greedy_and_rand_res)
    print(f"Time elapsed = {end - start}\n")

    print("UConn Greedy + Random (dist: 7.5):")
    start = timeit.default_timer()
    greedy_and_rand_res = algos.uconn_strat_2(g, k, 7.5)
    end = timeit.default_timer()
    benchmark.benchmark_partition(g, greedy_and_rand_res)
    print(f"Time elapsed = {end - start}\n")

    alpha: float = 0.5
    print("Greedy:")
    f: Callable[..., list[int]] = algos.greedy
    start = timeit.default_timer()
    transfer_res = algos.find_partition_with_heuristic(g, partition, f, alpha)
    end = timeit.default_timer()
    res = benchmark.solve_partition(g, transfer_res, f)
    benchmark.benchmark_partition(g, res)
    print(f"Time elapsed = {end - start}\n")

    print("Nearest Neighbor:")
    f = algos.nearest_neighbor
    start = timeit.default_timer()
    transfer_res = algos.find_partition_with_heuristic(g, partition, f, alpha)
    end = timeit.default_timer()
    res = benchmark.solve_partition(g, transfer_res, f)
    benchmark.benchmark_partition(g, res)
    print(f"Time elapsed = {end - start}\n")

    # print("TSP:")
    # f = algos.held_karp
    # start = timeit.default_timer()
    # transfer_res = algos.find_partition_with_heuristic(g, partition, f, alpha)
    # end = timeit.default_timer()
    # res = benchmark.solve_partition(g, transfer_res, f)
    # benchmark.benchmark_partition(g, res)
    # print(f"Time elapsed = {end - start}\n")

    # print("MWLP:")
    # f = algos.brute_force_mwlp
    # start = timeit.default_timer()
    # transfer_res = algos.find_partition_with_heuristic(g, partition, f, alpha)
    # end = timeit.default_timer()
    # res = benchmark.solve_partition(g, transfer_res, f)
    # benchmark.benchmark_partition(g, res)
    # print(f"Time elapsed = {end - start}\n")

    alpha = 0.75
    print("Average heuristic:")
    start = timeit.default_timer()
    transfer_res = algos.find_partition_with_average(g, partition, alpha)
    end = timeit.default_timer()
    res = benchmark.solve_partition(g, transfer_res)
    benchmark.benchmark_partition(g, res)
    print(f"Time elapsed = {end - start}\n")


if __name__ == "__main__":
    main()
