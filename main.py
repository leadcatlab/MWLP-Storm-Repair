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
    n: int = 15
    k: int = 3
    g = Graph.random_complete_metric(n, upper=10.0, directed=False)
    partition: list[set[int]] = Graph.create_agent_partition(g, k)

    print("Initial:")
    res = benchmark.solve_partition(g, partition)
    benchmark.benchmark_partition(g, res)

    # print("Greedy:")
    # f: Callable[..., list[int]] = algos.greedy
    # start = timeit.default_timer()
    # transfer_res = algos.transfers_and_swaps_mwlp(g, partition, f)
    # end = timeit.default_timer()
    # res = benchmark.solve_partition(g, transfer_res, f)
    # benchmark.benchmark_partition(g, res)
    # print(f"Time elapsed = {end - start}\n")

    # print("Nearest Neighbor:")
    # f = algos.nearest_neighbor
    # start = timeit.default_timer()
    # transfer_res = algos.transfers_and_swaps_mwlp(g, partition, f)
    # end = timeit.default_timer()
    # res = benchmark.solve_partition(g, transfer_res, f)
    # benchmark.benchmark_partition(g, res)
    # print(f"Time elapsed = {end - start}\n")

    # print("TSP:")
    # f = algos.held_karp
    # start = timeit.default_timer()
    # transfer_res = algos.transfers_and_swaps_mwlp(g, partition, f)
    # end = timeit.default_timer()
    # res = benchmark.solve_partition(g, transfer_res, f)
    # benchmark.benchmark_partition(g, res)
    # print(f"Time elapsed = {end - start}\n")

    # print("MWLP")
    # f = algos.brute_force_mwlp
    # start = timeit.default_timer()
    # transfer_res = algos.transfers_and_swaps_mwlp(g, partition, f)
    # end = timeit.default_timer()
    # res = benchmark.solve_partition(g, transfer_res, f)
    # benchmark.benchmark_partition(g, res)
    # print(f"Time elapsed = {end - start}\n")

    # print("Average heuristic")
    # start = timeit.default_timer()
    # transfer_res = algos.transfers_and_swaps_mwlp_with_average(g, partition)
    # end = timeit.default_timer()
    # res = benchmark.solve_partition(g, transfer_res)
    # benchmark.benchmark_partition(g, res)
    # print(f"Time elapsed = {end - start}\n")

    # print("UConn Greedy")
    # start = timeit.default_timer()
    # greedy_res = algos.uconn_strat_1(g, k)
    # end = timeit.default_timer()
    # benchmark.benchmark_partition(g, greedy_res)
    # print(f"Time elapsed = {end - start}\n")

    # print("UConn Greedy + Random (dist: 2.5)")
    # start = timeit.default_timer()
    # greedy_and_rand_res = algos.uconn_strat_2(g, k, 2.5)
    # end = timeit.default_timer()
    # benchmark.benchmark_partition(g, greedy_and_rand_res)
    # print(f"Time elapsed = {end - start}\n")

    # print("UConn Greedy + Random (dist: 5.0)")
    # start = timeit.default_timer()
    # greedy_and_rand_res = algos.uconn_strat_2(g, k, 5.0)
    # end = timeit.default_timer()
    # benchmark.benchmark_partition(g, greedy_and_rand_res)
    # print(f"Time elapsed = {end - start}\n")

    # print("UConn Greedy + Random (dist: 7.5)")
    # start = timeit.default_timer()
    # greedy_and_rand_res = algos.uconn_strat_2(g, k, 7.5)
    # end = timeit.default_timer()
    # benchmark.benchmark_partition(g, greedy_and_rand_res)
    # print(f"Time elapsed = {end - start}\n")

    alpha: float = 1.0
    print("Greedy:")
    f: Callable[..., list[int]] = algos.greedy
    transfer_res = algos.transfer_outliers_mwlp(g, partition, f, alpha)
    res = benchmark.solve_partition(g, transfer_res, f)
    benchmark.benchmark_partition(g, res)

    print("Nearest Neighbor:")
    f = algos.nearest_neighbor
    transfer_res = algos.transfer_outliers_mwlp(g, partition, f, alpha)
    res = benchmark.solve_partition(g, transfer_res, f)
    benchmark.benchmark_partition(g, res)

    # print("TSP:")
    # f = algos.held_karp
    # transfer_res = algos.transfer_outliers_mwlp(g, partition, f, alpha)
    # res = benchmark.solve_partition(g, transfer_res, f)
    # benchmark.benchmark_partition(g, res)

    # print("MWLP")
    # f = algos.brute_force_mwlp
    # transfer_res = algos.transfer_outliers_mwlp(g, partition, f, alpha)
    # res = benchmark.solve_partition(g, transfer_res, f)
    # benchmark.benchmark_partition(g, res)

    alpha: float = 1.0
    print("Average heuristic")
    transfer_res = algos.transfer_outliers_mwlp_with_average(g, partition, alpha)
    res = benchmark.solve_partition(g, transfer_res)
    benchmark.benchmark_partition(g, res)

if __name__ == "__main__":
    main()
