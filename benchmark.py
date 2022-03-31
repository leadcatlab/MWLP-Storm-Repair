import random
from collections import defaultdict
from typing import Callable

import algos
from graph import Graph

class Bcolors:
    """
    Helper class for adding colors to prints
    https://svn.blender.org/svnroot/bf-blender/trunk/blender/build_files/scons/tools/Bcolors.py
    """
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    CLEAR_LAST_LINE = (
        "\033[A                                                             \033[A"
    )

def solve_partition(
    g: Graph, part: list[set[int]], f: Callable[..., list[int]] = algos.brute_force_mwlp
) -> list[list[int]]:
    """
    Take the output of transfers and swaps and
    find optimal orders based on heuristic f
    """
    # creating a deep copy to be safe
    partition: list[set[int]] = [set(s) for s in part]

    res: list[list[int]] = []
    for p in partition:
        sub_g, sto, _ = Graph.subgraph(g, list(p))
        sub_res: list[int] = f(sub_g)
        remapped_res: list[int] = [sto[node] for node in sub_res]
        res.append(remapped_res)

    return res


def benchmark_partition(
    g: Graph, part: list[list[int]]
) -> tuple[float, float, float, float]:
    """
    Take solved partition and print
    """

    # creating a deep copy to be safe
    partition: list[list[int]] = [list(p) for p in part]

    vals: list[float] = [algos.wlp(g, p) for p in partition]
    output: str = ""
    for i in range(len(partition)):
        output += f"    Agent {i} = {vals[i]: >20}: {partition[i]}\n"

    res: tuple[float, float, float, float] = (
        max(vals),
        min(vals),
        max(vals) - min(vals),
        sum(vals) / len(vals),
    )

    output += f"{Bcolors.OKBLUE}Maximum: {Bcolors.ENDC}{res[0]}\n"
    output += f"{Bcolors.OKBLUE}Minimum: {Bcolors.ENDC}{res[1]}\n"
    output += f"{Bcolors.OKBLUE}Range:   {Bcolors.ENDC}{res[2]}\n"
    output += f"{Bcolors.OKBLUE}Average: {Bcolors.ENDC}{res[3]}\n"
    # print(output)

    return res


def mass_benchmark(
    count: int,
    k: int,
    n: int,
    edge_w: tuple[float, float] = (0.0, 1.0),
    metric: bool = True,
    upper: float = 1.0,
    node_w: tuple[int, int] = (0, 100),
) -> None:
    maximums = defaultdict(list)
    minimums = defaultdict(list)
    ranges = defaultdict(list)
    averages = defaultdict(list)

    for i in range(count):
        print(i)
        if metric:
            g = Graph.random_complete_metric(n, upper, node_w)
        else:
            g = Graph.random_complete(n, edge_w, node_w)

        partition: list[set[int]] = Graph.create_agent_partition(g, k)

        # Put all desired heuristics here

        curr = "UConn Greedy"
        print(curr)
        res = algos.uconn_strat_1(g, k)
        curr_max, curr_min, curr_range, curr_avg = benchmark_partition(g, res)
        maximums[curr].append(curr_max)
        minimums[curr].append(curr_min)
        ranges[curr].append(curr_range)
        averages[curr].append(curr_avg)
        print(Bcolors.CLEAR_LAST_LINE)

        curr = "UConn Greedy + Random (dist: 2.5)"
        print(curr)
        res = algos.uconn_strat_2(g, k, 2.5)
        curr_max, curr_min, curr_range, curr_avg = benchmark_partition(g, res)
        maximums[curr].append(curr_max)
        minimums[curr].append(curr_min)
        ranges[curr].append(curr_range)
        averages[curr].append(curr_avg)
        print(Bcolors.CLEAR_LAST_LINE)

        curr = "UConn Greedy + Random (dist: 5.0)"
        print(curr)
        res = algos.uconn_strat_2(g, k, 5.0)
        curr_max, curr_min, curr_range, curr_avg = benchmark_partition(g, res)
        maximums[curr].append(curr_max)
        minimums[curr].append(curr_min)
        ranges[curr].append(curr_range)
        averages[curr].append(curr_avg)
        print(Bcolors.CLEAR_LAST_LINE)

        curr = "UConn Greedy + Random (dist: 7.5)"
        print(curr)
        res = algos.uconn_strat_2(g, k, 7.5)
        curr_max, curr_min, curr_range, curr_avg = benchmark_partition(g, res)
        maximums[curr].append(curr_max)
        minimums[curr].append(curr_min)
        ranges[curr].append(curr_range)
        averages[curr].append(curr_avg)
        print(Bcolors.CLEAR_LAST_LINE)

        curr = "Greedy"
        print(curr)
        print("Finding partition")
        output = algos.find_partition_with_heuristic(g, partition, algos.greedy, 0.24)
        print(Bcolors.CLEAR_LAST_LINE)
        print("Solving partition")
        res = solve_partition(g, output, algos.greedy)
        print(Bcolors.CLEAR_LAST_LINE)
        curr_max, curr_min, curr_range, curr_avg = benchmark_partition(g, res)
        maximums[curr].append(curr_max)
        minimums[curr].append(curr_min)
        ranges[curr].append(curr_range)
        averages[curr].append(curr_avg)
        print(Bcolors.CLEAR_LAST_LINE)

        curr = "Nearest Neighbor"
        print(curr)
        print("Finding partition")
        output = algos.find_partition_with_heuristic(
            g, partition, algos.nearest_neighbor, 0.15
        )
        print(Bcolors.CLEAR_LAST_LINE)
        print("Solving partition")
        res = solve_partition(g, output, algos.nearest_neighbor)
        print(Bcolors.CLEAR_LAST_LINE)
        curr_max, curr_min, curr_range, curr_avg = benchmark_partition(g, res)
        maximums[curr].append(curr_max)
        minimums[curr].append(curr_min)
        ranges[curr].append(curr_range)
        averages[curr].append(curr_avg)
        print(Bcolors.CLEAR_LAST_LINE)

        curr = "Average Heuristic"
        print(curr)
        print("Finding partition")
        output = algos.find_partition_with_average(g, partition, 0.22)
        print(Bcolors.CLEAR_LAST_LINE)
        print(Bcolors.CLEAR_LAST_LINE)

        curr = "Average Heuristic: Greedy"
        print(curr)
        print("Solving partition")
        res = solve_partition(g, output, algos.greedy)
        print(Bcolors.CLEAR_LAST_LINE)
        curr_max, curr_min, curr_range, curr_avg = benchmark_partition(g, res)
        maximums[curr].append(curr_max)
        minimums[curr].append(curr_min)
        ranges[curr].append(curr_range)
        averages[curr].append(curr_avg)
        print(Bcolors.CLEAR_LAST_LINE)

        curr = "Average Heuristic: Nearest Neighbor"
        print(curr)
        print("Solving partition")
        res = solve_partition(g, output, algos.nearest_neighbor)
        print(Bcolors.CLEAR_LAST_LINE)
        curr_max, curr_min, curr_range, curr_avg = benchmark_partition(g, res)
        maximums[curr].append(curr_max)
        minimums[curr].append(curr_min)
        ranges[curr].append(curr_range)
        averages[curr].append(curr_avg)
        print(Bcolors.CLEAR_LAST_LINE)

        print(Bcolors.CLEAR_LAST_LINE)

    print(f"{Bcolors.OKBLUE}Maximums: {Bcolors.ENDC}")
    for key, val in maximums.items():
        print(f"\t{key = :40}{sum(val) / count}")
    print()

    print(f"{Bcolors.OKBLUE}Minimums: {Bcolors.ENDC}")
    for key, val in minimums.items():
        print(f"\t{key = :40}{sum(val) / count}")
    print()

    print(f"{Bcolors.OKBLUE}Ranges: {Bcolors.ENDC}")
    for key, val in ranges.items():
        print(f"\t{key = :40}{sum(val) / count}")
    print()

    print(f"{Bcolors.OKBLUE}Averages: {Bcolors.ENDC}")
    for key, val in averages.items():
        print(f"\t{key = :40}{sum(val) / count}")
    print()


def alpha_heuristic_search(
    f: Callable[..., list[int]],
    count: int,
    k: int,
    n: int,
    edge_w: tuple[float, float] = (0.0, 1.0),
    metric: bool = True,
    upper: float = 1.0,
    node_w: tuple[int, int] = (0, 100),
) -> float:

    graph_bank: list[Graph] = []
    for _ in range(count):
        if metric:
            g = Graph.random_complete_metric(n, upper, node_w)
        else:
            g = Graph.random_complete(n, edge_w, node_w)
        graph_bank.append(g)

    partition_bank: list[list[set[int]]] = []
    for g in graph_bank:
        partition: list[set[int]] = Graph.create_agent_partition(g, k)
        partition_bank.append(partition)

    averages: dict[float, float] = {}

    alpha: float = 0.0
    while alpha <= 1.0:
        print(alpha)
        maximums: list[float] = []
        for g, partition in zip(graph_bank, partition_bank):
            output = algos.find_partition_with_heuristic(g, partition, f, alpha)
            res = solve_partition(g, output)
            curr_max, _, _, _ = benchmark_partition(g, res)
            maximums.append(curr_max)
        averages[alpha] = sum(maximums) / count
        alpha = round(alpha + 0.01, 2)
        print(Bcolors.CLEAR_LAST_LINE)

    return min(averages.items(), key=lambda x: x[1])[0]


def avg_alpha_heuristic_search(
    count: int,
    k: int,
    n: int,
    edge_w: tuple[float, float] = (0.0, 1.0),
    metric: bool = True,
    upper: float = 1.0,
    node_w: tuple[int, int] = (0, 100),
) -> float:

    graph_bank: list[Graph] = []
    for _ in range(count):
        if metric:
            g = Graph.random_complete_metric(n, upper, node_w)
        else:
            g = Graph.random_complete(n, edge_w, node_w)
        graph_bank.append(g)

    partition_bank: list[list[set[int]]] = []
    for g in graph_bank:
        partition: list[set[int]] = Graph.create_agent_partition(g, k)
        partition_bank.append(partition)

    averages: dict[float, float] = {}

    alpha: float = 0.0
    while alpha <= 1.0:
        print(alpha)
        maximums: list[float] = []
        for g, partition in zip(graph_bank, partition_bank):
            output = algos.find_partition_with_average(g, partition, alpha)
            res = solve_partition(g, output)
            curr_max, _, _, _ = benchmark_partition(g, res)
            maximums.append(curr_max)
        averages[alpha] = sum(maximums) / count
        alpha = round(alpha + 0.01, 2)
        print(Bcolors.CLEAR_LAST_LINE)

    return min(averages.items(), key=lambda x: x[1])[0]
