"""
Benchmark Functions
"""
import time
from collections import defaultdict
from typing import Callable, DefaultDict, no_type_check

import matplotlib.pyplot as plt  # type: ignore
import mplcursors  # type: ignore
import networkx as nx  # type: ignore
import numpy as np

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
    Determine optimal orders for each subset in the partition
    according to a passed heuristic

    Parameters
    ----------
    g: Graph
        Input graph
        Assertions:
            g must be a complete graph

    part: list[set[int]]
        Starting unordered assignment of nodes for each agent
        Assertions:
            Must be an agent partition

    f: Callable[..., list[int]]
        Passed heuristic
        Default: brute force mwlp

    Returns
    -------
    list[list[int]]
        Solved orders of each agent

    """

    if not Graph.is_complete(g):
        raise ValueError("Passed graph is not complete")

    if Graph.is_agent_partition(g, part) is False:
        raise ValueError("Passed partition is invalid")

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
) -> tuple[float, float, float, float, float, float]:
    """
    Takes in a partition and path order of agents and benchmarks it

    Parameters
    ----------
    g: Graph
        Input graph
        Assertions:
            g must be a complete graph

    part: list[set[int]]
        Starting unordered assignment of nodes for each agent
        Assertions:
            Must be an agent partition

    Returns
    -------
    tuple[float, float, float, float, float, float]
        maximum, average wait, minimum, range, sum, average

    """

    #   Unsure this is a good idea since it may just invite alot of
    #   "well actually if you use this obsucre library no one has heard of it's faster"
    if not Graph.is_complete(g):
        raise ValueError("Passed graph is not complete")

    if Graph.is_agent_partition(g, [set(s) for s in part]) is False:
        raise ValueError("Passed partition is invalid")

    # creating a deep copy to be safe
    partition: list[list[int]] = [list(p) for p in part]

    vals: list[float] = [algos.wlp(g, p) for p in partition]
    # output: str = ""
    # for i in range(len(partition)):
    #     output += f"    Agent {i} = {vals[i]: >20}: {partition[i]}\n"

    # Calculate average wait times
    wait_times: list[float] = []
    for val, p in zip(vals, partition):
        wait_times.append(val / algos.num_visited_along_path(g, p)[-1])

    res: tuple[float, float, float, float, float] = (
        max(vals),
        sum(wait_times) / len(wait_times),
        min(vals),
        max(vals) - min(vals),
        sum(vals),
        sum(vals) / len(vals),
    )

    # output += f"{Bcolors.OKBLUE}Maximum: {Bcolors.ENDC}{res[0]}\n"
    # output += f"{Bcolors.OKBLUE}Minimum: {Bcolors.ENDC}{res[1]}\n"
    # output += f"{Bcolors.OKBLUE}Range:   {Bcolors.ENDC}{res[2]}\n"
    # output += f"{Bcolors.OKBLUE}Average: {Bcolors.ENDC}{res[3]}\n"
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
    """
    Benchmarks a large number of graphs randomly generated accord to the parameters

    Parameters
    ----------
    count: int
        The number of graphs to benchmark

    k: int
        The number of agents

    n: int
        The number of nodes per graph

    edge_w: tuple[float, float]
        The range of edge weights allowed
        Default: (0.0, 1.0)

    metric: bool
        Determine whether to test on metric or non-metric graphs
        Default: True

    upper: float
        Upper bound of edge weights for a metric graph
        Default: 1.0

    node_w: tuple[int, int]
        The range of node weights allowed
        Default: (0, 100)

    """

    maximums: DefaultDict[str, list[float]] = defaultdict(list)
    # WLP is a weighted average of wait times of sorts
    wait_times: DefaultDict[str, list[float]] = defaultdict(list)
    times: DefaultDict[str, list[float]] = defaultdict(list)
    minimums: DefaultDict[str, list[float]] = defaultdict(list)
    sums: DefaultDict[str, list[float]] = defaultdict(list)
    ranges: DefaultDict[str, list[float]] = defaultdict(list)
    averages: DefaultDict[str, list[float]] = defaultdict(list)
    bests: DefaultDict[str, int] = defaultdict(int)

    for i in range(count):
        print(i)
        if metric:
            g = Graph.random_complete_metric(n, upper, node_w)
        else:
            g = Graph.random_complete(n, edge_w, node_w)

        partition: list[set[int]] = Graph.create_agent_partition(g, k)

        # Put all desired heuristics here

        best, curr_best = "", float("inf")

        curr = "UConn Greedy"
        print(curr)
        start: float = time.perf_counter_ns()
        res = algos.uconn_strat_1(g, k)
        end: float = time.perf_counter_ns()
        curr_max, curr_wait, curr_min, curr_range, curr_sum, curr_avg = benchmark_partition(
            g, res
        )
        if curr_sum < curr_best:
            curr_best = curr_sum
            best = curr
        maximums[curr].append(curr_max)
        wait_times[curr].append(curr_wait)
        times[curr].append(end - start)
        minimums[curr].append(curr_min)
        ranges[curr].append(curr_range)
        sums[curr].append(curr_sum)
        averages[curr].append(curr_avg)
        print(Bcolors.CLEAR_LAST_LINE)

        curr = "UConn Nearest Neighbor"
        print(curr)
        start = time.perf_counter_ns()
        res = algos.uconn_strat_3(g, k)
        end = time.perf_counter_ns()
        curr_max, curr_wait, curr_min, curr_range, curr_sum, curr_avg = benchmark_partition(
            g, res
        )
        if curr_sum < curr_best:
            curr_best = curr_sum
            best = curr
        maximums[curr].append(curr_max)
        wait_times[curr].append(curr_wait)
        times[curr].append(end - start)
        minimums[curr].append(curr_min)
        ranges[curr].append(curr_range)
        sums[curr].append(curr_sum)
        averages[curr].append(curr_avg)
        print(Bcolors.CLEAR_LAST_LINE)

        curr = "UConn Greedy + Random (dist: 2.5)"
        print(curr)
        start = time.perf_counter_ns()
        res = algos.uconn_strat_2(g, k, 2.5)
        end = time.perf_counter_ns()
        curr_max, curr_wait, curr_min, curr_range, curr_sum, curr_avg = benchmark_partition(
            g, res
        )
        if curr_sum < curr_best:
            curr_best = curr_sum
            best = curr
        maximums[curr].append(curr_max)
        wait_times[curr].append(curr_wait)
        times[curr].append(end - start)
        minimums[curr].append(curr_min)
        ranges[curr].append(curr_range)
        sums[curr].append(curr_sum)
        averages[curr].append(curr_avg)
        print(Bcolors.CLEAR_LAST_LINE)

        curr = "UConn Greedy + Random (dist: 5.0)"
        print(curr)
        start = time.perf_counter_ns()
        res = algos.uconn_strat_2(g, k, 5.0)
        end = time.perf_counter_ns()
        curr_max, curr_wait, curr_min, curr_range, curr_sum, curr_avg = benchmark_partition(
            g, res
        )
        if curr_sum < curr_best:
            curr_best = curr_sum
            best = curr
        maximums[curr].append(curr_max)
        wait_times[curr].append(curr_wait)
        times[curr].append(end - start)
        minimums[curr].append(curr_min)
        ranges[curr].append(curr_range)
        sums[curr].append(curr_sum)
        averages[curr].append(curr_avg)
        print(Bcolors.CLEAR_LAST_LINE)

        curr = "UConn Greedy + Random (dist: 7.5)"
        print(curr)
        start = time.perf_counter_ns()
        res = algos.uconn_strat_2(g, k, 7.5)
        end = time.perf_counter_ns()
        curr_max, curr_wait, curr_min, curr_range, curr_sum, curr_avg = benchmark_partition(
            g, res
        )
        if curr_sum < curr_best:
            curr_best = curr_sum
            best = curr
        maximums[curr].append(curr_max)
        wait_times[curr].append(curr_wait)
        times[curr].append(end - start)
        minimums[curr].append(curr_min)
        ranges[curr].append(curr_range)
        sums[curr].append(curr_sum)
        averages[curr].append(curr_avg)
        print(Bcolors.CLEAR_LAST_LINE)

        curr = "Greedy"
        print(curr)
        print("Finding partition")
        start = time.perf_counter_ns()
        output = algos.find_partition_with_heuristic(g, partition, algos.greedy, 0.02)
        end = time.perf_counter_ns()
        print(Bcolors.CLEAR_LAST_LINE)
        print("Solving partition")
        res = solve_partition(g, output, algos.greedy)
        print(Bcolors.CLEAR_LAST_LINE)
        curr_max, curr_wait, curr_min, curr_range, curr_sum, curr_avg = benchmark_partition(
            g, res
        )
        if curr_sum < curr_best:
            curr_best = curr_sum
            best = curr
        maximums[curr].append(curr_max)
        wait_times[curr].append(curr_wait)
        times[curr].append(end - start)
        minimums[curr].append(curr_min)
        ranges[curr].append(curr_range)
        sums[curr].append(curr_sum)
        averages[curr].append(curr_avg)
        print(Bcolors.CLEAR_LAST_LINE)

        curr = "Nearest Neighbor"
        print(curr)
        print("Finding partition")
        start = time.perf_counter_ns()
        output = algos.find_partition_with_heuristic(
            g, partition, algos.nearest_neighbor, 0.18
        )
        end = time.perf_counter_ns()
        print(Bcolors.CLEAR_LAST_LINE)
        print("Solving partition")
        res = solve_partition(g, output, algos.nearest_neighbor)
        print(Bcolors.CLEAR_LAST_LINE)
        curr_max, curr_wait, curr_min, curr_range, curr_sum, curr_avg = benchmark_partition(
            g, res
        )
        if curr_sum < curr_best:
            curr_best = curr_sum
            best = curr
        maximums[curr].append(curr_max)
        wait_times[curr].append(curr_wait)
        times[curr].append(end - start)
        minimums[curr].append(curr_min)
        ranges[curr].append(curr_range)
        sums[curr].append(curr_sum)
        averages[curr].append(curr_avg)
        print(Bcolors.CLEAR_LAST_LINE)

        # curr = "Optimal after NN"
        # print(curr)
        # print("Solving partition")
        # start = time.perf_counter_ns()
        # # NOTE: This is the time to SOLVE for the optimal order in given partition.
        # #   Includes no time to find the partition (as this was done earlier)
        # res = solve_partition(g, output, algos.brute_force_mwlp)
        # end = time.perf_counter_ns()
        # print(Bcolors.CLEAR_LAST_LINE)
        # curr_max, curr_wait, curr_min, curr_range, curr_avg = benchmark_partition(
        #     g, res
        # )
        # if curr_sum < curr_best:
        #     curr_best = curr_sum
        #     best = curr
        # maximums[curr].append(curr_max)
        # wait_times[curr].append(curr_wait)
        # times[curr].append(end - start)
        # minimums[curr].append(curr_min)
        # ranges[curr].append(curr_range)
        # averages[curr].append(curr_avg)
        # print(Bcolors.CLEAR_LAST_LINE)

        # This doesn't have any real meaning so it is being depreciated
        # curr = "Alternate"
        # print(curr)
        # print("Finding partition")
        # start = time.perf_counter_ns()
        # output = algos.find_partition_with_heuristic(
        #     g, partition, algos.alternate, 0.18
        # )
        # end = time.perf_counter_ns()
        # print(Bcolors.CLEAR_LAST_LINE)
        # print("Solving partition")
        # res = solve_partition(g, output, algos.alternate)
        # print(Bcolors.CLEAR_LAST_LINE)
        # curr_max, curr_wait, curr_min, curr_range, curr_avg = benchmark_partition(
        #     g, res
        # )
        # if curr_sum < curr_best:
        #     curr_best = curr_sum
        #     best = curr
        # maximums[curr].append(curr_max)
        # wait_times[curr].append(curr_wait)
        # times[curr].append(end - start)
        # minimums[curr].append(curr_min)
        # ranges[curr].append(curr_range)
        # averages[curr].append(curr_avg)
        # print(Bcolors.CLEAR_LAST_LINE)

        bests[best] += 1

        print(Bcolors.CLEAR_LAST_LINE)

    print(f"{Bcolors.OKBLUE}Sums: {Bcolors.ENDC}")
    for key, vals in sums.items():
        print(f"\t{key:40}{sum(vals) / count}")
    print()
    
    print(f"{Bcolors.OKBLUE}Maximums: {Bcolors.ENDC}")
    for key, vals in maximums.items():
        print(f"\t{key:40}{sum(vals) / count}")
    print()

    print(f"{Bcolors.OKBLUE}Wait Times: {Bcolors.ENDC}")
    for key, vals in wait_times.items():
        print(f"\t{key:40}{sum(vals) / count}")
    print()

    print(f"{Bcolors.OKBLUE}Runtime in seconds: {Bcolors.ENDC}")
    for key, vals in times.items():
        print(f"\t{key:40}{sum(vals) / (count * (10 ** 9))}")
    print()

    print(f"{Bcolors.OKBLUE}Bests: {Bcolors.ENDC}")
    for key, val in bests.items():
        print(f"\t{key:40}{val}")
    print()

    print(f"{Bcolors.OKBLUE}Minimums: {Bcolors.ENDC}")
    for key, vals in minimums.items():
        print(f"\t{key:40}{sum(vals) / count}")
    print()

    print(f"{Bcolors.OKBLUE}Ranges: {Bcolors.ENDC}")
    for key, vals in ranges.items():
        print(f"\t{key:40}{sum(vals) / count}")
    print()

    print(f"{Bcolors.OKBLUE}Averages: {Bcolors.ENDC}")
    for key, vals in averages.items():
        print(f"\t{key:40}{sum(vals) / count}")
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
    """
    Search function to help determine ideal alpha values for transfers and swaps
    Used for nearest neighbor or greedy or other such heuristics

    Parameters
    ----------
    f: Callable[..., list[int]]
        Passed heuristic

    count: int
        The number of graphs to benchmark

    k: int
        The number of agents

    n: int
        The number of nodes per graph

    edge_w: tuple[float, float]
        The range of edge weights allowed
        Default: (0.0, 1.0)

    metric: bool
        Determine whether to test on metric or non-metric graphs
        Default: True

    upper: float
        Upper bound of edge weights for a metric graph
        Default: 1.0

    node_w: tuple[int, int]
        The range of node weights allowed
        Default: (0, 100)

    Returns
    -------
    float
        Ideal alpha

    """

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
            curr_max, _, _, _, _ = benchmark_partition(g, res)
            maximums.append(curr_max)
        averages[alpha] = sum(maximums) / count
        alpha = round(alpha + 0.01, 2)
        print(Bcolors.CLEAR_LAST_LINE)

    return min(averages.items(), key=lambda x: x[1])[0]


def line_plot(
    g: Graph,
    part: list[set[int]],
    x_range: tuple[int, int] = (0, 10),
) -> None:
    """
    Generate a graph of the given parameters and plot visited nodes

    Parameters
    ----------
    g: Graph
        Input graph
        Assertions:
            Must be complete
            Must be undirected

    part: list[set[int]]
        Starting agent partition
        Assertions:
            Must be an agent partition

    x_range: tuple[int, int]
        x-axis bounds for plotting
        Default: (0, 100)
    """

    if Graph.is_complete(g) is False:
        raise ValueError("Passed graph is not complete")

    if Graph.is_undirected(g) is False:
        raise ValueError("Passed graph is not undirected")

    if Graph.is_agent_partition(g, part) is False:
        raise ValueError("Passed partition is invalid")

    n: int = g.num_nodes
    k: int = len(part)
    low, high = x_range
    x = np.linspace(low, high, high * 10)
    _, ax = plt.subplots()
    total = sum(g.node_weight[x] for x in range(n))
    lines = []

    # List of matplotlib colors
    #   https://matplotlib.org/3.5.0/_images/sphx_glr_named_colors_003.png

    curr: str = "Random"
    output: list[set[int]] = [set(s) for s in part]
    paths: list[list[int]] = solve_partition(g, output, algos.random_order)
    curr_max: float = max(algos.wlp(g, path) for path in paths)
    f = algos.generate_partition_path_function(g, paths)
    y = [total - f(i) for i in x]
    (line,) = ax.plot(
        x, y, label=f"{curr}: {curr_max}", linewidth=2.0, color="darkviolet"
    )
    lines.append(line)

    curr = "UConn Greedy"
    paths = algos.uconn_strat_1(g, k)
    curr_max = max(algos.wlp(g, path) for path in paths)
    f = algos.generate_partition_path_function(g, paths)
    y = [total - f(i) for i in x]
    (line,) = ax.plot(x, y, label=f"{curr}: {curr_max}", color="lightsteelblue")
    lines.append(line)

    curr = "UConn Nearest Neighbor"
    paths = algos.uconn_strat_1(g, k)
    curr_max = max(algos.wlp(g, path) for path in paths)
    f = algos.generate_partition_path_function(g, paths)
    y = [total - f(i) for i in x]
    (line,) = ax.plot(x, y, label=f"{curr}: {curr_max}", color="aqua")
    lines.append(line)

    curr = "UConn Greedy + Rand (2.5)"
    paths = algos.uconn_strat_2(g, k, 2.5)
    curr_max = max(algos.wlp(g, path) for path in paths)
    f = algos.generate_partition_path_function(g, paths)
    y = [total - f(i) for i in x]
    (line,) = ax.plot(x, y, label=f"{curr}: {curr_max}", color="royalblue")
    lines.append(line)

    curr = "UConn Greedy + Rand (5.0)"
    paths = algos.uconn_strat_2(g, k, 5.0)
    curr_max = max(algos.wlp(g, path) for path in paths)
    f = algos.generate_partition_path_function(g, paths)
    y = [total - f(i) for i in x]
    (line,) = ax.plot(x, y, label=f"{curr}: {curr_max}", color="blue")
    lines.append(line)

    curr = "UConn Greedy + Rand (7.5)"
    paths = algos.uconn_strat_2(g, k, 7.5)
    curr_max = max(algos.wlp(g, path) for path in paths)
    f = algos.generate_partition_path_function(g, paths)
    y = [total - f(i) for i in x]
    (line,) = ax.plot(x, y, label=f"{curr}: {curr_max}", color="mediumslateblue")
    lines.append(line)

    curr = "Greedy"
    output = algos.find_partition_with_heuristic(g, part, algos.greedy, 0.02)
    paths = solve_partition(g, output, algos.greedy)
    curr_max = max(algos.wlp(g, path) for path in paths)
    f = algos.generate_partition_path_function(g, paths)
    y = [total - f(i) for i in x]
    (line,) = ax.plot(
        x, y, label=f"{curr}: {curr_max}", linewidth=2.0, color="limegreen"
    )
    lines.append(line)

    curr = "Alternate"
    output = algos.find_partition_with_heuristic(g, part, algos.alternate, 0.18)
    paths = solve_partition(g, output, algos.alternate)
    curr_max = max(algos.wlp(g, path) for path in paths)
    f = algos.generate_partition_path_function(g, paths)
    y = [total - f(i) for i in x]
    (line,) = ax.plot(
        x, y, label=f"{curr}: {curr_max}", linewidth=2.0, color="mediumspringgreen"
    )
    lines.append(line)

    curr = "Nearest Neighbor"
    output = algos.find_partition_with_heuristic(g, part, algos.nearest_neighbor, 0.18)
    paths = solve_partition(g, output, algos.nearest_neighbor)
    curr_max = max(algos.wlp(g, path) for path in paths)
    f = algos.generate_partition_path_function(g, paths)
    y = [total - f(i) for i in x]
    (line,) = ax.plot(
        x, y, label=f"{curr}: {curr_max}", linewidth=2.0, color="darkgreen"
    )
    lines.append(line)

    curr = "Optimal After NN transfers"
    paths = solve_partition(g, output, algos.brute_force_mwlp)
    curr_max = max(algos.wlp(g, path) for path in paths)
    f = algos.generate_partition_path_function(g, paths)
    y = [total - f(i) for i in x]
    (line,) = ax.plot(x, y, label=f"{curr}: {curr_max}", linewidth=2.0, color="red")
    lines.append(line)

    cap: int = (g.num_nodes // k) + 1
    curr = f"Best Solution with {cap = }"
    paths = algos.multi_agent_brute_force(g, k, f=algos.nearest_neighbor, max_size = cap)
    curr_max = max(algos.wlp(g, path) for path in paths)
    f = algos.generate_partition_path_function(g, paths)
    y = [total - f(i) for i in x]
    (line,) = ax.plot(
        x, y, label=f"{curr}: {curr_max}", linewidth=2.0, color="firebrick"
    )
    lines.append(line)
    
    cap: int = (g.num_nodes // k) + 2
    curr = f"Best Solution with {cap = }"
    paths = algos.multi_agent_brute_force(g, k, f=algos.nearest_neighbor, max_size = cap)
    curr_max = max(algos.wlp(g, path) for path in paths)
    f = algos.generate_partition_path_function(g, paths)
    y = [total - f(i) for i in x]
    (line,) = ax.plot(
        x, y, label=f"{curr}: {curr_max}", linewidth=2.0, color="firebrick"
    )
    lines.append(line)

    # This ended up performing poorly
    # curr = "TSP After NN"
    # paths = solve_partition(g, output, algos.held_karp)
    # curr_max = max(algos.wlp(g, path) for path in paths)
    # f = algos.generate_partition_path_function(g, paths)
    # y = [total - f(i) for i in x]
    # (line,) = ax.plot(
    #     x, y, label=f"{curr}: {curr_max}", linewidth=2.0, color="firebrick"
    # )
    # lines.append(line)

    mplcursors.cursor(lines, highlight=True)
    plt.legend()
    figure = plt.gcf()
    figure.set_size_inches(10, 7) # horizontal x vertical
    plt.savefig("most_recent_line_plot.png")
    # plt.show()


@no_type_check
def draw_graph_with_partitions(nx_g, assignments: list[list[int]], name=None) -> None:
    """
    Draws a graph using networkx
    Not all edges are drawn. Just the ones pertaining to the passted edges


    Parameters
    ----------
    nx_g: nx.DiGraph()
        Input Networkx Graph

    assignments: list[list[int]]
        Input agent assignment
        Assertions:
            Is agent assignment

    name: str
        Used to name output plot
        Default: None

    """

    # Color the nodes
    idx: int = 0
    color_list = plt.cm.get_cmap("tab20", 20)
    color_map = [None] * len(nx_g.nodes)
    for assignment in assignments:
        for node in assignment:
            color_map[node] = color_list.colors[idx]
        idx = (idx + 1) % 20

    # Choose edges
    edges = []
    for assignment in assignments:
        for (u, v) in zip(assignment, assignment[1:]):
            edges.append((u, v))

    plt.figure(name)
    nx.draw(
        nx_g,
        pos=nx.spring_layout(nx_g),
        edgelist=edges,
        with_labels=True,
        node_color=color_map,
    )
    # plt.show()
