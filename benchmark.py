"""
Benchmark Functions
"""
import json
import time
from collections import defaultdict
from typing import Any, Callable, DefaultDict, no_type_check

import matplotlib.pyplot as plt  # type: ignore
import mplcursors  # type: ignore
import networkx as nx  # type: ignore
import numpy as np

import algos
from graph import Graph, graph_dict


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
    g: Graph, assignment: list[list[int]]
) -> tuple[float, float, float, float, float, float]:
    """
    Takes in a partition and path order of agents and benchmarks it

    Parameters
    ----------
    g: Graph
        Input graph
        Assertions:
            g must be a complete graph

    assignment: list[set[int]]
        Assignment of nodes for each agent
        Assertions:
            Must be an agent assignment

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
    assign: list[list[int]] = [list(p) for p in assignment]

    vals: list[float] = [algos.wlp(g, p) for p in assign]
    
    # Calculate average wait times
    wait_times: list[float] = []
    for val, p in zip(vals, assign):
        wait_times.append(val / algos.num_visited_along_path(g, p)[-1])

    res: tuple[float, float, float, float, float, float] = (
        max(vals),
        sum(wait_times) / len(wait_times),
        min(vals),
        max(vals) - min(vals),
        sum(vals),
        sum(vals) / len(vals),
    )

    return res


def generate_graph_bank(
    count: int,
    n: int,
    edge_w: tuple[float, float] = (0.0, 1.0),
    metric: bool = True,
    upper: float = 1.0,
    node_w: tuple[int, int] = (0, 100),
) -> list[Graph]:
    """
    Generate a list of graphs based on passed parameters

    Parameters
    ----------
    count: int
        The number of graphs to benchmark

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
    list[Graph]:
        List of randomly generated graphs based on passed data

    """

    graph_bank: list[Graph] = []
    for _ in range(count):
        if metric:
            g = Graph.random_complete_metric(n, upper, node_w)
        else:
            g = Graph.random_complete(n, edge_w, node_w)
        graph_bank.append(g)

    return graph_bank


def graph_bank_from_file(loc: str) -> list[Graph]:
    """
    Generate a list of graphs based on a json file

    Parameters
    ----------
    loc: str
        location of json file of graph_dicts

    Returns
    -------
    list[Graph]:
        List of graphs based on json file

    """

    graph_dict_bank: dict[str, graph_dict] = {}
    with open(loc, encoding="utf-8") as gd_json:
        graph_dict_bank = json.load(gd_json)

    n: int = len(graph_dict_bank)
    graph_bank: list[Graph] = []
    for i in range(n):
        gd: graph_dict = graph_dict_bank[str(i)]
        graph_bank.append(Graph.from_dict(gd))
    return graph_bank


def generate_agent_partitions(graph_bank: list[Graph], k: int) -> list[list[set[int]]]:
    """
    Generate agent partitions based on passed graphs

    Parameters
    ----------
    graph_bank: list[Graph]
        Passed list of graphs
        Assertions:
            Each graph must be complete

    k: int
        Number of agents

    Returns
    -------
    list[list[set[int]]]
        Generated agent partitions

    """

    partition_bank: list[list[set[int]]] = []

    for g in graph_bank:
        assert Graph.is_complete(g)
        partition: list[set[int]] = Graph.create_agent_partition(g, k)
        partition_bank.append(partition)

    return partition_bank


def agent_partitions_from_file(loc: str) -> list[list[set[int]]]:
    """
    Generate a list of agent partitions based on a json file

    Parameters
    ----------
    loc: str
        location of json file of partitions

    Returns
    -------
    list[list[set[int]]]:
        List of agent partitions based on json file

    """

    serialized_partition_bank: dict[str, list[list[int]]] = {}
    with open(loc, encoding="utf8") as part_file:
        serialized_partition_bank = json.load(part_file)

    n: int = len(serialized_partition_bank)
    partition_bank: list[list[set[int]]] = []
    for i in range(n):
        serialized_part: list[list[int]] = serialized_partition_bank[str(i)]
        part: list[set[int]] = [set(s) for s in serialized_part]
        partition_bank.append(part)

    return partition_bank


def mass_benchmark(
    graph_bank: list[Graph],
    partition_bank: list[list[set[int]]],
    rand_dist_range: tuple[float, float],
) -> list[DefaultDict[Any, Any]]:
    """
    Benchmarks a large number of graphs randomly generated accord to the parameters

    Parameters
    ----------
    graph_bank: list[Graph]
        List of graphs to test over
        Assertions:
            complete graphs

    partition_bank: list[list[set[int]]]
        List of partitions associated with graphs in graph_bank
        Assertions:
            partition_bank[i] is an agent partition of graph_bank[i]

    rand_dist_range: tuple[float, float]
        Range of allowed distances for random aspect of prior strategies

    """

    assert len(graph_bank) == len(partition_bank)
    for g, p in zip(graph_bank, partition_bank):
        assert Graph.is_complete(g)
        assert Graph.is_agent_partition(g, p)

    maximums: DefaultDict[str, list[float]] = defaultdict(list)
    # WLP is a weighted average of wait times of sorts
    wait_times: DefaultDict[str, list[float]] = defaultdict(list)
    times: DefaultDict[str, list[float]] = defaultdict(list)
    minimums: DefaultDict[str, list[float]] = defaultdict(list)
    sums: DefaultDict[str, list[float]] = defaultdict(list)
    ranges: DefaultDict[str, list[float]] = defaultdict(list)
    averages: DefaultDict[str, list[float]] = defaultdict(list)
    bests: DefaultDict[str, int] = defaultdict(int)

    count: int = len(graph_bank)
    for i, (g, partition) in enumerate(zip(graph_bank, partition_bank)):
        print(i)
        k: int = len(partition)

        # Put all desired heuristics here

        best, curr_best = "", float("inf")

        curr = "UConn Greedy"
        print(curr)
        start: float = time.perf_counter_ns()
        res = algos.uconn_strat_1(g, k)
        end: float = time.perf_counter_ns()
        (
            curr_max,
            curr_wait,
            curr_min,
            curr_range,
            curr_sum,
            curr_avg,
        ) = benchmark_partition(g, res)
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
        (
            curr_max,
            curr_wait,
            curr_min,
            curr_range,
            curr_sum,
            curr_avg,
        ) = benchmark_partition(g, res)
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

        lo, hi = rand_dist_range
        dist_range: float = hi - lo

        curr = "UConn Greedy + Random (25%)"
        print(curr)
        start = time.perf_counter_ns()
        res = algos.uconn_strat_2(g, k, lo + (dist_range * 0.25))
        end = time.perf_counter_ns()
        (
            curr_max,
            curr_wait,
            curr_min,
            curr_range,
            curr_sum,
            curr_avg,
        ) = benchmark_partition(g, res)
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

        curr = "UConn Greedy + Random (50%)"
        print(curr)
        start = time.perf_counter_ns()
        res = algos.uconn_strat_2(g, k, lo + (dist_range * 0.50))
        end = time.perf_counter_ns()
        (
            curr_max,
            curr_wait,
            curr_min,
            curr_range,
            curr_sum,
            curr_avg,
        ) = benchmark_partition(g, res)
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

        curr = "UConn Greedy + Random (75%)"
        print(curr)
        start = time.perf_counter_ns()
        res = algos.uconn_strat_2(g, k, lo + (dist_range * 0.75))
        end = time.perf_counter_ns()
        (
            curr_max,
            curr_wait,
            curr_min,
            curr_range,
            curr_sum,
            curr_avg,
        ) = benchmark_partition(g, res)
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
        (
            curr_max,
            curr_wait,
            curr_min,
            curr_range,
            curr_sum,
            curr_avg,
        ) = benchmark_partition(g, res)
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
        (
            curr_max,
            curr_wait,
            curr_min,
            curr_range,
            curr_sum,
            curr_avg,
        ) = benchmark_partition(g, res)
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

    return [maximums, wait_times, times, minimums, sums, ranges, averages, bests]


def alpha_heuristic_given(
    f: Callable[..., list[int]],
    graph_bank: list[Graph],
    partition_bank: list[list[set[int]]],
) -> dict[float, float]:
    """
    Benchmark function to help determine ideal alpha values for transfers and swaps
    Creates a list of average resulting sums of weighted latencies for each alpha

    Parameters
    ----------
    f: Callable[..., list[int]]
        Passed heuristic

    graph_bank: list[Graph]
        List of graphs to test over
        Assertions:
            complete graphs

    partition_bank: list[list[set[int]]]
        List of partitions associated with graphs in graph_bank
        Assertions:
            partition_bank[i] is an agent partition of graph_bank[i]

    Returns
    -------
    dict[float, float]
        Averages of sums of weighted latencies
        One for each alpha value from 0.0 to 1.0 in increments of 0.01

    """

    assert len(graph_bank) == len(partition_bank)
    for g, p in zip(graph_bank, partition_bank):
        assert Graph.is_complete(g)
        assert Graph.is_agent_partition(g, p)

    count: int = len(graph_bank)
    averages: dict[float, float] = {}

    alpha: float = 0.0
    while alpha <= 1.0:
        print(f"Current Alpha Val = {alpha}")
        sums: list[float] = []
        for i, (g, partition) in enumerate(zip(graph_bank, partition_bank)):
            print(f"Current Graph = {i}")
            output = algos.find_partition_with_heuristic(g, partition, f, alpha)
            res = solve_partition(g, output)
            _, _, _, _, curr_sum, _ = benchmark_partition(g, res)
            sums.append(curr_sum)
            print(Bcolors.CLEAR_LAST_LINE)

        averages[alpha] = sum(sums) / count
        alpha = round(alpha + 0.01, 2)
        print(Bcolors.CLEAR_LAST_LINE)

    return averages


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

    # This is slow
    # curr = "Optimal After NN transfers"
    # paths = solve_partition(g, output, algos.brute_force_mwlp)
    # curr_max = max(algos.wlp(g, path) for path in paths)
    # f = algos.generate_partition_path_function(g, paths)
    # y = [total - f(i) for i in x]
    # (line,) = ax.plot(x, y, label=f"{curr}: {curr_max}", linewidth=2.0, color="red")
    # lines.append(line)

    # This is slow
    # Brute for solve with capped sizes
    cap: int = (g.num_nodes // k) + 1
    curr = f"Best Solution with {cap = }"
    paths = algos.multi_agent_brute_force(g, k, f=algos.nearest_neighbor, max_size=cap)
    curr_max = max(algos.wlp(g, path) for path in paths)
    f = algos.generate_partition_path_function(g, paths)
    y = [total - f(i) for i in x]
    (line,) = ax.plot(
        x, y, label=f"{curr}: {curr_max}", linewidth=2.0, color="firebrick"
    )
    lines.append(line)

    # This is slow and bad
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
    figure.set_size_inches(10, 7)  # horizontal x vertical
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
