import random
from collections import defaultdict
from typing import Callable

import algos
from graph import Graph


# https://svn.blender.org/svnroot/bf-blender/trunk/blender/build_files/scons/tools/Bcolors.py
class Bcolors:
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


def benchmark_single(
    n: int,
    rounds: int,
    metric: bool = False,
    edge_w: tuple[float, float] = (0, 1),
    node_w: tuple[int, int] = (1, 100),
    upper: float = 1,
) -> None:

    """Benchmarks various heuristics against bruteforce MWLP

    Tests the following heuristics: Random order, nearest neighbor, greedy, TSP
    on random complete (metric) graphs and prints comparisons

    Args:
        n: number of nodes for the graphs
        rounds: number of graphs to test
        metric: Create metric graphs if True, else False
        edge_w: interval for edge weights
        node_w: interval for node weights
        upper: upper bound edge weight for metric graphs
    """

    if n <= 0 or rounds <= 0:
        return
    assert edge_w[0] < edge_w[1]
    assert node_w[0] < node_w[1]
    assert upper >= 0

    # Print arguments
    print(f"{n = }")
    print(f"{rounds = }")
    print(f"{metric = }")
    if not metric:
        print(f"{edge_w = }")
    else:
        print(f"{upper = }")
    print(f"{node_w = }")
    print()

    # Lists to store results
    brute_forces: list[float] = []
    tsp_orders: list[float] = []
    random_orders: list[float] = []
    nearest_ns: list[float] = []
    greedy_orders: list[float] = []

    # Run heuristics
    for _ in range(rounds):
        g = (
            Graph.random_complete_metric(n, upper, node_w=node_w)
            if metric
            else Graph.random_complete(n, edge_w=edge_w, node_w=node_w)
        )
        brute_forces.append(algos.wlp(g, algos.brute_force_mwlp(g)))
        tsp_orders.append(algos.wlp(g, algos.held_karp(g)))
        random_orders.append(algos.wlp(g, algos.random_order(g)))
        nearest_ns.append(algos.wlp(g, algos.nearest_neighbor(g)))
        greedy_orders.append(algos.wlp(g, algos.greedy(g)))

    print(
        f"{'brute force':22}"
        + f"{'TSP':22}"
        + f"{'random order':22}"
        + f"{'nearest neighbor':22}"
        + f"{'greedy':22}"
    )

    # Print results
    for br, ts, ro, nn, gr in zip(
        brute_forces, tsp_orders, random_orders, nearest_ns, greedy_orders
    ):
        print(
            f"{br : <22}"
            + f"{ts : <22}"
            + f"{ro : <22}"
            + f"{nn : <22}"
            + f"{gr : <22}"
        )

    print(
        "\n"
        + f"{'' : <22}"
        + f"{'tsp % diff' : <22}"
        + f"{'random % diff' : <22}"
        + f"{'nearest n % diff' : <22}"
        + f"{'greedy % diff' : <22}"
    )

    # Calculate averages if needed
    if rounds > 1:
        # Find sums
        tsp_sum: float = 0.0
        rand_sum: float = 0.0
        nn_sum: float = 0.0
        greedy_sum: float = 0.0
        for i in range(rounds):
            tsp_order_percent: float = (
                100.0 * abs(tsp_orders[i] - brute_forces[i]) / brute_forces[i]
            )
            tsp_sum += tsp_order_percent

            random_percent: float = (
                100.0 * abs(random_orders[i] - brute_forces[i]) / brute_forces[i]
            )
            rand_sum += random_percent

            nearest_n_percent: float = (
                100.0 * abs(nearest_ns[i] - brute_forces[i]) / brute_forces[i]
            )
            nn_sum += nearest_n_percent

            greedy_percent: float = (
                100.0 * abs(greedy_orders[i] - brute_forces[i]) / brute_forces[i]
            )
            greedy_sum += greedy_percent

            print(
                f"{'' : <22}"
                + f"{str(tsp_order_percent) + '%' : <22}"
                + f"{str(random_percent) + '%' : <22}"
                + f"{str(nearest_n_percent) + '%' : <22}"
                + f"{str(greedy_percent) + '%' : <22}"
            )

        print()

        # Find averages
        print(" " * 22 + "Average of above % diffs")
        tsp_av = tsp_sum / rounds
        rand_av = rand_sum / rounds
        nn_av = nn_sum / rounds
        greedy_av = greedy_sum / rounds
        print(
            f"{'' : <22}"
            + f"{str(tsp_av) + '%' : <22}"
            + f"{str(rand_av) + '%' : <22}"
            + f"{str(nn_av) + '%' : <22}"
            + f"{str(greedy_av) + '%' : <22}"
        )


def benchmark_multi(
    n: int,
    k: int,
    rounds: int,
    metric: bool = False,
    edge_w: tuple[float, float] = (0, 1),
    node_w: tuple[int, int] = (1, 100),
    upper: float = 1,
) -> None:

    """Benchmarks various heuristics against bruteforce multi-agent MWLP

    Tests the following heuristics: Random order, nearest neighbor, greedy, TSP
    on random complete (metric) graphs and prints comparisons

    Args:
        n: number of nodes for the graphs
        k: number of agents
        rounds: number of graphs to test
        metric: Create metric graphs if True, else False
        edge_w: interval for edge weights
        node_w: interval for node weights
        upper: upper bound edge weight for metric graphs
    """

    if n <= 0 or rounds <= 0:
        return
    assert edge_w[0] < edge_w[1]
    assert node_w[0] < node_w[1]
    assert upper >= 0

    # Print arguments
    print(f"{n = }")
    print(f"{k = }")
    print(f"{rounds = }")
    print(f"{metric = }")
    if not metric:
        print(f"{edge_w = }")
    else:
        print(f"{upper = }")
    print(f"{node_w = }")
    print()

    # Lists to store results
    brute_forces: list[float] = []
    tsp_orders: list[float] = []
    random_orders: list[float] = []
    nearest_ns: list[float] = []
    greedy_orders: list[float] = []

    # Run heuristics
    for _ in range(rounds):
        g = (
            Graph.random_complete_metric(n, upper, node_w=node_w)
            if metric
            else Graph.random_complete(n, edge_w=edge_w, node_w=node_w)
        )

        brute_m, _ = algos.partition_heuristic(g, algos.brute_force_mwlp, k)
        brute_forces.append(brute_m)

        tsp_m, _ = algos.partition_heuristic(g, algos.held_karp, k)
        tsp_orders.append(tsp_m)

        partition: list[list[int]] = [[] for _ in range(k)]
        for i in range(n):
            partition[random.randint(0, 1000) % k].append(i)

        rand_total: float = 0.0
        for part in partition:
            rand_total += algos.wlp(g, part)

        random_orders.append(rand_total)

        nn_m, _ = algos.partition_heuristic(g, algos.nearest_neighbor, k)
        nearest_ns.append(nn_m)

        greedy_m, _ = algos.partition_heuristic(g, algos.greedy, k)
        greedy_orders.append(greedy_m)

    print(
        f"{'brute force':22}"
        + f"{'TSP':22}"
        + f"{'random order':22}"
        + f"{'nearest neighbor':22}"
        + f"{'greedy':22}"
    )

    # Print results
    for br, ts, ro, nn, gr in zip(
        brute_forces, tsp_orders, random_orders, nearest_ns, greedy_orders
    ):
        print(
            f"{br : <22}"
            + f"{ts : <22}"
            + f"{ro : <22}"
            + f"{nn : <22}"
            + f"{gr : <22}"
        )

    print(
        "\n"
        + f"{'' : <22}"
        + f"{'tsp % diff' : <22}"
        + f"{'random % diff' : <22}"
        + f"{'nearest n % diff' : <22}"
        + f"{'greedy % diff' : <22}"
    )

    # Calculate averages if needed
    if rounds > 1:
        # Find sums
        tsp_sum: float = 0.0
        rand_sum: float = 0.0
        nn_sum: float = 0.0
        greedy_sum: float = 0.0
        for i in range(rounds):
            tsp_order_percent: float = (
                100.0 * abs(tsp_orders[i] - brute_forces[i]) / brute_forces[i]
            )
            tsp_sum += tsp_order_percent

            random_percent: float = (
                100.0 * abs(random_orders[i] - brute_forces[i]) / brute_forces[i]
            )
            rand_sum += random_percent

            nearest_n_percent: float = (
                100.0 * abs(nearest_ns[i] - brute_forces[i]) / brute_forces[i]
            )
            nn_sum += nearest_n_percent

            greedy_percent: float = (
                100.0 * abs(greedy_orders[i] - brute_forces[i]) / brute_forces[i]
            )
            greedy_sum += greedy_percent

            print(
                f"{'' : <22}"
                + f"{str(tsp_order_percent) + '%' : <22}"
                + f"{str(random_percent) + '%' : <22}"
                + f"{str(nearest_n_percent) + '%' : <22}"
                + f"{str(greedy_percent) + '%' : <22}"
            )

        print()

        # Find averages
        print(" " * 22 + "Average of above % diffs")
        tsp_av = tsp_sum / rounds
        rand_av = rand_sum / rounds
        nn_av = nn_sum / rounds
        greedy_av = greedy_sum / rounds
        print(
            f"{'' : <22}"
            + f"{str(tsp_av) + '%' : <22}"
            + f"{str(rand_av) + '%' : <22}"
            + f"{str(nn_av) + '%' : <22}"
            + f"{str(greedy_av) + '%' : <22}"
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
