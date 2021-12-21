from graph import Graph
import algos
import numpy as np
from more_itertools import set_partitions


def benchmarkSingle(
    n: int,
    rounds: int,
    metric: bool = False,
    edgeW: tuple[float, float] = (0, 1),
    nodeW: tuple[int, int] = (1, 100),
    upper: float = 1,
) -> None:

    """Benchmarks various heuristics against bruteforce MWLP

    Tests the following heuristics: Random order, nearest neighbor, greedy, TSP
    Creates random complete (metric) graphs and runs the heuristics and prints comparisons

    Args:
        n: number of nodes for the graphs
        rounds: number of graphs to test
        metric: Create metric graphs if True, else False
        edgeW: interval for edge weights
        nodeW: interval for node weights
        upper: upper bound edge weight for metric graphs
    """

    if n <= 0 or rounds <= 0:
        return
    assert edgeW[0] < edgeW[1]
    assert nodeW[0] < nodeW[1]
    assert upper >= 0

    # Print arguments
    print(f"{n = }")
    print(f"{rounds = }")
    print(f"{metric = }")
    print(f"{edgeW = }")
    print(f"{nodeW = }")
    print(f"{upper = }")
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
            Graph.randomCompleteMetric(n, upper, nodeW=nodeW)
            if metric
            else Graph.randomComplete(n, edgeW=edgeW, nodeW=nodeW)
        )
        brute_forces.append(algos.WLP(g, algos.bruteForceMWLP(g)))
        tsp_orders.append(algos.WLP(g, algos.TSP(g)))
        random_orders.append(
            algos.WLP(g, [0] + list(np.random.permutation([i for i in range(1, n)])))
        )
        nearest_ns.append(algos.WLP(g, algos.nearestNeighbor(g)))
        greedy_orders.append(algos.WLP(g, algos.greedy(g)))

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


def benchmarkMulti(
    n: int,
    k: int,
    rounds: int,
    metric: bool = False,
    edgeW: tuple[float, float] = (0, 1),
    nodeW: tuple[int, int] = (1, 100),
    upper: float = 1,
) -> None:

    """Benchmarks various heuristics against bruteforce MWLP

    Tests the following heuristics: Random order, nearest neighbor, greedy, TSP
    Creates random complete (metric) graphs and runs the heuristics and prints comparisons

    Args:
        n: number of nodes for the graphs
        k: number of agents
        rounds: number of graphs to test
        metric: Create metric graphs if True, else False
        edgeW: interval for edge weights
        nodeW: interval for node weights
        upper: upper bound edge weight for metric graphs
    """

    if n <= 0 or rounds <= 0:
        return
    assert edgeW[0] < edgeW[1]
    assert nodeW[0] < nodeW[1]
    assert upper >= 0

    # Print arguments
    print(f"{n = }")
    print(f"{k = }")
    print(f"{rounds = }")
    print(f"{metric = }")
    print(f"{edgeW = }")
    print(f"{nodeW = }")
    print(f"{upper = }")
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
            Graph.randomCompleteMetric(n, upper, nodeW=nodeW)
            if metric
            else Graph.randomComplete(n, edgeW=edgeW, nodeW=nodeW)
        )

        brute_m, brute_order = algos.partitionHeuristic(g, algos.bruteForceMWLP, k)
        brute_forces.append(brute_m)

        tsp_m, tsp_order = algos.partitionHeuristic(g, algos.TSP, k)
        tsp_orders.append(tsp_m)

        # arbitrarily pick first possible partition as "random"
        nodes: list[int] = list(range(n))
        rand: list[list[int]] = next(set_partitions(nodes, k))
        print(rand)
        rand_total: float = 0.0
        for part in rand:
            rand_total += algos.WLP(g, part)

        random_orders.append(rand_total)

        nn_m, nn_order = algos.partitionHeuristic(g, algos.nearestNeighbor, k)
        nearest_ns.append(nn_m)

        greedy_m, greedy_order = algos.partitionHeuristic(g, algos.greedy, k)
        greedy_orders.append(tsp_m)

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
