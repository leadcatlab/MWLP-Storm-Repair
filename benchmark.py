import random
from graph import Graph
import algos


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
    Creates random complete (metric) graphs and runs the heuristics and prints comparisons

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
        brute_forces.append(algos.WLP(g, algos.brute_force_MWLP(g)))
        tsp_orders.append(algos.WLP(g, algos.held_karp(g)))
        random_orders.append(algos.WLP(g, algos.random_order(g)))
        nearest_ns.append(algos.WLP(g, algos.nearest_neighbor(g)))
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
    Creates random complete (metric) graphs and runs the heuristics and prints comparisons

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

        brute_m, _ = algos.partition_heuristic(g, algos.brute_force_MWLP, k)
        brute_forces.append(brute_m)

        tsp_m, _ = algos.partition_heuristic(g, algos.held_karp, k)
        tsp_orders.append(tsp_m)

        partition: list[list[int]] = [[] for _ in range(k)]
        for i in range(n):
            partition[random.randint(0, 1000) % k].append(i)

        rand_total: float = 0.0
        for part in partition:
            rand_total += algos.WLP(g, part)

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
