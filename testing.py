from graph import graph
import algos
import numpy as np

# TODO: Add test cases and configure pytest
# TODO: comments 

def benchmark(n: int, rounds: int) -> None:
    brute_forces: list[float] = []
    tsp_orders: list[float] = []
    random_orders: list[float] = []
    nearest_ns: list[float] = []
    greedy_orders: list[float] = []

    for _ in range(rounds):
        g = graph.randomComplete(n)
        brute_forces.append(algos.bruteForceMWLP(g))
        tsp_orders.append(algos.WLP(g, algos.TSP(g)))
        random_orders.append(
            algos.WLP(g, [0] + list(np.random.permutation([i for i in range(1, n)])))
        )
        nearest_ns.append(algos.nearestNeighbor(g))
        greedy_orders.append(algos.greedy(g))

    print(
        f"{'brute force':22}"
        + f"{'TSP':22}"
        + f"{'random order':22}"
        + f"{'nearest neighbor':22}"
        + f"{'greedy':22}"
    )

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

    print()
    print(
        f"{'' : <22}"
        + f"{'tsp % diff' : <22}"
        + f"{'random % diff' : <22}"
        + f"{'nearest n % diff' : <22}"
        + f"{'greedy % diff' : <22}"
    )

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
