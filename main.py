from graph import graph
import algos
import numpy as np

# graphDict = {
#    "numNodes": 4,
#    "nodeWeight": [10, 20, 30, 40],
#    "edges": [(0, 1, 1.0), (2, 3, 1.1)],
# }
#
# g = graph.FromDict(graphDict)
#
# print("================================")
# print("The adjacency list")
# g.printAdjacencyList()
# print("================================")
# print("The edge matrix")
# g.printEdgeStrength()
# print("================================")
# print("Node weight")
# g.printNodeWeight()
# print()
#
# completeGraph = {
#    "numNodes": 2,
#    "nodeWeight": [0, 1],
#    "edges": [(0, 1, 1.0), (1, 0, 2.0)]
# }
# complete = graph.FromDict(completeGraph)
# print(graph.isComplete(complete))
#
# incompleteGraph = {
#    "numNodes": 2,
#    "nodeWeight": [0, 1],
#    "edges": [(1, 0, 2.0)]
# }
# incomplete = graph.FromDict(incompleteGraph)
# print(graph.isComplete(incomplete))


def benchmark(n: int, rounds: int) -> None:
    inOrder: tuple[int, ...] = tuple(i for i in range(n))
    brute_forces: list[float] = []
    in_orders: list[float] = []
    random_orders: list[float] = []
    nearest_ns: list[float] = []
    greedy_orders: list[float] = []
    for _ in range(rounds):
        g = graph.randomComplete(n)
        brute_forces.append(algos.bruteForceMWLP(g))
        in_orders.append(algos.WLP(g, inOrder))
        random_orders.append(
            algos.WLP(g, [0] + list(np.random.permutation([i for i in range(1, n)])))
        )
        nearest_ns.append(algos.nearestNeighbor(g))
        greedy_orders.append(algos.greedy(g))

    print(
        f"{'brute force':22}{'in order':22}{'random order':22}{'nearest neighbor':22}{'greedy':22} "
    )
    for br, io, ro, nn, gr in zip(
        brute_forces, in_orders, random_orders, nearest_ns, greedy_orders
    ):
        print(f"{br : <22}{io : <22}{ro : <22}{nn : <22}{gr : <22}")

    print()
    print(
        f"{'' : <22}{'in order % diff' : <22}{'random % diff' : <22}{'nearest n % diff' : <22}{'greedy % diff' : <22}"
    )

    io_sum: float = 0.0
    rand_sum: float = 0.0
    nn_sum: float = 0.0
    greedy_sum: float = 0.0
    for i in range(rounds):
        in_order_percent: float = (
            100.0 * abs(in_orders[i] - brute_forces[i]) / brute_forces[i]
        )
        io_sum += in_order_percent

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
            f"{'' : <22}{str(in_order_percent) + '%' : <22}{str(random_percent) + '%' : <22}{str(nearest_n_percent) + '%' : <22}{str(greedy_percent) + '%' : <22}"
        )

    print()
    print(" " * 22 + "Average of above % diffs")
    io_av = io_sum / rounds
    rand_av = rand_sum / rounds
    nn_av = nn_sum / rounds
    greedy_av = greedy_sum / rounds
    print(
        f"{'' : <22}{str(io_av) + '%' : <22}{str(rand_av) + '%' : <22}{str(nn_av) + '%' : <22}{str(greedy_av) + '%' : <22}"
    )


if __name__ == "__main__":
    benchmark(8, 20)
