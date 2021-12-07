from graph import graph
import algos

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


def main() -> None:
    n: int = 11
    inOrder: tuple[int, ...] = tuple(i for i in range(n))
    brute_forces: list[float] = []
    path_orders: list[float] = []
    nearest_ns: list[float] = []
    greedy_orders: list[float] = []
    for _ in range(20):
        g = graph.randomComplete(n)
        brute_forces.append(algos.bruteForceMWLP(g))
        path_orders.append(algos.WLP(g, inOrder))
        nearest_ns.append(algos.nearestNeighbor(g))
        greedy_orders.append(algos.greedy(g))

    print(
        "%-20s %-20s %-20s %-20s"
        % ("brute force", "in order", "nearest neighbor", "greedy")
    )
    for br, io, nn, gr in zip(brute_forces, path_orders, nearest_ns, greedy_orders):
        print("%-20f %-20f %-20f %-20f" % (br, io, nn, gr))

    print()
    print(
        "%-20s %-20s %-20s " % ("in order % diff", "nearest n % diff", "greedy % diff")
    )
    for i in range(20):
        in_order_percent = (
            100.0 * abs(path_orders[i] - brute_forces[i]) / brute_forces[i]
        )
        nearest_n_percent = (
            100.0 * abs(nearest_ns[i] - brute_forces[i]) / brute_forces[i]
        )
        greedy_percent = (
            100.0 * abs(greedy_orders[i] - brute_forces[i]) / brute_forces[i]
        )
        print(
            "%-20s %-20s %-20s "
            % (
                str(in_order_percent) + "%",
                str(nearest_n_percent) + "%",
                str(greedy_percent) + "%",
            )
        )


if __name__ == "__main__":
    main()
