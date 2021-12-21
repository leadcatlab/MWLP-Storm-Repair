from graph import Graph
import algos
from typing_extensions import TypedDict

# import benchmark

graphDict = TypedDict(
    "graphDict",
    {
        "numNodes": int,
        "nodeWeight": list[int],
        "edges": list[tuple[int, int, float]],
    },
)


def main() -> None:
    # benchmark.benchmark(7, 20, metric=True, upper=5)

    g = Graph.randomComplete(12)

    mwlp_m, mwlp_order = algos.partitionHeuristic(g, algos.bruteForceMWLP, 3)
    print(f"Brute Force: {mwlp_m} with {mwlp_order}")

    nn_m, nn_order = algos.partitionHeuristic(g, algos.nearestNeighbor, 3)
    print(f"Nearest Neighbor: {nn_m} with {nn_order}")

    g_m, g_order = algos.partitionHeuristic(g, algos.greedy, 3)
    print(f"Greedy: {g_m} with {g_order}")

    tsp_m, tsp_order = algos.partitionHeuristic(g, algos.TSP, 3)
    print(f"TSP Approx: {tsp_m} with {tsp_order}")


if __name__ == "__main__":
    main()
