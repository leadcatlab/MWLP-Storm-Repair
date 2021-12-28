from graph import Graph
import algos
from typing_extensions import TypedDict
import benchmark

graphDict = TypedDict(
    "graphDict",
    {
        "numNodes": int,
        "nodeWeight": list[int],
        "edges": list[tuple[int, int, float]],
    },
)


def main() -> None:
    # n: int = 4

    # g = Graph.randomComplete(n)
    # optimal_m, optimal_order = algos.optimalNumberOfAgents(
    #     g, algos.bruteForceMWLP, 1, n - 1
    # )
    # print(f"Optimal solution is {optimal_m} with {len(optimal_order)} agents")
    # print(f"{optimal_order = }")

    # g = Graph.randomCompleteMetric(n)
    # optimal_m, optimal_order = algos.optimalNumberOfAgents(
    #     g, algos.bruteForceMWLP, 1, n - 1
    # )
    # print(f"Optimal solution is {optimal_m} with {len(optimal_order)} agents")
    # print(f"{optimal_order = }")

    # benchmark.benchmarkMulti(10, 2, 5, metric=False)

    g = Graph.randomCompleteMetric(8)
    mwlp_brute = algos.bruteForceMWLP(g)
    mwlp_dp = algos.MWLP_DP(g)
    print(mwlp_brute)
    print(algos.WLP(g, mwlp_brute))
    print(mwlp_dp)
    print(algos.WLP(g, mwlp_dp))


if __name__ == "__main__":
    main()
