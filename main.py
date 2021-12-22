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
    g = Graph.randomComplete(10)
    optimal_m, optimal_order = algos.optimalNumberOfAgents(
        g, algos.bruteForceMWLP, 1, 9
    )
    print(f"Optimal solution is {optimal_m} with {len(optimal_order)} agents")
    print(f"{optimal_order = }")

    g = Graph.randomCompleteMetric(10)
    optimal_m, optimal_order = algos.optimalNumberOfAgents(
        g, algos.bruteForceMWLP, 1, 9
    )
    print(f"Optimal solution is {optimal_m} with {len(optimal_order)} agents")
    print(f"{optimal_order = }")


if __name__ == "__main__":
    main()
