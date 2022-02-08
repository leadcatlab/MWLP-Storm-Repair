from graph import Graph
import algos
from typing_extensions import TypedDict
import random

graph_dict = TypedDict(
    "graph_dict",
    {
        "numNodes": int,
        "node_weight": list[int],
        "edges": list[tuple[int, int, float]],
    },
)


def main() -> None:
    n: int = 12
    k: int = 4
    g = Graph.random_complete_metric(n, directed=False)
    
    # TODO: Create function to generate partition
    # TODO: Create function to evalute before and after of transfers/swaps

    partition: list[set[int]] = [set() for _ in range(k)]
    for i in range(n):
        partition[random.randint(0, 1000) % k].add(i)

    # before: str = f"Before: {algos.nearest_neighbor(g, partition)}\n"
    before: str = ""
    for i in range(len(partition)):
        before += f"    Agent {i}: {partition[i]}\n"
    print(before)

    res = algos.transfers_and_swaps_mwlp(g, partition, algos.nearest_neighbor)

    # after: str = f"After: {algos.nearest_neighbor(g, res)}\n"
    after: str = ""
    for i in range(len(res)):
        after += f"    Agent {i}: {res[i]}\n"
    print(after)


if __name__ == "__main__":
    main()
