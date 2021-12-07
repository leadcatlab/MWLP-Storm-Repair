from graph import graph
from itertools import permutations
from collections.abc import Iterable


def WLP(g: graph, order: tuple[int, ...]) -> float:
    if len(order) <= 1:
        return 0.0

    n = g.numNodes
    for node in order:
        assert node < n
    wlp = 0.0
    for i, node in enumerate(order[1:]):
        # find length of path
        path: float = 0.0
        for j, before in enumerate(order[:i]):
            path += g.edgeWeight[before][order[j + 1]]

        wlp += g.nodeWeight[node] * path

    return wlp


def bruteForceMWLP(g: graph) -> float:
    # TODO: THIS RELIES ON WLP WORKING FULLY BUT I HAVE NOT TESTED WLP

    # for now assume complete
    assert graph.isComplete(g)

    mwlp = float("inf")
    nodes = [i for i in range(g.numNodes)]
    for order in permutations(nodes):
        mwlp = min(mwlp, WLP(g, order))

    return mwlp
