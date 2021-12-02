import graph
from helper import permutation

def WLP(g, order: list[int]) -> float:
    if len(order) <= 1:
        return 0.0

    n = g.numNodes
    for node in order:
        assert(node < n)
    wlp = 0.0
    for node in order[1:]:
        # find length of path
        path: float = 0.0
        for before in order[:node]:
            path += g.edgeWeight[before][before + 1]

        wlp += g.nodeWeight[node] * path

    return wlp


def brute_force(G):
    list = []

    #since its a complete graph, we could just take a permutation of the given nodes
    for i in range(G.numNodes):
        list.append(i)
    perm = permutation(list)

    


def random_order(G):
    return 0

def nearest_neighbour(G):
    return 0

def greedy_path(G):
    return 0


