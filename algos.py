import graph
from helper import permutation


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


