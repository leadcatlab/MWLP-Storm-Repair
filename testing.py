"""
Driver code for testing stuff
"""
import json
import random
from typing import Any, DefaultDict

import matplotlib.pyplot as plt  # type: ignore
import networkx as nx  # type: ignore
import osmnx as ox  # type: ignore

import algos
import benchmark
from graph import Graph


class Bcolors:
    """
    Helper class for adding colors to prints
    https://svn.blender.org/svnroot/bf-blender/trunk/blender/build_files/scons/tools/Bcolors.py
    """

    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    CLEAR_LAST_LINE = (
        "\033[A                                                             \033[A"
    )


def main() -> None:
    print("Loading graphml")
    G = ox.load_graphml("results/champaign/champaign.graphml")

    print("Fixing population numbers")
    for node in G.nodes():
        G.nodes[node]["pop"] = int(G.nodes[node]["pop"])

    # Find populated nodes in range
    node_list: list[int] = [int(node) for node in G.nodes()]
    populated: list[int] = list(
        filter(lambda node: G.nodes[node]["pop"] >= 1, node_list)
    )

    # # Find important and not important
    # high: list[int] = list(
    #     filter(lambda node: 500 <= G.nodes[node]["pop"] <= 1500, populated)
    # )
    # low: list[int] = list(
    #     filter(lambda node: 1 <= G.nodes[node]["pop"] <= 50, populated)
    # )

    # print(len(high), len(low))

    # # sort by 'y'
    # high = sorted(high, key=lambda node: G.nodes[node]['y'])
    # low = sorted(low, key=lambda node: G.nodes[node]['y'])

    # damaged: list[int] = []
    # damaged.extend(low[-10:])
    # damaged.extend(low[:11]) # Get a low value node to be our start node
    # damaged.extend(high[-10:])
    # damaged.extend(high[:10])
    # 
    # nc = ['b' if node == damaged[0] else 'r' if node in damaged else 'black' for node in G.nodes()]
    # ns = [40 if (node in damaged and node in high) else 20 if (node in damaged and node in low) else 1 for node in G.nodes()]
    # fig, ax = ox.plot_graph(G, node_size=ns, node_color=nc, node_zorder=2, bgcolor='w', edge_color="black", edge_linewidth=1.1)
    
    num_nodes: int = 41
    num_agents: int = 4

    g = Graph(num_nodes)
    for i in range(num_nodes):  # make complete
        g.adjacen_list[i] = list(range(num_nodes))
    print("Initializing edge weights")
    for i in range(num_nodes):
        g.edge_weight[i] = [-1.0 for _ in range(num_nodes)]
    print(f"Choosing {num_nodes} damaged nodes")
    damaged: list[int] = list(populated)
    random.shuffle(damaged)
    damaged = damaged[:num_nodes]
    print("Adding node weights to g")
    for i in range(1, num_nodes):
        g.node_weight[i] = G.nodes[damaged[i]]["pop"]
    g.node_weight[0] = 0
    distances = {}
    print("Finding shortest path travel times in hours")
    for u in range(num_nodes):
        for v in range(u + 1, num_nodes):
            u_prime, v_prime = damaged[u], damaged[v]
            if (u_prime, v_prime) in distances:
                time = distances[(u_prime, v_prime)]
            else:
                time = nx.shortest_path_length(
                    G, u_prime, v_prime, weight="travel_time"
                )
                distances[(u_prime, v_prime)] = time
            g.edge_weight[u][v] = time / (3600)
            g.edge_weight[v][u] = time / (3600)

    print("Adding repair times")
    # Ranges from "Predicting Outage Restoration ..."
    for v in range(num_nodes):
        repair_time = random.uniform(0.083333, 0.25)
        for u in range(num_nodes):
            if u != v:
                g.edge_weight[u][v] += repair_time
   
    # use the same parameters as above
    print(f"Creating partitions for {num_agents} agents")
    partition: list[set[int]] = Graph.create_agent_partition(g, num_agents)

    print("Calculating assignments")
    assignments: list[list[list[int]]] = []
    names = ["GA", "TSG"]
    colors: list[str] = ["royalblue", "limegreen"]

    paths = algos.greedy_assignment(g, num_agents)
    assignments.append(paths)
    (
        curr_max,
        curr_wait,
        curr_min,
        curr_range,
        curr_sum,
        curr_avg,
    ) = benchmark.benchmark_partition(g, paths)
    print("GA", curr_sum)
    print("GA", curr_wait)
    print("GA", curr_range)

    # paths = algos.nearest_neighbor_assignment(g, num_agents)
    # assignments.append(paths)

    # dist_range: float = 1.0 - 0.5
    # paths = algos.greedy_random_assignment(g, num_agents, 0.5 + (dist_range * 0.25))
    # assignments.append(paths)

    part = algos.find_partition_with_heuristic(g, partition, algos.greedy, 0.13)
    paths = benchmark.solve_partition(g, part, algos.greedy)
    assignments.append(paths)
    (
        curr_max,
        curr_wait,
        curr_min,
        curr_range,
        curr_sum,
        curr_avg,
    ) = benchmark.benchmark_partition(g, paths)
    print("TSG", curr_sum)
    print("TSG", curr_wait)
    print("TSG", curr_range)

    # part = algos.find_partition_with_heuristic(
    #     g, partition, algos.nearest_neighbor, 0.13
    # )
    # paths = benchmark.solve_partition(g, part, algos.nearest_neighbor)
    # assignments.append(paths)

    benchmark.line_plot(
        g,
        assignments,
        names,
        colors,
        x_range=(0, 100),
        loc="results/champaign/champaign_north_south_line_plot.png",
    )
    
if __name__ == "__main__":
    main()
