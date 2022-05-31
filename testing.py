"""
Driver code for Champaign Benchmarking
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
    ############################################################################
    ########################### Champaign Testing ##############################
    ############################################################################

    # Thanks Pranay
    # ox.config(log_console=True, use_cache=True)
    # place = "Champaign, Illinois, USA"
    # # gdf = ox.geocode_to_gdf(place)
    # # area = ox.projection.project_gdf(gdf).unary_union.area
    # # 'drive_service' := drivable and service roads both
    # G = ox.graph_from_place(place, network_type="drive_service", simplify=False)
    # G = ox.distance.add_edge_lengths(G, precision=5)

    # # From "Predicting Outage Restoration..."
    # # Agent speed was 25 mph
    # kph: float = 25.0 * 1.609344
    # print(f"Setting all travel speeds to {kph} kph")
    # for u, v, key in G.edges(keys=True):
    #     G[u][v][key]["speed_kph"] = kph
    # G = ox.add_edge_travel_times(G, precision=5)
    # print(max(G.edges(data=True),key= lambda x: x[2]['travel_time']))
    # print(min(G.edges(data=True),key= lambda x: x[2]['travel_time']))
    # # Remove unreachable /  empty nodes
    # print("Removing unreachable nodes")
    # components = list(nx.strongly_connected_components(G))
    # for item in components:
    #     if len(item) == 0 or len(item) == 1:
    #         G.remove_node(item.pop())

    # order: int = G.order()
    # print(f"{order} nodes")

    # # During repairs we do not care about one way roads
    # print("Turning G into undirected graph")
    # G = ox.utils_graph.get_undirected(G)

    # # Add population to the nearest points
    # pop_data = pd.read_csv("results/champaign/cus_blockdata.csv", index_col=0)
    # pop_points = list(pop_data.to_records(index=False))

    # def dist(x1: float, y1: float, x2: float, y2: float) -> float:
    #     x_diff: float = x1 - x2
    #     y_diff: float = y1 - y2
    #     return math.sqrt(x_diff**2 + y_diff**2)

    # print("Initializing population of each node to 0")
    # for i in G.nodes():
    #     G.nodes[i]["pop"] = 0

    # print("Adding populations")
    # for population, lat, long in pop_points:
    #     if population > 0:
    #         # Find the closest node in G to lat, long
    #         closest: int = min(
    #             G.nodes(),
    #             key=lambda i: dist(long, lat, G.nodes[i]["x"], G.nodes[i]["y"]),
    #         )

    #         # print(f"Adding {population} to node G.nodes[{closest}]['pop']")
    #         G.nodes[closest]["pop"] += population

    # print("Writing graphML")
    # ox.save_graphml(G, "results/champaign/champaign.graphml")

    print("Loading graphml")
    G = ox.load_graphml("results/champaign/champaign.graphml")

    print("Fixing population numbers")
    for node in G.nodes():
        G.nodes[node]["pop"] = int(G.nodes[node]["pop"])

    # Find populated nodes in range
    node_list: list[int] = [int(node) for node in G.nodes()]
    populated: list[int] = list(
        filter(lambda node: 1 <= G.nodes[node]["pop"] <= 1500, node_list)
    )

    print(list(G.nodes(data=True))[0])
    
    # Find important and not important
    high: list[int] = list(
        filter(lambda node: 500 <= G.nodes[node]["pop"] <= 1500, populated)
    )
    low: list[int] = list(
        filter(lambda node: 1 <= G.nodes[node]["pop"] <= 50, populated)
    )

    print(len(high), len(low))

    # sort by 'y'
    high = sorted(high, key=lambda node: G.nodes[node]['y'])
    low = sorted(low, key=lambda node: G.nodes[node]['y'])

    damaged: list[int] = []
    damaged.extend(high[-10:])
    damaged.extend(high[:10])
    damaged.extend(low[-10:])
    damaged.extend(low[:10])

    num_nodes: int = len(damaged)
    g = Graph(num_nodes)
    for i in range(num_nodes):  # make complete
        g.adjacen_list[i] = list(range(num_nodes))
    print("Initializing edge weights")
    for i in range(num_nodes):
        g.edge_weight[i] = [-1.0 for _ in range(num_nodes)]
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
        pop: int = g.node_weight[v]
        if pop <= 10:
            repair_time: float = random.uniform(2, 4)
        elif pop <= 100:
            repair_time = random.uniform(2, 6)
        elif pop <= 1000:
            repair_time = random.uniform(3, 8)
        else:
            repair_time = random.uniform(5, 10)
        for u in range(num_nodes):
            if u != v:
                g.edge_weight[u][v] += repair_time
    
    partition = [{0}, {0}, {0}, {0}]
    partition[0].update(range(0,10))
    partition[1].update(range(10,20))
    partition[2].update(range(20,30))
    partition[3].update(range(30,40))
    
    # damaged.extend(high[-10:])
    # damaged.extend(high[:10])
    # damaged.extend(low[-10:])
    # damaged.extend(low[:10])
   
    print(partition)

    transfers = algos.transfers_and_swaps_mwlp(g, partition, algos.greedy)
    assignment = benchmark.solve_partition(g, transfers, algos.greedy)
    greedy_assignment = algos.greedy_assignment(g, 4)
    
    print()

    (
        curr_max,
        curr_wait,
        curr_min,
        curr_range,
        curr_sum,
        curr_avg,
    ) = benchmark.benchmark_partition(g, assignment)
    print("T&S max", curr_max)
    print("T&S ran", curr_range)
    print("T&S avg", curr_avg)

    (
        curr_max,
        curr_wait,
        curr_min,
        curr_range,
        curr_sum,
        curr_avg,
    ) = benchmark.benchmark_partition(g, greedy_assignment)
    print("GA max", curr_max)
    print("GA ran", curr_range)
    print("GA avg", curr_avg)
    
    nc = ['r' if node in damaged else 'w' for node in G.nodes()]
    ns = [5 if node in high else 3 for node in G.nodes()]
    fig, ax = ox.plot_graph(G, node_size=ns, node_color=nc, node_zorder=2, bgcolor='k')
if __name__ == "__main__":
    main()
