"""
Driver code for Champaign Benchmarking
"""
import json
import random
from typing import Any, DefaultDict

from itertools import product
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

    # # Parameters for Graphs and Partitions
    num_graphs: int = 25
    num_nodes: int = 201  # 20 agents * 10 nodes per agent + start
    num_agents: int = 20

    # # Creating smaller graph bank
    # graph_bank: list[Graph] = []
    # distances: dict[tuple[int, int], float] = {}
    # for count in range(num_graphs):
    #     print(count)

    #     # Choose random nodes to be damaged
    #     g = Graph(num_nodes)

    #     # Initializing a bunch of empty nodes and edges is faster than calling add_edge
    #     print("Initializing adjacency lists")
    #     for i in range(num_nodes):  # make complete
    #         g.adjacen_list[i] = list(range(num_nodes))
    #     print("Initializing edge weights")
    #     for i in range(num_nodes):
    #         g.edge_weight[i] = [-1.0 for _ in range(num_nodes)]

    #     print(f"Choosing {num_nodes} damaged nodes")
    #     damaged: list[int] = list(populated)
    #     random.shuffle(damaged)
    #     damaged = damaged[:num_nodes]
    #     print("Adding node weights to g")
    #     for i in range(1, num_nodes):
    #         g.node_weight[i] = G.nodes[damaged[i]]["pop"]
    #     g.node_weight[0] = 0

    #     print("Finding shortest path travel times in hours")
    #     for u in range(num_nodes):
    #         for v in range(u + 1, num_nodes):
    #             u_prime, v_prime = damaged[u], damaged[v]
    #             if (u_prime, v_prime) in distances:
    #                 time = distances[(u_prime, v_prime)]
    #             else:
    #                 time = nx.shortest_path_length(
    #                     G, u_prime, v_prime, weight="travel_time"
    #                 )
    #                 distances[(u_prime, v_prime)] = time
    #             g.edge_weight[u][v] = time / (3600)
    #             g.edge_weight[v][u] = time / (3600)

    #     graph_bank.append(g)
    #     print(Bcolors.CLEAR_LAST_LINE)
    #     print(Bcolors.CLEAR_LAST_LINE)
    #     print(Bcolors.CLEAR_LAST_LINE)
    #     print(Bcolors.CLEAR_LAST_LINE)
    #     print(Bcolors.CLEAR_LAST_LINE)
    #     print(Bcolors.CLEAR_LAST_LINE)

    # print("Generating initial partitions")
    # partition_bank: list[list[set[int]]] = benchmark.generate_agent_partitions(
    #     graph_bank, num_agents
    # )

    # # save a representative graph
    # Graph.to_file(graph_bank[0], "results/champaign/champaign_rep.json")

    # # Mass benchmark of graphs given bank
    # # (0.0, 0.5) should be a big enough range for calculated travel times
    # benchmark_results: list[DefaultDict[Any, Any]] = benchmark.mass_benchmark(
    #     graph_bank, partition_bank, (0.0, 0.5)
    # )

    # # Write to files
    # names: list[str] = [
    #     "maximums",
    #     "wait_times",
    #     "times",
    #     "minimums",
    #     "sums",
    #     "ranges",
    #     "averages",
    #     "bests",
    # ]
    # for res, name in zip(benchmark_results, names):
    #     with open(f"results/champaign/{name}.json", "w", encoding="utf-8") as outfile:
    #         json.dump(res, outfile)

    # # Box Plot for sum of weighted latencies
    # with open("results/champaign/sums.json", encoding="utf-8") as file:
    #     sums: dict[str, list[float]] = json.load(file)

    # results: list[str] = [
    #     "Greedy Assignment",
    #     "Transfers and Swaps Greedy",
    # ]
    # 
    # # Multiplying by 60 to convert to minutes
    # boxes: list[list[float]] = [[60 * num for num in sums[name]] for name in results]
    colors: list[str] = ["royalblue", "limegreen"]

    # fig, ax = plt.subplots(figsize=(4, 6))

    # bp = ax.boxplot(boxes, patch_artist=True, widths=0.60)
    # for patch, color in zip(bp["boxes"], colors):
    #     patch.set_facecolor(color)
    # for median in bp["medians"]:
    #     median.set(color="black", linewidth=3)

    # frame1 = plt.gca()
    # frame1.axes.xaxis.set_ticklabels(["GA", "TSG"])
    # ax.tick_params(axis='both', which='major', labelsize=20)
    # ax.tick_params(axis='both', which='minor', labelsize=20)
    # plt.suptitle("Sum of Weighted Latencies:\n Champaign", fontsize=20)
    # plt.gcf().axes[0].yaxis.get_major_formatter().set_scientific(False)
    # fig.savefig("results/champaign/champaign_total_work", bbox_inches="tight")

    # # Bar Plot for average wait times
    # with open("results/champaign/wait_times.json", encoding="utf-8") as file:
    #     wait: dict[str, list[float]] = json.load(file)

    # # Multiplying by 60 to convert to minutes
    # boxes = [[60 * num for num in wait[name]] for name in results]

    # fig, ax = plt.subplots(figsize=(4, 6))

    # bp = ax.boxplot(boxes, patch_artist=True, widths=0.6)
    # for patch, color in zip(bp["boxes"], colors):
    #     patch.set_facecolor(color)
    # for median in bp["medians"]:
    #     median.set(color="black", linewidth=3)

    # frame1 = plt.gca()
    # frame1.axes.xaxis.set_ticklabels(["GA", "TSG"])
    # ax.tick_params(axis='both', which='major', labelsize=20)
    # ax.tick_params(axis='both', which='minor', labelsize=20)
    # plt.suptitle("Average Wait Time:\n Champaign (Minutes)", fontsize=20)
    # plt.gcf().axes[0].yaxis.get_major_formatter().set_scientific(False)
    # fig.savefig("results/champaign/champaign_wait_time", bbox_inches="tight")

    # # Bar Plot for ranges
    # with open("results/champaign/ranges.json", encoding="utf-8") as file:
    #     ranges: dict[str, list[float]] = json.load(file)

    # # Multiplying by 60 to convert to minutes
    # boxes = [[60 * num for num in ranges[name]] for name in results]

    # fig, ax = plt.subplots(figsize=(4, 6))

    # bp = ax.boxplot(boxes, patch_artist=True, widths=0.6)
    # for patch, color in zip(bp["boxes"], colors):
    #     patch.set_facecolor(color)
    # for median in bp["medians"]:
    #     median.set(color="black", linewidth=3)

    # frame1 = plt.gca()
    # frame1.axes.xaxis.set_ticklabels(["GA", "TSG"])
    # ax.tick_params(axis='both', which='major', labelsize=20)
    # ax.tick_params(axis='both', which='minor', labelsize=20)
    # plt.suptitle("Range of Weighted Latencies:\n Champaign", fontsize=20)
    # plt.gcf().axes[0].yaxis.get_major_formatter().set_scientific(False)
    # fig.savefig("results/champaign/champaign_ranges", bbox_inches="tight")

    
    # Choose random nodes to be damaged
    g = Graph(num_nodes)

    # Initializing a bunch of empty nodes and edges is faster than calling add_edge
    print("Initializing adjacency lists")
    for i in range(num_nodes):  # make complete
        g.adjacen_list[i] = list(range(num_nodes))
    print("Initializing edge weights")
    for i in range(num_nodes):
        g.edge_weight[i] = [-1.0 for _ in range(num_nodes)]

    print(f"Choosing {num_nodes} damaged nodes")
    damaged = list(populated)
    random.shuffle(damaged)
    damaged = damaged[:num_nodes]
    print("Adding node weights to g")
    for i in range(1, num_nodes):
        g.node_weight[i] = G.nodes[damaged[i]]["pop"]
    g.node_weight[0] = 0

    print("Finding shortest path travel times in minutes")
    for u in range(num_nodes):
        for v in range(u + 1, num_nodes):
            u_prime, v_prime = damaged[u], damaged[v]
            time = nx.shortest_path_length(
                G, u_prime, v_prime, weight="travel_time"
            )
            g.edge_weight[u][v] = time / (60)
            g.edge_weight[v][u] = time / (60)
    
    nc = ['#43bf6e' if node == damaged[0] else 'r' if node in damaged else 'black' for node in G.nodes()]
    ns = [50 if node == damaged[0] else 20 if node in damaged else 1 for node in G.nodes()] 
    fig, ax = ox.plot_graph(G, node_size=ns, node_color=nc, node_zorder=2, bgcolor='w', edge_color="black", edge_linewidth=1.1)
    
    # use the same parameters as above
    print(f"Creating partitions for {num_agents} agents")
    partition: list[set[int]] = Graph.create_agent_partition(g, num_agents)

    print("Calculating assignments")
    assignments: list[list[list[int]]] = []
    names = ["GA", "TSG"]

    paths = algos.greedy_assignment(g, num_agents)
    assignments.append(paths)


    part = algos.find_partition_with_heuristic(g, partition, algos.greedy, 0.13)
    paths = benchmark.solve_partition(g, part, algos.greedy)
    assignments.append(paths)

    benchmark.line_plot(
        g,
        assignments,
        names,
        colors,
        x_range=(0, 80),
        loc="results/champaign/champaign_unvisited.png",
    )


if __name__ == "__main__":
    main()
