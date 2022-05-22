"""
Driver code for testing functions
"""
import json
import math
import random
from itertools import product
from typing import Any, DefaultDict

import matplotlib.pyplot as plt  # type: ignore
import networkx as nx  # type: ignore
import osmnx as ox  # type: ignore
import pandas as pd  # type: ignore

import algos
import benchmark
from graph import Graph, graph_dict


def main() -> None:
    # Messing with plotting
    # n = 100
    # k = 8
    # g: Graph = Graph.random_complete_metric(n)
    # nx_g = Graph.to_networkx(g)

    # part: list[set[int]] = Graph.create_agent_partition(g, k)
    # initial_assignment: list[list[int]] = benchmark.solve_partition(
    #     g, part, algos.nearest_neighbor
    # )
    # benchmark.draw_graph_with_partitions(nx_g, initial_assignment, "Initial")

    # output = algos.find_partition_with_heuristic(
    #         g,
    #         part,
    #         algos.nearest_neighbor,
    #         0.18
    #     )
    # final_assignment = benchmark.solve_partition(g, output, algos.nearest_neighbor)
    # benchmark.draw_graph_with_partitions(nx_g, final_assignment, "Nearest Neighbor")
    # plt.show()

    # Testing Champaign data
    # Thanks Pranay
    ox.config(log_console=True, use_cache=True)
    place = "Champaign, Illinois, USA"
    # gdf = ox.geocode_to_gdf(place)
    # area = ox.projection.project_gdf(gdf).unary_union.area
    # 'drive_service' := drivable and service roads both
    G = ox.graph_from_place(place, network_type="drive_service", simplify=True)

    # Remove unreachable /  empty nodes
    print("Removing unreachable nodes")
    components = list(nx.strongly_connected_components(G))
    for item in components:
        if len(item) == 0 or len(item) == 1:
            G.remove_node(item.pop())

    # During repairs we do not care about one way roads
    print("Turning G into undirected graph")
    G = ox.utils_graph.get_undirected(G)

    # Set speed of repair crews to 30 km / h
    G = ox.add_edge_speeds(G, fallback=30)
    G = ox.add_edge_travel_times(G)

    # Add population to the nearest points
    pop_data = pd.read_csv("cus_blockdata.csv", index_col=0)
    pop_points = list(pop_data.to_records(index=False))

    def dist(x1: float, y1: float, x2: float, y2: float) -> float:
        x_diff: float = x1 - x2
        y_diff: float = y1 - y2
        return math.sqrt(x_diff**2 + y_diff**2)

    print("Initializing population of each node to 0")
    for i in G.nodes():
        G.nodes[i]["pop"] = 0

    print("Adding populations")
    for population, lat, long in pop_points:
        # Find the closest node in G to lat, long
        closest: int = min(
            G.nodes(), key=lambda i: dist(long, lat, G.nodes[i]["x"], G.nodes[i]["y"])
        )

        # print(f"Adding {population} to node G.nodes[{closest}]['pop']")
        G.nodes[closest]["pop"] += population
    
    print("Writing graphML")
    ox.save_graphml(G, "champaign.graphml")

    n: int = G.order()
    # Initializing a bunch of empty nodes and edges is faster than calling add_edge
    g = Graph(n)
    print("Initializing adjacency lists")
    for i in range(n):  # make complete
        g.adjacen_list[i] = list(range(n))
    print("Initializing edge weights")
    for i in range(n):
        g.edge_weight[i] = [-1.0 for _ in range(n)]

    # create a random ordering of nodes with node 0 = start
    rand_order: list[int] = list(G.nodes)
    random.shuffle(rand_order)

    # map to relate nodes between g and G
    # node_map[i] = x := node i in g corresponding to node x in G
    node_map: dict[int, int] = {i: rand_order[i] for i in range(n)}
    
    print("Adding node weights to g")
    for i in range(n):
        g.node_weight[i] = G.nodes[rand_order[i]]['pop']

    # Use APSP algorithm to add edge weights
    print("Solving APSP")
    apsp = dict(nx.all_pairs_shortest_path_length(G))
    print("Adding edges to g")
    for u, v in product(range(n), range(n)):
        if u != v:
            u_prime, v_prime = node_map[u], node_map[v]
            g.edge_weight[u][v] = apsp[u_prime][v_prime]

    print("Checking completeness of g")
    for u, v in product(range(n), range(n)):
        if u != v: 
            if g.edge_weight[u][v] <= 0.0:
                print("Graph is incomplete")
                break
    else:
        print("Graph is complete")

    print("Checking directedness of g")
    for u, v in product(range(n), range(n)):
        if u != v:
            if g.edge_weight[u][v] != g.edge_weight[v][u]:
                print("Graph is directed")
                break
    else:
        print("Graph is undirected")
    
    # print("Writing graph json")
    # gd: graph_dict = Graph.dict_from_graph(g)
    # loc: str = "champaign_graph.json"
    # with open(loc, "w", encoding="utf-8") as outfile:
    #     json.dump(gd, outfile)


    # sorted_pop = sorted(list(g.nodes), key=lambda i: g.nodes[i]["pop"])
    # # for i in sorted_pop:
    # #     print(f"{i}: {G.nodes[i]['pop']}")

    # max_pop = g.nodes[sorted_pop[-1]]["pop"]

    # ox.plot_graph(
    #     g,
    #     node_size=[(G.nodes[i]["pop"] / max_pop) * 100 for i in g.nodes],
    #     node_color="#261CE9",
    # )

    ############################################################################
    ###################### Finalizing Mass Benchmarking ########################
    ############################################################################

    # # Read from generated bank files
    # graphs_from_file: list[Graph] = benchmark.graph_bank_from_file(
    #     "results/alpha/final_graph.json"
    # )
    # parts_from_file: list[list[set[int]]] = benchmark.agent_partitions_from_file(
    #     "results/alpha/final_part.json"
    # )
    # assert len(graphs_from_file) == len(parts_from_file)
    # for g, p in zip(graphs_from_file, parts_from_file):
    #     assert Graph.is_agent_partition(g, p)

    # # Mass benchmark of graphs given bank
    # results: list[DefaultDict[Any, Any]] = benchmark.mass_benchmark(
    #     graphs_from_file, parts_from_file, (5.0, 10.0)
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
    # for res, name in zip(results, names):
    #     with open(
    #         f"results/mass_benchmark/{name}.json", "w", encoding="utf-8"
    #     ) as outfile:
    #         json.dump(res, outfile)

    ############################################################################
    ############### Code used for finalizing Alpha benchmarking ################
    ############################################################################

    # # Parameters for Graphs and Partitions
    # num_graphs: int = 20
    # num_nodes: int = 56
    # edge_w: tuple[float, float] = (5.0, 10.0)
    # metric = False
    # node_w: tuple[int, int] = (10, 100)
    # num_agents: int = 8
    # repair_time: float = 2.0

    # # Generating Graphs
    # graph_bank: list[Graph] = benchmark.generate_graph_bank(
    #     count=num_graphs,
    #     n=num_nodes,
    #     edge_w=edge_w,
    #     metric=metric,
    #     node_w=node_w
    # )

    # # Adding Repair Times
    # for i, g in enumerate(graph_bank):
    #     g.add_repair_time(repair_time)
    #     graph_bank[i] = g
    # # Generating Partitions
    # partition_bank: list[list[set[int]]] = benchmark.generate_agent_partitions(
    #     graph_bank, num_agents
    # )

    # # Serializing Graphs and Partitions
    # graphs: dict[int, graph_dict] = {}
    # parts: dict[int, list[list[int]]] = {}
    # graph_dict_bank: list[graph_dict] = [Graph.dict_from_graph(g) for g in graph_bank]
    # # sets and frozensets are both unserializable
    # serializable_partition_bank: list[list[list[int]]] = [
    #     [list(s) for s in part] for part in partition_bank
    # ]
    # for i in range(num_graphs):
    #     graphs[i] = graph_dict_bank[i]
    #     parts[i] = serializable_partition_bank[i]
    # # Saving Graphs and Partitions to files
    # loc: str = "results/alpha/final_graph.json"
    # with open(loc, "w", encoding="utf-8") as outfile:
    #     json.dump(graphs, outfile)
    # loc = "results/alpha/final_part.json"
    # with open(loc, "w", encoding="utf-8") as outfile:
    #     json.dump(parts, outfile)

    # # Read from generated bank files
    # graphs_from_file: list[Graph] = benchmark.graph_bank_from_file(
    #     "results/alpha/final_graph.json"
    # )
    # parts_from_file: list[list[set[int]]] = benchmark.agent_partitions_from_file(
    #     "results/alpha/final_part.json"
    # )
    # assert len(graphs_from_file) == len(parts_from_file)
    # for g, p in zip(graphs_from_file, parts_from_file):
    #     assert Graph.is_agent_partition(g, p)

    # # Run Alpha Heuristic Benchmark
    # greedy_alpha_dict: dict[float, float] = benchmark.alpha_heuristic_given(
    #     algos.greedy, graphs_from_file, parts_from_file
    # )
    # for alpha, val in greedy_alpha_dict.items():
    #     print(f"{alpha:.2f}: {val}")
    # print()
    # with open(
    #     "results/alpha/alpha_greedy_final_results.json", "w", encoding="utf-8"
    # ) as outfile:
    #     json.dump(greedy_alpha_dict, outfile)

    # nn_alpha_dict: dict[float, float] = benchmark.alpha_heuristic_given(
    #     algos.nearest_neighbor, graphs_from_file, parts_from_file
    # )
    # for alpha, val in nn_alpha_dict.items():
    #     print(f"{alpha:.2f}: {val}")
    # print()
    # with open(
    #     "results/alpha/alpha_nn_final_results.json", "w", encoding="utf-8"
    # ) as outfile:
    #     json.dump(nn_alpha_dict, outfile)

    # # Plot results
    # plt.plot(
    #     list(greedy_alpha_dict.keys()), list(greedy_alpha_dict.values()),
    #     label="greedy"
    # )
    # plt.plot(list(nn_alpha_dict.keys()), list(nn_alpha_dict.values()), label="nn")
    # plt.legend()
    # plt.show()


if __name__ == "__main__":
    main()
