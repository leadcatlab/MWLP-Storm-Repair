"""
Driver code for testing functions
"""
from typing_extensions import TypedDict

from graph import Graph
import matplotlib.pyplot as plt
import mplcursors
import algos
import benchmark
import numpy as np

graph_dict = TypedDict(
    "graph_dict",
    {
        "numNodes": int,
        "node_weight": list[int],
        "edges": list[tuple[int, int, float]],
    },
)


def main() -> None:
    # Mass benchmark of graphs given parameters
    # num_graphs: int = 100
    # num_agents: int = 20
    # num_nodes: int = 200
    # upper_bound: float = 10.0

    # benchmark.mass_benchmark(
    #     count=num_graphs, k=num_agents, n=num_nodes, metric=True, upper=upper_bound
    # )

    # Alpha threshold benchmarking code
    # best_for_greedy: float = benchmark.alpha_heuristic_search(
    #     f=algos.greedy, count=num_graphs, k=num_agents, n=num_nodes, upper=upper_bound
    # )
    # print(f"{best_for_greedy = }")

    # best_for_nn: float = benchmark.alpha_heuristic_search(
    #     f=algos.nearest_neighbor,
    #     count=num_graphs,
    #     k=num_agents,
    #     n=num_nodes,
    #     upper=upper_bound,
    # )
    # print(f"{best_for_nn = }")

    # best_for_avg_with_greedy: float = benchmark.avg_alpha_heuristic_search(
    #     f=algos.greedy, count=num_graphs, k=num_agents, n=num_nodes, upper=upper_bound
    # )
    # print(f"{best_for_avg_with_greedy = }")

    # best_for_avg_with_nn: float = benchmark.avg_alpha_heuristic_search(
    #     f=algos.nearest_neighbor,
    #     count=num_graphs,
    #     k=num_agents,
    #     n=num_nodes,
    #     upper=upper_bound,
    # )
    # print(f"{best_for_avg_with_nn = }")

    # Messing with plotting
    g = Graph.random_complete_metric(50, upper=10.0)
    x = np.linspace(0, 100, 1000)
    _, ax = plt.subplots()
    total = sum(g.node_weight[x] for x in range(50))
    lines = []

    paths: list[list[int]] = algos.uconn_strat_1(g, 5)
    f = algos.generate_partition_path_function(g, paths)
    y = [total - f(i) for i in x]
    line, = ax.plot(x, y, label="UConn Greedy", color="lightsteelblue")
    lines.append(line)

    paths = algos.uconn_strat_2(g, 5, 2.5)
    f = algos.generate_partition_path_function(g, paths)
    y = [total - f(i) for i in x]
    line, = ax.plot(x, y, label="UConn Greedy + Rand (2.5)", color="royalblue")
    lines.append(line)

    paths = algos.uconn_strat_2(g, 5, 5.0)
    f = algos.generate_partition_path_function(g, paths)
    y = [total - f(i) for i in x]
    line, = ax.plot(x, y, label="UConn Greedy + Rand (5.0)", color="blue")
    lines.append(line)
    
    paths = algos.uconn_strat_2(g, 5, 7.5)
    f = algos.generate_partition_path_function(g, paths)
    y = [total - f(i) for i in x]
    line, = ax.plot(x, y, label="UConn Greedy + Rand (7.5)", color="mediumslateblue")
    lines.append(line)
    
    part: list[set[int]] = Graph.create_agent_partition(g, 5)
    output = algos.find_partition_with_heuristic(g, part, algos.greedy, 0.02)
    paths = benchmark.solve_partition(g, output, algos.greedy)
    f = algos.generate_partition_path_function(g, paths)
    y = [total - f(i) for i in x]
    line, = ax.plot(x, y, label="Greedy", linewidth=2.0, color="limegreen")
    lines.append(line)
    
    output = algos.find_partition_with_heuristic(g, part, algos.nearest_neighbor, 0.18)
    paths = benchmark.solve_partition(g, output, algos.nearest_neighbor)
    f = algos.generate_partition_path_function(g, paths)
    y = [total - f(i) for i in x]
    line, = ax.plot(x, y, label="Nearest Neighbor", linewidth=2.0, color="darkgreen")
    lines.append(line)
        
    output = algos.find_partition_with_average(g, part, 0.0)
    paths = benchmark.solve_partition(g, output, algos.greedy)
    f = algos.generate_partition_path_function(g, paths)
    y = [total - f(i) for i in x]
    line, = ax.plot(x, y, label="Avg + Greedy", color="firebrick")
    lines.append(line)

    paths = benchmark.solve_partition(g, output, algos.nearest_neighbor)
    f = algos.generate_partition_path_function(g, paths)
    y = [total - f(i) for i in x]
    line, = ax.plot(x, y, label="Avg + Nearest Neighbor", color="orangered")
    lines.append(line)
    
    mplcursors.cursor(lines, highlight=True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
