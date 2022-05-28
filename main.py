"""
Driver code for testing functions
"""

import matplotlib.pyplot as plt  # type: ignore

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
    # Messing with plotting
    n = 100
    k = 8
    g: Graph = Graph.random_complete_metric(n)
    nx_g = Graph.to_networkx(g)

    part: list[set[int]] = Graph.create_agent_partition(g, k)
    initial_assignment: list[list[int]] = benchmark.solve_partition(
        g, part, algos.nearest_neighbor
    )
    benchmark.draw_graph_with_partitions(nx_g, initial_assignment, "Initial")

    output = algos.find_partition_with_heuristic(g, part, algos.nearest_neighbor, 0.18)
    final_assignment = benchmark.solve_partition(g, output, algos.nearest_neighbor)
    benchmark.draw_graph_with_partitions(nx_g, final_assignment, "Nearest Neighbor")
    plt.show()


if __name__ == "__main__":
    main()
