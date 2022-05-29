# [Paper Title will go here eventually]

## Authors: Anakin Dey, Melkior Ornik

## Overview:
Benchmarking code for heuristic algorithms for the multiagent MWLP. `graph.py` defines a class for a graph and `algos.py` defines algorithms for that graph. There are two test files that can be run with pytest and 100% code coverage of `graph.py` and `algos.py` can be verified with the coverage module. `benchmark.py` has helper code for running various benchmarks and creating some graphs. `mass_benchmark.py` and `champaign.py` have code used to create the benchmarks presented in the paper. `alpha.py` has some testing code to test parameters for the transfer-and-swaps algorithms.

The code has been linted for readablity using `flake8` and `pylint` and the `black` autoformatter was also used to aid in readability. Type hints were used in the majority of the code to aid with understanding of what is going on.

The code was developed using Python 3.9.5 so as of now that is the recommended version that should be used. MatPlotLib 3.5.1, NetworkX 2.6.3, and OSMNX 1.1.2 are also needed.

## Acknowledgment

We thank Pranay Thangeda for helping identify some bugs in an early verison of this code and also with providing some reference code for some of the `champaign.py` benchmarking.
