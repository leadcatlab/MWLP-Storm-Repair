# Post-Disaster Repair Crew Assignment Optimization Using Minimum Latency
[arXiv](https://arxiv.org/abs/2206.00597)

## Authors: Anakin Dey, Melkior Ornik

## Overview:
This repositiry contains the code used to benchmark for a preliminary paper on heuristic algorithms for the multiagent MWLP. `graph.py` defines a custom graph class and `algos.py` defines the core algorithms in the paper. There are two test files that can be run with `pytest` and 100% code coverage of `graph.py` and `algos.py` can be verified with the `Coverage.py` module. `benchmark.py` contains helper code for running various benchmarks and creating some plots used in the paper. `mass_benchmark.py` and `champaign.py` have code used to create the actual benchmarks presented in the paper. `alpha.py` contains some testing code to test parameters for the transfer-and-swaps algorithms.

The code has been linted for readablity using `flake8` and `pylint` and the `black` autoformatter was also used to aid in readability. Type hints, checked with `mypy`, were used in the majority of the code to also aid with readability.

The code was developed using Python 3.9.5 so as of now that is the recommended version that should be used. MatPlotLib 3.5.1, NetworkX 2.6.3, and OSMNX 1.1.2 are also needed.

###### This code will be as-is at the time of uploading the paper, barring minor non-functional changes

## Acknowledgment

We thank Pranay Thangeda for his help in identifying bugs in early versions of the code for this paper as well as providing baseline code for the Champaign simulations. We also thank Diego Cerrai for clarifying previous work on this topic and kindly sharing his recent results.
