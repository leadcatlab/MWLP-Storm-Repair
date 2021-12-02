from graph import graph
import algos

#graphDict = {
#    "numNodes": 4,
#    "nodeWeight": [10, 20, 30, 40],
#    "edges": [(0, 1, 1.0), (2, 3, 1.1)],
#}
#
#g = graph.FromDict(graphDict)
#
#print("================================")
#print("The adjacency list")
#g.printAdjacencyList()
#print("================================")
#print("The edge matrix")
#g.printEdgeStrength()
#print("================================")
#print("Node weight")
#g.printNodeWeight()
#print()
#
#completeGraph = {
#    "numNodes": 2,
#    "nodeWeight": [0, 1],
#    "edges": [(0, 1, 1.0), (1, 0, 2.0)]
#}
#complete = graph.FromDict(completeGraph)
#print(graph.isComplete(complete))
#
#incompleteGraph = {
#    "numNodes": 2,
#    "nodeWeight": [0, 1],
#    "edges": [(1, 0, 2.0)]
#}
#incomplete = graph.FromDict(incompleteGraph)
#print(graph.isComplete(incomplete))

g = graph.randomComplete(4)
print(g)
