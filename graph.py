import networkx as nx
import matplotlib.pyplot as plt
from networkx.generators import directed
from networkx.generators.random_graphs import erdos_renyi_graph
from networkx.utils.decorators import random_state

G = nx.Graph()
N = 10
for n in range(N):
    G.add_node(n)

#edge between one and two node

#G.add_edge(1,2)

g = erdos_renyi_graph(N, 0.3,seed = None, directed= True)

nx.draw(g)

A = nx.adjacency_matrix(g)
print(A.todense())

plt.show()