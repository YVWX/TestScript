import scipy
from time import time
import numpy as np
import networkx as nx
from scipy.sparse import rand
from scipy.sparse.csgraph import maximum_flow
from networkx.algorithms.flow import edmonds_karp
from networkx.algorithms.flow import shortest_augmenting_path
from networkx.algorithms.flow import preflow_push
from networkx.algorithms.flow import dinitz
from scipy.sparse import coo_matrix, csr_matrix
"""
n = 1000
density = 0.1
for k in range(100):
    m = (scipy.sparse.rand(n, n, density=density, format='csr',
                           random_state=k)*100).astype(np.int32)
    G = nx.from_numpy_matrix(m.toarray(), create_using=nx.DiGraph())
    Edmonds_Karp_max_flow = nx.algorithms.flow.maximum_flow_value(G, 0, n-1, capacity='weight', flow_func=edmonds_karp)
    Dinitz_max_flow = nx.algorithms.flow.maximum_flow_value(G, 0, n-1, capacity='weight', flow_func=dinitz)
    assert Edmonds_Karp_max_flow == Dinitz_max_flow
"""
"""
n = 1000
density = 0.1
m = (scipy.sparse.rand(n, n, density=density, format='csr', random_state=42)*100).astype(np.int32)
G = nx.from_numpy_matrix(m.toarray(), create_using=nx.DiGraph())

begin = time()
for itr in range(3):
    flow = nx.algorithms.flow.maximum_flow_value(G, 0, n - 1, capacity='weight', flow_func=edmonds_karp)
end = time()
print(f"Edmonds Karp: {(end - begin) / 3}")

begin = time()
for itr in range(3):
    flow = nx.algorithms.flow.maximum_flow_value(G, 0, n - 1, capacity='weight', flow_func=shortest_augmenting_path)
end = time()
print(f"Shortest augmenting path: {(end - begin) / 3}")

begin = time()
for itr in range(3):
    flow = nx.algorithms.flow.maximum_flow_value(G, 0, n - 1, capacity='weight', flow_func=dinitz)
end = time()
print(f"New Dinitz: {(end - begin) / 3}")
"""
"""
n = 500
np.random.seed(42)
a = np.zeros((n, n), dtype=np.int32)
for k in range(n - 1):
    for j in range(-50, 50):
        if j != 0 and k + j >= 0 and k + j < n:
            a[k, k + j] = np.random.randint(1, 1000)
m = csr_matrix(a)
G = nx.from_numpy_matrix(a, create_using=nx.DiGraph())

begin = time()
for itr in range(3):
    flow = nx.algorithms.flow.maximum_flow_value(G, 0, n - 1, capacity='weight', flow_func=edmonds_karp)
end = time()
print(f"Edmonds Karp: {(end - begin) / 3}")

begin = time()
for itr in range(3):
    flow = nx.algorithms.flow.maximum_flow_value(G, 0, n - 1, capacity='weight', flow_func=shortest_augmenting_path)
end = time()
print(f"Shortest augmenting path: {(end - begin) / 3}")

begin = time()
for itr in range(3):
    flow = nx.algorithms.flow.maximum_flow_value(G, 0, n - 1, capacity='weight', flow_func=dinitz)
end = time()
print(f"New Dinitz: {(end - begin) / 3}")
"""

def make_data(density):
    m = (rand(1000, 1000, density=density, format='coo', random_state=42)*100).astype(np.int32)
    return np.vstack([m.row, m.col, m.data]).T

data01 = make_data(0.1)
data03 = make_data(0.3)
data05 = make_data(0.5)

def networkx_max_flow(data, primitive):
    m = coo_matrix((data[:, 2], (data[:, 0], data[:, 1])))
    G = nx.from_numpy_array(m.toarray(), create_using=nx.DiGraph())
    return nx.maximum_flow_value(G, 0, 999, capacity='weight', flow_func=primitive)

def scipy_max_flow(data):
    m = csr_matrix((data[:, 2], (data[:, 0], data[:, 1])))
    return maximum_flow(m, 0, 999).flow_value

begin = time()
for itr in range(3):
    networkx_max_flow(data01, nx.algorithms.flow.edmonds_karp)
end = time()
print(f"Edmonds Karp: {(end - begin) / 3}")

begin = time()
for itr in range(3):
    networkx_max_flow(data03, nx.algorithms.flow.edmonds_karp)
end = time()
print(f"Edmonds Karp: {(end - begin) / 3}")

begin = time()
for itr in range(3):
    networkx_max_flow(data05, nx.algorithms.flow.edmonds_karp)
end = time()
print(f"Edmonds Karp: {(end - begin) / 3}")

begin = time()
for itr in range(3):
    networkx_max_flow(data01, nx.algorithms.flow.shortest_augmenting_path)
end = time()
print(f"Shortest augmenting path: {(end - begin) / 3}")

begin = time()
for itr in range(3):
    networkx_max_flow(data03, nx.algorithms.flow.shortest_augmenting_path)
end = time()
print(f"Shortest augmenting path: {(end - begin) / 3}")

begin = time()
for itr in range(3):
    networkx_max_flow(data05, nx.algorithms.flow.shortest_augmenting_path)
end = time()
print(f"Shortest augmenting path: {(end - begin) / 3}")

begin = time()
for itr in range(3):
    networkx_max_flow(data01, nx.algorithms.flow.dinitz)
end = time()
print(f"New Dinitz: {(end - begin) / 3}")

begin = time()
for itr in range(3):
    networkx_max_flow(data03, nx.algorithms.flow.dinitz)
end = time()
print(f"New Dinitz: {(end - begin) / 3}")

begin = time()
for itr in range(3):
    networkx_max_flow(data05, nx.algorithms.flow.dinitz)
end = time()
print(f"New Dinitz: {(end - begin) / 3}")

