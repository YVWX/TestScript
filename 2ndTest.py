import bz2
import importlib.resources
import pickle

import inspect

from time import time

import networkx as nx
from networkx.algorithms.flow import (
    dinitz,
    edmonds_karp,
    shortest_augmenting_path,
)

flow_funcs = [
    dinitz,
    edmonds_karp,
    shortest_augmenting_path,
]

def gen_pyramid(N):
    # This graph admits a flow of value 1 for which every arc is at
    # capacity (except the arcs incident to the sink which have
    # infinite capacity).
    G = nx.DiGraph()

    for i in range(N - 1):
        cap = 1.0 / (i + 2)
        for j in range(i + 1):
            G.add_edge((i, j), (i + 1, j), capacity=cap)
            cap = 1.0 / (i + 1) - cap
            G.add_edge((i, j), (i + 1, j + 1), capacity=cap)
            cap = 1.0 / (i + 2) - cap

    for j in range(N):
        G.add_edge((N - 1, j), "t")

    return G

def read_graph(name):
    fname = (
        importlib.resources.files("networkx.algorithms.flow.tests")
        / f"{name}.gpickle.bz2"
    )

    with bz2.BZ2File(fname, "rb") as f:
        G = pickle.load(f)
    return G

class TestGomoryHuTree:
    def test_karate_club_graph(self):
        G = nx.karate_club_graph()
        nx.set_edge_attributes(G, 1, "capacity")
        for flow_func in flow_funcs:
            begin = time()
            T = nx.gomory_hu_tree(G, flow_func=flow_func)
            end = time()
            print(flow_func.__name__ + ": " + str(end - begin))
            assert nx.is_tree(T)

    def test_davis_southern_women_graph(self):
        G = nx.davis_southern_women_graph()
        nx.set_edge_attributes(G, 1, "capacity")
        for flow_func in flow_funcs:
            begin = time()
            T = nx.gomory_hu_tree(G, flow_func=flow_func)
            end = time()
            print(flow_func.__name__ + ": " + str(end - begin))
            assert nx.is_tree(T)

    def test_florentine_families_graph(self):
        G = nx.florentine_families_graph()
        nx.set_edge_attributes(G, 1, "capacity")
        for flow_func in flow_funcs:
            begin = time()
            T = nx.gomory_hu_tree(G, flow_func=flow_func)
            end = time()
            print(flow_func.__name__ + ": " + str(end - begin))
            assert nx.is_tree(T)
            
    def test_les_miserables_graph_cutset(self):
        G = nx.les_miserables_graph()
        nx.set_edge_attributes(G, 1, "capacity")
        for flow_func in flow_funcs:
            begin = time()
            T = nx.gomory_hu_tree(G, flow_func=flow_func)
            end = time()
            print(flow_func.__name__ + ": " + str(end - begin))
            assert nx.is_tree(T)

    def test_wikipedia_example(self):
        # Example from https://en.wikipedia.org/wiki/Gomory%E2%80%93Hu_tree
        G = nx.Graph()
        G.add_weighted_edges_from(
            (
                (0, 1, 1),
                (0, 2, 7),
                (1, 2, 1),
                (1, 3, 3),
                (1, 4, 2),
                (2, 4, 4),
                (3, 4, 1),
                (3, 5, 6),
                (4, 5, 2),
            )
        )
        for flow_func in flow_funcs:
            begin = time()
            T = nx.gomory_hu_tree(G, capacity="weight", flow_func=flow_func)
            end = time()
            print(flow_func.__name__ + ": " + str(end - begin))
            assert nx.is_tree(T)

class TestMaxflowLargeGraph:
    def test_complete_graph(self):
        N = 50
        G = nx.complete_graph(N)
        nx.set_edge_attributes(G, 5, "capacity")
        for flow_func in flow_funcs:
            errmsg = f"Assertion failed in function: {flow_func.__name__}"
            begin = time()
            flow_value = nx.maximum_flow_value(G, 1, 2, flow_func=flow_func)
            end = time()
            print(flow_func.__name__ + ": " + str(end - begin))
            assert flow_value == 5 * (N - 1), errmsg

    def test_pyramid(self):
        N = 100 # this gives a graph with 5051 nodes
        G = gen_pyramid(N)
        for flow_func in flow_funcs:
            begin = time()
            flow_value = nx.maximum_flow_value(G, (0, 0), "t", flow_func=flow_func)
            end = time()
            print(flow_func.__name__ + ": " + str(end - begin))

    def test_gl1(self):
        G = read_graph("gl1")
        s = 1
        t = len(G)
        for flow_func in flow_funcs:
            begin = time()
            flow_value = nx.maximum_flow_value(G, s, t, flow_func=flow_func)
            end = time()
            print(flow_func.__name__ + ": " + str(end - begin))

    def test_gw1(self):
        G = read_graph("gw1")
        s = 1
        t = len(G)
        for flow_func in flow_funcs:
            begin = time()
            flow_value = nx.maximum_flow_value(G, s, t, flow_func=flow_func)
            end = time()
            print(flow_func.__name__ + ": " + str(end - begin))

    def test_wlm3(self):
        G = read_graph("wlm3")
        s = 1
        t = len(G)
        for flow_func in flow_funcs:
            begin = time()
            flow_value = nx.maximum_flow_value(G, s, t, flow_func=flow_func)
            end = time()
            print(flow_func.__name__ + ": " + str(end - begin))
"""
testGomoryHuTree = TestGomoryHuTree()
for tests in dir(testGomoryHuTree):
    if tests[0 : 4] == "test":
        test_func = getattr(testGomoryHuTree, tests)
        print(test_func.__name__)
        test_func()
"""

testMaxflowLargeGraph = TestMaxflowLargeGraph()
for tests in dir(testMaxflowLargeGraph):
    if tests[0 : 4] == "test":
        test_func = getattr(testMaxflowLargeGraph, tests)
        print(test_func.__name__)
        test_func()

