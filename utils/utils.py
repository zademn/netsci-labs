import networkx as nx
import numpy as np
import scipy
import itertools
import os, sys
import random


def average_shortest_path_length_sampled(G, n_samples=500):
    G0 = giant_component(G)
    if n_samples >= len(G0.edges):
        return nx.average_shortest_path_length(G0)
    
    possible_pairs = np.array(list(itertools.combinations(G0.nodes, 2)))
    idxs = np.random.choice(len(possible_pairs), n_samples, replace = False)
    pairs = possible_pairs[idxs]
                     
    lengths = []
    for u, v in pairs:
        length = nx.shortest_path_length(G0, source=u, target=v)
        lengths.append(length)
    return np.mean(length)


def average_degree(G):
    return sum(G.degree(n) for n in G.nodes) / len(G.nodes)
def average_in(G):
    return sum(G.in_degree(n) for n in G.nodes) / len(G.nodes)
def average_out(G):
    return sum(G.out_degree(n) for n in G.nodes) / len(G.nodes)


def giant_component(G):
    Gcc = sorted(nx.connected_components(G), key = len, reverse = True)
    G0 = G.subgraph(Gcc[0])
    return G0
def connectivity_perc(G):
    """How many nodes are in the giant component"""
    G0 = giant_component(G)
    connectivity_perc = G0.number_of_nodes() / G.number_of_nodes()  
    return connectivity_perc

def average_clustering(G):
    return np.mean(list(nx.clustering(G).values()))
def average_degree(G):
    return sum(G.degree(n) for n in G.nodes) / len(G.nodes)


def print_stats(G, n_samples = 500):
    print(f"{G.number_of_nodes() = :}")
    print(f"{G.number_of_edges() = }")
    print(f"{average_degree(G) = :.2f}")
    print(f"{average_clustering(G) = :.4f}")
    print(f"{connectivity_perc(G) = :.2f}")
    print(f"{average_shortest_path_length_sampled(G, n_samples) = }") 
    
# Helper functions
def communities_to_dict(communities):
    """Transforms a communities list formed from a list of sets [{}, {}, ...] into a node:community dict"""
    d = {}
    for v in range(len(communities)):
        for k in communities[v]:
            d[k] = v
    return d
def dict_to_communities(d):
    """ Transforms a dict from node: community to a communities where each set is a community of nodes [{}, {}]"""
    return [{u for u, si in d.items() if si == s } for s in np.unique(list(d.values()))]
    