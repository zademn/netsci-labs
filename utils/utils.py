from typing import Any, List
import networkx as nx
import numpy as np
import itertools
import io
import zipfile
import pandas as pd
import requests
import gzip
from scipy.io import mmread
from six.moves import urllib
from scipy.sparse import coo_matrix
from joblib import Parallel, delayed

def average_shortest_path_length_sampled(G: nx.Graph, n_samples: int = 500) -> float:
    """
    Return the average shortest path length of a graph
    by sampling nodes and calculating the shortest path between them.
    If the number of samples is higher than the number of edges use networkx intrinsic function

    Parameters:
    -----------
        G: nx.Graph
            Input graph
        n_samples: int
            number of node pairs to sample

    Returns:
    --------
        average shortes path length :int

    """
    G0 = giant_component(G)
    if n_samples >= len(G0.edges):
        return nx.average_shortest_path_length(G0)

    # Get all possible pairs of 2 nodes
    possible_pairs = np.array(list(itertools.combinations(G0.nodes, 2)))
    # Select `n_samples` pairs indexes
    idxs = np.random.choice(len(possible_pairs), n_samples, replace=False)
    # Get the pairs
    pairs = possible_pairs[idxs]

    lengths = []
    # For each pair, calculate the shortest path.
    for u, v in pairs:
        length = nx.shortest_path_length(G0, source=u, target=v)
        lengths.append(length)
    # return the mean
    return np.mean(length)


def average_degree(G: nx.Graph) -> float:
    return sum(G.degree(n) for n in G.nodes) / len(G.nodes)


def average_in(G: nx.Graph) -> float:
    return sum(G.in_degree(n) for n in G.nodes) / len(G.nodes)


def average_out(G: nx.Graph) -> float:
    return sum(G.out_degree(n) for n in G.nodes) / len(G.nodes)


def giant_component(G: nx.Graph) -> nx.Graph:
    """Return the biggest component of a graph

    Parameters:
    -----------
    G: nx.Graph
        Input graph

    Returns:
    --------
    biggest component: nx.graph
    """

    # Get all connected components and sort them by the number of nodes
    Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
    # Get the subgraph corresponding to the first giant component
    G0 = G.subgraph(Gcc[0])
    return G0


def connectivity_perc(G: nx.Graph) -> float:
    """Returns the percentage of nodes found in the giant component

    Parameters:
    -----------
    G: nx.Graph
        Input graph

    Returns:
    --------
        :float
    """
    G0 = giant_component(G)
    connectivity_perc = G0.number_of_nodes() / G.number_of_nodes()
    return connectivity_perc


def average_clustering(G: nx.Graph) -> float:
    return np.mean(list(nx.clustering(G).values()))


def print_stats(G: nx.Graph, n_samples=500):
    """Prints statistics about the graph."""
    print(f"{G.number_of_nodes() = :}")
    print(f"{G.number_of_edges() = }")
    print(f"{average_degree(G) = :.2f}")
    print(f"{average_clustering(G) = :.4f}")
    print(f"{connectivity_perc(G) = :.2f}")
    print(f"{average_shortest_path_length_sampled(G, n_samples) = }")


# Helper functions
def communities_to_dict(communities: list):
    """Transforms a communities list formed from a list of sets
    [{u1, u2, ...}, {v1, v2, ...}, ...] into a {node:community} dict
    """
    d = {}
    for v in range(len(communities)):
        for k in communities[v]:
            d[k] = v
    return d


def dict_to_communities(d: dict):
    """Transforms a dict from {node: community} to a communities where each set is a community of nodes [{}, {}]"""
    return [{u for u, si in d.items() if si == s} for s in np.unique(list(d.values()))]


def load_graph(name: str) -> nx.Graph:
    """Downloads and initiates a nx.Graph

    Parameters
    ----------
    name : str
        one of ["cora", "cora_labels", "football", "arvix", "power-us-grid", "facebook", "wiki"]

    Returns
    -------
    nx.Graph

    Raises
    ------
    ValueError
        If the str is not in one of the possible variants
    """
    if name.lower() == "cora":
        download_url = "https://temprl.com/cora.graphml"
        res = requests.get(download_url)  # Download
        G = nx.read_graphml(io.BytesIO(res.content))
        G = nx.to_undirected(G)
        return G

    elif name.lower() == "cora_labels":
        # Load cora
        download_url = "https://nrvis.com/download/data/labeled/cora.zip"
        res = requests.get(download_url)  # Download
        zf = zipfile.ZipFile(io.BytesIO(res.content))  # zipfile from downloaded content

        f_edges = io.BytesIO(zf.read("cora.edges"))
        f_labels = io.BytesIO(zf.read("cora.node_labels"))

        G = nx.read_weighted_edgelist(f_edges, delimiter=",")
        df = pd.read_csv(
            f_labels,
            delimiter=",",
            names=["nodes", "label"],
        )
        d = {str(k): {"label": v} for k, v in zip(df["nodes"], df["label"])}
        nx.set_node_attributes(G, d)
        G = nx.relabel_nodes(G, {k: v for k, v in zip(G.nodes, range(len(G.nodes)))})
        G = giant_component(G)
        return G

    elif name.lower() == "football":
        download_url = "http://www-personal.umich.edu/~mejn/netdata/football.zip"

        res = requests.get(download_url)
        s = io.BytesIO(res.content)  # read into BytesIO "file"

        zf = zipfile.ZipFile(s)  # zipfile object
        txt = zf.read("football.txt").decode()  # read info file
        gml = zf.read("football.gml").decode()  # read gml data
        # throw away bogus first line with # from mejn files
        gml = gml.split("\n")[1:]
        G_football = nx.parse_gml(gml)  # parse gml data
        G_football = nx.relabel_nodes(
            G_football,
            {k: v for k, v in zip(G_football.nodes, range(len(G_football.nodes)))},
        )
        return G_football

    elif name.lower() == "power-us-grid":
        download_url = "https://nrvis.com/download/data/power/power-US-Grid.zip"
        res = requests.get(download_url)  # Download
        zf = zipfile.ZipFile(io.BytesIO(res.content))  # zipfile from downloaded content
        G = nx.from_scipy_sparse_array(
            mmread(zf.open("power-US-Grid.mtx"))
        )  # open the file pointer and mmread.
        node_map = {u: int(u) for u in G.nodes}
        G = nx.relabel_nodes(G, node_map, copy = True)

        return G

    elif name.lower() in ["wiki", "arvix", "facebook"]:
        if name.lower() == "wiki":
            download_url = "https://snap.stanford.edu/data/wiki-Vote.txt.gz"
        elif name.lower() == "arvix":
            download_url = "https://snap.stanford.edu/data/ca-GrQc.txt.gz"
        elif name.lower() == "facebook":
            download_url = "https://snap.stanford.edu/data/facebook_combined.txt.gz"

        res = requests.get(download_url)  # Download
        with gzip.open(io.BytesIO(res.content), "rb") as f:
            G = nx.read_edgelist(f)
        return G

    else:
        raise ValueError("Graph not found")

# From https://github.com/benedekrozemberczki/karateclub/blob/d750b33e8faaeb95ac2ed6d6ed4f0b470bd506ce/karateclub/dataset/dataset_reader.py#L12
class GraphReader(object):
    r"""Class to read benchmark datasets for the community detection or node embedding task.

    Args:
        dataset (str): Dataset of interest, one of:
            (:obj:`"facebook"`, :obj:`"twitch"`, :obj:`"wikipedia"`, :obj:`"github"`, :obj:`"lastfm"`, :obj:`"deezer"`). Default is 'wikipedia'.
    """

    def __init__(self, dataset: str = "wikipedia"):
        assert dataset in [
            "wikipedia",
            "twitch",
            "github",
            "facebook",
            "lastfm",
            "deezer",
        ], "Wrong dataset."
        self.dataset = dataset
        self.base_url = "https://github.com/benedekrozemberczki/karateclub/raw/master/dataset/node_level/"

    def _pandas_reader(self, bytes):
        """
        Reading bytes as a Pandas dataframe.
        """
        tab = pd.read_csv(
            io.BytesIO(bytes), encoding="utf8", sep=",", dtype={"switch": np.int32}
        )
        return tab

    def _dataset_reader(self, end):
        """
        Reading the dataset from the web.
        """
        path = self.base_url + self.dataset + "/" + end
        data = urllib.request.urlopen(path).read()
        data = self._pandas_reader(data)
        return data

    def get_graph(self) -> nx.classes.graph.Graph:
        r"""Getting the graph.

        Return types:
            * **graph** *(NetworkX graph)* - Graph of interest.
        """
        data = self._dataset_reader("edges.csv")
        graph = nx.convert_matrix.from_pandas_edgelist(data, "id_1", "id_2")
        return graph

    def get_features(self) -> coo_matrix:
        r"""Getting the node features Scipy matrix.

        Return types:
            * **features** *(COO Scipy array)* - Node feature matrix.
        """
        data = self._dataset_reader("features.csv")
        row = np.array(data["node_id"])
        col = np.array(data["feature_id"])
        values = np.array(data["value"])
        node_count = max(row) + 1
        feature_count = max(col) + 1
        shape = (node_count, feature_count)
        features = coo_matrix((values, (row, col)), shape=shape)
        return features

    def get_target(self) -> np.array:
        r"""Getting the class membership of nodes.

        Return types:
            * **target** *(Numpy array)* - Class membership vector.
        """
        data = self._dataset_reader("target.csv")
        target = np.array(data["target"])
        return target
    

def walk_node(graph: nx.Graph, node: Any, walk_length: int) -> List[Any]:
    """Given a graph, a node and a walk_length, walks the graph starting from that node"""
    # Add the initial node to the walk
    walk = [node]

    for _ in range(walk_length - 1):
        # Pick neighbours from the last node in the walk
        neighbours = list(graph.neighbors(walk[-1]))
        # If the node has neighbours, sample and add to the walk
        if len(neighbours) > 0:
            next_node = np.random.choice(neighbours)
            walk.append(next_node)
    return walk


def walk_graph(
    graph: nx.Graph, walks_per_node: int, walk_length: int
) -> List[List[Any]]:
    """Given a graph, how many walks_per_node and the walk_length, for each node do
    walks_per_node walks of length walk_length starting from it. Add the walk to a list
    and return it"""
    walks = []
    # For each node, do `walks_per_node` random walks
    for node in graph.nodes():
        for _ in range(walks_per_node):
            walk = walk_node(graph, node, walk_length)
            walks.append(walk)
    return walks