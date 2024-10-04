import json
import random
from collections import Counter

import networkx as nx
import numpy as np
import torch
import torch_geometric
from torch_geometric.utils import (
    from_networkx,
    homophily,
    scatter,
    to_edge_index,
    to_networkx,
    to_torch_csr_tensor,
)

from data import DATASET_TO_CLS, get_dataset


class Args:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def num_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def mad_stats(x, edge_index, n=500):
    x_i = x[edge_index[0]]
    x_j = x[edge_index[1]]

    cosine_sim = torch.cosine_similarity(x_i, x_j)
    dist = 1 - cosine_sim
    nonzero_mask = dist > 0
    mad_nei = scatter(
        dist[nonzero_mask],
        edge_index[0][nonzero_mask],
        dim=0,
        dim_size=x.size(0),
        reduce="mean",
    )
    mad_nei = mad_nei[:n]
    nodes = set(np.arange(x.shape[0]))
    mad_rmt = torch.zeros_like(mad_nei)
    for i in list(nodes)[:n]:
        neighbors = edge_index[:, (edge_index[0] == i)]
        remote = list(nodes.difference(neighbors))
        cosine_sim = torch.cosine_similarity(x[i], x[remote])
        dist = 1 - cosine_sim
        dist = dist[dist > 0].mean().item()
        mad_rmt[i] = dist

    mad_gap_total = (mad_rmt.mean() - mad_nei.mean()).item()
    mad_ratio_total = (mad_rmt.mean() / mad_nei.mean()).item()

    return mad_gap_total, mad_ratio_total


def calculate_dataset_stats(save=True):
    dataset_stats = {}
    for d in DATASET_TO_CLS:
        dataset = get_dataset(d)
        dataset_stats[d] = {}
        dataset_stats[d]["num_nodes"] = dataset._data.num_nodes
        dataset_stats[d]["num_edges"] = dataset._data.num_edges
        dataset_stats[d]["num_features"] = dataset.num_features
        dataset_stats[d]["num_classes"] = dataset.num_classes
        dataset_stats[d]["homophily_edge"] = round(
            homophily(dataset._data.edge_index, dataset._data.y, method="edge"), 2
        )
        dataset_stats[d]["homophily_node"] = round(
            homophily(dataset._data.edge_index, dataset._data.y, method="node"), 2
        )
        dataset_stats[d]["homophily_edge_insensitive"] = round(
            homophily(
                dataset._data.edge_index, dataset._data.y, method="edge_insensitive"
            ),
            2,
        )
        dataset_stats[d]["num_splits"] = (
            int(dataset._data.train_mask.shape[-1])
            if dataset._data.train_mask.ndim > 1
            else 1
        )

        class_edge_index = dataset._data.y[dataset._data.edge_index]
        inter_class_edge_mask = class_edge_index[0] != class_edge_index[1]
        class_edge_index = class_edge_index[:, inter_class_edge_mask]
        class_edge_index = class_edge_index.T.tolist()
        class_edge_index = [str(tuple(set(x))) for x in class_edge_index]
        inter_class_edge_dict = dict(Counter(class_edge_index))

        dataset_stats[d]["inter_class_edges"] = inter_class_edge_mask.sum().item()
        dataset_stats[d]["inter_class_edge_ratio"] = round(
            dataset_stats[d]["inter_class_edges"] / dataset_stats[d]["num_edges"], 2
        )
        for key in inter_class_edge_dict:
            inter_class_edge_dict[key] = round(
                inter_class_edge_dict[key] / dataset_stats[d]["inter_class_edges"], 2
            )
        dataset_stats[d]["inter_class_edge_stats"] = inter_class_edge_dict

    if save:
        json.dump(
            dataset_stats,
            open("dataset_stats.json", "w"),
            indent=4,
            separators=(",", ": "),
            sort_keys=True,
        )
    return dataset_stats


def two_hop_edge_index(edge_index, num_nodes):
    N = num_nodes
    adj = to_torch_csr_tensor(edge_index, size=(N, N))
    edge_index2, _ = to_edge_index((adj @ adj))
    idx = edge_index[0] * N + edge_index[1]
    idx2 = edge_index2[0] * N + edge_index2[1]
    mask = torch.isin(idx2, idx)
    edge_index2 = edge_index2[:, ~mask]
    return edge_index2


def complement(data, num_nodes):
    graph = to_networkx(data)
    graph_complement = nx.algorithms.operators.complement(graph)
    data = from_networkx(graph_complement)
    return data.edge_index


def create_boolean_mask(n, p):
    random_values = torch.rand(n)
    mask = (random_values < p).type(torch.bool)
    return mask


def cosine_similarity(vectors_a, vectors_b):
    dot_products = np.sum(vectors_a * vectors_b, axis=1)
    magnitude_a = np.linalg.norm(vectors_a, axis=1)
    magnitude_b = np.linalg.norm(vectors_b, axis=1)
    cosine_similarities = dot_products / (magnitude_a * magnitude_b)
    return cosine_similarities


def dirichlet_energy(x, edge_index):
    dist = torch.sum((x[edge_index[0]] - x[edge_index[1]]) ** 2, dim=-1)
    nodewise_sums = scatter(dist, edge_index[0], 0, dim_size=x.size(0), reduce="sum")
    return float(nodewise_sums.mean())


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch_geometric.seed_everything(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
