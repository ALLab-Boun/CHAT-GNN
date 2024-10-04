import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import (
    Actor,
    HeterophilousGraphDataset,
    Planetoid,
    WebKB,
    WikipediaNetwork,
)
from torch_geometric.utils import remove_self_loops

DATASET_TO_CLS = {
    "cornell": WebKB,
    "texas": WebKB,
    "wisconsin": WebKB,
    "chameleon": WikipediaNetwork,
    "squirrel": WikipediaNetwork,
    "cora": Planetoid,
    "citeseer": Planetoid,
    "pubmed": Planetoid,
    "actor": Actor,
    "roman-empire": HeterophilousGraphDataset,
    "amazon-ratings": HeterophilousGraphDataset,
    "minesweeper": HeterophilousGraphDataset,
    "tolokers": HeterophilousGraphDataset,
    "questions": HeterophilousGraphDataset,
}


DATASET_TO_METRIC = {
    "cornell": "acc",
    "texas": "acc",
    "wisconsin": "acc",
    "chameleon": "acc",
    "squirrel": "acc",
    "cora": "acc",
    "citeseer": "acc",
    "pubmed": "acc",
    "actor": "acc",
    "roman-empire": "acc",
    "amazon-ratings": "acc",
    "minesweeper": "auc",
    "tolokers": "auc",
    "questions": "auc",
}


def get_dataset(
    name, root_path="./data/", device="cpu", undirected=True, one_hot=False, **kwargs
):
    name = name.lower()

    transforms = [T.ToDevice(device)]
    if undirected:
        transforms.extend([T.ToUndirected()])

    constructor = DATASET_TO_CLS[name]
    if constructor != HeterophilousGraphDataset:
        transforms.extend([T.NormalizeFeatures()])

    transform = T.Compose(transforms)
    args = dict(root=root_path, transform=transform, **kwargs)
    if constructor == Planetoid:
        args["split"] = "full"
    if "name" in constructor.__init__.__annotations__:
        args["name"] = name
    dataset = constructor(**args)
    if one_hot:
        dataset._data.x = torch.eye(dataset._data.x.shape[0])
    dataset._data.edge_index, dataset._data.edge_attr = remove_self_loops(
        dataset._data.edge_index, dataset._data.edge_attr
    )
    return dataset
