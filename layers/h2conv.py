from typing import Optional

import torch
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.typing import Adj, OptPairTensor, OptTensor
from torch_geometric.utils import (
    remove_self_loops,
    to_edge_index,
    to_torch_csr_tensor,
)


def two_hop_edge_index(edge_index, num_nodes):
    N = num_nodes
    adj = to_torch_csr_tensor(edge_index, size=(N, N))
    edge_index2, _ = to_edge_index(adj @ adj)
    edge_index2, _ = remove_self_loops(edge_index2)

    return edge_index2


class H2Conv(MessagePassing):
    _cached_edge_index: Optional[OptPairTensor]
    _cached_edge_index2: Optional[OptPairTensor]

    def __init__(
        self,
        cached: bool = True,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super().__init__(**kwargs)

        self.cached = cached

        self._cached_edge_index = None
        self._cached_edge_index2 = None

        self.reset_parameters()

    def reset_parameters(self):
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
    ):
        if isinstance(edge_index, Tensor):
            cache = self._cached_edge_index
            cache2 = self._cached_edge_index2
            if cache is None:
                edge_index, edge_weight = gcn_norm(  # yapf: disable
                    edge_index,
                    None,
                    x.size(self.node_dim),
                    False,
                    False,
                    self.flow,
                    dtype=x.dtype,
                )
                edge_index2, edge_weight2 = gcn_norm(  # yapf: disable
                    two_hop_edge_index(edge_index, x.size(self.node_dim)),
                    None,
                    x.size(self.node_dim),
                    False,
                    False,
                    self.flow,
                    dtype=x.dtype,
                )
                if self.cached:
                    self._cached_edge_index = (edge_index, edge_weight)
                    self._cached_edge_index2 = (edge_index2, edge_weight2)
            else:
                edge_index, edge_weight = cache[0], cache[1]
                edge_index2, edge_weight2 = cache2[0], cache2[1]

        # propagate_type: (x: Tensor, alpha: PairTensor, edge_weight: OptTensor)  # noqa
        msg = self.propagate(
            edge_index,
            x=x,
            edge_weight=edge_weight,
            size=None,
        )

        msg2 = self.propagate(
            edge_index2,
            x=x,
            edge_weight=edge_weight2,
            size=None,
        )

        return torch.cat([msg, msg2], dim=1)

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j * edge_weight.view(-1, 1)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
