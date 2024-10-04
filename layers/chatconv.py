from typing import Optional

import torch.nn.functional as F
from torch import Tensor
from torch.nn import Identity
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, SparseTensor
from torch_geometric.utils import is_torch_sparse_tensor
from torch_geometric.utils.sparse import set_sparse_value


class CHATConv(MessagePassing):
    _cached_edge_index: Optional[OptPairTensor]
    _cached_adj_t: Optional[SparseTensor]
    _alpha: OptTensor

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        post_transform: bool = False,
        post_transform_shared: bool = False,
        cached: bool = False,
        add_self_loops: bool = True,
        normalize: bool = True,
        shared: bool = False,
        activation: str = "tanh",
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.post_transform = post_transform
        self.post_transform_shared = post_transform_shared
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        self.shared = shared
        self.activation = activation

        self._cached_edge_index = None
        self._cached_adj_t = None
        self._alpha = None

        if self.shared:
            self.att_l = self.att_r = Linear(in_channels, out_channels, bias=False)
        else:
            self.att_l = Linear(in_channels, out_channels, bias=False)
            self.att_r = Linear(in_channels, out_channels, bias=False)

        if self.post_transform:
            if self.post_transform_shared:
                self.fuse_l = self.fuse_r = Linear(
                    in_channels, out_channels, bias=False
                )
            else:
                self.fuse_l = Linear(in_channels, out_channels, bias=False)
                self.fuse_r = Linear(in_channels, out_channels, bias=False)
        else:
            self.fuse_l = Identity()
            self.fuse_r = Identity()

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self._cached_edge_index = None
        self._cached_adj_t = None
        glorot(self.att_l)
        glorot(self.att_r)

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        edge_weight: OptTensor = None,
        return_attention_weights=None,
        x_init=None,
    ):
        if self.normalize:
            if isinstance(edge_index, Tensor):
                assert edge_weight is None
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index,
                        None,
                        x.size(self.node_dim),
                        False,
                        self.add_self_loops,
                        self.flow,
                        dtype=x.dtype,
                    )
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                assert not edge_index.has_value()
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index,
                        None,
                        x.size(self.node_dim),
                        False,
                        self.add_self_loops,
                        self.flow,
                        dtype=x.dtype,
                    )
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache
        else:
            if isinstance(edge_index, Tensor) and not is_torch_sparse_tensor(
                edge_index
            ):
                assert edge_weight is not None
            elif isinstance(edge_index, SparseTensor):
                assert edge_index.has_value()

        alpha_l = self.att_l(x)
        alpha_r = self.att_r(x)
        # propagate_type: (x: Tensor, alpha: PairTensor, edge_weight: OptTensor)  # noqa
        msg = self.propagate(
            edge_index,
            x=x,
            alpha=(alpha_l, alpha_r),
            edge_weight=edge_weight,
            size=None,
        )

        alpha = self._alpha
        self._alpha = None
        out = self.fuse_l(x) + self.fuse_r(msg)
        if x_init is not None:
            out = out + x_init

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                if is_torch_sparse_tensor(edge_index):
                    # TODO TorchScript requires to return a tuple
                    adj = set_sparse_value(edge_index, alpha)
                    return out, (adj, alpha)
                else:
                    return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout="coo")
        else:
            return out

    def message(
        self, x_j: Tensor, alpha_j: Tensor, alpha_i: Tensor, edge_weight: OptTensor
    ) -> Tensor:
        assert edge_weight is not None
        alpha = getattr(F, self.activation)(alpha_j + alpha_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        self._alpha = alpha
        message = x_j * alpha * edge_weight.view(-1, 1)
        return message

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.in_channels}, {self.out_channels})"
