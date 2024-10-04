from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    FAConv,
    GATConv,
    GatedGraphConv,
    GATv2Conv,
    GCN2Conv,
    GCNConv,
    JumpingKnowledge,
    SAGEConv,
)
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, ones
from torch_geometric.utils import add_self_loops
from torch_scatter import scatter_add

from layers import CHATConv, ContraNorm, DirGNNConv, H2Conv, MixHopConv, ONGNNConv


class BaseModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        kwargs = defaultdict(lambda: None, kwargs)
        self.in_features = kwargs["in_features"]
        self.hidden_size = kwargs["hidden_size"]
        self.num_layers = kwargs["num_layers"]
        self.out_features = kwargs["out_features"]
        self.dropout = kwargs["dropout"]
        self.initial_dropout = kwargs["initial_dropout"]
        self.conv_layers = nn.ModuleList()
        self.dropout_layer = nn.Dropout(kwargs["dropout"])
        self.n_heads = kwargs["n_heads"]
        self.multi_out_heads = kwargs["multi_out_heads"]
        self.activation_func = kwargs["activation_func"]
        self.aggr = kwargs["aggr"]
        self.jk_mode = kwargs["jk_mode"]
        self.iterations = kwargs["iterations"]
        self.lambd = kwargs["lambd"]
        self.add_dropout = kwargs["add_dropout"]
        self.alpha = kwargs["alpha"]
        self.rgnn_prop = kwargs["rgnn_prop"]
        self.rgnn_rnn = kwargs["rgnn_rnn"]
        self.eps = kwargs["eps"]
        self.alpha_gcn2 = kwargs["alpha_gcn2"]
        self.theta_gcn2 = kwargs["theta_gcn2"]
        # attention layers accept None for "don't return the attention weights"
        if not kwargs["return_attention_weights"]:
            self.return_attention_weights = None
        else:
            self.return_attention_weights = True
        self.cn_scale = kwargs["cn_scale"]
        self.cn_tau = kwargs["cn_tau"]
        self.dir_alpha = kwargs["dir_alpha"]
        self.dir_conv = kwargs["dir_conv"]
        self.og_chunk_size = kwargs["og_chunk_size"]
        self.og_num_input_layers = kwargs["og_num_input_layers"]
        self.og_simple_gating = kwargs["og_simple_gating"]
        self.og_global_gating = kwargs["og_global_gating"]
        self.post_transform = kwargs["post_transform"]
        self.layer_norm = kwargs["layer_norm"]
        self.post_transform_shared = kwargs["post_transform_shared"]
        self.ca_activation = kwargs["ca_activation"]

    @property
    def representation_signature(self):
        if self.num_layers < 2:
            return self.conv_layers[0], "output"  # what choice do i have?
        return self.conv_layers[-1], "input"

    def activate_store_emb(self):
        representation_layer, side = self.representation_signature
        self.x_emb = None

        def store_out(l, inp, out):
            if side == "input":
                reps = inp[0]
            else:
                reps = out
            self.x_emb = reps.view(reps.size(0), -1)

        self.emb_hook_handle = representation_layer.register_forward_hook(store_out)

    def deactivate_store_emb(self):
        del self.x_emb
        self.emb_hook_handle.remove()


class MLP(BaseModel):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        for i in range(self.num_layers):
            inp = self.in_features if i == 0 else self.hidden_size
            out = self.out_features if i == self.num_layers - 1 else self.hidden_size
            self.conv_layers.append(Linear(inp, out))

    def forward(self, x, edge_index):
        for i, layer in enumerate(self.conv_layers):
            x = self.dropout_layer(x)
            x = layer(x)
            if i < self.num_layers - 1:
                x = getattr(F, self.activation_func)(x)
        return x


class GCN(BaseModel):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        for i in range(self.num_layers):
            inp = self.in_features if i == 0 else self.hidden_size
            out = self.out_features if i == self.num_layers - 1 else self.hidden_size
            self.conv_layers.append(GCNConv(inp, out))

    def forward(self, x, edge_index):
        for i, layer in enumerate(self.conv_layers):
            x = self.dropout_layer(x)
            x = layer(x, edge_index)
            if i < self.num_layers - 1:
                x = getattr(F, self.activation_func)(x)
        return x


class GAT(BaseModel):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        for i in range(self.num_layers):
            inp = self.in_features if i == 0 else self.hidden_size
            out = (
                self.out_features
                if i == self.num_layers - 1
                else self.hidden_size // self.n_heads
            )
            n_heads = (
                1
                if (i == self.num_layers - 1 and not self.multi_out_heads)
                else self.n_heads
            )
            concat = i < self.num_layers - 1
            self.conv_layers.append(
                GATConv(inp, out, n_heads, dropout=self.dropout, concat=concat)
            )

    def forward(self, x, edge_index):
        for i, layer in enumerate(self.conv_layers):
            x = self.dropout_layer(x)
            x = layer(x, edge_index)
            if i < self.num_layers - 1:
                x = getattr(F, self.activation_func)(x)
        return x


class GraphSAGE(BaseModel):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        for i in range(self.num_layers):
            inp = self.in_features if i == 0 else self.hidden_size
            out = self.out_features if i == self.num_layers - 1 else self.hidden_size
            self.conv_layers.append(SAGEConv(inp, out, aggr=self.aggr))

    def forward(self, x, edge_index):
        for i, layer in enumerate(self.conv_layers):
            x = self.dropout_layer(x)
            x = layer(x, edge_index)
            if i < self.num_layers - 1:
                x = getattr(F, self.activation_func)(x)
        return x


class GATv2(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for i in range(self.num_layers):
            inp = self.in_features if i == 0 else self.hidden_size
            out = (
                self.out_features
                if i == self.num_layers - 1
                else self.hidden_size // self.n_heads
            )
            n_heads = (
                1
                if (i == self.num_layers - 1 and not self.multi_out_heads)
                else self.n_heads
            )
            concat = i < self.num_layers - 1
            self.conv_layers.append(
                GATv2Conv(
                    inp,
                    out,
                    n_heads,
                    dropout=self.dropout,
                    share_weights=True,
                    concat=concat,
                )
            )

    def forward(self, x, edge_index):
        for i, layer in enumerate(self.conv_layers):
            x = self.dropout_layer(x)
            x = layer(x, edge_index)
            if i < self.num_layers - 1:
                x = getattr(F, self.activation_func)(x)
        return x


class GATv2Res(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for i in range(self.num_layers):
            inp = self.in_features if i == 0 else self.hidden_size
            out = (
                self.out_features
                if i == self.num_layers - 1
                else self.hidden_size // self.n_heads
            )
            n_heads = (
                1
                if (i == self.num_layers - 1 and not self.multi_out_heads)
                else self.n_heads
            )
            concat = i < self.num_layers - 1
            self.conv_layers.append(
                GATv2Conv(
                    inp,
                    out,
                    n_heads,
                    dropout=self.dropout,
                    share_weights=True,
                    concat=concat,
                )
            )

    def forward(self, x, edge_index):
        x = self.dropout_layer(x)
        x = self.conv_layers[0](x, edge_index)
        x = getattr(F, self.activation_func)(x)
        x_init = x
        x = self.dropout_layer(x)
        for i in range(1, self.num_layers):
            x = self.conv_layers[i](x, edge_index)
            if i < self.num_layers - 1:
                x = x * (1 - self.alpha) + x_init * self.alpha
            x = getattr(F, self.activation_func)(x)
            x = self.dropout_layer(x)

        return x


class JKNet(BaseModel):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        for i in range(self.num_layers):
            self.conv_layers.append(GCNConv(self.hidden_size, self.hidden_size))
        self.jk = JumpingKnowledge(
            mode=self.jk_mode, channels=self.hidden_size, num_layers=self.num_layers
        )
        self.out_layer_inp = (
            (self.num_layers + 1) * self.hidden_size
            if self.jk_mode == "cat"
            else self.hidden_size
        )
        self.input_layer = nn.Linear(self.in_features, self.hidden_size)
        self.output_layer = nn.Linear(self.out_layer_inp, self.out_features)

    @property
    def representation_signature(self):
        return self.jk, "output"

    def forward(self, x, edge_index):
        x = self.dropout_layer(x)
        x = self.input_layer(x)
        x = getattr(F, self.activation_func)(x)
        x = self.dropout_layer(x)
        xs = [x]
        for i, layer in enumerate(self.conv_layers):
            x = layer(x, edge_index)
            x = getattr(F, self.activation_func)(x)
            x = self.dropout_layer(x)
            xs.append(x)

        x = self.jk(xs)
        x = self.dropout_layer(x)
        x = self.output_layer(x)
        return x


class GCN2(BaseModel):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.input_layer = nn.Linear(self.in_features, self.hidden_size)
        self.output_layer = nn.Linear(self.hidden_size, self.out_features)
        for i in range(self.num_layers):
            self.conv_layers.append(
                GCN2Conv(
                    self.hidden_size,
                    alpha=self.alpha_gcn2,
                    theta=self.theta_gcn2,
                    layer=i + 1,
                )
            )

    @property
    def representation_signature(self):
        return self.output_layer, "input"

    def forward(self, x, edge_index):
        x = self.dropout_layer(x)
        x = self.input_layer(x)
        x = getattr(F, self.activation_func)(x)
        x_0 = x
        for i, layer in enumerate(self.conv_layers):
            x = self.dropout_layer(x)
            x = layer(x, x_0, edge_index)
            if i < self.num_layers - 1:
                x = getattr(F, self.activation_func)(x)

        x = self.dropout_layer(x)
        x = self.output_layer(x)
        return x


class AERO(BaseModel, MessagePassing):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.node_dim = 0
        self.hidden_size_ = self.n_heads * self.hidden_size
        self.K = self.iterations
        self.cached_edge_index = None

        self.setup_layers()
        self.reset_parameters()

    def setup_layers(self):
        self.dropout = nn.Dropout(self.dropout)
        self.elu = nn.ELU()
        self.softplus = nn.Softplus()

        self.dense_lins = nn.ModuleList()
        self.atts = nn.ParameterList()
        self.hop_atts = nn.ParameterList()
        self.hop_biases = nn.ParameterList()
        self.decay_weights = []

        self.dense_lins.append(
            Linear(
                self.in_features,
                self.hidden_size_,
                bias=True,
                weight_initializer="glorot",
            )
        )
        for _ in range(self.num_layers - 1):
            self.dense_lins.append(
                Linear(
                    self.hidden_size_,
                    self.hidden_size_,
                    bias=True,
                    weight_initializer="glorot",
                )
            )
        self.dense_lins.append(
            Linear(
                self.hidden_size_,
                self.out_features,
                bias=True,
                weight_initializer="glorot",
            )
        )

        self.hop_atts.append(
            nn.Parameter(torch.Tensor(1, self.n_heads, self.hidden_size))
        )
        for k in range(self.K + 1):
            if k > 0:
                self.atts.append(
                    nn.Parameter(torch.Tensor(1, self.n_heads, self.hidden_size))
                )
                self.hop_atts.append(
                    nn.Parameter(torch.Tensor(1, self.n_heads, self.hidden_size * 2))
                )
            self.hop_biases.append(nn.Parameter(torch.Tensor(1, self.n_heads)))
            self.decay_weights.append(np.log((self.lambd / (k + 1)) + (1 + 1e-6)))

    def reset_parameters(self):
        for lin in self.dense_lins:
            lin.reset_parameters()
        for att in self.atts:
            glorot(att)
        for att in self.hop_atts:
            glorot(att)
        for bias in self.hop_biases:
            ones(bias)

    def hid_feat_init(self, x):
        x = self.dropout(x)
        x = self.dense_lins[0](x)

        for l in range(self.num_layers - 1):
            x = self.elu(x)
            x = self.dropout(x)
            x = self.dense_lins[l + 1](x)

        return x

    def aero_propagate(self, h, edge_index):
        self.k = 0
        h = h.view(-1, self.n_heads, self.hidden_size)
        g = self.hop_att_pred(h, z_scale=None)
        z = h * g
        z_scale = z * self.decay_weights[self.k]

        for k in range(self.K):
            self.k = k + 1
            h = self.propagate(edge_index, x=h, z_scale=z_scale)
            g = self.hop_att_pred(h, z_scale)
            z = z + h * g
            z_scale = z * self.decay_weights[self.k]

        return z

    def node_classifier(self, z):
        z = z.view(-1, self.n_heads * self.hidden_size)
        z = self.elu(z)
        if self.add_dropout:
            z = self.dropout(z)
        z = self.dense_lins[-1](z)

        return z

    def hop_att_pred(self, h, z_scale):
        if z_scale is None:
            x = h
        else:
            x = torch.cat((h, z_scale), dim=-1)

        g = x.view(-1, self.n_heads, int(x.shape[-1]))
        g = self.elu(g)
        g = (self.hop_atts[self.k] * g).sum(dim=-1) + self.hop_biases[self.k]

        return g.unsqueeze(-1)

    def edge_att_pred(self, z_scale_i, z_scale_j, edge_index):
        a_ij = z_scale_i + z_scale_j
        a_ij = self.elu(a_ij)
        a_ij = (self.atts[self.k - 1] * a_ij).sum(dim=-1)
        a_ij = self.softplus(a_ij) + 1e-6

        row, col = edge_index
        deg = scatter_add(a_ij, col, dim=0, dim_size=z_scale_i.shape[0])
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)
        a_ij = deg_inv_sqrt[row] * a_ij * deg_inv_sqrt[col]

        return a_ij

    def message(self, edge_index, x_j, z_scale_i, z_scale_j):
        a = self.edge_att_pred(z_scale_i, z_scale_j, edge_index)
        return a.unsqueeze(-1) * x_j

    @property
    def representation_signature(self):
        return self.dense_lins[-1], "input"

    def forward(self, x, edge_index):
        if self.cached_edge_index is None:
            edge_index, _ = add_self_loops(edge_index)
        else:
            edge_index = self.cached_edge_index
        h0 = self.hid_feat_init(x)
        z_k_max = self.aero_propagate(h0, edge_index)
        z_star = self.node_classifier(z_k_max)
        return z_star


class RGNN(BaseModel):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        for i in range(self.num_layers):
            if self.rgnn_prop == "gcn":
                inp = self.hidden_size
                out = self.hidden_size
                self.conv_layers.append(GCNConv(inp, out))
            elif self.rgnn_prop == "gat":
                inp = self.hidden_size
                out = self.hidden_size // self.n_heads
                n_heads = (
                    1
                    if (i == self.num_layers - 1 and not self.multi_out_heads)
                    else self.n_heads
                )
                concat = i < self.num_layers - 1
                self.conv_layers.append(
                    GATConv(inp, out, n_heads, dropout=self.dropout, concat=concat)
                )
            else:
                raise ValueError("Wrong conv layer")

        self.input_linear = nn.Linear(self.in_features, self.hidden_size)
        self.output_linear = nn.Linear(self.hidden_size, self.out_features)

        if self.rgnn_rnn == "lstm":
            self.rnn = nn.LSTMCell(self.hidden_size, self.hidden_size)
        elif self.rgnn_rnn == "gru":
            self.rnn = nn.GRUCell(self.hidden_size, self.hidden_size)
        else:
            raise ValueError("Wrong rnn layer")

    def rnn_step(self, x, x_prev=None):
        if self.rgnn_rnn == "lstm":
            rnn_out = self.rnn(x, x_prev)
            x_prev = rnn_out[0]
        else:
            rnn_out = self.rnn(x, x_prev)
            x_prev = rnn_out

        return x_prev, rnn_out

    @property
    def representation_signature(self):
        return self.output_linear, "input"

    def forward(self, x, edge_index):
        x = self.dropout_layer(x)
        x = self.input_linear(x)
        x, out = self.rnn_step(x)
        for i, layer in enumerate(self.conv_layers):
            x = self.dropout_layer(x)
            x = layer(x, edge_index)
            x, out = self.rnn_step(x, out)
            x = getattr(F, self.activation_func)(x)

        x = self.output_linear(x)
        return x


class GGNN(BaseModel):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.input_linear = nn.Linear(self.in_features, self.hidden_size)
        self.ggnn = GatedGraphConv(self.hidden_size, self.num_layers, self.aggr)
        self.output_linear = nn.Linear(self.hidden_size, self.out_features)

    @property
    def representation_signature(self):
        return self.output_linear, "input"

    def forward(self, x, edge_index):
        x = self.dropout_layer(x)
        x = self.input_linear(x)
        x = self.ggnn(x, edge_index)
        x = self.dropout_layer(x)
        x = self.output_linear(x)
        return x


class FAGCN(BaseModel):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.input_linear = nn.Linear(self.in_features, self.hidden_size, bias=True)
        self.output_linear = nn.Linear(self.hidden_size, self.out_features, bias=True)
        for _ in range(self.num_layers):
            self.conv_layers.append(
                FAConv(
                    self.hidden_size,
                    eps=self.eps,
                    dropout=self.dropout,
                    add_self_loops=False,
                )
            )

    @property
    def representation_signature(self):
        return self.output_linear, "input"

    def forward(self, x, edge_index):
        x = self.dropout_layer(x)
        x = self.input_linear(x)
        x = getattr(F, self.activation_func)(x)
        x = self.dropout_layer(x)
        x_0 = x
        all_att_weights = []
        for layer in self.conv_layers:
            x = layer(
                x,
                x_0,
                edge_index,
                return_attention_weights=self.return_attention_weights,
            )
            if self.return_attention_weights:
                x, (_, att_weights) = x
                all_att_weights.append(att_weights)

        x = self.output_linear(x)

        if self.return_attention_weights:
            return x, all_att_weights
        return x


class H2GCN(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_linear = nn.Linear(self.in_features, self.hidden_size, bias=False)
        self.output_linear = nn.Linear(self.hidden_size, self.out_features, bias=False)
        self.combine = nn.Linear(
            (2 ** (self.num_layers + 1) - 1) * self.hidden_size,
            self.out_features,
            bias=False,
        )
        self.conv = H2Conv()

    @property
    def representation_signature(self):
        return self.combine, "input"

    def forward(self, x, edge_index):
        x = self.input_linear(x)
        x = getattr(F, self.activation_func)(x)

        rs = [x]
        for _ in range(self.num_layers):
            res = self.conv(rs[-1], edge_index)
            rs.append(res)
        x = torch.cat(rs, dim=1)
        del rs
        x = self.dropout_layer(x)
        x = self.combine(x)
        return x


class MixHop(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_layers.append(
            MixHopConv(
                self.in_features,
                self.hidden_size,
                add_self_loops=False,
                powers=list(range(self.num_layers + 1)),
            )
        )
        self.conv_layers.append(
            MixHopConv(
                (self.num_layers + 1) * self.hidden_size,
                self.out_features,
                add_self_loops=False,
                powers=list(range(self.num_layers + 1)),
            ),
        )

    @property
    def representation_signature(self):
        return self.output_linear, "input"

    def forward(self, x, edge_index):
        x = self.conv_layers[0](x, edge_index)
        getattr(F, self.activation_func)(x)
        x = self.dropout_layer(x)
        x = self.conv_layers[1](x, edge_index)
        getattr(F, self.activation_func)(x)
        x = self.dropout_layer(x)
        return x


class CNGCN(BaseModel):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.cn = ContraNorm(self.cn_scale, self.cn_tau)
        for i in range(self.num_layers):
            inp = self.in_features if i == 0 else self.hidden_size
            out = self.out_features if i == self.num_layers - 1 else self.hidden_size
            self.conv_layers.append(GCNConv(inp, out))

    @property
    def representation_signature(self):
        return self.cn, "output"

    def forward(self, x, edge_index):
        for i, layer in enumerate(self.conv_layers):
            x = self.dropout_layer(x)
            x = layer(x, edge_index)
            x = self.cn(x, edge_index)
            if i < self.num_layers - 1:
                x = getattr(F, self.activation_func)(x)
        return x


class DirGNN(BaseModel):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.input_linear = nn.Linear(self.in_features, self.hidden_size, bias=True)
        self.jk = JumpingKnowledge(
            mode=self.jk_mode,
            channels=self.hidden_size,
            num_layers=self.num_layers,
        )
        output_size = (
            self.num_layers * self.hidden_size
            if self.jk_mode == "cat"
            else self.hidden_size
        )
        self.output_linear = nn.Linear(output_size, self.out_features, bias=True)

        if self.layer_norm:
            self.layer_norms = nn.ModuleList()
        for i in range(self.num_layers):
            if self.dir_conv == "gcn":
                conv_layer = GCNConv(
                    self.hidden_size, self.hidden_size, add_self_loops=False
                )
            elif self.dir_conv == "chatgnn":
                conv_layer = CHATConv(
                    self.hidden_size,
                    self.hidden_size,
                    cached=True,
                    dropout=self.dropout,
                    post_transform=self.post_transform,
                    add_self_loops=False,
                )
                if self.layer_norm:
                    self.layer_norms.append(nn.LayerNorm(self.hidden_size))

            self.conv_layers.append(
                DirGNNConv(conv_layer, alpha=self.dir_alpha, root_weight=False)
            )

    @property
    def representation_signature(self):
        return self.output_linear, "input"

    def forward(self, x, edge_index):
        if self.dir_conv == "gcn":
            if self.initial_dropout:
                x = self.dropout_layer(x)
            x = self.input_linear(x)
            x = getattr(F, self.activation_func)(x)
            x = self.dropout_layer(x)
            x_0 = x
            xs = []

            for i, layer in enumerate(self.conv_layers):
                x = layer(x, edge_index)
                x = getattr(F, self.activation_func)(x)
                x = self.dropout_layer(x)
                if self.layer_norm:
                    x = F.normalize(x, p=2, dim=1)
                xs.append(x)

            x = self.jk(xs)
            x = self.output_linear(x)

        elif self.dir_conv == "chatgnn":
            if self.initial_dropout:
                x = self.dropout_layer(x)
            x = self.input_linear(x)
            x = getattr(F, self.activation_func)(x)
            x = self.dropout_layer(x)
            x_0 = x
            for i, layer in enumerate(self.conv_layers):
                x = layer(x, edge_index)
                x = x + x_0
                if self.layer_norm:
                    x = self.layer_norms[i](x)

            x = self.dropout_layer(x)
            x = self.output_linear(x)
        return x


class GONN(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_linear_net = nn.ModuleList()
        self.output_linear = Linear(self.hidden_size, self.out_features)
        self.norm_input = nn.ModuleList()
        self.conv_layers = nn.ModuleList()
        self.tm_norm = nn.ModuleList()
        self.tm_net = nn.ModuleList()

        self.input_linear_net.append(Linear(self.in_features, self.hidden_size))

        self.norm_input.append(nn.LayerNorm(self.hidden_size))

        for i in range(self.og_num_input_layers - 1):
            self.input_linear_net.append(Linear(self.hidden_size, self.hidden_size))
            self.norm_input.append(nn.LayerNorm(self.hidden_size))

        if self.og_global_gating:
            gate = Linear(2 * self.hidden_size, self.og_chunk_size)

        for i in range(self.num_layers):
            self.tm_norm.append(nn.LayerNorm(self.hidden_size))
            if self.og_global_gating:
                self.tm_net.append(gate)
            else:
                self.tm_net.append(Linear(2 * self.hidden_size, self.og_chunk_size))

            self.conv_layers.append(
                ONGNNConv(
                    self.hidden_size,
                    self.og_chunk_size,
                    tm_net=self.tm_net[i],
                    tm_norm=self.tm_norm[i],
                    simple_gating=self.og_simple_gating,
                )
            )

    @property
    def representation_signature(self):
        return self.output_linear, "input"

    def forward(self, x, edge_index):
        for i in range(len(self.input_linear_net)):
            x = self.dropout_layer(x)
            x = F.relu(self.input_linear_net[i](x))
            x = self.norm_input[i](x)

        tm_signal = x.new_zeros(self.og_chunk_size)

        for j in range(len(self.conv_layers)):
            x = self.dropout_layer(x)
            x, tm_signal = self.conv_layers[j](x, edge_index, last_tm_signal=tm_signal)

        x = self.dropout_layer(x)
        x = self.output_linear(x)

        return x


class CHATGNN(BaseModel):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.input_linear = nn.Linear(self.in_features, self.hidden_size, bias=True)
        self.output_linear = nn.Linear(self.hidden_size, self.out_features, bias=True)
        if self.layer_norm:
            self.layer_norms = nn.ModuleList()
        for i in range(self.num_layers):
            self.conv_layers.append(
                CHATConv(
                    self.hidden_size,
                    self.hidden_size,
                    dropout=self.dropout,
                    post_transform=self.post_transform,
                    post_transform_shared=self.post_transform_shared,
                    activation=self.ca_activation,
                    add_self_loops=False,
                    cached=True,
                )
            )
            if self.layer_norm:
                self.layer_norms.append(nn.LayerNorm(self.hidden_size))

    @property
    def representation_signature(self):
        return self.output_linear, "input"

    def forward(self, x, edge_index):
        if self.initial_dropout:
            x = self.dropout_layer(x)
        x = self.input_linear(x)
        x = getattr(F, self.activation_func)(x)
        x = self.dropout_layer(x)
        x_0 = x
        all_att_weights = []
        for i, layer in enumerate(self.conv_layers):
            x = layer(
                x,
                edge_index,
                return_attention_weights=self.return_attention_weights,
                x_init=x_0,
            )
            if self.return_attention_weights:
                x, (_, att_weights) = x
                all_att_weights.append(att_weights)

            if self.layer_norm:
                x = self.layer_norms[i](x)

        x = self.output_linear(x)

        if self.return_attention_weights:
            return x, all_att_weights

        return x


MODEL_TO_CLS = {
    "mlp": MLP,
    "gcn": GCN,
    "gat": GAT,
    "gatv2": GATv2,
    "gatv2res": GATv2Res,
    "graphsage": GraphSAGE,
    "jknet": JKNet,
    "gcn2": GCN2,
    "aero": AERO,
    "rgnn": RGNN,
    "ggnn": GGNN,
    "fagcn": FAGCN,
    "chatgnn": CHATGNN,
    "h2gcn": H2GCN,
    "mixhop": MixHop,
    "cngcn": CNGCN,
    "dirgnn": DirGNN,
    "gonn": GONN,
}
