import numpy as np
import torch
from torch.nn import ELU, Parameter, Softplus
from torch_geometric.nn.conv import MessagePassing
from torch_scatter import scatter_add


class AEROConv(MessagePassing):
    def __init__(
        self,
        hidden_size,
        n_heads=1,
        lambd=0.25,
        k=0,
    ):
        super(AEROConv, self).__init__()
        self.node_dim = 0
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.lambd = lambd
        self.k = k
        self.softplus = Softplus()
        self.elu = ELU()

        if self.k == 0:
            self.hop_att = Parameter(torch.Tensor(1, self.n_heads, self.hidden_size))
        else:
            self.hop_att = Parameter(
                torch.Tensor(1, self.n_heads, 2 * self.hidden_size)
            )

        self.hop_bias = Parameter(torch.Tensor(1, self.n_heads))
        self.att = Parameter(torch.Tensor(1, self.n_heads, self.hidden_size))
        self.decay_weight = np.log((self.lambd / (self.k + 1)) + (1 + 1e-6))

    def hop_att_pred(self, h, z_scale):
        if z_scale is None:
            x = h
        else:
            x = torch.cat((h, z_scale), dim=-1)

        g = x.view(-1, self.n_heads, int(x.shape[-1]))
        g = self.elu(g)
        g = (self.hop_att * g).sum(dim=-1) + self.hop_bias

        return g.unsqueeze(-1)

    def edge_att_pred(self, z_scale_i, z_scale_j, edge_index):
        a_ij = z_scale_i + z_scale_j
        a_ij = self.elu(a_ij)
        a_ij = (self.att * a_ij).sum(dim=-1)
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

    def aero_propagate(self, h, edge_index, z, z_scale):
        h = h.view(-1, self.n_heads, self.hidden_size)

        if self.k > 0:
            h = self.propagate(edge_index, x=h, z_scale=z_scale)

        g = self.hop_att_pred(h, z_scale)
        if z is None:
            z = h * g
        else:
            z = z + h * g
        z_scale = z * self.decay_weight

        return h, z, z_scale

    def forward(self, x, edge_index, z=None, z_scale=None):
        h, z, z_scale = self.aero_propagate(x, edge_index, z, z_scale)
        h = h.view(-1, self.n_heads * self.hidden_size)

        return h, z, z_scale
