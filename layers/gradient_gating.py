import torch
import torch.nn.functional as F
from torch_geometric.utils import scatter


class G2(torch.nn.Module):
    def __init__(self, conv, p=2.0, conv_type="gcn"):
        super(G2, self).__init__()
        self.conv = conv
        self.p = p
        self.conv_type = conv_type

    def forward(self, X, edge_index):
        n_nodes = X.size(0)
        if self.conv_type == "gat":
            X = F.elu(self.conv(X, edge_index)).view(n_nodes, -1, 4).mean(dim=-1)
        else:
            X = F.relu(self.conv(X, edge_index))

        gg = torch.tanh(
            scatter(
                (torch.abs(X[edge_index[0]] - X[edge_index[1]]) ** self.p).squeeze(-1),
                edge_index[0],
                0,
                dim_size=X.size(0),
                reduce="mean",
            )
        )

        return gg
