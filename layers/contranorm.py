import torch
import torch.nn as nn
import torch.nn.functional as F


class ContraNorm(nn.Module):
    def __init__(self, scale, tau):
        super(ContraNorm, self).__init__()
        self.scale = scale
        self.tau = tau

    def forward(self, x, edge_index):
        norm_x = F.normalize(x, dim=1)
        sim = norm_x @ norm_x.T / self.tau
        sim[edge_index[0], edge_index[1]] = -torch.inf
        sim = F.softmax(sim, dim=1)
        x_neg = sim @ x
        x = (1 + self.scale) * x - self.scale * x_neg

        return x
