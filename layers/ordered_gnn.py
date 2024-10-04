import torch
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing


class ONGNNConv(MessagePassing):
    def __init__(
        self,
        hidden_channel,
        chunk_size,
        tm_net,
        tm_norm,
        simple_gating=True,
    ):
        super(ONGNNConv, self).__init__("mean")
        self.hidden_channel = hidden_channel
        self.chunk_size = chunk_size
        self.simple_gating = simple_gating
        self.tm_net = tm_net
        self.tm_norm = tm_norm

    def forward(self, x, edge_index, last_tm_signal):
        m = self.propagate(edge_index, x=x)
        if self.simple_gating:
            tm_signal_raw = F.sigmoid(self.tm_net(torch.cat((x, m), dim=1)))
        else:
            tm_signal_raw = F.softmax(self.tm_net(torch.cat((x, m), dim=1)), dim=-1)
            tm_signal_raw = torch.cumsum(tm_signal_raw, dim=-1)
            tm_signal_raw = last_tm_signal + (1 - last_tm_signal) * tm_signal_raw
        tm_signal = tm_signal_raw.repeat_interleave(
            repeats=int(self.hidden_channel / self.chunk_size), dim=1
        )
        out = x * tm_signal + m * (1 - tm_signal)
        out = self.tm_norm(out)

        return out, tm_signal_raw
