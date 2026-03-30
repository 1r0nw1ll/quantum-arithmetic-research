# qagcn.py
import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class QAGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, mod=24):
        super().__init__(aggr="add")  # sum aggregation
        self.lin = nn.Linear(in_channels, out_channels)
        self.mod = mod

    def forward(self, x, edge_index):
        # Linear projection
        x = self.lin(x)

        # Embed features into modular residue space
        if self.mod is not None:
            x = torch.remainder(x, self.mod) / float(self.mod)

        # Add self-loops to adjacency
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Compute degree for normalization
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Propagate messages
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        # scale messages by norm
        return norm.view(-1, 1) * x_j
