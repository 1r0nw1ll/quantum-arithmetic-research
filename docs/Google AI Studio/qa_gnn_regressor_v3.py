# qa_gnn_regressor_v3.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from qa_qm9_converter import QAQM9Dataset
from torch_geometric.nn import global_mean_pool, MessagePassing
from torch_geometric.utils import add_self_loops

# -------------------------------
# QA-aware GCN Layer
# -------------------------------
class QAGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(QAGCNConv, self).__init__(aggr='add')
        self.lin = nn.Linear(in_channels, out_channels)
        self.qa_lin = nn.Linear(out_channels, out_channels)

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        return self.lin(x_j)

    def update(self, aggr_out, x):
        h = aggr_out + self.lin(x)
        # enforce QA consistency
        b, e, d, a = h.chunk(4, dim=-1)
        d_hat = b + e
        a_hat = b + 2 * e
        h = torch.cat([b, e, d_hat, a_hat], dim=-1)
        return F.relu(self.qa_lin(h))

# -------------------------------
# Residue Embedding Module
# -------------------------------
class ResidueEmbedding(nn.Module):
    def __init__(self, modulo, embed_dim):
        super().__init__()
        self.modulo = modulo
        self.embed = nn.Embedding(modulo, embed_dim)

    def forward(self, x):
        # x is [N,4] QA tuple
        residues = (x % self.modulo).long()  # project into Z_mod
        embeds = self.embed(residues)        # [N,4,embed_dim]
        return embeds.view(x.size(0), -1)    # flatten

# -------------------------------
# QA-GNN with Residues + Multi-heads
# -------------------------------
class QAGNNMultiHead(nn.Module):
    def __init__(self, num_node_features, hidden_dim, num_properties):
        super().__init__()
        # residue embeddings
        self.res24 = ResidueEmbedding(24, 8)
        self.res72 = ResidueEmbedding(72, 8)

        # base QA-GCN layers
        in_dim = num_node_features + 4*8 + 4*8  # raw + residues
        self.conv1 = QAGCNConv(in_dim, hidden_dim)
        self.conv2 = QAGCNConv(hidden_dim, hidden_dim)

        # multi-head outputs (one small MLP per property)
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim//2),
                nn.ReLU(),
                nn.Linear(hidden_dim//2, 1)
            )
            for _ in range(num_properties)
        ])

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # concatenate raw + residue embeddings
        res24 = self.res24(x)
        res72 = self.res72(x)
        x = torch.cat([x, res24, res72], dim=-1)

        # QA GCN layers
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)

        # global pooling
        x = global_mean_pool(x, batch)

        # multi-head outputs
        outs = [head(x) for head in self.heads]  # list of [B,1]
        return torch.cat(outs, dim=-1)  # [B, num_properties]

# -------------------------------
# Training / Evaluation
# -------------------------------
def train(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.mse_loss(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)

def test(model, loader, device):
    model.eval()
    total_loss = 0
    per_property_loss = None
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            loss = F.mse_loss(out, data.y, reduction='none')
            total_loss += loss.mean().item() * data.num_graphs
            if per_property_loss is None:
                per_property_loss = loss.sum(dim=0)
            else:
                per_property_loss += loss.sum(dim=0)
    per_property_loss /= len(loader.dataset)
    return total_loss / len(loader.dataset), per_property_loss.cpu().numpy()

# -------------------------------
# Main
# -------------------------------
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = QAQM9Dataset(root='data/qa_qm9')

    # 80/10/10 split
    train_dataset = dataset[:8000]
    test_dataset = dataset[8000:9000]
    val_dataset = dataset[9000:]

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)
    val_loader = DataLoader(val_dataset, batch_size=64)

    model = QAGNNMultiHead(num_node_features=4, hidden_dim=128, num_properties=19).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1, 31):
        train_loss = train(model, train_loader, optimizer, device)
        test_loss, per_prop = test(model, test_loader, device)
        print(f"Epoch {epoch:02d} | Train MSE: {train_loss:.4f} | Test MSE: {test_loss:.4f}")
        print("Per-property MSE:", per_prop)
        print("-" * 70)

if __name__ == "__main__":
    main()
