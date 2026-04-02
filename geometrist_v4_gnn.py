import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import time

# --- 1. Core Geometric & Dataset Functions (from previous work) ---

def archimedes(Q1, Q2, Q3):
    s = Q1 + Q2 + Q3
    return s * s - 2 * (Q1 * Q1 + Q2 * Q2 + Q3 * Q3)

def face_quadrea_from_edges(Q, i, j, k):
    Q1 = Q[min(j, k), max(j, k)]; Q2 = Q[min(i, k), max(i, k)]; Q3 = Q[min(i, j), max(i, j)]
    return archimedes(Q1, Q2, Q3)

def quadrume(Q):
    Q01, Q02, Q03, Q12, Q13, Q23 = Q[0,1], Q[0,2], Q[0,3], Q[1,2], Q[1,3], Q[2,3]
    M = np.array([
        [2*Q01, Q01+Q02-Q12, Q01+Q03-Q13],
        [Q01+Q02-Q12, 2*Q02, Q02+Q03-Q23],
        [Q01+Q03-Q13, Q02+Q03-Q23, 2*Q03],
    ], float)
    return 0.5 * np.linalg.det(M)

def get_invariants_from_points(points):
    """Returns a dictionary of all invariants."""
    Q = np.zeros((4,4))
    for i in range(4):
        for j in range(i + 1, 4): Q[i,j] = np.sum((points[i] - points[j])**2)
    faces = [(0,1,2), (0,1,3), (0,2,3), (1,2,3)]
    A = {face: face_quadrea_from_edges(Q, *face) for face in faces}
    V = quadrume(Q)
    return {
        'Q': Q, 'A': A, 'V': V,
        'full_vector': np.nan_to_num([
            Q[0,1], Q[0,2], Q[0,3], Q[1,2], Q[1,3], Q[2,3],
            A[(0,1,2)], A[(0,1,3)], A[(0,2,3)], A[(1,2,3)], V
        ])
    }

def generate_gnn_dataset(num_samples=2000):
    node_data, adj_data, target_data, full_data, label_data = [], [], [], [], []
    adj_matrix = np.ones((4,4)) - np.eye(4)
    
    for _ in range(num_samples):
        # Positive example
        b, e, d = np.random.randint(1, 10, size=3)
        points_pos = np.array([[0,0,0], [b,0,0], [0,e,0], [0,0,d]], dtype=float)
        invariants = get_invariants_from_points(points_pos)
        Q_pos = invariants['Q']
        
        node_features = np.array([
            [Q_pos[0,1], Q_pos[0,2], Q_pos[0,3]], [Q_pos[0,1], Q_pos[1,2], Q_pos[1,3]],
            [Q_pos[0,2], Q_pos[1,2], Q_pos[2,3]], [Q_pos[0,3], Q_pos[1,3], Q_pos[2,3]],
        ])
        
        node_data.append(node_features)
        adj_data.append(adj_matrix)
        target_data.append(np.nan_to_num(list(invariants['A'].values()) + [invariants['V']]))
        full_data.append(invariants['full_vector'])
        label_data.append(1)

    return np.array(node_data), np.array(adj_data), np.array(target_data), np.array(full_data), np.array(label_data)

# --- 2. The GNN Generator Model (PyTorch) ---

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, node_features, adj_matrix):
        deg = torch.pow(adj_matrix.sum(1), -0.5)
        deg[torch.isinf(deg)] = 0
        adj_norm = deg.view(-1, 1) * adj_matrix * deg.view(1, -1)
        support = self.linear(node_features)
        output = torch.matmul(adj_norm, support)
        return output

class GNNGenerator(nn.Module):
    def __init__(self, node_feature_dim=3, gcn_hidden_dim=16, final_hidden_dim=64, output_dim=5):
        super(GNNGenerator, self).__init__()
        self.gcn1 = GCNLayer(node_feature_dim, gcn_hidden_dim)
        self.relu = nn.ReLU()
        self.mlp = nn.Sequential(
            nn.Linear(gcn_hidden_dim, final_hidden_dim), nn.ReLU(),
            nn.Linear(final_hidden_dim, output_dim)
        )

    def forward(self, node_features, adj_matrix):
        x = self.gcn1(node_features, adj_matrix)
        x = self.relu(x)
        graph_vector = x.mean(dim=1)
        output = self.mlp(graph_vector)
        return output

# --- 3. Main Execution ---

if __name__ == "__main__":
    start_time = time.time()
    
    print("--- Building the Geometrist v4.0 ---")
    
    # Corrected Unpacking: The function returns 5 values.
    # We don't need all of them for the verifier setup, so we use _ to ignore the ones we don't need.
    _, _, _, X_full_verifier, y_label_verifier = generate_gnn_dataset(num_samples=1000)
    verifier = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1).fit(X_full_verifier, y_label_verifier)
    print("Verifier ('The Geometrist') is ready.")

    # Prepare training data for the GNN Generator
    node_data, adj_data, target_data, _, _ = generate_gnn_dataset(num_samples=3000)
    
    scaler_X = StandardScaler().fit(node_data.reshape(-1, 3))
    scaler_y = StandardScaler().fit(target_data)
    
    node_data_scaled = scaler_X.transform(node_data.reshape(-1, 3)).reshape(node_data.shape)
    node_tensors = torch.tensor(node_data_scaled, dtype=torch.float32)
    adj_tensors = torch.tensor(adj_data, dtype=torch.float32)
    target_tensors = torch.tensor(scaler_y.transform(target_data), dtype=torch.float32)
    
    print("Training the GNN Generator...")
    generator = GNNGenerator()
    optimizer = torch.optim.Adam(generator.parameters(), lr=0.005)
    criterion = nn.MSELoss()
    
    epochs = 300; batch_size = 128
    
    for epoch in range(epochs):
        for i in range(0, len(node_tensors), batch_size):
            nodes_batch, adj_batch, targets_batch = node_tensors[i:i+batch_size], adj_tensors[i:i+batch_size], target_tensors[i:i+batch_size]
            
            optimizer.zero_grad()
            outputs = generator(nodes_batch, adj_batch)
            loss = criterion(outputs, targets_batch)
            loss.backward()
            optimizer.step()
            
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")
    
    print("Training complete.\n")

    print("--- Evaluating the GNN Generator ---")
    generator.eval()
    
    node_test, adj_test, _, full_test, _ = generate_gnn_dataset(num_samples=500)
    node_test_scaled = scaler_X.transform(node_test.reshape(-1, 3)).reshape(node_test.shape)
    node_test_tensors = torch.tensor(node_test_scaled, dtype=torch.float32)
    adj_test_tensors = torch.tensor(adj_test, dtype=torch.float32)

    with torch.no_grad():
        predicted_targets_scaled = generator(node_test_tensors, adj_test_tensors)
    
    predicted_targets = scaler_y.inverse_transform(predicted_targets_scaled.numpy())
    generated_samples = np.hstack([full_test[:, :6], predicted_targets])
    
    final_verdict = verifier.predict(generated_samples)
    num_valid = np.sum(final_verdict)
    success_rate = num_valid / len(generated_samples)
    
    print(f"Generated {len(generated_samples)} new geometric vectors using the GNN.")
    print(f"The 'Geometrist' discriminator classified {num_valid} of them as 'Valid Geometry'.")
    print(f"Generator Success Rate: {success_rate:.0%}")

    if success_rate > 0.9:
        print("\n✅ BREAKTHROUGH: The GNN has learned to generate valid geometric theorems with near-perfect fidelity.")
    elif success_rate > 0.6:
        print("\n✅ STRONG SUCCESS: The GNN architecture provided a major leap in performance.")
    else:
        print("\n❌ ANALYSIS REQUIRED: The GNN did not achieve an immediate breakthrough.")

    print(f"\nTotal execution time: {time.time() - start_time:.2f} seconds.")
