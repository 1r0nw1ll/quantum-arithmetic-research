# qa_qm9_converter.py

import torch
from torch_geometric.datasets import QM9
from torch_geometric.data import InMemoryDataset, Data
from tqdm import tqdm
from rdkit.Chem import GetPeriodicTable

# === QA Mapping: (b, e, d, a) ===
def map_atom_to_qa(atom_type):
    atom_bases = {
        'H': (1, 1),
        'C': (2, 1),
        'N': (3, 2),
        'O': (4, 2),
        'F': (5, 3),
    }
    b, e = atom_bases.get(atom_type, (1, 1))  # default if unknown
    d = b + e
    a = b + 2 * e
    return [b, e, d, a]

# === QA-QM9 Dataset ===
class QAQM9Dataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(QAQM9Dataset, self).__init__(root, transform, pre_transform)

        import torch.serialization
        from torch_geometric.data import Data
        torch.serialization.add_safe_globals([Data])

        self.data, self.slices, self.mean, self.std = torch.load(
            self.processed_paths[0], weights_only=False
        )

    @property
    def raw_file_names(self):
        return []  # handled internally by PyG

    @property
    def processed_file_names(self):
        return ['data_qaqm9.pt']

    def download(self):
        pass

    def process(self):
        dataset = QM9(root=self.root)
        pt = GetPeriodicTable()
        data_list = []

        print("📦 Processing QM9 → QA representation")
        for data in tqdm(dataset[:10000], desc="Building QA molecules"):
            atom_symbols = [pt.GetElementSymbol(int(z)) for z in data.z]
            x = torch.tensor([map_atom_to_qa(a) for a in atom_symbols], dtype=torch.float)

            edge_index = data.edge_index
            y = data.y.view(-1)[:19]  # force shape [19]

            data_q = Data(x=x, edge_index=edge_index, y=y.view(1, -1))  # shape: [1, 19]
            data_list.append(data_q)

        # === Normalize Targets ===
        ys = torch.stack([d.y for d in data_list])
        mean = ys.mean(dim=0)
        std = ys.std(dim=0)

        for d in data_list:
            d.y = (d.y - mean) / std

        data, slices = self.collate(data_list)
        torch.save((data, slices, mean, std), self.processed_paths[0])
        print("✅ Saved QA dataset with normalized targets")

# === Run Script ===
if __name__ == "__main__":
    dataset = QAQM9Dataset(root='data/qa_qm9')
    print(f"✅ Loaded {len(dataset)} QA molecules")
    print("📊 QA molecule sample:")
    print(dataset[0])
