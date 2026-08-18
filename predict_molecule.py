import torch
import torch.nn.functional as F

from rdkit import Chem
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool


# ==========================================
# 1. Convert SMILES to graph
# ==========================================

def smiles_to_graph(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError("Invalid SMILES string!")

    node_features = []

    # 9 features matching the ESOL dataset format
    for atom in mol.GetAtoms():

        atomic_num = atom.GetAtomicNum()
        degree = atom.GetDegree()
        formal_charge = atom.GetFormalCharge()
        num_hydrogens = atom.GetTotalNumHs()

        # Additional features
        aromatic = int(atom.GetIsAromatic())
        hybridization = int(atom.GetHybridization())
        num_radical_electrons = atom.GetNumRadicalElectrons()
        chiral_tag = int(atom.GetChiralTag())
        isotope = atom.GetIsotope()

        node_features.append([
            atomic_num,
            degree,
            formal_charge,
            num_hydrogens,
            aromatic,
            hybridization,
            num_radical_electrons,
            chiral_tag,
            isotope
        ])

    x = torch.tensor(node_features, dtype=torch.float)

    # --------------------------------------
    # Create edges
    # --------------------------------------

    edges = []
    edge_features = []

    for bond in mol.GetBonds():

        start = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()

        bond_type = bond.GetBondType()

        if bond_type == Chem.BondType.SINGLE:
            bond_value = 1.0
        elif bond_type == Chem.BondType.DOUBLE:
            bond_value = 2.0
        elif bond_type == Chem.BondType.TRIPLE:
            bond_value = 3.0
        elif bond_type == Chem.BondType.AROMATIC:
            bond_value = 1.5
        else:
            bond_value = 0.0

        # Add both directions
        edges.append([start, end])
        edges.append([end, start])

        edge_features.append([bond_value])
        edge_features.append([bond_value])

    edge_index = torch.tensor(
        edges,
        dtype=torch.long
    ).t().contiguous()

    edge_attr = torch.tensor(
        edge_features,
        dtype=torch.float
    )

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr
    )

    return data


# ==========================================
# 2. Define the same GNN model
# ==========================================

class MolecularGNN(torch.nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1 = GCNConv(9, 64)
        self.conv2 = GCNConv(64, 64)
        self.conv3 = GCNConv(64, 64)

        self.fc1 = torch.nn.Linear(64, 32)
        self.fc2 = torch.nn.Linear(32, 1)

    def forward(self, data):

        x = data.x.float()
        edge_index = data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = self.conv3(x, edge_index)
        x = F.relu(x)

        # One graph only
        batch = torch.zeros(
            x.size(0),
            dtype=torch.long,
            device=x.device
        )

        x = global_mean_pool(x, batch)

        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)

        return x


# ==========================================
# 3. Ask user for molecule
# ==========================================

smiles = input("Enter a SMILES string: ").strip()

if not smiles:
    print("No SMILES string entered.")
    exit()


# ==========================================
# 4. Convert molecule to graph
# ==========================================

try:

    data = smiles_to_graph(smiles)

except Exception as e:

    print("Error:", e)
    exit()


# ==========================================
# 5. Create model
# ==========================================

model = MolecularGNN()


# ==========================================
# 6. Load trained model
# ==========================================

try:

    model.load_state_dict(
        torch.load(
            "molecular_gnn_model.pth",
            map_location="cpu"
        )
    )

except FileNotFoundError:

    print(
        "\nERROR: molecular_gnn_model.pth was not found."
    )

    print(
        "Make sure this Python file is in the same folder "
        "as molecular_gnn_model.pth."
    )

    exit()


model.eval()


# ==========================================
# 7. Make prediction
# ==========================================

with torch.no_grad():

    prediction = model(data)

    predicted_value = prediction.item()


# ==========================================
# 8. Display result
# ==========================================

print("\n========================================")
print("MOLECULE PREDICTION")
print("========================================")

print("Molecule:", smiles)

print("Number of atoms:", data.x.shape[0])

print("Number of edges:", data.edge_index.shape[1])

print(
    f"Predicted ESOL value: {predicted_value:.4f}"
)

print("========================================")