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
        raise ValueError("Invalid SMILES string")

    node_features = []

    for atom in mol.GetAtoms():

        # 9 features
        atomic_number = atom.GetAtomicNum()
        degree = atom.GetDegree()
        formal_charge = atom.GetFormalCharge()
        hydrogens = atom.GetTotalNumHs()

        aromatic = 1 if atom.GetIsAromatic() else 0
        mass = atom.GetMass()
        valence = atom.GetTotalValence()
        implicit_valence = atom.GetImplicitValence()
        explicit_valence = atom.GetExplicitValence()

        features = [
            atomic_number,
            degree,
            formal_charge,
            hydrogens,
            aromatic,
            mass,
            valence,
            implicit_valence,
            explicit_valence
        ]

        node_features.append(features)

    x = torch.tensor(
        node_features,
        dtype=torch.float
    )

    # ==========================================
    # Edges
    # ==========================================

    edge_list = []
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

        # Forward edge
        edge_list.append([start, end])
        edge_features.append([bond_value])

        # Reverse edge
        edge_list.append([end, start])
        edge_features.append([bond_value])

    edge_index = torch.tensor(
        edge_list,
        dtype=torch.long
    ).t().contiguous()

    edge_attr = torch.tensor(
        edge_features,
        dtype=torch.float
    )

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr
    )


# ==========================================
# 2. GNN model
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

        batch = torch.zeros(
            x.size(0),
            dtype=torch.long
        )

        x = global_mean_pool(x, batch)

        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)

        return x


# ==========================================
# 3. Load trained model
# ==========================================

model = MolecularGNN()

model.load_state_dict(
    torch.load(
        "molecular_gnn_model.pth",
        map_location="cpu"
    )
)

model.eval()

print("========================================")
print("BATCH MOLECULE PREDICTION")
print("========================================")

print("Model loaded successfully!")


# ==========================================
# 4. Enter molecules
# ==========================================

print()
print("Enter SMILES strings.")
print("Enter one molecule at a time.")
print("Type DONE when finished.")
print()

smiles_list = []

while True:

    smiles = input("Enter SMILES: ").strip()

    if smiles.upper() == "DONE":
        break

    if smiles == "":
        continue

    smiles_list.append(smiles)


# ==========================================
# 5. Predictions
# ==========================================

print()
print("========================================")
print("PREDICTION RESULTS")
print("========================================")

results = []

with torch.no_grad():

    for smiles in smiles_list:

        try:

            data = smiles_to_graph(smiles)

            prediction = model(data)

            predicted_value = prediction.item()

            num_atoms = data.x.shape[0]
            num_edges = data.edge_index.shape[1]

            print(
                f"Molecule: {smiles} | "
                f"Atoms: {num_atoms} | "
                f"Edges: {num_edges} | "
                f"Predicted ESOL: {predicted_value:.4f}"
            )

            results.append(
                (
                    smiles,
                    num_atoms,
                    num_edges,
                    predicted_value
                )
            )

        except Exception as e:

            print(
                f"Molecule: {smiles} | ERROR: {e}"
            )


# ==========================================
# 6. Save CSV
# ==========================================

with open(
    "batch_predictions.csv",
    "w"
) as file:

    file.write(
        "SMILES,Atoms,Edges,Predicted_ESOL\n"
    )

    for result in results:

        smiles, atoms, edges, prediction = result

        file.write(
            f"{smiles},{atoms},{edges},{prediction:.4f}\n"
        )


print()
print("========================================")
print("DONE")
print("========================================")

print("Results saved to:")
print("batch_predictions.csv")