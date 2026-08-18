from rdkit import Chem
import torch
from torch_geometric.data import Data


def smiles_to_graph(smiles):
    # Convert SMILES into a molecule
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        print("Invalid SMILES!")
        return None

    # -----------------------------
    # 1. Get atom features
    # -----------------------------

    node_features = []

    for atom in mol.GetAtoms():
        atomic_number = atom.GetAtomicNum()
        degree = atom.GetDegree()
        formal_charge = atom.GetFormalCharge()
        hydrogens = atom.GetTotalNumHs()

        node_features.append([
            atomic_number,
            degree,
            formal_charge,
            hydrogens
        ])

    # -----------------------------
    # 2. Get bonds and bond features
    # -----------------------------

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

        # Add start -> end
        edges.append([start, end])
        edge_features.append([bond_value])

        # Add end -> start
        edges.append([end, start])
        edge_features.append([bond_value])

    # -----------------------------
    # 3. Convert to PyTorch tensors
    # -----------------------------

    x = torch.tensor(
        node_features,
        dtype=torch.float
    )

    edge_index = torch.tensor(
        edges,
        dtype=torch.long
    ).t().contiguous()

    edge_attr = torch.tensor(
        edge_features,
        dtype=torch.float
    )

    # -----------------------------
    # 4. Create PyTorch Geometric graph
    # -----------------------------

    graph = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr
    )

    return graph


# -----------------------------
# Ask the user for a molecule
# -----------------------------

smiles = input("Enter a SMILES string: ")

graph = smiles_to_graph(smiles)


# -----------------------------
# Display the graph
# -----------------------------

if graph is not None:

    print("\n", graph)

    print("\nNode features:")
    print(graph.x)

    print("\nEdges:")
    print(graph.edge_index)

    print("\nBond features:")
    print(graph.edge_attr)