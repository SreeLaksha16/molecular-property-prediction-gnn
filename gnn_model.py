import torch
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, global_mean_pool

from smiles_to_graph import smiles_to_graph


class MolecularGNN(torch.nn.Module):

    def __init__(self):
        super().__init__()

        # First graph convolution layer
        self.conv1 = GCNConv(4, 16)

        # Second graph convolution layer
        self.conv2 = GCNConv(16, 16)

        # Final prediction layer
        self.fc = torch.nn.Linear(16, 1)

    def forward(self, data):

        # Get atom features
        x = data.x

        # Get graph connections
        edge_index = data.edge_index

        # First GNN layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        # Second GNN layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # Put all atom information together
        x = x.mean(dim=0, keepdim=True)

        # Make prediction
        output = self.fc(x)

        return output


# --------------------------------
# Test the GNN
# --------------------------------

smiles = "CCO"

graph = smiles_to_graph(smiles)

model = MolecularGNN()

prediction = model(graph)

print("Molecule:", smiles)
print("Graph:", graph)

print("\nPrediction:")
print(prediction)