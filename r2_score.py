import torch
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F
from sklearn.metrics import r2_score


# ==========================================
# Load ESOL dataset
# ==========================================

print("Loading ESOL dataset...")

dataset = MoleculeNet(
    root="data",
    name="ESOL"
)

print("Total molecules:", len(dataset))


# ==========================================
# Same train/validation/test split
# ==========================================

total = len(dataset)

train_size = int(0.8 * total)
validation_size = int(0.1 * total)
test_size = total - train_size - validation_size

generator = torch.Generator().manual_seed(42)

_, _, test_dataset = torch.utils.data.random_split(
    dataset,
    [train_size, validation_size, test_size],
    generator=generator
)

print("Test molecules:", len(test_dataset))


# ==========================================
# Test loader
# ==========================================

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)


# ==========================================
# Define model
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

        x = global_mean_pool(x, data.batch)

        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)

        return x


# ==========================================
# Load trained model
# ==========================================

model = MolecularGNN()

model.load_state_dict(
    torch.load(
        "molecular_gnn_model.pth",
        map_location="cpu"
    )
)

model.eval()

print("Trained model loaded successfully!")


# ==========================================
# Generate predictions
# ==========================================

predictions = []
actual_values = []

with torch.no_grad():

    for batch in test_loader:

        output = model(batch)

        predictions.extend(
            output.view(-1).tolist()
        )

        actual_values.extend(
            batch.y.view(-1).tolist()
        )


# ==========================================
# Calculate R²
# ==========================================

r2 = r2_score(
    actual_values,
    predictions
)


# ==========================================
# Print result
# ==========================================

print()
print("========================================")
print("R² SCORE")
print("========================================")

print(f"R²: {r2:.4f}")