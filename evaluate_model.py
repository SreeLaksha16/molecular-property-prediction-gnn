import torch
import torch.nn.functional as F

from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

# ==========================================
# 1. Load dataset
# ==========================================

print("Loading ESOL dataset...")

dataset = MoleculeNet(
    root="data",
    name="ESOL"
)

print("Total molecules:", len(dataset))


# ==========================================
# 2. Create SAME dataset split
# ==========================================

total = len(dataset)

train_size = int(0.8 * total)
validation_size = int(0.1 * total)
test_size = total - train_size - validation_size

generator = torch.Generator().manual_seed(42)

train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(
    dataset,
    [train_size, validation_size, test_size],
    generator=generator
)

print("Training molecules:", len(train_dataset))
print("Validation molecules:", len(validation_dataset))
print("Test molecules:", len(test_dataset))


# ==========================================
# 3. Test DataLoader
# ==========================================

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)


# ==========================================
# 4. Define the SAME GNN model
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

        # GCN layer 1
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        # GCN layer 2
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # GCN layer 3
        x = self.conv3(x, edge_index)
        x = F.relu(x)

        # Convert node features into molecule-level features
        x = global_mean_pool(x, data.batch)

        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)

        return x


# ==========================================
# 5. Create model
# ==========================================

model = MolecularGNN()


# ==========================================
# 6. Load trained model
# ==========================================

model.load_state_dict(
    torch.load(
        "molecular_gnn_model.pth",
        map_location="cpu"
    )
)

model.eval()

print("\nTrained model loaded successfully!")


# ==========================================
# 7. Evaluate model
# ==========================================

all_predictions = []
all_targets = []

with torch.no_grad():

    for batch in test_loader:

        output = model(batch)

        target = batch.y.float()
        target = target.view(-1, 1)

        all_predictions.append(output)
        all_targets.append(target)


# ==========================================
# 8. Combine all results
# ==========================================

predictions = torch.cat(all_predictions, dim=0)
targets = torch.cat(all_targets, dim=0)

print("\nTotal predictions:", len(predictions))


# ==========================================
# 9. Calculate metrics
# ==========================================

mse = torch.mean((predictions - targets) ** 2)

rmse = torch.sqrt(mse)

mae = torch.mean(torch.abs(predictions - targets))


# ==========================================
# 10. Display metrics
# ==========================================

print("\n========================================")
print("MODEL TEST RESULTS")
print("========================================")

print(f"Test MSE:  {mse.item():.4f}")
print(f"Test RMSE: {rmse.item():.4f}")
print(f"Test MAE:  {mae.item():.4f}")


# ==========================================
# 11. Show first 10 predictions
# ==========================================

print("\n========================================")
print("FIRST 10 PREDICTIONS")
print("========================================")

for i in range(min(10, len(predictions))):

    actual = targets[i].item()
    predicted = predictions[i].item()

    print(
        f"Molecule {i + 1:02d} | "
        f"Actual: {actual:8.3f} | "
        f"Predicted: {predicted:8.3f}"
    )


# ==========================================
# 12. Save ALL predictions
# ==========================================

with open("test_predictions.csv", "w") as file:

    file.write("Actual,Predicted\n")

    for actual, predicted in zip(targets, predictions):

        file.write(
            f"{actual.item():.6f},{predicted.item():.6f}\n"
        )


print("\nAll predictions saved to:")
print("test_predictions.csv")


# ==========================================
# 13. Save predictions as PyTorch tensors
# ==========================================

torch.save(
    {
        "actual": targets,
        "predicted": predictions
    },
    "test_predictions.pt"
)

print("PyTorch prediction file saved to:")
print("test_predictions.pt")


# ==========================================
# 14. Final message
# ==========================================

print("\nEvaluation completed successfully!")