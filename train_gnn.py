import torch
import torch.nn.functional as F

from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool


# ==========================================
# 1. Load ESOL dataset
# ==========================================

print("Loading ESOL dataset...")

dataset = MoleculeNet(
    root="data",
    name="ESOL"
)

print("Total molecules:", len(dataset))


# ==========================================
# 2. Split dataset
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

print("Training:", len(train_dataset))
print("Validation:", len(validation_dataset))
print("Test:", len(test_dataset))


# ==========================================
# 3. Create DataLoaders
# ==========================================

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=32,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)


# ==========================================
# 4. Define GNN model
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

        # Convert node features to float
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

        # Convert node representations
        # into one representation per molecule
        x = global_mean_pool(x, data.batch)

        # Fully connected layer
        x = self.fc1(x)
        x = F.relu(x)

        # Final prediction
        x = self.fc2(x)

        return x


# ==========================================
# 5. Create model
# ==========================================

model = MolecularGNN()

print("\nModel:")
print(model)


# ==========================================
# 6. Loss function
# ==========================================

criterion = torch.nn.MSELoss()


# ==========================================
# 7. Optimizer
# ==========================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# ==========================================
# 8. Training function
# ==========================================

def train():

    model.train()

    total_loss = 0

    for batch in train_loader:

        # Clear old gradients
        optimizer.zero_grad()

        # Model prediction
        output = model(batch)

        # Target values
        target = batch.y.float()

        # Make sure target has same shape
        target = target.view(-1, 1)

        # Calculate loss
        loss = criterion(output, target)

        # Calculate gradients
        loss.backward()

        # Update model
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


# ==========================================
# 9. Validation function
# ==========================================

def validate():

    model.eval()

    total_loss = 0

    with torch.no_grad():

        for batch in validation_loader:

            output = model(batch)

            target = batch.y.float()
            target = target.view(-1, 1)

            loss = criterion(output, target)

            total_loss += loss.item()

    return total_loss / len(validation_loader)


# ==========================================
# 10. Train the model
# ==========================================

epochs = 50

print("\nStarting training...\n")

for epoch in range(1, epochs + 1):

    train_loss = train()

    validation_loss = validate()

    print(
        f"Epoch {epoch:03d} | "
        f"Training Loss: {train_loss:.4f} | "
        f"Validation Loss: {validation_loss:.4f}"
    )


# ==========================================
# 11. Save trained model
# ==========================================

torch.save(
    model.state_dict(),
    "molecular_gnn_model.pth"
)

print("\nTraining complete!")

print("Model saved as:")
print("molecular_gnn_model.pth")