import copy
import random
import numpy as np
import torch
import torch.nn as nn

from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (
    GCNConv,
    SAGEConv,
    GATConv,
    global_mean_pool
)

# ============================================================
# 1. SETTINGS
# ============================================================

SEED = 42
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("GNN ARCHITECTURE COMPARISON")
print("=" * 60)
print(f"Device: {DEVICE}")


# ============================================================
# 2. LOAD ESOL DATASET
# ============================================================

print("\nLoading ESOL dataset...")

dataset = MoleculeNet(
    root="data",
    name="ESOL"
)

print(f"Total molecules: {len(dataset)}")


# ============================================================
# 3. TRAIN / VALIDATION / TEST SPLIT
# ============================================================

# Same 80/10/10 style split used in your project.
# We use a fixed seed so the comparison is reproducible.

generator = torch.Generator().manual_seed(SEED)

indices = torch.randperm(
    len(dataset),
    generator=generator
).tolist()

train_size = int(0.80 * len(dataset))
val_size = int(0.10 * len(dataset))

train_indices = indices[:train_size]
val_indices = indices[train_size:train_size + val_size]
test_indices = indices[train_size + val_size:]

train_dataset = [dataset[i] for i in train_indices]
val_dataset = [dataset[i] for i in val_indices]
test_dataset = [dataset[i] for i in test_indices]

print("\nDataset split:")
print(f"Training:   {len(train_dataset)}")
print(f"Validation: {len(val_dataset)}")
print(f"Test:       {len(test_dataset)}")


# ============================================================
# 4. DATA LOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# 5. GCN MODEL
# ============================================================

class GCNModel(nn.Module):

    def __init__(self, input_dim):
        super().__init__()

        self.conv1 = GCNConv(input_dim, 64)
        self.conv2 = GCNConv(64, 64)
        self.conv3 = GCNConv(64, 64)

        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 1)

        self.dropout = nn.Dropout(0.2)

    def forward(self, data):

        x = data.x.float()
        edge_index = data.edge_index
        batch = data.batch

        x = self.conv1(x, edge_index)
        x = torch.relu(x)

        x = self.conv2(x, edge_index)
        x = torch.relu(x)

        x = self.conv3(x, edge_index)
        x = torch.relu(x)

        x = global_mean_pool(x, batch)

        x = self.fc1(x)
        x = torch.relu(x)

        x = self.dropout(x)

        x = self.fc2(x)

        return x


# ============================================================
# 6. GRAPHSAGE MODEL
# ============================================================

class GraphSAGEModel(nn.Module):

    def __init__(self, input_dim):
        super().__init__()

        self.conv1 = SAGEConv(input_dim, 64)
        self.conv2 = SAGEConv(64, 64)
        self.conv3 = SAGEConv(64, 64)

        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 1)

        self.dropout = nn.Dropout(0.2)

    def forward(self, data):

        x = data.x.float()
        edge_index = data.edge_index
        batch = data.batch

        x = self.conv1(x, edge_index)
        x = torch.relu(x)

        x = self.conv2(x, edge_index)
        x = torch.relu(x)

        x = self.conv3(x, edge_index)
        x = torch.relu(x)

        x = global_mean_pool(x, batch)

        x = self.fc1(x)
        x = torch.relu(x)

        x = self.dropout(x)

        x = self.fc2(x)

        return x


# ============================================================
# 7. GAT MODEL
# ============================================================

class GATModel(nn.Module):

    def __init__(self, input_dim):
        super().__init__()

        self.conv1 = GATConv(
            input_dim,
            32,
            heads=2,
            concat=True
        )

        self.conv2 = GATConv(
            64,
            32,
            heads=2,
            concat=True
        )

        self.conv3 = GATConv(
            64,
            32,
            heads=1,
            concat=False
        )

        self.fc1 = nn.Linear(32, 32)
        self.fc2 = nn.Linear(32, 1)

        self.dropout = nn.Dropout(0.2)

    def forward(self, data):

        x = data.x.float()
        edge_index = data.edge_index
        batch = data.batch

        x = self.conv1(x, edge_index)
        x = torch.relu(x)

        x = self.conv2(x, edge_index)
        x = torch.relu(x)

        x = self.conv3(x, edge_index)
        x = torch.relu(x)

        x = global_mean_pool(x, batch)

        x = self.fc1(x)
        x = torch.relu(x)

        x = self.dropout(x)

        x = self.fc2(x)

        return x


# ============================================================
# 8. TRAINING FUNCTION
# ============================================================

def train_model(model, train_loader, val_loader):

    model = model.to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_model_state = None

    for epoch in range(1, EPOCHS + 1):

        # -----------------------------
        # Training
        # -----------------------------

        model.train()

        total_train_loss = 0.0
        total_train_samples = 0

        for batch in train_loader:

            batch = batch.to(DEVICE)

            optimizer.zero_grad()

            output = model(batch)

            target = batch.y.view(-1, 1).float()

            loss = criterion(output, target)

            loss.backward()

            optimizer.step()

            batch_size = batch.num_graphs

            total_train_loss += loss.item() * batch_size
            total_train_samples += batch_size

        train_loss = (
            total_train_loss /
            total_train_samples
        )

        # -----------------------------
        # Validation
        # -----------------------------

        model.eval()

        total_val_loss = 0.0
        total_val_samples = 0

        with torch.no_grad():

            for batch in val_loader:

                batch = batch.to(DEVICE)

                output = model(batch)

                target = batch.y.view(-1, 1).float()

                loss = criterion(output, target)

                batch_size = batch.num_graphs

                total_val_loss += loss.item() * batch_size
                total_val_samples += batch_size

        val_loss = (
            total_val_loss /
            total_val_samples
        )

        # Save best model
        if val_loss < best_val_loss:

            best_val_loss = val_loss

            best_model_state = copy.deepcopy(
                model.state_dict()
            )

        # Print progress
        if epoch == 1 or epoch % 10 == 0:

            print(
                f"Epoch {epoch:03d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f}"
            )

    # Restore best validation model

    model.load_state_dict(best_model_state)

    return model


# ============================================================
# 9. EVALUATION FUNCTION
# ============================================================

def evaluate_model(model, loader):

    model.eval()

    predictions = []
    actuals = []

    with torch.no_grad():

        for batch in loader:

            batch = batch.to(DEVICE)

            output = model(batch)

            target = batch.y.view(-1, 1).float()

            predictions.extend(
                output.cpu().numpy().flatten()
            )

            actuals.extend(
                target.cpu().numpy().flatten()
            )

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    mse = np.mean(
        (actuals - predictions) ** 2
    )

    rmse = np.sqrt(mse)

    mae = np.mean(
        np.abs(actuals - predictions)
    )

    # R²
    ss_res = np.sum(
        (actuals - predictions) ** 2
    )

    ss_tot = np.sum(
        (actuals - np.mean(actuals)) ** 2
    )

    r2 = 1 - (ss_res / ss_tot)

    return mse, rmse, mae, r2


# ============================================================
# 10. CREATE MODELS
# ============================================================

input_dim = dataset.num_node_features

print("\nInput node features:", input_dim)

models = {

    "GCN": GCNModel(input_dim),

    "GraphSAGE": GraphSAGEModel(input_dim),

    "GAT": GATModel(input_dim)
}


# ============================================================
# 11. TRAIN AND EVALUATE EACH MODEL
# ============================================================

results = {}

for model_name, model in models.items():

    print("\n")
    print("=" * 60)
    print(f"MODEL: {model_name}")
    print("=" * 60)

    print(model)

    print("\nStarting training...")

    trained_model = train_model(
        model,
        train_loader,
        val_loader
    )

    print("\nTraining complete.")

    mse, rmse, mae, r2 = evaluate_model(
        trained_model,
        test_loader
    )

    results[model_name] = {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    }

    print("\nTest Results:")
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R²:   {r2:.4f}")

    # Save each model
    filename = (
        model_name.lower()
        .replace(" ", "_")
        + "_model.pth"
    )

    torch.save(
        trained_model.state_dict(),
        filename
    )

    print(f"\nSaved model: {filename}")


# ============================================================
# 12. FINAL COMPARISON
# ============================================================

print("\n")
print("=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)

print(
    f"{'Model':<15}"
    f"{'MSE':>12}"
    f"{'RMSE':>12}"
    f"{'MAE':>12}"
    f"{'R²':>12}"
)

print("-" * 70)

for model_name, metrics in results.items():

    print(
        f"{model_name:<15}"
        f"{metrics['MSE']:>12.4f}"
        f"{metrics['RMSE']:>12.4f}"
        f"{metrics['MAE']:>12.4f}"
        f"{metrics['R2']:>12.4f}"
    )


# ============================================================
# 13. FIND BEST MODEL
# ============================================================

best_model_name = min(
    results,
    key=lambda name: results[name]["RMSE"]
)

print("\n")
print("=" * 70)
print("BEST MODEL")
print("=" * 70)

print(f"Best architecture: {best_model_name}")
print(
    f"Best RMSE: "
    f"{results[best_model_name]['RMSE']:.4f}"
)

print(
    f"Best MAE:  "
    f"{results[best_model_name]['MAE']:.4f}"
)

print(
    f"Best R²:   "
    f"{results[best_model_name]['R2']:.4f}"
)

print("\nComparison complete!")