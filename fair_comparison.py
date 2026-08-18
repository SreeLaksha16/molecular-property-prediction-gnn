import copy
import random
import numpy as np
import torch
import torch.nn as nn

from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, global_mean_pool


# ============================================================
# SETTINGS
# ============================================================

SEED = 42
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("FAIR GNN ARCHITECTURE COMPARISON")
print("=" * 70)
print(f"Device: {DEVICE}")


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading ESOL dataset...")

dataset = MoleculeNet(
    root="data",
    name="ESOL"
)

print(f"Total molecules: {len(dataset)}")


# ============================================================
# CREATE ONE FIXED SPLIT
# ============================================================

generator = torch.Generator().manual_seed(SEED)

indices = torch.randperm(
    len(dataset),
    generator=generator
).tolist()

train_size = int(0.80 * len(dataset))
val_size = int(0.10 * len(dataset))

train_indices = indices[:train_size]

val_indices = indices[
    train_size:train_size + val_size
]

test_indices = indices[
    train_size + val_size:
]

train_dataset = [dataset[i] for i in train_indices]
val_dataset = [dataset[i] for i in val_indices]
test_dataset = [dataset[i] for i in test_indices]

print("\nFIXED DATASET SPLIT")
print("-" * 40)
print(f"Training:   {len(train_dataset)}")
print(f"Validation: {len(val_dataset)}")
print(f"Test:       {len(test_dataset)}")


# ============================================================
# DATA LOADERS
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
# BASE GCN
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

        return self.fc2(x)


# ============================================================
# GRAPHSAGE
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

        return self.fc2(x)


# ============================================================
# GAT
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

        return self.fc2(x)


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_model(model):

    model = model.to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, EPOCHS + 1):

        model.train()

        train_loss_sum = 0.0
        train_count = 0

        for batch in train_loader:

            batch = batch.to(DEVICE)

            optimizer.zero_grad()

            prediction = model(batch)

            target = batch.y.view(-1, 1).float()

            loss = criterion(
                prediction,
                target
            )

            loss.backward()

            optimizer.step()

            n = batch.num_graphs

            train_loss_sum += loss.item() * n
            train_count += n

        train_loss = (
            train_loss_sum /
            train_count
        )

        # ----------------------------
        # Validation
        # ----------------------------

        model.eval()

        val_loss_sum = 0.0
        val_count = 0

        with torch.no_grad():

            for batch in val_loader:

                batch = batch.to(DEVICE)

                prediction = model(batch)

                target = batch.y.view(-1, 1).float()

                loss = criterion(
                    prediction,
                    target
                )

                n = batch.num_graphs

                val_loss_sum += loss.item() * n
                val_count += n

        val_loss = (
            val_loss_sum /
            val_count
        )

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            best_state = copy.deepcopy(
                model.state_dict()
            )

        if (
            epoch == 1
            or epoch % 10 == 0
        ):

            print(
                f"Epoch {epoch:03d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f}"
            )

    model.load_state_dict(best_state)

    return model


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(model):

    model.eval()

    predictions = []
    actuals = []

    with torch.no_grad():

        for batch in test_loader:

            batch = batch.to(DEVICE)

            prediction = model(batch)

            target = batch.y.view(-1, 1).float()

            predictions.extend(
                prediction.cpu().numpy().flatten()
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

    ss_res = np.sum(
        (actuals - predictions) ** 2
    )

    ss_tot = np.sum(
        (actuals - np.mean(actuals)) ** 2
    )

    r2 = 1 - (
        ss_res / ss_tot
    )

    return mse, rmse, mae, r2


# ============================================================
# MODEL COMPARISON
# ============================================================

input_dim = dataset.num_node_features

print("\nInput node features:", input_dim)

model_classes = {
    "GCN": GCNModel,
    "GraphSAGE": GraphSAGEModel,
    "GAT": GATModel
}

results = {}


# ============================================================
# TRAIN EACH MODEL
# ============================================================

for name, model_class in model_classes.items():

    print("\n")
    print("=" * 70)
    print(f"TRAINING {name}")
    print("=" * 70)

    # Reset seed before each model so initialization
    # is reproducible.
    torch.manual_seed(SEED)

    model = model_class(input_dim)

    print(model)

    print("\nStarting training...")

    model = train_model(model)

    print("\nTraining complete.")

    mse, rmse, mae, r2 = evaluate_model(model)

    results[name] = {
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

    filename = (
        name.lower()
        + "_fair_model.pth"
    )

    torch.save(
        model.state_dict(),
        filename
    )

    print(f"Saved model: {filename}")


# ============================================================
# FINAL COMPARISON TABLE
# ============================================================

print("\n")
print("=" * 75)
print("FAIR MODEL COMPARISON")
print("=" * 75)

print(
    f"{'Model':<15}"
    f"{'MSE':>12}"
    f"{'RMSE':>12}"
    f"{'MAE':>12}"
    f"{'R²':>12}"
)

print("-" * 75)

for name, metrics in results.items():

    print(
        f"{name:<15}"
        f"{metrics['MSE']:>12.4f}"
        f"{metrics['RMSE']:>12.4f}"
        f"{metrics['MAE']:>12.4f}"
        f"{metrics['R2']:>12.4f}"
    )


# ============================================================
# BEST MODEL
# ============================================================

best_model = min(
    results,
    key=lambda name: results[name]["RMSE"]
)

print("\n")
print("=" * 75)
print("BEST MODEL")
print("=" * 75)

print(f"Architecture: {best_model}")
print(
    f"RMSE: {results[best_model]['RMSE']:.4f}"
)
print(
    f"MAE:  {results[best_model]['MAE']:.4f}"
)
print(
    f"R²:   {results[best_model]['R2']:.4f}"
)

print("\nFair comparison complete!")