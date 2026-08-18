import torch
from torch_geometric.datasets import MoleculeNet
from torch.utils.data import random_split


# --------------------------------
# Load ESOL dataset
# --------------------------------

dataset = MoleculeNet(
    root="data",
    name="ESOL"
)

print("Total molecules:", len(dataset))


# --------------------------------
# Calculate split sizes
# --------------------------------

total = len(dataset)

train_size = int(0.8 * total)
validation_size = int(0.1 * total)
test_size = total - train_size - validation_size


# --------------------------------
# Split dataset
# --------------------------------

generator = torch.Generator().manual_seed(42)

train_dataset, validation_dataset, test_dataset = random_split(
    dataset,
    [train_size, validation_size, test_size],
    generator=generator
)


# --------------------------------
# Print results
# --------------------------------

print("\nDataset split:")
print("----------------")
print("Training molecules:", len(train_dataset))
print("Validation molecules:", len(validation_dataset))
print("Test molecules:", len(test_dataset))

print("\nTotal:", 
      len(train_dataset) +
      len(validation_dataset) +
      len(test_dataset))