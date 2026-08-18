from torch_geometric.datasets import MoleculeNet

print("Downloading/loading ESOL dataset...")

dataset = MoleculeNet(
    root="data",
    name="ESOL"
)

print("Dataset loaded successfully!")
print("Number of molecules:", len(dataset))

print("\nFirst molecule:")
print(dataset[0])