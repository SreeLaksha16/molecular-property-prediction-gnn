from torch_geometric.datasets import MoleculeNet

# Load ESOL dataset
dataset = MoleculeNet(
    root="data",
    name="ESOL"
)

print("Number of molecules:", len(dataset))

print("\nFirst 10 molecules:")
print("--------------------")

for i in range(10):
    data = dataset[i]

    print(
        "Molecule", i + 1,
        "| Atoms:", data.x.shape[0],
        "| Edges:", data.edge_index.shape[1],
        "| Target:", data.y.item()
    )

# Collect target values
targets = []

for data in dataset:
    targets.append(data.y.item())

print("\nTarget information:")
print("-------------------")
print("Minimum target:", min(targets))
print("Maximum target:", max(targets))
print("Average target:", sum(targets) / len(targets))