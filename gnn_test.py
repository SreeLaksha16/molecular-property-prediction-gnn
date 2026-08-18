import torch
from torch_geometric.nn import GCNConv

print("PyTorch version:", torch.__version__)

layer = GCNConv(4, 8)

print("GCN layer created successfully!")
print(layer)