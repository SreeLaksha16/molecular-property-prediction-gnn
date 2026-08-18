# Molecular Property Prediction Using Graph Neural Networks

## Overview

This project predicts the aqueous solubility (ESOL) of molecules using Graph Neural Networks (GNNs). Molecules are represented as graphs where atoms are nodes and chemical bonds are edges. Different GNN architectures are trained and compared to determine the best model for molecular property prediction.

The project uses:

- RDKit for molecular processing
- PyTorch Geometric for Graph Neural Networks
- ESOL dataset from MoleculeNet
- Streamlit for the web application

---

## Problem Statement

Traditional machine learning models require handcrafted molecular descriptors. Graph Neural Networks can directly learn from molecular structures by treating molecules as graphs.

The goal of this project is to predict molecular solubility from molecular structure and compare different GNN architectures.

---

## Features

- Convert SMILES strings into molecular graphs
- Extract atom and bond features
- Train Graph Neural Networks on the ESOL dataset
- Compare multiple GNN architectures
- Evaluate performance using standard regression metrics
- Predict properties of custom molecules
- Interactive Streamlit web application

---

## Dataset

Dataset: ESOL (Delaney)

Source: MoleculeNet

Number of molecules: 1128

Target property:
- Water solubility (logS)

Dataset split:

| Split | Molecules |
|---------|---------|
| Training | 902 |
| Validation | 112 |
| Test | 114 |

---

## Molecular Graph Representation

### Node Features

Each atom is represented using:

- Atomic number
- Degree
- Formal charge
- Number of hydrogens
- Additional atom features from MoleculeNet

### Edge Features

Each bond is represented using:

- Bond type
- Bond characteristics

---

## Model Architectures

### 1. Graph Convolutional Network (GCN)

Three graph convolution layers followed by fully connected layers.

### 2. GraphSAGE

Neighborhood aggregation-based graph neural network.

### 3. Graph Attention Network (GAT)

Uses attention mechanisms to learn the importance of neighboring atoms.

---

## Experimental Results

### Model Comparison

| Model | MSE | RMSE | MAE | R² |
|---------|---------|---------|---------|---------|
| GCN | 1.4444 | 1.2018 | 0.9434 | 0.6559 |
| GraphSAGE | 1.4369 | 1.1987 | 0.9584 | 0.6577 |
| GAT | **1.1882** | **1.0900** | **0.8420** | **0.7169** |

### Best Model

**Graph Attention Network (GAT)**

Performance:

- RMSE: 1.0900
- MAE: 0.8420
- R²: 0.7169

---

## Example Prediction

Input SMILES:

```text
CCO
```

Output:

```text
Predicted ESOL value: -0.1960
```

---

## Project Structure

```text
GNN_Molecular_Project/
│
├── app/
│   └── app.py
│
├── data/
│
├── models/
│
├── compare_gnn_models.py
├── train_gnn.py
├── evaluate_model.py
├── predict_molecule.py
├── load_dataset.py
├── inspect_dataset.py
├── split_dataset.py
│
├── molecular_gnn_model.pth
├── gat_model.pth
│
├── actual_vs_predicted.png
├── prediction_error_distribution.png
│
├── README.md
│
└── requirements.txt
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/GNN_Molecular_Project.git

cd GNN_Molecular_Project
```

### Create Virtual Environment

```bash
python -m venv venv
```

Activate environment:

Windows:

```bash
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Project

### Train Model

```bash
python train_gnn.py
```

### Evaluate Model

```bash
python evaluate_model.py
```

### Predict a Molecule

```bash
python predict_molecule.py
```

### Compare GNN Architectures

```bash
python compare_gnn_models.py
```

### Launch Web Application

```bash
streamlit run app/app.py
```

---

## Web Application

### Home Page

![Home Page](screenshots/home_page.png)

### Prediction Example

![Prediction](screenshots/prediction_result.png)

### Model Performance

![Performance](screenshots/model_performance.png)

---

## Visualizations

### Actual vs Predicted

![Actual vs Predicted](screenshots/actual_vs_predicted.png)

### Error Distribution

![Error Distribution](screenshots/prediction_error_distribution.png)

---

## Technologies Used

- Python
- PyTorch
- PyTorch Geometric
- RDKit
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Streamlit

---

## Applications

This project can be applied in:

- Drug Discovery
- Computational Chemistry
- Material Science
- Molecular Property Prediction
- Pharmaceutical Research

---

## Future Improvements

- Hyperparameter tuning
- GNNExplainability (GNNExplainer)
- Additional MoleculeNet datasets
- Molecular embedding visualization
- Online deployment
- Multi-task molecular property prediction

---

## Author

Your Name

Graph Neural Networks for Molecular Property Prediction

Built using PyTorch Geometric and RDKit.
