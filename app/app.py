import os

import streamlit as st
import torch
import torch.nn as nn

from torch_geometric.data import Data
from torch_geometric.nn import GATConv, global_mean_pool

from rdkit import Chem
from rdkit.Chem import Draw


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="GAT Molecular ESOL Predictor",
    page_icon="🧪",
    layout="wide"
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device("cpu")


# ============================================================
# GAT MODEL
#
# This architecture MUST match the architecture used when
# gat_fair_model.pth was trained.
#
# GAT:
#   GATConv(9, 32, heads=2)  -> 64 features
#   GATConv(64, 32, heads=2) -> 64 features
#   GATConv(64, 32, heads=1) -> 32 features
#   Linear(32, 32)
#   Linear(32, 1)
# ============================================================

class GATModel(nn.Module):

    def __init__(self, input_dim=9):

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

        self.fc1 = nn.Linear(
            32,
            32
        )

        self.fc2 = nn.Linear(
            32,
            1
        )

        self.dropout = nn.Dropout(0.2)

    def forward(self, data):

        x = data.x.float()

        edge_index = data.edge_index

        batch = data.batch

        # ----------------------------------------------------
        # GAT Layer 1
        # ----------------------------------------------------

        x = self.conv1(
            x,
            edge_index
        )

        x = torch.relu(x)

        # ----------------------------------------------------
        # GAT Layer 2
        # ----------------------------------------------------

        x = self.conv2(
            x,
            edge_index
        )

        x = torch.relu(x)

        # ----------------------------------------------------
        # GAT Layer 3
        # ----------------------------------------------------

        x = self.conv3(
            x,
            edge_index
        )

        x = torch.relu(x)

        # ----------------------------------------------------
        # Global graph pooling
        # ----------------------------------------------------

        x = global_mean_pool(
            x,
            batch
        )

        # ----------------------------------------------------
        # Fully connected layers
        # ----------------------------------------------------

        x = self.fc1(x)

        x = torch.relu(x)

        x = self.dropout(x)

        x = self.fc2(x)

        return x


# ============================================================
# LOAD TRAINED GAT MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = GATModel(
        input_dim=9
    )

    # app folder:
    # C:\...\GNN_Molecular_Project\app
    #
    # project root:
    # C:\...\GNN_Molecular_Project

    project_root = os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )

    model_path = os.path.join(
        project_root,
        "gat_fair_model.pth"
    )

    if not os.path.exists(model_path):

        raise FileNotFoundError(
            f"Model file not found:\n{model_path}\n\n"
            "Make sure gat_fair_model.pth is in the "
            "main GNN_Molecular_Project folder."
        )

    checkpoint = torch.load(
        model_path,
        map_location=DEVICE
    )

    # Support both:
    # 1. state_dict
    # 2. checkpoint containing model_state_dict

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

    else:

        model.load_state_dict(
            checkpoint
        )

    model.to(DEVICE)

    model.eval()

    return model


# ============================================================
# SMILES -> GRAPH
# ============================================================

def smiles_to_graph(smiles):

    mol = Chem.MolFromSmiles(
        smiles
    )

    if mol is None:

        return None

    # --------------------------------------------------------
    # Node features
    #
    # 1. Atomic number
    # 2. Degree
    # 3. Formal charge
    # 4. Number of hydrogens
    # 5. Aromatic
    # 6. Hybridization
    # 7. Ring membership
    # 8. Chiral tag
    # 9. Radical electrons
    # --------------------------------------------------------

    node_features = []

    for atom in mol.GetAtoms():

        atomic_number = atom.GetAtomicNum()

        degree = atom.GetDegree()

        formal_charge = atom.GetFormalCharge()

        hydrogens = atom.GetTotalNumHs()

        aromatic = int(
            atom.GetIsAromatic()
        )

        hybridization = int(
            atom.GetHybridization()
        )

        in_ring = int(
            atom.IsInRing()
        )

        chiral_tag = int(
            atom.GetChiralTag()
        )

        radical_electrons = (
            atom.GetNumRadicalElectrons()
        )

        features = [
            atomic_number,
            degree,
            formal_charge,
            hydrogens,
            aromatic,
            hybridization,
            in_ring,
            chiral_tag,
            radical_electrons
        ]

        node_features.append(
            features
        )

    x = torch.tensor(
        node_features,
        dtype=torch.float
    )

    # --------------------------------------------------------
    # Edges
    # --------------------------------------------------------

    edges = []

    edge_features = []

    for bond in mol.GetBonds():

        start = bond.GetBeginAtomIdx()

        end = bond.GetEndAtomIdx()

        bond_type = bond.GetBondType()

        if bond_type == Chem.BondType.SINGLE:

            bond_value = 1.0

        elif bond_type == Chem.BondType.DOUBLE:

            bond_value = 2.0

        elif bond_type == Chem.BondType.TRIPLE:

            bond_value = 3.0

        elif bond_type == Chem.BondType.AROMATIC:

            bond_value = 1.5

        else:

            bond_value = 1.0

        # Add forward edge

        edges.append(
            [start, end]
        )

        edge_features.append(
            [bond_value]
        )

        # Add reverse edge

        edges.append(
            [end, start]
        )

        edge_features.append(
            [bond_value]
        )

    # --------------------------------------------------------
    # Handle molecules with bonds
    # --------------------------------------------------------

    if len(edges) > 0:

        edge_index = torch.tensor(
            edges,
            dtype=torch.long
        ).t().contiguous()

        edge_attr = torch.tensor(
            edge_features,
            dtype=torch.float
        )

    # --------------------------------------------------------
    # Handle molecule with no bonds
    # --------------------------------------------------------

    else:

        edge_index = torch.empty(
            (2, 0),
            dtype=torch.long
        )

        edge_attr = torch.empty(
            (0, 1),
            dtype=torch.float
        )

    return x, edge_index, edge_attr, mol


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_esol(
    model,
    smiles
):

    result = smiles_to_graph(
        smiles
    )

    if result is None:

        return None

    x, edge_index, edge_attr, mol = result

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr
    )

    # --------------------------------------------------------
    # One molecule = one graph
    # --------------------------------------------------------

    data.batch = torch.zeros(
        x.shape[0],
        dtype=torch.long
    )

    data = data.to(DEVICE)

    with torch.no_grad():

        prediction = model(
            data
        )

    predicted_value = (
        prediction.item()
    )

    return (
        predicted_value,
        mol,
        data
    )


# ============================================================
# HEADER
# ============================================================

st.title(
    "🧪 GAT Molecular ESOL Predictor"
)

st.markdown(
    """
    ### Graph Attention Network for Molecular Solubility Prediction

    Enter a molecule as a **SMILES string** and the trained
    Graph Attention Network will predict its **ESOL molecular
    solubility value**.

    The final GAT model was selected after comparing
    **GCN, GraphSAGE, and GAT** architectures on the same
    fixed ESOL test split.
    """
)


# ============================================================
# MODEL INFORMATION
# ============================================================

with st.expander(
    "ℹ️ About the final model"
):

    st.write(
        """
        **Dataset:** ESOL

        **Task:** Molecular solubility prediction

        **Final architecture:** Graph Attention Network (GAT)

        **Input features:** 9 molecular node features

        **Test molecules:** 114

        The GAT model was selected because it achieved the
        best performance among the three evaluated GNN
        architectures.
        """
    )


# ============================================================
# INPUT SECTION
# ============================================================

st.subheader(
    "🔬 Enter Molecular Information"
)

smiles = st.text_input(
    "SMILES string",
    value="CCO",
    help=(
        "Example: CCO represents ethanol."
    )
)

predict_button = st.button(
    "🔬 Predict ESOL",
    type="primary"
)


# ============================================================
# PREDICTION SECTION
# ============================================================

if predict_button:

    if not smiles.strip():

        st.error(
            "Please enter a SMILES string."
        )

    else:

        smiles = smiles.strip()

        mol_check = Chem.MolFromSmiles(
            smiles
        )

        if mol_check is None:

            st.error(
                "Invalid SMILES string. "
                "Please enter a valid molecular SMILES."
            )

        else:

            try:

                # ------------------------------------------------
                # Load GAT model
                # ------------------------------------------------

                model = load_model()

                # ------------------------------------------------
                # Prediction
                # ------------------------------------------------

                result = predict_esol(
                    model,
                    smiles
                )

                if result is None:

                    st.error(
                        "Could not convert the SMILES "
                        "string into a molecular graph."
                    )

                else:

                    (
                        predicted_value,
                        mol,
                        graph_data
                    ) = result

                    # ============================================
                    # MOLECULE PREDICTION
                    # ============================================

                    st.markdown("---")

                    st.subheader(
                        "🎯 Prediction Result"
                    )

                    col1, col2, col3 = st.columns(
                        3
                    )

                    with col1:

                        st.metric(
                            "Predicted ESOL",
                            f"{predicted_value:.4f}"
                        )

                    with col2:

                        st.metric(
                            "Number of Atoms",
                            mol.GetNumAtoms()
                        )

                    with col3:

                        st.metric(
                            "Number of Bonds",
                            mol.GetNumBonds()
                        )

                    # ============================================
                    # MOLECULAR STRUCTURE
                    # ============================================

                    st.subheader(
                        "🧬 Molecular Structure"
                    )

                    image = Draw.MolToImage(
                        mol,
                        size=(500, 400)
                    )

                    st.image(
                        image,
                        caption=(
                            f"Molecular structure: "
                            f"{smiles}"
                        )
                    )

                    # ============================================
                    # ESOL INTERPRETATION
                    # ============================================

                    st.subheader(
                        "📋 ESOL Interpretation"
                    )

                    if predicted_value >= 0:

                        st.success(
                            f"""
                            **Predicted ESOL value:
                            {predicted_value:.4f}**

                            The molecule is predicted to have
                            relatively high aqueous solubility.
                            """
                        )

                    elif predicted_value >= -2:

                        st.info(
                            f"""
                            **Predicted ESOL value:
                            {predicted_value:.4f}**

                            The molecule is predicted to have
                            moderate aqueous solubility.
                            """
                        )

                    elif predicted_value >= -4:

                        st.warning(
                            f"""
                            **Predicted ESOL value:
                            {predicted_value:.4f}**

                            The molecule is predicted to have
                            relatively low aqueous solubility.
                            """
                        )

                    else:

                        st.error(
                            f"""
                            **Predicted ESOL value:
                            {predicted_value:.4f}**

                            The molecule is predicted to have
                            very low aqueous solubility.
                            """
                        )

                    # ============================================
                    # GRAPH INFORMATION
                    # ============================================

                    st.subheader(
                        "🕸️ Molecular Graph Information"
                    )

                    graph_col1, graph_col2, graph_col3 = (
                        st.columns(3)
                    )

                    with graph_col1:

                        st.metric(
                            "Nodes",
                            graph_data.x.shape[0]
                        )

                    with graph_col2:

                        st.metric(
                            "Graph Edges",
                            graph_data.edge_index.shape[1]
                        )

                    with graph_col3:

                        st.metric(
                            "Chemical Bonds",
                            mol.GetNumBonds()
                        )

                    # ============================================
                    # GRAPH DATA
                    # ============================================

                    with st.expander(
                        "🔍 View Graph Data"
                    ):

                        st.write(
                            "Node feature shape:"
                        )

                        st.write(
                            graph_data.x.shape
                        )

                        st.write(
                            "Node features:"
                        )

                        st.write(
                            graph_data.x
                        )

                        st.write(
                            "Edge index:"
                        )

                        st.write(
                            graph_data.edge_index
                        )

                        st.write(
                            "Bond features:"
                        )

                        st.write(
                            graph_data.edge_attr
                        )

            except Exception as e:

                st.error(
                    "Prediction failed."
                )

                st.exception(e)


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.markdown("---")

st.subheader(
    "📊 Final Model Performance"
)

st.write(
    """
    The following results are from the **fair architecture
    comparison**, where GCN, GraphSAGE, and GAT were evaluated
    using the same fixed train/validation/test split.
    """
)


metric1, metric2, metric3, metric4 = st.columns(
    4
)

with metric1:

    st.metric(
        "MSE",
        "1.0581"
    )

with metric2:

    st.metric(
        "RMSE",
        "1.0286"
    )

with metric3:

    st.metric(
        "MAE",
        "0.7630"
    )

with metric4:

    st.metric(
        "R²",
        "0.7479"
    )


st.write(
    "**Test molecules:** 114"
)

st.success(
    """
    🏆 **Best architecture: GAT**

    The GAT model achieved the best test performance among
    the GCN, GraphSAGE, and GAT architectures evaluated in
    the controlled comparison.
    """
)


# ============================================================
# ARCHITECTURE COMPARISON
# ============================================================

st.subheader(
    "⚖️ GNN Architecture Comparison"
)

comparison_data = {
    "Architecture": [
        "GCN",
        "GraphSAGE",
        "GAT"
    ],
    "MSE": [
        1.7113,
        1.2559,
        1.0581
    ],
    "RMSE": [
        1.3082,
        1.1206,
        1.0286
    ],
    "MAE": [
        1.0186,
        0.9126,
        0.7630
    ],
    "R²": [
        0.5923,
        0.7008,
        0.7479
    ]
}

st.dataframe(
    comparison_data,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# PERFORMANCE VISUALIZATIONS
# ============================================================

st.subheader(
    "📈 Model Evaluation Plots"
)

project_root = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


# ------------------------------------------------------------
# Possible plot locations
# ------------------------------------------------------------

actual_predicted_paths = [
    os.path.join(
        project_root,
        "actual_vs_predicted.png"
    ),
    os.path.join(
        project_root,
        "figures",
        "actual_vs_predicted.png"
    )
]

error_distribution_paths = [
    os.path.join(
        project_root,
        "prediction_error_distribution.png"
    ),
    os.path.join(
        project_root,
        "figures",
        "prediction_error_distribution.png"
    )
]


def find_existing_file(paths):

    for path in paths:

        if os.path.exists(path):

            return path

    return None


actual_predicted_path = find_existing_file(
    actual_predicted_paths
)

error_distribution_path = find_existing_file(
    error_distribution_paths
)


plot_col1, plot_col2 = st.columns(
    2
)


with plot_col1:

    if actual_predicted_path:

        st.image(
            actual_predicted_path,
            caption=(
                "Actual vs Predicted ESOL"
            ),
            use_container_width=True
        )

    else:

        st.info(
            "Actual vs predicted plot not found."
        )


with plot_col2:

    if error_distribution_path:

        st.image(
            error_distribution_path,
            caption=(
                "Prediction Error Distribution"
            ),
            use_container_width=True
        )

    else:

        st.info(
            "Prediction error distribution "
            "plot not found."
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    """
    Molecular ESOL Prediction using Graph Neural Networks |
    Final Architecture: GAT |
    ESOL Test R²: 0.7479
    """
)