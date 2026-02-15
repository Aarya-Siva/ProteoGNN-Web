"""
ProteoGNN Model - Graph Neural Network for protein misfolding prediction.
Optimized for web deployment with minimal dependencies.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Union


class NodeEncoder(nn.Module):
    """Encode raw node features with MLP projection."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        layers = []
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))

        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class EdgeEncoder(nn.Module):
    """Encode edge features with MLP projection."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        layers = []
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))

        self.encoder = nn.Sequential(*layers)

    def forward(self, edge_attr: torch.Tensor) -> torch.Tensor:
        return self.encoder(edge_attr)


class GNNLayer(nn.Module):
    """
    Graph Convolutional Layer using message passing.
    Simplified version that doesn't require torch_geometric.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.linear_self = nn.Linear(in_channels, out_channels)
        self.linear_neighbor = nn.Linear(in_channels, out_channels)
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass using message passing.

        Args:
            x: (N, in_channels) node features
            edge_index: (2, E) edge indices
            edge_attr: (E, edge_dim) edge features (unused in this simple version)

        Returns:
            (N, out_channels) updated node features
        """
        num_nodes = x.size(0)

        # Self transformation
        out = self.linear_self(x)

        # Aggregate neighbor features
        row, col = edge_index
        neighbor_features = x[col]  # Features of neighbor nodes

        # Sum aggregation with degree normalization
        neighbor_sum = torch.zeros(num_nodes, x.size(1), device=x.device)
        neighbor_sum.index_add_(0, row, neighbor_features)

        # Compute degree for normalization
        degree = torch.zeros(num_nodes, device=x.device)
        degree.index_add_(0, row, torch.ones(row.size(0), device=x.device))
        degree = degree.clamp(min=1)  # Avoid division by zero

        # Normalize and transform
        neighbor_mean = neighbor_sum / degree.unsqueeze(-1)
        out = out + self.linear_neighbor(neighbor_mean)

        # Apply normalization, activation, dropout
        out = self.norm(out)
        out = self.activation(out)
        out = self.dropout(out)

        return out


class ResidualGNNBlock(nn.Module):
    """GNN block with residual connection."""

    def __init__(
        self,
        channels: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.gnn = GNNLayer(channels, channels, dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return x + self.gnn(x, edge_index, edge_attr)


class GNNStack(nn.Module):
    """Stack of GNN layers with residual connections."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        dropout: float = 0.1,
        residual: bool = True,
    ):
        super().__init__()

        self.input_proj = nn.Linear(in_channels, hidden_channels)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            if residual:
                self.layers.append(ResidualGNNBlock(hidden_channels, dropout))
            else:
                self.layers.append(GNNLayer(hidden_channels, hidden_channels, dropout))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.input_proj(x)

        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)

        return x


class ProteoGNN(nn.Module):
    """
    Graph Neural Network for predicting residue-level misfolding propensity.

    Architecture:
    1. Node encoder: Projects input features to hidden dimension
    2. Edge encoder (optional): Encodes edge features
    3. GNN stack: Multiple message passing layers
    4. Prediction head: Per-node binary classification
    """

    def __init__(
        self,
        node_input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        layer_type: str = "graphconv",
        heads: int = 4,
        dropout: float = 0.1,
        edge_dim: Optional[int] = None,
        use_edge_features: bool = True,
        residual: bool = True,
        jk_mode: Optional[str] = None,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.use_edge_features = use_edge_features and edge_dim is not None

        # Node encoder
        self.node_encoder = NodeEncoder(
            input_dim=node_input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=2,
            dropout=dropout,
        )

        # Edge encoder (optional)
        self.edge_encoder = None
        if self.use_edge_features and edge_dim:
            self.edge_encoder = EdgeEncoder(
                input_dim=edge_dim,
                hidden_dim=hidden_dim // 2,
                output_dim=hidden_dim,
                num_layers=2,
                dropout=dropout,
            )

        # GNN stack
        self.gnn = GNNStack(
            in_channels=hidden_dim,
            hidden_channels=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            residual=residual,
        )

        # Prediction head for per-node classification
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, data) -> torch.Tensor:
        """
        Forward pass.

        Args:
            data: Object with x, edge_index, and optionally edge_attr

        Returns:
            (N,) per-node logits
        """
        x = data.x
        edge_index = data.edge_index

        # Encode nodes
        x = self.node_encoder(x)

        # Encode edges (if available)
        edge_attr = None
        if self.use_edge_features and hasattr(data, 'edge_attr') and data.edge_attr is not None:
            edge_attr = self.edge_encoder(data.edge_attr)

        # GNN forward
        x = self.gnn(x, edge_index, edge_attr)

        # Per-node prediction
        logits = self.head(x).squeeze(-1)

        return logits

    def predict_proba(self, data) -> torch.Tensor:
        """Get probability predictions."""
        logits = self.forward(data)
        return torch.sigmoid(logits)

    def predict(self, data, threshold: float = 0.5) -> torch.Tensor:
        """Get binary predictions."""
        probs = self.predict_proba(data)
        return (probs >= threshold).long()


def create_model(config: Dict) -> ProteoGNN:
    """Factory function to create ProteoGNN from configuration."""
    return ProteoGNN(
        node_input_dim=config['node_input_dim'],
        hidden_dim=config.get('hidden_dim', 128),
        num_layers=config.get('num_layers', 4),
        layer_type=config.get('layer_type', 'graphconv'),
        heads=config.get('heads', 4),
        dropout=config.get('dropout', 0.1),
        edge_dim=config.get('edge_dim'),
        use_edge_features=config.get('use_edge_features', True),
        residual=config.get('residual', True),
        jk_mode=config.get('jk_mode'),
    )


def load_model(
    checkpoint_path: str,
    config: Optional[Dict] = None,
    device: str = "cpu",
) -> ProteoGNN:
    """Load a trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if config is None:
        config = checkpoint.get('config', {}).get('model', {})

    model = create_model(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model
