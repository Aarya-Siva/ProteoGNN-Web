"""
Protein Graph Builder - Converts PDB structures to graph representations.
Optimized for web deployment without heavy dependencies.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)


# Standard amino acids
STANDARD_AMINO_ACIDS = {
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY',
    'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER',
    'THR', 'TRP', 'TYR', 'VAL'
}


@dataclass
class ResidueInfo:
    """Container for residue-level information."""
    chain_id: str
    residue_id: int
    residue_name: str
    ca_coords: np.ndarray
    b_factor: float = 100.0


@dataclass
class GraphData:
    """PyTorch-like data container for graph."""
    x: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: Optional[torch.Tensor]
    pos: torch.Tensor
    num_nodes: int
    structure_id: str
    chain_id: str
    residue_ids: List[int]
    residue_names: List[str]


class SimplePDBParser:
    """Simple PDB parser without BioPython dependency."""

    def parse(self, file_path: Union[str, Path], chain_id: str = "A") -> List[ResidueInfo]:
        """
        Parse PDB file and extract residue information.

        Args:
            file_path: Path to PDB file
            chain_id: Chain to extract

        Returns:
            List of ResidueInfo objects
        """
        residues = []
        current_res = None
        ca_coords = None
        b_factor = 100.0

        with open(file_path, 'r') as f:
            for line in f:
                if not line.startswith(('ATOM', 'HETATM')):
                    continue

                # Parse ATOM record
                atom_name = line[12:16].strip()
                res_name = line[17:20].strip()
                chain = line[21].strip()
                res_id = int(line[22:26].strip())
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                b = float(line[60:66].strip()) if len(line) > 66 else 100.0

                # Filter by chain
                if chain != chain_id:
                    continue

                # Only process standard amino acids
                if res_name not in STANDARD_AMINO_ACIDS:
                    continue

                # Track Cα atoms
                if atom_name == 'CA':
                    if current_res is not None and ca_coords is not None:
                        residues.append(ResidueInfo(
                            chain_id=chain_id,
                            residue_id=current_res[0],
                            residue_name=current_res[1],
                            ca_coords=ca_coords,
                            b_factor=b_factor
                        ))

                    current_res = (res_id, res_name)
                    ca_coords = np.array([x, y, z])
                    b_factor = b

        # Add last residue
        if current_res is not None and ca_coords is not None:
            residues.append(ResidueInfo(
                chain_id=chain_id,
                residue_id=current_res[0],
                residue_name=current_res[1],
                ca_coords=ca_coords,
                b_factor=b_factor
            ))

        return residues

    def get_chains(self, file_path: Union[str, Path]) -> List[str]:
        """Get list of chain IDs in the PDB file."""
        chains = set()
        with open(file_path, 'r') as f:
            for line in f:
                if line.startswith(('ATOM', 'HETATM')):
                    chain = line[21].strip()
                    if chain:
                        chains.add(chain)
        return sorted(chains)


class ProteinGraphBuilder:
    """Build graph representations from protein structures."""

    def __init__(
        self,
        distance_cutoff: float = 8.0,
        add_sequential_edges: bool = True,
        use_edge_features: bool = True,
    ):
        """
        Initialize the graph builder.

        Args:
            distance_cutoff: Cα distance threshold for edges (Angstroms)
            add_sequential_edges: Ensure backbone connectivity
            use_edge_features: Whether to compute edge features
        """
        self.distance_cutoff = distance_cutoff
        self.add_sequential_edges = add_sequential_edges
        self.use_edge_features = use_edge_features
        self.parser = SimplePDBParser()
        self.node_extractor = NodeFeatureExtractor()
        self.edge_extractor = EdgeFeatureExtractor()

    def build_edge_index(
        self,
        ca_coords: np.ndarray,
        residue_ids: List[int],
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """Build edge index based on Cα distance threshold."""
        n = len(ca_coords)

        # Compute pairwise distances
        diff = ca_coords[:, np.newaxis, :] - ca_coords[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.sum(diff ** 2, axis=2))

        # Find edges within cutoff
        mask = (dist_matrix < self.distance_cutoff) & (dist_matrix > 0)
        edges_i, edges_j = np.where(mask)

        # Add sequential edges
        if self.add_sequential_edges:
            seq_edges = set()
            for idx in range(n - 1):
                if abs(residue_ids[idx + 1] - residue_ids[idx]) == 1:
                    seq_edges.add((idx, idx + 1))
                    seq_edges.add((idx + 1, idx))

            existing = set(zip(edges_i.tolist(), edges_j.tolist()))
            new_edges = seq_edges - existing

            if new_edges:
                new_i, new_j = zip(*new_edges)
                edges_i = np.concatenate([edges_i, list(new_i)])
                edges_j = np.concatenate([edges_j, list(new_j)])

        edge_index = torch.tensor(
            np.array([edges_i, edges_j]),
            dtype=torch.long
        )
        distances = dist_matrix[edges_i, edges_j]

        return edge_index, distances

    def build_graph(
        self,
        structure_file: Union[str, Path],
        chain_id: str = "A",
    ) -> GraphData:
        """
        Build complete graph from structure file.

        Args:
            structure_file: Path to PDB file
            chain_id: Chain to extract

        Returns:
            GraphData object with node features, edges, etc.
        """
        structure_file = Path(structure_file)
        structure_id = structure_file.stem

        # Check available chains
        chains = self.parser.get_chains(structure_file)
        if chain_id not in chains and chains:
            logger.warning(f"Chain '{chain_id}' not found. Using '{chains[0]}'")
            chain_id = chains[0]

        # Parse structure
        residues = self.parser.parse(structure_file, chain_id)

        if len(residues) == 0:
            raise ValueError(f"No valid residues found in {structure_id} chain {chain_id}")

        # Extract coordinates
        ca_coords = np.array([r.ca_coords for r in residues])
        residue_ids = [r.residue_id for r in residues]
        residue_names = [r.residue_name for r in residues]

        # Build node features
        x = self.node_extractor.extract_features(residues)

        # Build edge index
        edge_index, distances = self.build_edge_index(ca_coords, residue_ids)

        # Build edge features
        edge_attr = None
        if self.use_edge_features and edge_index.shape[1] > 0:
            edge_attr = self.edge_extractor.compute_edge_features(
                edge_index, ca_coords, residue_ids
            )

        return GraphData(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            pos=torch.tensor(ca_coords, dtype=torch.float32),
            num_nodes=len(residues),
            structure_id=structure_id,
            chain_id=chain_id,
            residue_ids=residue_ids,
            residue_names=residue_names,
        )


# ============================================================================
# BIOPHYSICAL PROPERTY SCALES
# ============================================================================

AMINO_ACIDS = [
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY',
    'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER',
    'THR', 'TRP', 'TYR', 'VAL', 'UNK'
]
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

HYDROPHOBICITY = {
    'ALA': 0.700, 'ARG': 0.000, 'ASN': 0.111, 'ASP': 0.111, 'CYS': 0.778,
    'GLN': 0.111, 'GLU': 0.111, 'GLY': 0.456, 'HIS': 0.144, 'ILE': 1.000,
    'LEU': 0.922, 'LYS': 0.067, 'MET': 0.711, 'PHE': 0.811, 'PRO': 0.322,
    'SER': 0.411, 'THR': 0.422, 'TRP': 0.400, 'TYR': 0.356, 'VAL': 0.967,
    'UNK': 0.500
}

CHARGE = {
    'ALA': 0.5, 'ARG': 1.0, 'ASN': 0.5, 'ASP': 0.0, 'CYS': 0.5,
    'GLN': 0.5, 'GLU': 0.0, 'GLY': 0.5, 'HIS': 0.6, 'ILE': 0.5,
    'LEU': 0.5, 'LYS': 1.0, 'MET': 0.5, 'PHE': 0.5, 'PRO': 0.5,
    'SER': 0.5, 'THR': 0.5, 'TRP': 0.5, 'TYR': 0.5, 'VAL': 0.5,
    'UNK': 0.5
}

MOLECULAR_WEIGHT = {
    'ALA': 0.127, 'ARG': 0.811, 'ASN': 0.377, 'ASP': 0.382, 'CYS': 0.315,
    'GLN': 0.468, 'GLU': 0.475, 'GLY': 0.000, 'HIS': 0.589, 'ILE': 0.406,
    'LEU': 0.406, 'LYS': 0.520, 'MET': 0.507, 'PHE': 0.646, 'PRO': 0.270,
    'SER': 0.217, 'THR': 0.304, 'TRP': 1.000, 'TYR': 0.744, 'VAL': 0.310,
    'UNK': 0.400
}

BETA_PROPENSITY = {
    'ALA': 0.417, 'ARG': 0.583, 'ASN': 0.333, 'ASP': 0.250, 'CYS': 0.667,
    'GLN': 0.583, 'GLU': 0.167, 'GLY': 0.333, 'HIS': 0.500, 'ILE': 1.000,
    'LEU': 0.708, 'LYS': 0.417, 'MET': 0.625, 'PHE': 0.750, 'PRO': 0.250,
    'SER': 0.417, 'THR': 0.667, 'TRP': 0.708, 'TYR': 0.833, 'VAL': 0.958,
    'UNK': 0.500
}

ALPHA_PROPENSITY = {
    'ALA': 0.909, 'ARG': 0.636, 'ASN': 0.394, 'ASP': 0.636, 'CYS': 0.364,
    'GLN': 0.727, 'GLU': 0.939, 'GLY': 0.273, 'HIS': 0.636, 'ILE': 0.667,
    'LEU': 0.818, 'LYS': 0.758, 'MET': 0.909, 'PHE': 0.727, 'PRO': 0.273,
    'SER': 0.424, 'THR': 0.455, 'TRP': 0.667, 'TYR': 0.364, 'VAL': 0.636,
    'UNK': 0.500
}

AGGREGATION_PROPENSITY = {
    'ALA': 0.45, 'ARG': 0.10, 'ASN': 0.25, 'ASP': 0.05, 'CYS': 0.55,
    'GLN': 0.30, 'GLU': 0.05, 'GLY': 0.35, 'HIS': 0.35, 'ILE': 0.85,
    'LEU': 0.80, 'LYS': 0.10, 'MET': 0.65, 'PHE': 0.90, 'PRO': 0.15,
    'SER': 0.30, 'THR': 0.35, 'TRP': 0.75, 'TYR': 0.70, 'VAL': 0.80,
    'UNK': 0.50
}

POLARITY = {
    'ALA': 0.0, 'ARG': 1.0, 'ASN': 1.0, 'ASP': 1.0, 'CYS': 0.2,
    'GLN': 1.0, 'GLU': 1.0, 'GLY': 0.0, 'HIS': 0.8, 'ILE': 0.0,
    'LEU': 0.0, 'LYS': 1.0, 'MET': 0.0, 'PHE': 0.0, 'PRO': 0.0,
    'SER': 0.7, 'THR': 0.7, 'TRP': 0.3, 'TYR': 0.5, 'VAL': 0.0,
    'UNK': 0.5
}

AROMATICITY = {
    'ALA': 0.0, 'ARG': 0.0, 'ASN': 0.0, 'ASP': 0.0, 'CYS': 0.0,
    'GLN': 0.0, 'GLU': 0.0, 'GLY': 0.0, 'HIS': 1.0, 'ILE': 0.0,
    'LEU': 0.0, 'LYS': 0.0, 'MET': 0.0, 'PHE': 1.0, 'PRO': 0.0,
    'SER': 0.0, 'THR': 0.0, 'TRP': 1.0, 'TYR': 1.0, 'VAL': 0.0,
    'UNK': 0.0
}

VDW_VOLUME = {
    'ALA': 0.167, 'ARG': 0.596, 'ASN': 0.315, 'ASP': 0.296, 'CYS': 0.278,
    'GLN': 0.407, 'GLU': 0.389, 'GLY': 0.000, 'HIS': 0.463, 'ILE': 0.407,
    'LEU': 0.407, 'LYS': 0.463, 'MET': 0.463, 'PHE': 0.593, 'PRO': 0.278,
    'SER': 0.185, 'THR': 0.296, 'TRP': 1.000, 'TYR': 0.648, 'VAL': 0.315,
    'UNK': 0.400
}

FLEXIBILITY = {
    'ALA': 0.357, 'ARG': 0.529, 'ASN': 0.463, 'ASP': 0.511, 'CYS': 0.346,
    'GLN': 0.493, 'GLU': 0.497, 'GLY': 0.544, 'HIS': 0.323, 'ILE': 0.462,
    'LEU': 0.365, 'LYS': 0.466, 'MET': 0.295, 'PHE': 0.314, 'PRO': 0.509,
    'SER': 0.507, 'THR': 0.444, 'TRP': 0.305, 'TYR': 0.420, 'VAL': 0.386,
    'UNK': 0.450
}

TURN_PROPENSITY = {
    'ALA': 0.318, 'ARG': 0.591, 'ASN': 1.000, 'ASP': 0.909, 'CYS': 0.545,
    'GLN': 0.591, 'GLU': 0.409, 'GLY': 0.818, 'HIS': 0.636, 'ILE': 0.182,
    'LEU': 0.227, 'LYS': 0.591, 'MET': 0.318, 'PHE': 0.273, 'PRO': 0.909,
    'SER': 0.773, 'THR': 0.636, 'TRP': 0.500, 'TYR': 0.545, 'VAL': 0.227,
    'UNK': 0.500
}


class NodeFeatureExtractor:
    """Extract node (residue) features for GNN."""

    def __init__(self, context_window: int = 5):
        self.context_window = context_window

    def encode_amino_acid(self, residue_name: str) -> torch.Tensor:
        """One-hot encode amino acid type."""
        idx = AA_TO_IDX.get(residue_name, AA_TO_IDX['UNK'])
        encoding = torch.zeros(len(AMINO_ACIDS))
        encoding[idx] = 1.0
        return encoding

    def get_physicochemical_features(self, residue_name: str) -> torch.Tensor:
        """Get physicochemical property features."""
        return torch.tensor([
            HYDROPHOBICITY.get(residue_name, 0.5),
            CHARGE.get(residue_name, 0.5),
            POLARITY.get(residue_name, 0.5),
            MOLECULAR_WEIGHT.get(residue_name, 0.4),
            AROMATICITY.get(residue_name, 0.0),
            VDW_VOLUME.get(residue_name, 0.4),
        ], dtype=torch.float32)

    def get_aggregation_features(self, residue_name: str) -> torch.Tensor:
        """Get aggregation-related features."""
        return torch.tensor([
            BETA_PROPENSITY.get(residue_name, 0.5),
            ALPHA_PROPENSITY.get(residue_name, 0.5),
            AGGREGATION_PROPENSITY.get(residue_name, 0.5),
            TURN_PROPENSITY.get(residue_name, 0.5),
            FLEXIBILITY.get(residue_name, 0.45),
        ], dtype=torch.float32)

    def compute_sequence_context(
        self,
        residues: List[ResidueInfo],
        idx: int,
    ) -> torch.Tensor:
        """Compute local sequence context features."""
        n = len(residues)
        start = max(0, idx - self.context_window)
        end = min(n, idx + self.context_window + 1)

        window_residues = [residues[i].residue_name for i in range(start, end)]

        local_hydro = np.mean([HYDROPHOBICITY.get(r, 0.5) for r in window_residues])
        local_beta = np.mean([BETA_PROPENSITY.get(r, 0.5) for r in window_residues])
        local_charge = np.mean([CHARGE.get(r, 0.5) for r in window_residues])
        local_aggreg = np.mean([AGGREGATION_PROPENSITY.get(r, 0.5) for r in window_residues])

        return torch.tensor([
            local_hydro,
            local_beta,
            local_charge,
            local_aggreg,
        ], dtype=torch.float32)

    def extract_features(self, residues: List[ResidueInfo]) -> torch.Tensor:
        """
        Extract all node features for a protein.

        Returns tensor of shape (num_residues, 46):
        - 21: one-hot amino acid
        - 8: secondary structure placeholder (zeros for now)
        - 1: SASA placeholder
        - 1: pLDDT/B-factor
        - 6: physicochemical properties
        - 5: aggregation features
        - 4: local sequence context
        """
        features = []

        for idx, res in enumerate(residues):
            res_features = []

            # One-hot amino acid (21)
            res_features.append(self.encode_amino_acid(res.residue_name))

            # Secondary structure placeholder (8) - would need DSSP
            res_features.append(torch.zeros(8))

            # SASA placeholder (1)
            res_features.append(torch.tensor([0.5]))

            # pLDDT/B-factor normalized (1)
            b_factor = res.b_factor / 100.0
            res_features.append(torch.tensor([b_factor]))

            # Physicochemical properties (6)
            res_features.append(self.get_physicochemical_features(res.residue_name))

            # Aggregation features (5)
            res_features.append(self.get_aggregation_features(res.residue_name))

            # Local sequence context (4)
            res_features.append(self.compute_sequence_context(residues, idx))

            features.append(torch.cat(res_features))

        return torch.stack(features)


class EdgeFeatureExtractor:
    """Extract edge features for GNN."""

    def __init__(self, distance_cutoff: float = 8.0):
        self.distance_cutoff = distance_cutoff
        self.distance_bins = [0.0, 4.0, 6.0, 8.0]

    def compute_edge_features(
        self,
        edge_index: torch.Tensor,
        ca_coords: np.ndarray,
        residue_ids: List[int],
    ) -> torch.Tensor:
        """
        Compute edge features.

        Returns tensor of shape (num_edges, 13):
        - 1: normalized distance
        - 4: distance bins
        - 3: contact type (sequential, local, long-range)
        - 1: normalized sequence separation
        - 4: direction features
        """
        num_edges = edge_index.shape[1]
        features = []

        max_seq_len = max(residue_ids) - min(residue_ids) + 1 if residue_ids else 1

        for e in range(num_edges):
            i, j = edge_index[0, e].item(), edge_index[1, e].item()
            edge_feat = []

            # Distance and direction
            vec_ij = ca_coords[j] - ca_coords[i]
            dist = np.linalg.norm(vec_ij)

            # Normalized distance (1)
            norm_dist = min(dist / self.distance_cutoff, 1.0)
            edge_feat.append(torch.tensor([norm_dist]))

            # Binned distance (4)
            bin_feat = torch.zeros(len(self.distance_bins))
            for b, threshold in enumerate(self.distance_bins):
                if dist <= threshold:
                    bin_feat[b] = 1.0
                    break
            else:
                bin_feat[-1] = 1.0
            edge_feat.append(bin_feat)

            # Contact type (3)
            seq_dist = abs(residue_ids[i] - residue_ids[j])
            contact_type = torch.zeros(3)
            if seq_dist <= 1:
                contact_type[0] = 1.0  # Sequential
            elif seq_dist <= 4:
                contact_type[1] = 1.0  # Local
            else:
                contact_type[2] = 1.0  # Long-range
            edge_feat.append(contact_type)

            # Normalized sequence separation (1)
            norm_seq_dist = min(seq_dist / max_seq_len, 1.0)
            edge_feat.append(torch.tensor([norm_seq_dist]))

            # Direction features (4)
            if dist > 1e-6:
                dir_vec = vec_ij / dist
            else:
                dir_vec = np.zeros(3)
            pos_sign = 1.0 if residue_ids[j] > residue_ids[i] else -1.0
            edge_feat.append(torch.tensor([dir_vec[0], dir_vec[1], dir_vec[2], pos_sign]))

            features.append(torch.cat(edge_feat))

        if features:
            return torch.stack(features)
        else:
            return torch.zeros((0, 13))
