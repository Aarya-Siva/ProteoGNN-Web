"""
Feature extraction module - re-exported for compatibility.
All feature extraction is now in graph_builder.py
"""

from graph_builder import (
    NodeFeatureExtractor,
    EdgeFeatureExtractor,
    AMINO_ACIDS,
    AA_TO_IDX,
    HYDROPHOBICITY,
    CHARGE,
    POLARITY,
    MOLECULAR_WEIGHT,
    AROMATICITY,
    VDW_VOLUME,
    BETA_PROPENSITY,
    ALPHA_PROPENSITY,
    AGGREGATION_PROPENSITY,
    TURN_PROPENSITY,
    FLEXIBILITY,
)

__all__ = [
    'NodeFeatureExtractor',
    'EdgeFeatureExtractor',
    'AMINO_ACIDS',
    'AA_TO_IDX',
    'HYDROPHOBICITY',
    'CHARGE',
    'POLARITY',
    'MOLECULAR_WEIGHT',
    'AROMATICITY',
    'VDW_VOLUME',
    'BETA_PROPENSITY',
    'ALPHA_PROPENSITY',
    'AGGREGATION_PROPENSITY',
    'TURN_PROPENSITY',
    'FLEXIBILITY',
]
