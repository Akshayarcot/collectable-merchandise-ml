"""
preprocessing.py
Feature engineering and transformation pipeline:
- Log transformation of search volume
- Min-max scaling of continuous attributes to [0, 1]
- One-Hot Encoding for baseline models
- Ordinal Encoding and marginal distribution calculation for Custom Kernel SVM
- Stratified splitting for imbalanced multi-class data
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any


class CollectablesPreprocessor:
    """
    Transforms raw collectible merchandise attributes into feature matrices
    suitable for Linear SVM, Decision Trees, Bagged Ensembles, and Custom Kernel SVM.
    """
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.min_log_vol = None
        self.max_log_vol = None
        
        # Categorical feature vocabulary mappings
        self.categorical_cols = ["category", "material", "special_version"]
        self.cat_to_idx: Dict[str, Dict[str, int]] = {}
        self.idx_to_cat: Dict[str, Dict[int, str]] = {}
        self.marginal_probs: Dict[str, np.ndarray] = {}
        
        # One-hot column names
        self.one_hot_columns = []
        
    def fit(self, df: pd.DataFrame):
        """
        Fits transformation parameters (log bounds, ordinal maps, marginal probabilities).
        """
        # 1. Continuous feature scaling: Log transform + Min-Max
        log_vols = np.log1p(df["search_volume"].values.astype(float))
        self.min_log_vol = float(np.min(log_vols))
        self.max_log_vol = float(np.max(log_vols))
        if self.max_log_vol == self.min_log_vol:
            self.max_log_vol += 1.0
            
        # 2. Categorical ordinal maps and marginal probabilities
        for col in self.categorical_cols:
            unique_vals = sorted(df[col].unique().tolist())
            self.cat_to_idx[col] = {val: idx for idx, val in enumerate(unique_vals)}
            self.idx_to_cat[col] = {idx: val for idx, val in enumerate(unique_vals)}
            
            # Marginal empirical probabilities P(Z = z)
            val_counts = df[col].value_counts(normalize=True)
            probs = np.array([val_counts.get(val, 1e-4) for val in unique_vals])
            # Normalize to sum strictly to 1.0
            probs = probs / np.sum(probs)
            self.marginal_probs[col] = probs
            
        # 3. Fit One-hot column template
        ohe_df = pd.get_dummies(df[self.categorical_cols], drop_first=False)
        self.one_hot_columns = list(ohe_df.columns)
        
        return self

    def transform_continuous(self, search_volumes: np.ndarray) -> np.ndarray:
        """Log transform + Min-Max scale to [0, 1]."""
        log_vols = np.log1p(np.maximum(search_volumes, 0).astype(float))
        norm_vols = (log_vols - self.min_log_vol) / (self.max_log_vol - self.min_log_vol)
        return np.clip(norm_vols, 0.0, 1.0).reshape(-1, 1)

    def transform_one_hot(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transforms dataframe into feature matrix with continuous normalized feature
        plus one-hot encoded categorical attributes.
        Used for Linear SVM, Vanilla Decision Tree, and Bagged Decision Tree.
        """
        norm_vol = self.transform_continuous(df["search_volume"].values)
        
        ohe_df = pd.get_dummies(df[self.categorical_cols], drop_first=False)
        # Align with fitted columns
        for col in self.one_hot_columns:
            if col not in ohe_df.columns:
                ohe_df[col] = 0
        ohe_mat = ohe_df[self.one_hot_columns].values.astype(float)
        
        return np.hstack([norm_vol, ohe_mat])

    def transform_ordinal(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transforms features for Custom Kernel SVM:
        Returns:
            continuous_feats: normalized search volume array (N, 1)
            categorical_indices: integer indices for (category, material, special_version) (N, 3)
        """
        norm_vol = self.transform_continuous(df["search_volume"].values)
        
        ord_matrix = np.zeros((len(df), len(self.categorical_cols)), dtype=int)
        for c_idx, col in enumerate(self.categorical_cols):
            mapping = self.cat_to_idx[col]
            ord_matrix[:, c_idx] = df[col].map(lambda x: mapping.get(x, 0)).values
            
        return norm_vol, ord_matrix
