"""
preprocessing.py
Feature engineering and transformation pipeline:
- Log transformation of search volume
- Normalization of community hype score (Novel Engineered Feature)
- Min-max scaling of continuous attributes to [0, 1]
- One-Hot Encoding for baseline models
- Ordinal Encoding and marginal distribution calculation for Custom Kernel SVM
- Stratified splitting for imbalanced multi-class data
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List


class CollectablesPreprocessor:
    """
    Transforms raw collectible merchandise attributes into feature matrices
    suitable for Linear SVM, Decision Trees, Bagged Ensembles, and Custom Kernel SVM.
    Supports both standard Stanford CS229 features and our novel engineered features.
    """
    def __init__(self, include_novel_features: bool = True, random_state: int = 42):
        self.include_novel_features = include_novel_features
        self.random_state = random_state
        self.min_log_vol = None
        self.max_log_vol = None
        
        # Categorical feature vocabulary mappings
        self.categorical_cols = ["category", "material", "special_version"]
        if self.include_novel_features:
            self.categorical_cols.append("is_vaulted")
            
        self.cat_to_idx: Dict[str, Dict[Any, int]] = {}
        self.idx_to_cat: Dict[str, Dict[int, Any]] = {}
        self.marginal_probs: Dict[str, np.ndarray] = {}
        
        # One-hot column names
        self.one_hot_columns = []
        
    def fit(self, df: pd.DataFrame):
        """
        Fits transformation parameters (log bounds, ordinal maps, marginal probabilities).
        """
        # 1. Continuous feature scaling: Log transform + Min-Max for search volume
        log_vols = np.log1p(df["search_volume"].values.astype(float))
        self.min_log_vol = float(np.min(log_vols))
        self.max_log_vol = float(np.max(log_vols))
        if self.max_log_vol == self.min_log_vol:
            self.max_log_vol += 1.0
            
        # 2. Categorical ordinal maps and marginal probabilities
        for col in self.categorical_cols:
            if col not in df.columns:
                continue
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
        ohe_cols = [c for c in self.categorical_cols if c in df.columns]
        ohe_df = pd.get_dummies(df[ohe_cols].astype(str), drop_first=False)
        self.one_hot_columns = list(ohe_df.columns)
        
        return self

    def transform_continuous(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transforms continuous features to [0, 1].
        If include_novel_features=True, returns [norm_search_volume, norm_community_hype].
        Otherwise returns [norm_search_volume].
        """
        search_volumes = df["search_volume"].values
        log_vols = np.log1p(np.maximum(search_volumes, 0).astype(float))
        norm_vols = (log_vols - self.min_log_vol) / (self.max_log_vol - self.min_log_vol)
        norm_vols = np.clip(norm_vols, 0.0, 1.0).reshape(-1, 1)
        
        if self.include_novel_features and "community_hype_score" in df.columns:
            # Scale 0 to 10 hype score to [0, 1]
            hype_scores = df["community_hype_score"].values.astype(float)
            norm_hype = np.clip(hype_scores / 10.0, 0.0, 1.0).reshape(-1, 1)
            return np.hstack([norm_vols, norm_hype])
            
        return norm_vols

    def transform_one_hot(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transforms dataframe into feature matrix with continuous normalized feature(s)
        plus one-hot encoded categorical attributes.
        Used for Linear SVM, Vanilla Decision Tree, and Bagged Decision Tree.
        """
        cont_mat = self.transform_continuous(df)
        
        ohe_cols = [c for c in self.categorical_cols if c in df.columns]
        ohe_df = pd.get_dummies(df[ohe_cols].astype(str), drop_first=False)
        # Align with fitted columns
        for col in self.one_hot_columns:
            if col not in ohe_df.columns:
                ohe_df[col] = 0
        ohe_mat = ohe_df[self.one_hot_columns].values.astype(float)
        
        return np.hstack([cont_mat, ohe_mat])

    def transform_ordinal(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transforms features for Custom Kernel SVM:
        Returns:
            continuous_feats: normalized array of shape (N, D_cont)
            categorical_indices: integer indices array of shape (N, D_cat)
        """
        cont_mat = self.transform_continuous(df)
        
        ohe_cols = [c for c in self.categorical_cols if c in df.columns]
        ord_matrix = np.zeros((len(df), len(ohe_cols)), dtype=int)
        for c_idx, col in enumerate(ohe_cols):
            mapping = self.cat_to_idx[col]
            ord_matrix[:, c_idx] = df[col].map(lambda x: mapping.get(x, 0)).values
            
        return cont_mat, ord_matrix
