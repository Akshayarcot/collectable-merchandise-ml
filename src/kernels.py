"""
kernels.py
Custom Composite Kernel for Support Vector Machines (SVM)
Combines continuous Gaussian/RBF kernels with Marco's probability-adjusted categorical overlap kernel:
1. Continuous similarity (Search Volume & Community Hype Score):
   U_cont(x_i, x_j) = exp( - ||x_i - x_j||^2 / (2 * theta^2) )
2. Categorical similarity (Category, Material, Special Version, Vaulted Status):
   U_k(z_i, z_j) = h(P(z_i)) if z_i == z_j else 0
   where h(z) = (1 - z^alpha)^(1/alpha) for 0 < alpha < 1
3. Composite weighted kernel:
   K(i, j) = sum_l w_l * U_l(i, j)
"""

import numpy as np
from typing import List, Dict


class CustomCollectableKernel:
    """
    Computes custom precomputed Gram matrix for SVM handling mixed continuous and categorical data.
    """
    def __init__(self,
                 marginal_probs: Dict[str, np.ndarray],
                 weights: List[float] = None,
                 theta: float = 0.45,
                 alpha: float = 0.5,
                 categorical_cols: List[str] = None):
        """
        Args:
            marginal_probs: Dictionary with marginal probabilities P(Z=z) for each categorical feature
            weights: Weight array for each similarity kernel component
            theta: RBF bandwidth parameter for normalized continuous features
            alpha: Power parameter for Marco's convex function h(z) = (1 - z^alpha)^(1/alpha)
            categorical_cols: Ordered list of categorical column names
        """
        self.marginal_probs = marginal_probs
        self.theta = theta
        self.alpha = alpha
        self.categorical_cols = categorical_cols or ["category", "material", "special_version"]
        
        # Default weights if not specified
        if weights is None:
            if len(self.categorical_cols) > 3:
                # 2 continuous + 4 categorical: [Search, Hype, Category, Material, Version, Vaulted]
                self.weights = np.array([1.5, 2.5, 1.0, 1.0, 3.0, 2.0], dtype=float)
            else:
                # 1 continuous + 3 categorical: [Search, Category, Material, Version]
                self.weights = np.array([2.0, 1.0, 1.0, 3.0], dtype=float)
        else:
            self.weights = np.array(weights, dtype=float)
        
        # Precompute h(P(z)) lookup tables for each categorical feature
        self.h_tables = {}
        for col, probs in marginal_probs.items():
            p_clipped = np.clip(probs, 1e-6, 1.0 - 1e-6)
            h_vals = (1.0 - (p_clipped ** self.alpha)) ** (1.0 / self.alpha)
            self.h_tables[col] = h_vals

    def _compute_rbf_single(self, x1_col: np.ndarray, x2_col: np.ndarray) -> np.ndarray:
        """
        1D Gaussian RBF similarity.
        x1_col: (N1,), x2_col: (N2,)
        """
        diff_sq = (x1_col[:, None] - x2_col[None, :]) ** 2
        gamma = 1.0 / (2.0 * (self.theta ** 2))
        return np.exp(-gamma * diff_sq)

    def _compute_categorical_similarity(self, z1: np.ndarray, z2: np.ndarray, col: str) -> np.ndarray:
        """
        Vectorized computation of Marco's categorical overlap similarity.
        z1: shape (N1,), z2: shape (N2,)
        U = h(P(z)) if z1 == z2 else 0
        """
        match_mask = (z1[:, None] == z2[None, :])
        if col not in self.h_tables:
            return match_mask.astype(float)
            
        h_table = self.h_tables[col]
        h_z1 = h_table[np.clip(z1, 0, len(h_table) - 1)]
        return match_mask * h_z1[:, None]

    def compute_kernel_matrix(self,
                              x1_cont: np.ndarray, z1_cat: np.ndarray,
                              x2_cont: np.ndarray, z2_cat: np.ndarray) -> np.ndarray:
        """
        Computes composite kernel Gram matrix between dataset 1 and dataset 2.
        """
        n1 = len(x1_cont)
        n2 = len(x2_cont)
        k_matrix = np.zeros((n1, n2), dtype=float)
        
        w_idx = 0
        
        # 1. Continuous feature kernels
        d_cont = x1_cont.shape[1]
        for c in range(d_cont):
            u_cont = self._compute_rbf_single(x1_cont[:, c], x2_cont[:, c])
            w = self.weights[w_idx] if w_idx < len(self.weights) else 1.0
            k_matrix += w * u_cont
            w_idx += 1
            
        # 2. Categorical feature kernels
        d_cat = z1_cat.shape[1]
        for c in range(d_cat):
            col_name = self.categorical_cols[c] if c < len(self.categorical_cols) else f"cat_{c}"
            u_cat = self._compute_categorical_similarity(z1_cat[:, c], z2_cat[:, c], col_name)
            w = self.weights[w_idx] if w_idx < len(self.weights) else 1.0
            k_matrix += w * u_cat
            w_idx += 1
            
        return k_matrix
