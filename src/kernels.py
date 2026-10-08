"""
kernels.py
Custom Composite Kernel for Support Vector Machines (SVM)
Combines continuous Gaussian/RBF kernel with Marco's probability-adjusted categorical overlap kernel:
1. Continuous similarity (Search Volume):
   U_1(x_i, x_j) = exp( - (x_i - x_j)^2 / (2 * theta^2) )
2. Categorical similarity (Category, Material, Special Version):
   U_k(z_i, z_j) = h(P(z_i)) if z_i == z_j else 0
   where h(z) = (1 - z^alpha)^(1/alpha) for 0 < alpha < 1
3. Composite weighted kernel:
   K(i, j) = sum_{l=1}^4 w_l * U_l(i, j)
   with domain knowledge weights w = [2, 1, 1, 3]
"""

import numpy as np
from typing import List, Dict


class CustomCollectableKernel:
    """
    Computes custom precomputed Gram matrix for SVM handling mixed continuous and categorical data.
    """
    def __init__(self,
                 marginal_probs: Dict[str, np.ndarray],
                 weights: List[float] = [2.0, 1.0, 1.0, 3.0],
                 theta: float = 0.5,
                 alpha: float = 0.5):
        """
        Args:
            marginal_probs: Dictionary with marginal probabilities P(Z=z) for each categorical feature
            weights: [w_search_volume, w_category, w_material, w_special_version]
            theta: RBF bandwidth parameter for normalized search volume
            alpha: Power parameter for Marco's convex function h(z) = (1 - z^alpha)^(1/alpha)
        """
        self.marginal_probs = marginal_probs
        self.weights = np.array(weights, dtype=float)
        self.theta = theta
        self.alpha = alpha
        self.categorical_cols = ["category", "material", "special_version"]
        
        # Precompute h(P(z)) lookup tables for each categorical feature
        self.h_tables = {}
        for col, probs in marginal_probs.items():
            # clip probabilities strictly in (0, 1)
            p_clipped = np.clip(probs, 1e-6, 1.0 - 1e-6)
            h_vals = (1.0 - (p_clipped ** self.alpha)) ** (1.0 / self.alpha)
            self.h_tables[col] = h_vals

    def _compute_rbf(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """
        Vectorized computation of continuous RBF similarity matrix U1.
        x1: shape (N1, 1), x2: shape (N2, 1)
        """
        # Squared difference matrix (N1, N2)
        diff_sq = (x1[:, None, 0] - x2[None, :, 0]) ** 2
        gamma = 1.0 / (2.0 * (self.theta ** 2))
        return np.exp(-gamma * diff_sq)

    def _compute_categorical_similarity(self, z1: np.ndarray, z2: np.ndarray, col: str) -> np.ndarray:
        """
        Vectorized computation of Marco's categorical overlap similarity.
        z1: shape (N1,), z2: shape (N2,)
        U = h(P(z)) if z1 == z2 else 0
        """
        # Boolean match matrix (N1, N2)
        match_mask = (z1[:, None] == z2[None, :])
        
        # Values from precomputed h_table
        h_table = self.h_tables[col]
        # For each element in z1, look up its h value
        h_z1 = h_table[np.clip(z1, 0, len(h_table) - 1)]
        
        # Similarity is h(P(z1)) wherever z1 matches z2
        sim_matrix = match_mask * h_z1[:, None]
        return sim_matrix

    def compute_kernel_matrix(self,
                              x1_cont: np.ndarray, z1_cat: np.ndarray,
                              x2_cont: np.ndarray, z2_cat: np.ndarray) -> np.ndarray:
        """
        Computes the composite kernel Gram matrix between dataset 1 and dataset 2.
        
        Args:
            x1_cont: Continuous normalized search volume (N1, 1)
            z1_cat: Ordinal categorical indices (N1, 3) [category, material, special_version]
            x2_cont: Continuous normalized search volume (N2, 1)
            z2_cat: Ordinal categorical indices (N2, 3)
            
        Returns:
            Gram matrix K of shape (N1, N2)
        """
        # U1: Search volume RBF
        u1 = self._compute_rbf(x1_cont, x2_cont)
        
        # U2, U3, U4: Categorical overlaps
        u2 = self._compute_categorical_similarity(z1_cat[:, 0], z2_cat[:, 0], "category")
        u3 = self._compute_categorical_similarity(z1_cat[:, 1], z2_cat[:, 1], "material")
        u4 = self._compute_categorical_similarity(z1_cat[:, 2], z2_cat[:, 2], "special_version")
        
        # Composite weighted sum: sum(w_l * U_l)
        k_matrix = (self.weights[0] * u1 +
                    self.weights[1] * u2 +
                    self.weights[2] * u3 +
                    self.weights[3] * u4)
                    
        return k_matrix
