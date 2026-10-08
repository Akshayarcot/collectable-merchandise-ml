"""
models.py
Model implementations for Collectable Merchandise Evaluation:
1. Baseline Linear SVM (One-vs-One binarization)
2. Custom Kernel SVM (Precomputed kernel with domain knowledge weights)
3. Vanilla Decision Tree (CART)
4. Bagged Decision Tree Ensemble (BaggingClassifier across tree counts)
5. Random Forest Benchmark
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from src.kernels import CustomCollectableKernel


class BaselineLinearSVM(BaseEstimator, ClassifierMixin):
    """
    Linear Support Vector Classifier using One-vs-One (OvO) multi-class binarization
    as specified in Section 4 of the paper.
    """
    def __init__(self, C: float = 1.0, random_state: int = 42):
        self.C = C
        self.random_state = random_state
        self.clf = SVC(kernel="linear", C=self.C, decision_function_shape="ovo", random_state=self.random_state)
        
    def fit(self, X, y):
        self.clf.fit(X, y)
        self.classes_ = self.clf.classes_
        return self
        
    def predict(self, X):
        return self.clf.predict(X)


class CustomKernelSVM(BaseEstimator, ClassifierMixin):
    """
    Support Vector Classifier powered by the custom domain-weighted
    RBF + Marco Categorical Overlap Kernel (Section 5.1).
    """
    def __init__(self, kernel_calculator: CustomCollectableKernel, C: float = 1.0, random_state: int = 42):
        self.kernel_calculator = kernel_calculator
        self.C = C
        self.random_state = random_state
        self.clf = SVC(kernel="precomputed", C=self.C, decision_function_shape="ovo", random_state=self.random_state)
        self.train_x_cont = None
        self.train_z_cat = None
        
    def fit(self, X_tuple, y):
        """
        X_tuple is (x_cont, z_cat):
            x_cont: (N, 1) normalized continuous search volume
            z_cat: (N, 3) ordinal categorical features
        """
        x_cont, z_cat = X_tuple
        self.train_x_cont = np.array(x_cont, copy=True)
        self.train_z_cat = np.array(z_cat, copy=True)
        
        # Compute N x N Gram matrix
        K_train = self.kernel_calculator.compute_kernel_matrix(
            self.train_x_cont, self.train_z_cat,
            self.train_x_cont, self.train_z_cat
        )
        self.clf.fit(K_train, y)
        self.classes_ = self.clf.classes_
        return self
        
    def predict(self, X_tuple):
        x_cont, z_cat = X_tuple
        # Compute M x N Test-Train Gram matrix
        K_test = self.kernel_calculator.compute_kernel_matrix(
            x_cont, z_cat,
            self.train_x_cont, self.train_z_cat
        )
        return self.clf.predict(K_test)


class VanillaDecisionTree(BaseEstimator, ClassifierMixin):
    """
    Unpruned / standard Decision Tree Classifier (Section 5.2).
    """
    def __init__(self, criterion: str = "gini", max_depth: int = None, random_state: int = 42):
        self.criterion = criterion
        self.max_depth = max_depth
        self.random_state = random_state
        self.clf = DecisionTreeClassifier(
            criterion=self.criterion,
            max_depth=self.max_depth,
            random_state=self.random_state
        )
        
    def fit(self, X, y):
        self.clf.fit(X, y)
        self.classes_ = self.clf.classes_
        return self
        
    def predict(self, X):
        return self.clf.predict(X)


class BaggedDecisionTree(BaseEstimator, ClassifierMixin):
    """
    Bagged Decision Tree Ensemble with customizable number of base estimators (Section 5.3).
    """
    def __init__(self, n_estimators: int = 100, max_depth: int = None, random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.clf = BaggingClassifier(
            estimator=DecisionTreeClassifier(max_depth=self.max_depth, random_state=self.random_state),
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1
        )
        
    def fit(self, X, y):
        self.clf.fit(X, y)
        self.classes_ = self.clf.classes_
        return self
        
    def predict(self, X):
        return self.clf.predict(X)


class RandomForestModel(BaseEstimator, ClassifierMixin):
    """
    Random Forest Classifier reference baseline.
    """
    def __init__(self, n_estimators: int = 100, max_depth: int = None, random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.clf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1
        )
        
    def fit(self, X, y):
        self.clf.fit(X, y)
        self.classes_ = self.clf.classes_
        return self
        
    def predict(self, X):
        return self.clf.predict(X)
