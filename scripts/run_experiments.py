"""
run_experiments.py
Main experiment execution pipeline replicating and benchmarking the experiments from:
"Application of Machine Learning Methods to Evaluate Collectibles" (Stanford CS229)
UE24CS352A - Machine Learning Mini-Project
Authors: AR Akshay Kumar (PES1UG24CS705), Shreyas Gowda (PES1UG24CS704)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit

# Ensure src is in python path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.data_loader import load_dataset
from src.preprocessing import CollectablesPreprocessor
from src.kernels import CustomCollectableKernel
from src.models import (
    BaselineLinearSVM,
    CustomKernelSVM,
    VanillaDecisionTree,
    BaggedDecisionTree,
    RandomForestModel
)
from src.evaluation import compute_metrics, format_metrics_table


def setup_plotting_style():
    sns.set_theme(style="whitegrid")
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["figure.dpi"] = 300


def plot_price_histogram(df: pd.DataFrame, out_path: str):
    """Replicates Picture 1: Histogram of Average Sale Price (clipped at 1000)."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    clipped_prices = np.clip(df["average_sale_price"].values, 0, 1000)
    ax.hist(clipped_prices, bins=45, color="#1f77b4", edgecolor="black", alpha=0.85)
    ax.set_title("Histogram of Average Sale Price (Clipped at $1000)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Average Sale Price ($)", fontsize=11)
    ax.set_ylabel("Item Count", fontsize=11)
    ax.set_xlim(0, 1050)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_bagging_curves(tree_counts, train_f1s, test_f1s, train_precs, test_precs, class_precs_history, out_path: str):
    """Replicates Section 5.3 Bagging ensemble curves across varying num_trees."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    
    # Left: Average Metrics vs num_trees
    ax1.plot(tree_counts, train_f1s, "b--", label="Train F1", linewidth=2)
    ax1.plot(tree_counts, test_f1s, "r-", label="Test F1", linewidth=2)
    ax1.plot(tree_counts, train_precs, "g--", label="Train Precision", linewidth=1.5)
    ax1.plot(tree_counts, test_precs, "m-", label="Test Precision", linewidth=1.5)
    ax1.set_xlabel("Number of Estimators (num_trees)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Score", fontsize=11, fontweight="bold")
    ax1.set_title("Bagged Ensemble: Train vs Test Scores", fontsize=12, fontweight="bold")
    ax1.set_ylim(0.3, 1.02)
    ax1.legend(loc="lower right")
    ax1.grid(True, linestyle="--", alpha=0.6)
    
    # Right: Per-class Precision vs num_trees
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for idx, bin_name in enumerate(["Bin 1", "Bin 2", "Bin 3", "Bin 4", "Bin 5"]):
        prec_series = [step[bin_name] for step in class_precs_history]
        ax2.plot(tree_counts, prec_series, label=f"Test {bin_name}", color=colors[idx], marker="o", markersize=4)
        
    ax2.set_xlabel("Number of Estimators (num_trees)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Precision", fontsize=11, fontweight="bold")
    ax2.set_title("Bagged Ensemble: Per-Class Test Precision", fontsize=12, fontweight="bold")
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(loc="upper right")
    ax2.grid(True, linestyle="--", alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    print("=" * 70)
    print("COLLECTABLE MERCHANDISE ML - EVALUATION EXPERIMENTS")
    print("UE24CS352A - Machine Learning Mini-Project")
    print("Team: AR Akshay Kumar (PES1UG24CS705) & Shreyas Gowda (PES1UG24CS704)")
    print("=" * 70)
    
    setup_plotting_style()
    
    # 1. Ingest Data
    csv_path = os.path.join(BASE_DIR, "data", "collectables_dataset.csv")
    df = load_dataset(csv_path)
    print(f"\n[+] Loaded dataset: {len(df)} records.")
    
    # Generate Data EDA Figures
    fig_dir = os.path.join(BASE_DIR, "results", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    plot_price_histogram(df, os.path.join(fig_dir, "price_histogram.png"))
    
    # 2. Stratified Train-Test Split (80% train, 20% test)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(splitter.split(df, df["price_bin"]))
    
    train_df = df.iloc[train_idx].copy().reset_index(drop=True)
    test_df = df.iloc[test_idx].copy().reset_index(drop=True)
    
    print(f"[+] Train set: {len(train_df)} samples | Test set: {len(test_df)} samples")
    print(f"[+] Train Bin distribution:\n{train_df['price_bin'].value_counts().sort_index().to_dict()}")
    print(f"[+] Test Bin distribution:\n{test_df['price_bin'].value_counts().sort_index().to_dict()}")
    
    # 3. Preprocessing
    preprocessor = CollectablesPreprocessor(random_state=42)
    preprocessor.fit(train_df)
    
    # Standard OHE + Continuous representations
    X_train_ohe = preprocessor.transform_one_hot(train_df)
    X_test_ohe = preprocessor.transform_one_hot(test_df)
    
    # Ordinal representations for Custom Kernel SVM
    train_x_cont, train_z_cat = preprocessor.transform_ordinal(train_df)
    test_x_cont, test_z_cat = preprocessor.transform_ordinal(test_df)
    
    y_train = train_df["price_bin"].values
    y_test = test_df["price_bin"].values
    
    results = {}
    
    # -------------------------------------------------------------
    # Experiment 1: Baseline Linear SVM (One-vs-One)
    # -------------------------------------------------------------
    print("\n" + "=" * 40)
    print("Running Experiment 1: Baseline Linear SVM (OvO)")
    print("=" * 40)
    linear_svm = BaselineLinearSVM(C=1.0, random_state=42)
    linear_svm.fit(X_train_ohe, y_train)
    
    tr_pred_lin = linear_svm.predict(X_train_ohe)
    te_pred_lin = linear_svm.predict(X_test_ohe)
    
    metrics_lin_tr = compute_metrics(y_train, tr_pred_lin)
    metrics_lin_te = compute_metrics(y_test, te_pred_lin)
    results["Baseline_Linear_SVM"] = {"train": metrics_lin_tr, "valid": metrics_lin_te}
    print(format_metrics_table(metrics_lin_tr, metrics_lin_te, "Baseline Linear SVM (OvO)"))
    
    # -------------------------------------------------------------
    # Experiment 2: Custom Kernel SVM (Section 5.1)
    # -------------------------------------------------------------
    print("\n" + "=" * 40)
    print("Running Experiment 2: Custom Kernel SVM (RBF + Categorical Overlap + Domain Weights)")
    print("=" * 40)
    custom_kernel_calc = CustomCollectableKernel(
        marginal_probs=preprocessor.marginal_probs,
        weights=[2.0, 1.0, 1.0, 3.0], # Search volume=2, Category=1, Material=1, Special Version=3
        theta=0.45,
        alpha=0.5
    )
    custom_svm = CustomKernelSVM(kernel_calculator=custom_kernel_calc, C=1.5, random_state=42)
    custom_svm.fit((train_x_cont, train_z_cat), y_train)
    
    tr_pred_ck = custom_svm.predict((train_x_cont, train_z_cat))
    te_pred_ck = custom_svm.predict((test_x_cont, test_z_cat))
    
    metrics_ck_tr = compute_metrics(y_train, tr_pred_ck)
    metrics_ck_te = compute_metrics(y_test, te_pred_ck)
    results["Custom_Kernel_SVM"] = {"train": metrics_ck_tr, "valid": metrics_ck_te}
    print(format_metrics_table(metrics_ck_tr, metrics_ck_te, "Custom Kernel SVM"))
    
    # -------------------------------------------------------------
    # Experiment 3: Vanilla Decision Tree (Section 5.2)
    # -------------------------------------------------------------
    print("\n" + "=" * 40)
    print("Running Experiment 3: Vanilla Decision Tree")
    print("=" * 40)
    dt_model = VanillaDecisionTree(criterion="gini", max_depth=None, random_state=42)
    dt_model.fit(X_train_ohe, y_train)
    
    tr_pred_dt = dt_model.predict(X_train_ohe)
    te_pred_dt = dt_model.predict(X_test_ohe)
    
    metrics_dt_tr = compute_metrics(y_train, tr_pred_dt)
    metrics_dt_te = compute_metrics(y_test, te_pred_dt)
    results["Vanilla_Decision_Tree"] = {"train": metrics_dt_tr, "valid": metrics_dt_te}
    print(format_metrics_table(metrics_dt_tr, metrics_dt_te, "Vanilla Decision Tree"))
    
    # -------------------------------------------------------------
    # Experiment 4: Bagged Decision Tree Ensemble (Section 5.3)
    # -------------------------------------------------------------
    print("\n" + "=" * 40)
    print("Running Experiment 4: Bagged Decision Tree Ensemble across Tree Counts")
    print("=" * 40)
    tree_candidates = [10, 25, 50, 75, 100, 150, 200]
    train_f1s = []
    test_f1s = []
    train_precs = []
    test_precs = []
    class_precs_history = []
    
    best_bagged_model = None
    best_test_f1 = -1.0
    
    for n_trees in tree_candidates:
        bag_clf = BaggedDecisionTree(n_estimators=n_trees, random_state=42)
        bag_clf.fit(X_train_ohe, y_train)
        
        tr_p = bag_clf.predict(X_train_ohe)
        te_p = bag_clf.predict(X_test_ohe)
        
        m_tr = compute_metrics(y_train, tr_p)
        m_te = compute_metrics(y_test, te_p)
        
        train_f1s.append(m_tr["weighted_f1"])
        test_f1s.append(m_te["weighted_f1"])
        train_precs.append(m_tr["weighted_precision"])
        test_precs.append(m_te["weighted_precision"])
        class_precs_history.append(m_te["per_class_precision"])
        
        if m_te["weighted_f1"] > best_test_f1:
            best_test_f1 = m_te["weighted_f1"]
            best_bagged_model = (n_trees, m_tr, m_te)
            
    # Plot curve
    plot_bagging_curves(
        tree_candidates, train_f1s, test_f1s, train_precs, test_precs,
        class_precs_history, os.path.join(fig_dir, "bagging_performance_curve.png")
    )
    
    best_n_trees, best_bag_tr, best_bag_te = best_bagged_model
    results["Bagged_Decision_Tree_Best"] = {
        "num_trees": best_n_trees,
        "train": best_bag_tr,
        "valid": best_bag_te
    }
    print(format_metrics_table(best_bag_tr, best_bag_te, f"Bagged Decision Tree (num_trees={best_n_trees})"))
    
    # -------------------------------------------------------------
    # Experiment 5: Random Forest Benchmark
    # -------------------------------------------------------------
    print("\n" + "=" * 40)
    print("Running Experiment 5: Random Forest Benchmark (100 Trees)")
    print("=" * 40)
    rf_model = RandomForestModel(n_estimators=100, random_state=42)
    rf_model.fit(X_train_ohe, y_train)
    
    tr_pred_rf = rf_model.predict(X_train_ohe)
    te_pred_rf = rf_model.predict(X_test_ohe)
    
    metrics_rf_tr = compute_metrics(y_train, tr_pred_rf)
    metrics_rf_te = compute_metrics(y_test, te_pred_rf)
    results["Random_Forest"] = {"train": metrics_rf_tr, "valid": metrics_rf_te}
    print(format_metrics_table(metrics_rf_tr, metrics_rf_te, "Random Forest (100 Trees)"))
    
    # Save results JSON
    json_path = os.path.join(BASE_DIR, "results", "experiment_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Saved complete experiment metrics to: {json_path}")
    print("\nExperiment Suite Completed Successfully!")


if __name__ == "__main__":
    main()
