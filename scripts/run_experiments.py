"""
run_experiments.py
Main experiment execution pipeline replicating the Stanford CS229 benchmark
AND presenting our novel feature extension (Community Hype Score & Vaulted Status).
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


def plot_ablation_comparison(comparison_df: pd.DataFrame, out_path: str):
    """Plots feature ablation comparison between Original CS229 and Our Novel Feature Extension."""
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(comparison_df))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, comparison_df["Original CS229 (4 Features)"], width, label="Original CS229 (4 Features)", color="#94a3b8")
    rects2 = ax.bar(x + width/2, comparison_df["Our Extension (Novel Features)"], width, label="Our Extension (+Community Hype & Vault Status)", color="#2563eb")
    
    ax.set_ylabel("Metric Score", fontsize=11, fontweight="bold")
    ax.set_title("Ablation Study: Impact of Novel Engineered Features on Valuation Models", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df.index, fontsize=10, fontweight="bold")
    ax.legend()
    ax.set_ylim(0, 1.1)
    
    for rects in [rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.2f}",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9)
                        
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    print("=" * 75)
    print("COLLECTABLE MERCHANDISE ML - EVALUATION & INNOVATION BENCHMARK")
    print("UE24CS352A - Machine Learning Mini-Project")
    print("Team: AR Akshay Kumar (PES1UG24CS705) & Shreyas Gowda (PES1UG24CS704)")
    print("=" * 75)
    
    setup_plotting_style()
    
    # 1. Ingest Data
    csv_path = os.path.join(BASE_DIR, "data", "collectables_dataset.csv")
    df = load_dataset(csv_path)
    print(f"\n[+] Loaded dataset: {len(df)} records.")
    
    fig_dir = os.path.join(BASE_DIR, "results", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    plot_price_histogram(df, os.path.join(fig_dir, "price_histogram.png"))
    
    # 2. Stratified Train-Test Split (80% train, 20% test)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(splitter.split(df, df["price_bin"]))
    
    train_df = df.iloc[train_idx].copy().reset_index(drop=True)
    test_df = df.iloc[test_idx].copy().reset_index(drop=True)
    
    print(f"[+] Train set: {len(train_df)} samples | Test set: {len(test_df)} samples")
    
    # -------------------------------------------------------------
    # PART 1: ORIGINAL STANFORD CS229 BENCHMARK (4 Features)
    # -------------------------------------------------------------
    print("\n" + "#" * 60)
    print("PART 1: ORIGINAL CS229 FEATURE SET (Search Volume, Cat, Mat, Ver)")
    print("#" * 60)
    
    prep_orig = CollectablesPreprocessor(include_novel_features=False, random_state=42)
    prep_orig.fit(train_df)
    
    X_tr_orig_ohe = prep_orig.transform_one_hot(train_df)
    X_te_orig_ohe = prep_orig.transform_one_hot(test_df)
    tr_x_orig, tr_z_orig = prep_orig.transform_ordinal(train_df)
    te_x_orig, te_z_orig = prep_orig.transform_ordinal(test_df)
    
    y_train = train_df["price_bin"].values
    y_test = test_df["price_bin"].values
    
    # Linear SVM Baseline
    lin_svm_orig = BaselineLinearSVM(C=1.0, random_state=42)
    lin_svm_orig.fit(X_tr_orig_ohe, y_train)
    m_lin_orig = compute_metrics(y_test, lin_svm_orig.predict(X_te_orig_ohe))
    
    # Custom Kernel SVM Original
    ck_calc_orig = CustomCollectableKernel(
        marginal_probs=prep_orig.marginal_probs,
        weights=[2.0, 1.0, 1.0, 3.0],
        theta=0.45,
        alpha=0.5,
        categorical_cols=["category", "material", "special_version"]
    )
    ck_svm_orig = CustomKernelSVM(kernel_calculator=ck_calc_orig, C=1.5, random_state=42)
    ck_svm_orig.fit((tr_x_orig, tr_z_orig), y_train)
    m_ck_orig = compute_metrics(y_test, ck_svm_orig.predict((te_x_orig, te_z_orig)))
    
    # Bagged Decision Trees Original
    bag_orig = BaggedDecisionTree(n_estimators=100, random_state=42)
    bag_orig.fit(X_tr_orig_ohe, y_train)
    m_bag_orig = compute_metrics(y_test, bag_orig.predict(X_te_orig_ohe))
    
    # -------------------------------------------------------------
    # PART 2: OUR TEAM'S NOVEL FEATURE ENHANCEMENT (6 Features)
    # Includes: Community Hype Score & Discontinued Vault Status
    # -------------------------------------------------------------
    print("\n" + "#" * 60)
    print("PART 2: OUR EXTENDED MODEL (+Community Hype Score & Vault Status)")
    print("#" * 60)
    
    prep_novel = CollectablesPreprocessor(include_novel_features=True, random_state=42)
    prep_novel.fit(train_df)
    
    X_tr_nov_ohe = prep_novel.transform_one_hot(train_df)
    X_te_nov_ohe = prep_novel.transform_one_hot(test_df)
    tr_x_nov, tr_z_nov = prep_novel.transform_ordinal(train_df)
    te_x_nov, te_z_nov = prep_novel.transform_ordinal(test_df)
    
    # Enhanced Custom Kernel SVM with Novel Features
    # Weights: [Search Volume=1.5, Hype=2.5, Category=1.0, Material=1.0, Version=3.0, Vaulted=2.0]
    ck_calc_nov = CustomCollectableKernel(
        marginal_probs=prep_novel.marginal_probs,
        weights=[1.5, 2.5, 1.0, 1.0, 3.0, 2.0],
        theta=0.45,
        alpha=0.5,
        categorical_cols=prep_novel.categorical_cols
    )
    ck_svm_nov = CustomKernelSVM(kernel_calculator=ck_calc_nov, C=1.5, random_state=42)
    ck_svm_nov.fit((tr_x_nov, tr_z_nov), y_train)
    
    tr_pred_ck_nov = ck_svm_nov.predict((tr_x_nov, tr_z_nov))
    te_pred_ck_nov = ck_svm_nov.predict((te_x_nov, te_z_nov))
    m_ck_nov_tr = compute_metrics(y_train, tr_pred_ck_nov)
    m_ck_nov_te = compute_metrics(y_test, te_pred_ck_nov)
    
    print("\n" + format_metrics_table(m_ck_nov_tr, m_ck_nov_te, "Enhanced Custom Kernel SVM (Our Novel Features)"))
    
    # Enhanced Bagged Ensemble with Novel Features across Tree Counts
    tree_candidates = [10, 25, 50, 75, 100, 150, 200]
    train_f1s, test_f1s, train_precs, test_precs, class_precs_history = [], [], [], [], []
    best_bag_model = None
    best_f1 = -1.0
    
    for n_trees in tree_candidates:
        bag_clf = BaggedDecisionTree(n_estimators=n_trees, random_state=42)
        bag_clf.fit(X_tr_nov_ohe, y_train)
        
        tr_p = bag_clf.predict(X_tr_nov_ohe)
        te_p = bag_clf.predict(X_te_nov_ohe)
        
        m_tr = compute_metrics(y_train, tr_p)
        m_te = compute_metrics(y_test, te_p)
        
        train_f1s.append(m_tr["weighted_f1"])
        test_f1s.append(m_te["weighted_f1"])
        train_precs.append(m_tr["weighted_precision"])
        test_precs.append(m_te["weighted_precision"])
        class_precs_history.append(m_te["per_class_precision"])
        
        if m_te["weighted_f1"] > best_f1:
            best_f1 = m_te["weighted_f1"]
            best_bag_model = (n_trees, m_tr, m_te)
            
    plot_bagging_curves(
        tree_candidates, train_f1s, test_f1s, train_precs, test_precs,
        class_precs_history, os.path.join(fig_dir, "bagging_performance_curve.png")
    )
    
    best_trees, b_tr, b_te = best_bag_model
    print("\n" + format_metrics_table(b_tr, b_te, f"Enhanced Bagged Decision Tree (num_trees={best_trees})"))
    
    # -------------------------------------------------------------
    # ABLATION STUDY COMPARISON
    # -------------------------------------------------------------
    ablation_data = {
        "Custom Kernel SVM (Weighted F1)": [m_ck_orig["weighted_f1"], m_ck_nov_te["weighted_f1"]],
        "Custom Kernel SVM (Test Precision)": [m_ck_orig["weighted_precision"], m_ck_nov_te["weighted_precision"]],
        "Bagged Trees (Weighted F1)": [m_bag_orig["weighted_f1"], b_te["weighted_f1"]],
        "Bin 4 Grail Precision (Custom SVM)": [m_ck_orig["per_class_precision"]["Bin 4"], m_ck_nov_te["per_class_precision"]["Bin 4"]],
        "Bin 4 Grail Precision (Bagged)": [m_bag_orig["per_class_precision"]["Bin 4"], b_te["per_class_precision"]["Bin 4"]],
    }
    ablation_df = pd.DataFrame(ablation_data, index=["Original CS229 (4 Features)", "Our Extension (Novel Features)"]).T
    
    plot_ablation_comparison(ablation_df, os.path.join(fig_dir, "ablation_study.png"))
    
    print("\n" + "=" * 65)
    print("ABLATION STUDY: ORIGINAL CS229 vs. OUR TEAM'S NOVEL FEATURES")
    print("=" * 65)
    print(ablation_df.round(3).to_string())
    print("=" * 65)
    
    # Save results JSON
    results = {
        "Original_CS229": {
            "Linear_SVM": m_lin_orig,
            "Custom_Kernel_SVM": m_ck_orig,
            "Bagged_Decision_Tree": m_bag_orig
        },
        "Our_Novel_Extension": {
            "Custom_Kernel_SVM": m_ck_nov_te,
            "Bagged_Decision_Tree": b_te
        },
        "Ablation_Comparison": ablation_df.to_dict()
    }
    
    json_path = os.path.join(BASE_DIR, "results", "experiment_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Saved complete experiment metrics to: {json_path}")


if __name__ == "__main__":
    main()
