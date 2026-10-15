"""
demo_predict.py
Interactive CLI demo for live evaluation and viva demonstration.
Allows user/evaluator to input collectible merchandise attributes
and obtain price bracket predictions across all trained models,
including our novel features: Community Hype Score, Vaulted Status,
and Box Condition Grading appraisal.
UE24CS352A - Machine Learning Mini-Project
Authors: AR Akshay Kumar (PES1UG24CS705) & Shreyas Gowda (PES1UG24CS704)
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.data_loader import load_dataset, PRICE_BIN_EDGES
from src.preprocessing import CollectablesPreprocessor
from src.kernels import CustomCollectableKernel
from src.models import (
    BaselineLinearSVM,
    CustomKernelSVM,
    VanillaDecisionTree,
    BaggedDecisionTree,
    RandomForestModel
)

PRESETS = {
    "1": {
        "name": "Beerus Metallic SDCC Figure (High-End Holy Grail)",
        "category": "Anime",
        "material": "Metallic",
        "special_version": "SDCC",
        "search_volume": 42000,
        "community_hype_score": 9.4,
        "is_vaulted": 1,
        "box_condition": "Mint_10",
        "true_price": "$1250 (Bin 5)"
    },
    "2": {
        "name": "Iron Man Glow-in-the-dark Exclusive (Mid-Tier Rare)",
        "category": "Marvel",
        "material": "Glow-in-the-dark",
        "special_version": "Special Edition",
        "search_volume": 12500,
        "community_hype_score": 6.8,
        "is_vaulted": 1,
        "box_condition": "Near_Mint_9",
        "true_price": "$185 (Bin 3)"
    },
    "3": {
        "name": "Simba Spirit Standard Release (Common Mass Market)",
        "category": "Disney",
        "material": "Standard/Vinyl",
        "special_version": "Standard",
        "search_volume": 850,
        "community_hype_score": 2.2,
        "is_vaulted": 0,
        "box_condition": "Mint_10",
        "true_price": "$18 (Bin 1)"
    },
    "4": {
        "name": "Tony the Tiger Flocked Chase (Very Rare Vaulted Icon)",
        "category": "Ad-Icons",
        "material": "Flocked",
        "special_version": "Chase",
        "search_volume": 28000,
        "community_hype_score": 8.5,
        "is_vaulted": 1,
        "box_condition": "Mint_10",
        "true_price": "$520 (Bin 4)"
    }
}

BIN_DESCRIPTIONS = {
    1: f"Bin 1: [${PRICE_BIN_EDGES[0]:.2f} - ${PRICE_BIN_EDGES[1]:.2f}] (Common / Base Retail)",
    2: f"Bin 2: (${PRICE_BIN_EDGES[1]:.2f} - ${PRICE_BIN_EDGES[2]:.2f}] (Moderate Demand / Slight Premium)",
    3: f"Bin 3: (${PRICE_BIN_EDGES[2]:.2f} - ${PRICE_BIN_EDGES[3]:.2f}] (Collectible / Con Exclusive)",
    4: f"Bin 4: (${PRICE_BIN_EDGES[3]:.2f} - ${PRICE_BIN_EDGES[4]:.2f}] (High Rarity / Coveted Grail)",
    5: f"Bin 5: (${PRICE_BIN_EDGES[4]:.2f} - ${PRICE_BIN_EDGES[5]:.2f}] (Ultra Rare / Museum Tier)"
}

CONDITION_MULTIPLIERS = {
    "Mint_10": (1.00, "10/10 Mint in Box (Full Market Value)"),
    "Near_Mint_9": (0.85, "9/10 Near Mint (Minor shelf wear, 15% discount)"),
    "Damaged_Box": (0.60, "6/10 Box Creased/Damaged (40% discount)"),
    "Loose_OOB": (0.35, "Out of Box / Loose Figure (65% discount)")
}


def train_models_for_demo():
    df = load_dataset()
    preprocessor = CollectablesPreprocessor(include_novel_features=True, random_state=42)
    preprocessor.fit(df)
    
    X_ohe = preprocessor.transform_one_hot(df)
    x_cont, z_cat = preprocessor.transform_ordinal(df)
    y = df["price_bin"].values
    
    # Train models
    lin_svm = BaselineLinearSVM(C=1.0, random_state=42)
    lin_svm.fit(X_ohe, y)
    
    custom_kernel_calc = CustomCollectableKernel(
        marginal_probs=preprocessor.marginal_probs,
        weights=[1.5, 2.5, 1.0, 1.0, 3.0, 2.0],
        theta=0.45,
        alpha=0.5,
        categorical_cols=preprocessor.categorical_cols
    )
    custom_svm = CustomKernelSVM(kernel_calculator=custom_kernel_calc, C=1.5, random_state=42)
    custom_svm.fit((x_cont, z_cat), y)
    
    dt = VanillaDecisionTree(criterion="gini", max_depth=None, random_state=42)
    dt.fit(X_ohe, y)
    
    bagged_tree = BaggedDecisionTree(n_estimators=100, random_state=42)
    bagged_tree.fit(X_ohe, y)
    
    rf = RandomForestModel(n_estimators=100, random_state=42)
    rf.fit(X_ohe, y)
    
    return preprocessor, {
        "Baseline Linear SVM": lin_svm,
        "Custom Kernel SVM (Our Proposed)": custom_svm,
        "Vanilla Decision Tree": dt,
        "Bagged Decision Tree (100 Trees)": bagged_tree,
        "Random Forest Benchmark": rf
    }


def predict_sample(item_dict, preprocessor, models):
    sample_df = pd.DataFrame([item_dict])
    X_sample_ohe = preprocessor.transform_one_hot(sample_df)
    x_sample_cont, z_sample_cat = preprocessor.transform_ordinal(sample_df)
    
    box_condition = item_dict.get("box_condition", "Mint_10")
    mult, cond_desc = CONDITION_MULTIPLIERS.get(box_condition, (1.0, "Mint 10"))
    
    print("\n" + "=" * 70)
    print("EVALUATION RESULT FOR COLLECTABLE ITEM:")
    print(f" Category             : {item_dict['category']}")
    print(f" Material             : {item_dict['material']}")
    print(f" Special Version      : {item_dict['special_version']}")
    print(f" Search Volume        : {item_dict['search_volume']} monthly queries")
    print(f" Community Hype Score : {item_dict.get('community_hype_score', 5.0)} / 10.0 (Novel Feature)")
    print(f" Vaulted Status       : {'Discontinued / Vaulted (Rarity Locked)' if item_dict.get('is_vaulted', 0) == 1 else 'Active Production'}")
    print(f" Physical Condition   : {cond_desc}")
    if "true_price" in item_dict:
        print(f" Market Benchmark Ref : {item_dict['true_price']}")
    print("-" * 70)
    print(f"{'Model Architecture':<36} | {'Predicted Price Tier'}")
    print("-" * 70)
    
    predicted_bins = []
    for name, model in models.items():
        if "Custom Kernel" in name:
            pred = model.predict((x_sample_cont, z_sample_cat))[0]
        else:
            pred = model.predict(X_sample_ohe)[0]
        desc = BIN_DESCRIPTIONS.get(int(pred), f"Bin {pred}")
        predicted_bins.append(int(pred))
        print(f"{name:<36} | {desc}")
        
    # Consensus appraisal
    consensus_bin = int(np.median(predicted_bins))
    low_price = PRICE_BIN_EDGES[consensus_bin - 1]
    high_price = PRICE_BIN_EDGES[consensus_bin]
    adj_low = low_price * mult
    adj_high = high_price * mult
    
    print("-" * 70)
    print(f"Consensus Valuation Tier: Bin {consensus_bin}")
    print(f"Estimated Mint Fair Market Value: ${low_price:.2f} - ${high_price:.2f}")
    if mult < 1.0:
        print(f"Condition-Adjusted Value ({box_condition}): ${adj_low:.2f} - ${adj_high:.2f}")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Collectable Merchandise Valuation Demo")
    parser.add_argument("--preset", type=str, choices=["1", "2", "3", "4"], help="Run with preset example")
    parser.add_argument("--category", type=str, default="Anime", help="Category: Anime, Marvel, Disney, Ad-Icons, Game-of-Thrones")
    parser.add_argument("--material", type=str, default="Metallic", help="Material: Standard/Vinyl, Metallic, Flocked, Glow-in-the-dark, Diamond Collection, Chrome")
    parser.add_argument("--version", type=str, default="SDCC", help="Special Version: Standard, SDCC, NYCC, Chase, Special Edition, Vaulted")
    parser.add_argument("--search_volume", type=int, default=35000, help="Search Volume (queries/month)")
    parser.add_argument("--hype_score", type=float, default=8.5, help="Community Hype Score (0.0 to 10.0)")
    parser.add_argument("--is_vaulted", type=int, default=1, choices=[0, 1], help="Is Vaulted: 1=Yes, 0=No")
    parser.add_argument("--box_condition", type=str, default="Mint_10", choices=list(CONDITION_MULTIPLIERS.keys()), help="Box Condition")
    args = parser.parse_args()
    
    print("Initializing Collectable Merchandise ML System...")
    preprocessor, models = train_models_for_demo()
    print("Models trained and loaded successfully!")
    
    if args.preset:
        preset_data = PRESETS[args.preset]
        print(f"\nRunning Preset {args.preset}: {preset_data['name']}")
        predict_sample(preset_data, preprocessor, models)
    else:
        sample_item = {
            "category": args.category,
            "material": args.material,
            "special_version": args.version,
            "search_volume": args.search_volume,
            "community_hype_score": args.hype_score,
            "is_vaulted": args.is_vaulted,
            "box_condition": args.box_condition
        }
        predict_sample(sample_item, preprocessor, models)


if __name__ == "__main__":
    main()
