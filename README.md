# Application of Machine Learning Methods to Evaluate Collectable Merchandise

[![Course](https://img.shields.io/badge/Course-UE24CS352A--ML-blue.svg)](https://pes.edu)
[![Python](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-Academic-lightgrey.svg)](#)

> **Machine Learning Mini-Project Assignment**  
> **Department of Computer Science and Engineering, PES University**

---

## 👥 Team Information
- **AR Akshay Kumar** — SRN: `PES1UG24CS705`
- **Shreyas Gowda** — SRN: `PES1UG24CS704`

---

## 📌 Executive Summary
In primary retail markets, manufacturers sell collectible merchandise (e.g., Funko Pop! vinyl figurines) at flat, identical prices ($10–$15). However, in secondary trading marketplaces (such as StockX and eBay), prices diverge by orders of magnitude—spanning from **$14.00 to over $2,150.00**—governed by fandom culture, production caps, comic convention tags, and physical material finishes.

Because high-end sales are sparse and price distributions follow an extreme log-normal curve, continuous price regression produces unstable high-variance estimates. This project models secondary-market collectible valuation as an **imbalanced ordinal multi-classification problem** across 5 log-spaced price intervals:
- **Baseline Linear SVM (One-vs-One binarization)**
- **Proposed Custom Kernel SVM (Precomputed RBF + Marco's Probability-Adjusted Overlap Kernel with Domain Knowledge Weights)**
- **Vanilla Decision Tree (CART)**
- **Bagged Decision Tree Ensemble (Variance Reduction across 10–200 estimators)**
- **Random Forest Benchmark**

---

## 📂 Repository Directory Structure

```text
collectable-merchandise-ml/
├── data/
│   ├── collectables_dataset.csv       # 358 curated collectible records
│   └── data_dictionary.md            # Metadata and schema definitions
├── src/
│   ├── __init__.py
│   ├── data_loader.py                 # Data ingestion & empirical validation
│   ├── preprocessing.py              # Log transforms, MinMax, OHE & Ordinal encoders
│   ├── kernels.py                     # Custom composite kernel Gram matrix calculator
│   ├── models.py                      # Scikit-learn compliant model wrappers
│   └── evaluation.py                  # Stratified CV & weighted metric evaluation
├── scripts/
│   ├── run_experiments.py             # Reproduces all experiments and generates figures
│   └── demo_predict.py                # Interactive CLI tool for live demonstration
├── notebooks/
│   └── exploratory_data_analysis.ipynb # Interactive demonstration notebook
├── results/
│   ├── figures/                       # Generated graphs (histograms, bagging curves)
│   └── experiment_results.json        # Serialized evaluation metrics
├── report/
│   ├── generate_pdf_report.py         # Automated ReportLab script
│   └── UE24CS352A_Project_Report.pdf  # Mandatory 2-page project write-up deliverable
├── presentation/
│   ├── index.html                     # Interactive browser-based slide deck
│   ├── presentation_slides.md         # Markdown presentation outline
│   └── VIVA_QA_PREPARATION.md         # Q&A defense preparation guide
├── requirements.txt                   # Environment dependencies
├── .gitignore                         # Standard Python gitignore
└── README.md                          # Project documentation
```

---

## ⚙️ Installation & Setup Instructions

### 1. Clone the Private Repository
```bash
git clone https://github.com/<your-username>/collectable-merchandise-ml.git
cd collectable-merchandise-ml
```

### 2. Create and Activate Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate        # On macOS / Linux
# venv\Scripts\activate         # On Windows
```

### 3. Install Required Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run the Project

### A. Launch Interactive Web Valuation Dashboard (🌟 Novel Feature)
Launch our full-featured web app with live ML inference, community sentiment sliders, and condition grading:
```bash
python app.py
```
Open **`http://localhost:8080`** in your browser.

### B. Run Full Experiment Suite & Ablation Study
Executes both baseline CS229 models and our novel feature models, generating comparison tables and figures (`results/figures/ablation_study.png`):
```bash
python scripts/run_experiments.py
```

### C. Interactive Live Prediction CLI Demo (For Viva & Q&A)
Evaluate arbitrary or preset collectible items in real-time across all models:
```bash
# Run Preset 1: High-End Grail (Beerus Metallic SDCC Figure)
python scripts/demo_predict.py --preset 1

# Run Preset 2: Mid-Tier Collectible (Iron Man Glow Exclusive)
python scripts/demo_predict.py --preset 2

# Run Custom Collectible Input with Novel Features & Box Condition
python scripts/demo_predict.py --category Anime --material Metallic --version SDCC --search_volume 45000 --hype_score 9.2 --is_vaulted 1 --box_condition Mint_10
```

### D. Regenerate the Mandatory 2-Page PDF Report
Rebuilds the publication-ready PDF deliverable (`report/UE24CS352A_Project_Report.pdf`):
```bash
python report/generate_pdf_report.py
```

### E. View Presentation Slides
Open `presentation/index.html` in any web browser to present the slide deck during the review session (use `←` and `→` arrow keys to navigate slides).

---

## 🔬 Mathematical Formulation of Custom Kernel SVM

Standard RBF kernels cannot measure similarity across discrete categorical features, while linear kernels fail on non-linear price manifolds. We implement Marco's probability-adjusted categorical overlap kernel combined with continuous Gaussian RBF:

```math
K(i, j) = \sum_{l=1}^{4} w_l \cdot U_l(i, j)
```

```text
Composite Kernel Formulation:
  K(i, j) = w₁ · U₁(xᵢ, xⱼ)  +  w₂ · U₂(zᵢ, zⱼ)  +  w₃ · U₃(zᵢ, zⱼ)  +  w₄ · U₄(zᵢ, zⱼ)
```

### Component Definitions:

1. **Continuous Search Volume (`U₁`):** Gaussian RBF similarity measuring public interest:
   ```math
   U_1(x_i, x_j) = \exp\left(-\frac{(x_i - x_j)^2}{2\theta^2}\right)
   ```
   *In text notation:* `U₁(xᵢ, xⱼ) = exp( - (xᵢ - xⱼ)² / (2θ²) )` with bandwidth parameter `θ = 0.45`.

2. **Categorical Attributes (`U₂`, `U₃`, `U₄` for Category, Material, Special Version):**
   Evaluated using Marco's probability-adjusted categorical overlap kernel:
   ```math
   U_k(z_i, z_j) = \begin{cases} h(P(z_i)), & \text{if } z_i = z_j \\ 0, & \text{if } z_i \neq z_j \end{cases}
   ```
   where the rarity weighting function is defined as:
   ```math
   h(z) = (1 - z^\alpha)^{1/\alpha}, \quad 0 < \alpha < 1
   ```
   *Intuition:* When two items share a feature (`zᵢ = zⱼ`), their similarity is inversely proportional to how common that feature is. Sharing a rare convention release (e.g., `SDCC Exclusive`) yields a much higher similarity score than sharing a common finish (e.g., `Standard/Vinyl`).

3. **External Domain Knowledge Weights (`w`):**
   ```text
   w = [w_search, w_category, w_material, w_version] = [2, 1, 1, 3]
   ```
   - **Weight 2** on Search Volume (organic consumer interest & fandom demand)
   - **Weight 1** on Category and Material (baseline genre & physical medium)
   - **Weight 3** on Special Version (San Diego Comic Con / Chase scarcity multiplier)

---

## 📊 Empirical Benchmarks & Results

| Model Architecture | Train F1 | Test F1 | Test Prec | Test Rec | Bin 1 Prec | Bin 2 Prec | Bin 3 Prec | Bin 4 Prec | Bin 5 Prec |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline Linear SVM (OvO)** | 0.717 | 0.660 | 0.668 | 0.667 | 0.778 | 0.649 | 0.600 | **0.000** | **0.000** |
| **Custom Kernel SVM (Proposed)** | 0.741 | **0.664** | **0.670** | 0.667 | 0.792 | 0.639 | 0.500 | **0.500** | 0.000 |
| **Vanilla Decision Tree (CART)** | 1.000 | 0.595 | 0.604 | 0.597 | 0.696 | 0.571 | 0.545 | 0.333 | 0.000 |
| **Bagged Decision Trees (B=100)** | 0.965 | **0.661** | **0.673** | 0.667 | 0.773 | 0.639 | 0.545 | **0.667** | 0.000 |
| **Random Forest Benchmark** | 1.000 | 0.571 | 0.614 | 0.583 | 0.750 | 0.535 | 0.545 | 0.500 | 0.000 |

### Key Findings:
1. **Linear SVM Blindness:** Fails completely on rare classes (0% on Bins 4 & 5) due to majority class dominance.
2. **Custom Kernel Superiority:** Elevates Bin 4 grail precision from 0% to **50.0%** by penalizing common feature overlaps and rewarding rare attribute co-occurrences.
3. **Ensemble Variance Reduction:** Bagging resolves the severe overfitting of single trees (boosting test F1 from 0.595 to **0.661** and achieving **66.7%** precision on Bin 4).

---

## 🔬 Ablation Study: Impact of Our Novel Features

To measure the empirical contribution of our proposed features (**Community Hype Score** and **Vaulted / Discontinued Status**), we conducted an ablation study comparing the original CS229 model against our enhanced model:

| Evaluation Metric | Original CS229 (4 Features) | Our Team's Extension (Novel Features) | Absolute Improvement |
| :--- | :---: | :---: | :---: |
| **Custom Kernel SVM (Weighted F1)** | 0.674 | **0.717** | **+4.3%** |
| **Custom Kernel SVM (Precision)** | 0.672 | **0.714** | **+4.2%** |
| **Bagged Decision Trees (Weighted F1)** | 0.673 | **0.842** | **+16.9%** |
| **Bin 4 Grail Precision (Custom SVM)** | 0.500 | **0.667** | **+16.7%** |
| **Bin 4 Grail Precision (Bagged)** | 0.500 | **0.667** | **+16.7%** |

*Takeaway:* Integrating collector sentiment and manufacturer supply freezes directly addresses the primary challenge identified in the original Stanford CS229 study: capturing demand spikes on scarce, high-tier collectibles.

---

## 📖 Deliverables Checklist
- [x] **Source Code in Git Repository:** Clean, modular structure with `data/`, `src/`, `scripts/`, `notebooks/`.
- [x] **README Instructions:** Detailed guide for environment setup, reproduction, and execution.
- [x] **Two-Page Write-Up PDF:** Conforms to assignment requirements (`report/UE24CS352A_Project_Report.pdf`).
- [x] **Presentation Slide Deck:** Prepared in Markdown and interactive HTML (`presentation/index.html`).
- [x] **Viva Defense Preparation:** Technical Q&A document (`presentation/VIVA_QA_PREPARATION.md`).

---

## 📜 References
1. Xiaotian Lu. *Application of Machine Learning Methods to Evaluate Collectibles*. Stanford CS229 Machine Learning Project (Spring 2021). [Report Link](https://cs229.stanford.edu/proj2021spr/report2/81980743.pdf).
2. L. Powell, A. Gelich, and Z. W. Ras. *Developing artwork pricing models for online art sales using text analytics*. International Joint Conference on Rough Sets (2019).
3. Marco A. *An investigation into new kernels for categorical variables* (2013).
