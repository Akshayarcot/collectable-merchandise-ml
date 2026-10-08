# Application of Machine Learning Methods to Evaluate Collectable Merchandise
**Course:** UE24CS352A — Machine Learning Mini-Project  
**Institution:** PES University, Department of Computer Science and Engineering  
**Team Members:**
- **AR Akshay Kumar** (SRN: `PES1UG24CS705`)
- **Shreyas Gowda** (SRN: `PES1UG24CS704`)

---

## Slide 1: Title & Overview
- **Title:** Application of Machine Learning Methods to Evaluate Collectable Merchandise
- **Problem Domain:** Quantitative Finance, E-commerce, & Machine Learning
- **Target Merchandise:** Pop-Culture Vinyl Collectibles (Funko Pop! secondary marketplace)
- **Objective:** Predict secondary-market price valuation tiers under extreme price skewness and transactional illiquidity.

---

## Slide 2: Problem Statement & Motivation
- **Primary vs. Secondary Market Disconnect:**
  - Manufacturers release merchandise at flat retail pricing ($10 - $15).
  - Secondary markets (StockX, eBay, Grail platforms) exhibit dramatic valuation divergence ($14 to $2,150+).
- **Core ML Challenges:**
  - **Severe Transaction Scarcity:** Limited sales records for top-tier collectibles.
  - **Extreme Lognormal Skewness:** Heavy right tail makes continuous regression prone to high variance.
  - **Heterogeneous Mixed Features:** Continuous demand proxies (search interest) combined with sparse categorical tags (conventions, materials).
- **Formulation:** Multi-class classification into 5 log-scaled price intervals.

---

## Slide 3: Dataset Architecture & Features
- **Dataset Size:** 358 curated collectible items across 5 core fandom categories:
  - *Anime:* 198 items (55.3%)
  - *Marvel:* 69 items (19.3%)
  - *Ad-Icons:* 43 items (12.0%)
  - *Disney:* 37 items (10.3%)
  - *Game-of-Thrones:* 11 items (3.1%)
- **Target Price Bins (Log-spaced intervals):**
  - **Bin 1:** `[$14.00, $38.13]` — 146 items (Common / Retail)
  - **Bin 2:** `($38.13, $104.87]` — 153 items (Modest Premium)
  - **Bin 3:** `($104.87, $287.02]` — 46 items (Con Exclusive)
  - **Bin 4:** `($287.02, $785.55]` — 10 items (Grail Tier)
  - **Bin 5:** `($785.55, $2150.00]` — 3 items (Ultra-Rare Museum Tier)
- **Engineered Features:**
  - *Continuous:* Search Volume (Google Trends / Ahrefs) normalized via $\log(1 + x)$ and Min-Max scaling to $[0, 1]$.
  - *Categorical:* Category, Material (Standard, Metallic, Flocked, Glow, Diamond, Chrome), Special Version (Standard, SDCC, NYCC, Chase, Vaulted).

---

## Slide 4: Proposed Technical Methodology
We evaluate 4 model families designed to explore the bias-variance tradeoff and kernel behavior:

1. **Baseline Linear SVM (One-vs-One):**
   - Pairwise binarization to mitigate class imbalance effects.
   - Evaluates linear separability in one-hot encoded space.
2. **Proposed Custom Kernel SVM:**
   - Standard RBF fails on categorical data; standard linear fails on non-linear boundaries.
   - We construct a hybrid composite kernel:
     $$K(i, j) = \sum_{l=1}^4 w_l \cdot U_l(i, j)$$
     - $U_1$: Continuous Gaussian RBF kernel on search interest.
     - $U_2, U_3, U_4$: Marco's probability-adjusted categorical overlap kernel:
       $$U_k(z_i, z_j) = h(P(z_i)) \cdot \mathbb{I}(z_i = z_j), \quad h(z) = (1 - z^\alpha)^{1/\alpha}$$
     - External domain weights: $w = [2, 1, 1, 3]$ (emphasizing search volume & convention rarity).
3. **Vanilla Decision Tree (CART):**
   - Non-parametric recursive splitting using Gini impurity.
4. **Bagged Decision Tree Ensemble:**
   - Bootstrap aggregating over $B \in [10, 200]$ trees to slash high variance.

---

## Slide 5: Experimental Results & Benchmarks

| Model Architecture | Train F1 | Test F1 | Test Prec | Bin 1 Prec | Bin 2 Prec | Bin 3 Prec | Bin 4 Prec | Bin 5 Prec |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline Linear SVM** | 0.717 | 0.660 | 0.668 | 0.778 | 0.649 | 0.600 | **0.000** | **0.000** |
| **Custom Kernel SVM (Ours)** | 0.741 | **0.664** | **0.670** | 0.792 | 0.639 | 0.500 | **0.500** | 0.000 |
| **Vanilla Decision Tree** | 1.000 | 0.595 | 0.604 | 0.696 | 0.571 | 0.545 | 0.333 | 0.000 |
| **Bagged Decision Trees (B=100)**| 0.965 | **0.661** | **0.673** | 0.773 | 0.639 | 0.545 | **0.667** | 0.000 |
| **Random Forest (100 Trees)** | 1.000 | 0.571 | 0.614 | 0.750 | 0.535 | 0.545 | 0.500 | 0.000 |

---

## Slide 6: Key Insights & Discussion
1. **Linear SVM Blindness to High-Tier Grails:**
   - Linear SVM achieves 0% precision on Bins 4 & 5 because linear hyperplanes in one-hot space are overwhelmed by the 83% majority samples.
2. **Domain-Weighted Custom Kernel Advantage:**
   - By penalizing common feature matches and amplifying rare feature matches (e.g. SDCC + Metallic), Custom Kernel SVM lifts Bin 4 precision from 0% to **50.0%**.
3. **Variance Reduction via Bagging:**
   - Single decision tree severely overfits (100% train F1 vs. 59.5% test F1).
   - Bagging stabilizes boundaries, boosting test F1 to **66.1%** and achieving **66.7%** precision on Bin 4 with rapid convergence at 50-100 trees.

---

## Slide 7: Live System Demonstration
- **Interactive Prediction Tool (`scripts/demo_predict.py`):**
  - Live query: Input Category, Material, Special Edition, and Monthly Search Volume.
  - Multi-model evaluation and real-time price tier estimation.
- **Reproducibility:**
  - Automated experiment suite (`scripts/run_experiments.py`).
  - Automated PDF report generator (`report/generate_pdf_report.py`).

---

## Slide 8: Conclusions & Future Work
- **Summary:**
  - Designed an end-to-end ML valuation pipeline tailored for illiquid collectible markets.
  - Validated that domain-weighted custom kernels and bagged ensembles dramatically outperform standard baselines on rare high-value tiers.
- **Future Directions:**
  - **NLP:** Extract collector sentiment from Reddit (`r/funkopop`) discussions to detect emerging trends before sales occur.
  - **Multimodal Computer Vision:** ResNet/ViT feature extractors on box condition and hologram authenticity to detect counterfeits.

---

## Slide 9: Q&A & Defense
- Thank you!
- Open for questions from the evaluation panel.
