# Viva & Review Defense Preparation Guide
**Course:** UE24CS352A — Machine Learning Mini-Project  
**Topic:** Application of Machine Learning Methods to Evaluate Collectable Merchandise  
**Authors:** AR Akshay Kumar (`PES1UG24CS705`), Shreyas Gowda (`PES1UG24CS704`)

This document prepares you for all anticipated technical and theoretical questions from the faculty review panel.

---

### Q1: Why formulate price evaluation as a classification problem instead of direct regression?
**Answer:**
- **Extreme Variance & Lognormal Skew:** Secondary market collectible prices range from \$14 to \$2,150. Most items cluster around \$15–\$40, while a tiny handful of "grail" items sell for thousands. Continuous regression (e.g., MSE loss) penalizes large dollar errors quadratically, causing the model to over-index on outliers or produce volatile predictions.
- **Sparse Transactions:** High-end items transact very infrequently. Continuous price regression requires rich continuous trading volume, which does not exist for rare collectibles.
- **Commercial Practicality:** In real-world secondary market platforms (e.g., StockX, eBay), sellers and buyers evaluate goods in discrete valuation brackets (e.g., "Common Retail", "Moderate Demand", "Convention Grail", "Museum Piece") rather than predicting exact cents.

---

### Q2: Why is severe class imbalance a problem for standard SVM, and why did you use One-vs-One (OvO) binarization?
**Answer:**
- **The Imbalance Challenge:** In our dataset, Bins 1 and 2 account for ~83% of all items (146 and 153 items), while Bins 4 and 5 represent only 10 and 3 items respectively. Standard classifiers maximize overall accuracy by simply predicting the majority classes, leading to **0.000 precision and recall on high-value items**.
- **OvO vs. OvR (One-vs-Rest):**
  - In *One-vs-Rest*, a classifier separating Bin 5 against the rest trains on 3 positive vs. 355 negative samples (a 1:118 ratio). The majority class completely dominates the soft-margin slack penalty $\sum \xi_i$.
  - In *One-vs-One*, $K(K-1)/2 = 10$ binary classifiers are trained only on pairs of classes. When training the Bin 4 vs. Bin 5 classifier, only those 13 samples are used, keeping pairwise sample sizes balanced and preventing the common classes from drowning out rare class boundaries.

---

### Q3: Explain the mathematical formulation of your Custom Kernel. How does it handle categorical features?
**Answer:**
Standard RBF kernels compute Euclidean distances: $\|x_i - x_j\|^2$. While valid for continuous features like search volume, Euclidean distance has no natural geometric meaning for discrete nominal categories (e.g., Anime vs Marvel, or Metallic vs Flocked).
We implement a hybrid composite kernel:
$$K(i, j) = \sum_{l=1}^4 w_l \cdot U_l(i, j)$$
1. **Continuous Feature ($U_1$ - Search Volume):** Gaussian RBF similarity:
   $$U_1(x_i, x_j) = \exp\left(-\frac{(x_i - x_j)^2}{2\theta^2}\right)$$
2. **Categorical Features ($U_2, U_3, U_4$ - Category, Material, Special Version):** Marco's probability-adjusted categorical overlap kernel:
   $$U_k(z_i, z_j) = h(P(z_i)) \cdot \mathbb{I}(z_i = z_j)$$
   where $h(z) = (1 - z^\alpha)^{1/\alpha}$ for $0 < \alpha < 1$.
- **Why this function?** If two items share a feature ($z_i = z_j$), their similarity score is inversely related to the marginal probability $P(z_i)$ of that feature.
  - If two items share a *very common* attribute (e.g., `Standard/Vinyl`, $P \approx 0.75$), $h(P(z))$ is small.
  - If two items share an *exceptionally rare* attribute (e.g., `SDCC Exclusive`, $P \approx 0.05$), $h(P(z))$ is large.
  - If they do not match ($z_i \neq z_j$), similarity is 0.

---

### Q4: How were the domain weights $w = [2, 1, 1, 3]$ chosen?
**Answer:**
- In collectible merchandise valuation, domain knowledge indicates that **Special Version / Exclusivity** (e.g. San Diego Comic Con, Chase) and **Fandom Search Interest** are the predominant drivers of secondary market price explosions.
- Weight assignment:
  - $w_1 = 2$: Search Volume (quantifies organic demand).
  - $w_2 = 1$: Category (baseline fandom genre).
  - $w_3 = 1$: Material (finish variant).
  - $w_4 = 3$: Special Version (convention release / production cap).
- Giving $w_4 = 3$ ensures that items sharing rare convention tags have high kernel similarity, allowing the SVM to successfully identify Bin 4 items (boosting test precision from 0% in Linear SVM to 50.0% in Custom SVM).

---

### Q5: Why did the Vanilla Decision Tree achieve 1.000 F1 on Train but only 0.595 on Test?
**Answer:**
- Decision trees are non-parametric estimators that partition feature space recursively. Without pruning or max-depth constraints, a tree splits until leaves are pure ($Gini = 0$).
- Because of sample sparsity and rare combinations, the tree memorizes individual training samples, leading to **high variance and severe overfitting**.

---

### Q6: How does Bagged Decision Trees solve this, and why did it perform so well?
**Answer:**
- **Bootstrap Aggregating (Bagging):** Creates $B$ bootstrap datasets by sampling $N$ samples with replacement from the training set, trains an independent decision tree on each, and aggregates their predictions via majority voting.
- **Bias-Variance Decomposition:**
  $$\text{Var}(\bar{f}) = \rho \sigma^2 + \frac{1 - \rho}{B} \sigma^2$$
  As the number of trees $B$ increases, the second term $\frac{1 - \rho}{B}\sigma^2 \to 0$. By averaging over $B=100$ trees, the individual tree variance is drastically reduced without increasing bias.
- **Empirical Confirmation:** In our experiment, Bagging raised validation F1 from 0.595 to **0.661**, and achieved **0.667 precision on Bin 4**.

---

### Q7: Why did Random Forest have slightly lower test F1 (0.571) than Bagged Decision Trees (0.661)?
**Answer:**
- In Random Forest, at each split node, only a random subset of features (typically $\sqrt{p}$) is considered.
- In our dataset, we have only 4 primary features (Search Volume, Category, Material, Version), where Search Volume and Version are heavily dominant. Subsampling features often forces nodes to split on less informative features, weakening the tree's split quality.
- Standard Bagging uses all features at every split, which is superior when the feature dimensionality is small and specific features are critical.

---

### Q8: What metrics did you use to evaluate, and why?
**Answer:**
- We tracked **Weighted Precision, Weighted Recall, and Weighted F1-score**, along with **Per-Class Precision across Bins 1 through 5**.
- **Why not standard Accuracy?** With 83% of samples residing in Bins 1 and 2, a dummy classifier predicting only Bins 1 and 2 would achieve >80% accuracy while failing completely on all collectibles worth over \$105.
- Weighted F1 weights each class's score by its support, and per-class precision provides explicit visibility into whether the model can accurately spot rare high-value collectibles.
