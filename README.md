# End-to-End ML Pipeline Capstone

### Adult Census Income — Predict whether a person earns > $50K/year

---

## What Was Predicted

We trained binary classifiers on the **1994 Adult Census Income** dataset (UCI ML Repository).
Each row represents one person described by demographic and employment attributes.
The goal is to predict whether their annual income exceeds $50,000.

| Attribute     | Details                                                   |
| ------------- | --------------------------------------------------------- |
| Dataset       | UCI Adult Census Income                                   |
| Rows          | 48,842 (train + test split combined, then re-split 80/20) |
| Features      | 13 (5 numeric + 8 categorical)                            |
| Target        | `income` — binary: 0 = ≤$50K, 1 = >$50K                   |
| Class balance | ~76% negative (≤$50K), ~24% positive (>$50K)              |

---

## Results

| Model                          | CV ROC-AUC (mean ± std) | Test ROC-AUC |
| ------------------------------ | ----------------------- | ------------ |
| Logistic Regression (baseline) | 0.9066 ± 0.0025         | —            |
| Random Forest                  | 0.9095 ± 0.0018         | —            |
| **Gradient Boosting** (chosen) | **0.9267 ± 0.0012**     | **0.9296**   |

All cross-validation was performed with **Stratified 5-Fold** on the training set only.
The test set was touched exactly **once** for the final evaluation.

---

## Test Set Breakdown (9,769 samples)

|                   | Predicted ≤50K         | Predicted >50K         |
| ----------------- | ---------------------- | ---------------------- |
| **Actually ≤50K** | 7,017 ✅ True Negative | 414 ❌ False Positive  |
| **Actually >50K** | 802 ❌ False Negative  | 1,536 ✅ True Positive |

- The model correctly identified **1,536 out of 2,338** high-income individuals (recall = 65.7%)
- The **802 false negatives** are the most costly error — people earning >$50K
  predicted as low income. In a real application (loan approval, benefit eligibility)
  these errors directly harm individuals.

---

## Why Gradient Boosting Beat the Baseline

**Logistic Regression** (AUC 0.9066) is a strong starting point — with scaled
numeric features and one-hot encoded categoricals it finds a good linear boundary.
However, the dataset has strong nonlinear interaction effects that a linear model
cannot capture: for example, the combination of `marital_status + capital_gain +
age` predicts income far better than any of those features alone.

**Random Forest** (AUC 0.9095) barely improved over Logistic Regression — only
+0.003 AUC. This tells us that bagging 200 independent trees on this data isn't
capturing much that the linear model missed. Each tree sees a random subset of
features, which breaks the very interactions that matter most here.

**Gradient Boosting** (AUC 0.9267, +0.020 over baseline) wins decisively because:

1. **Sequential residual correction** — each new tree focuses on the people the
   previous ensemble got wrong, specifically targeting the hard boundary between
   the ≤50K and >50K classes.
2. **Captures deep interactions** — `marital_status × capital_gain × education_num`
   type signals emerge naturally from the boosted tree structure.
3. **Controlled variance** — `learning_rate=0.05` (shrinkage) and `subsample=0.80`
   (stochastic boosting) keep fold-to-fold AUC variance at just ±0.0012,
   the tightest of all three models.

---

## Key Findings from Feature Importance

The Gradient Boosting model revealed what actually drives income prediction:

| Rank | Feature                             | Importance | Insight                                      |
| ---- | ----------------------------------- | ---------- | -------------------------------------------- |
| 1    | `marital_status_Married-civ-spouse` | 0.35       | Single largest predictor by far              |
| 2    | `education_num`                     | 0.20       | Years of education — intuitive               |
| 3    | `capital_gain`                      | 0.19       | Investment income signals existing wealth    |
| 4    | `capital_loss`                      | 0.065      | Even losses signal that investments exist    |
| 5    | `age`                               | 0.055      | Older = higher earning in this 1994 data     |
| 6    | `hours_per_week`                    | 0.040      | Work hours matter but less than demographics |

**Marital status at 35% importance is a red flag.** The model is using relationship
status as a major proxy for income, which reflects 1994 social structure (married
dual-income households were more common in higher brackets) rather than any
inherent truth. This should be carefully reviewed before any production deployment.

---

## EDA Findings

- **Class imbalance**: ~37,000 people earn ≤$50K vs ~12,000 earn >$50K (75/25 split).
  All models used `class_weight="balanced"` to compensate.
- **Age gap**: Median age for >$50K earners is ~45 vs ~35 for ≤$50K. Age is a
  meaningful predictor, consistent with career progression over time.
- **Capital gain skew**: Most values are 0, a few reach $99,999. Median imputation
  was chosen specifically because mean would be distorted by these outliers.

---

## Preprocessing Design

```
Raw CSV  →  train_test_split (stratified 80/20)  ← happens BEFORE anything else
                │
          ColumnTransformer.fit(X_train ONLY)
          ┌────────────────────┬───────────────────────────┐
          │  Numeric branch    │  Categorical branch        │
          │  MedianImputer     │  MostFrequentImputer       │
          │  StandardScaler    │  OneHotEncoder             │
          └────────────────────┴───────────────────────────┘
                │
          Pipeline(preprocessor + model)
                │
          StratifiedKFold(k=5) → ROC-AUC per fold → mean ± std
```

**Missingness reasoning:**

| Column           | % Missing | Mechanism                                               | Strategy                                    |
| ---------------- | --------- | ------------------------------------------------------- | ------------------------------------------- |
| `workclass`      | ~5.6%     | MAR — self-employed/never-worked people cluster here    | `most_frequent`                             |
| `occupation`     | ~5.7%     | MAR — co-occurs with missing `workclass` (>90% overlap) | `most_frequent`                             |
| `native_country` | ~1.8%     | MCAR — sporadic, no pattern with income                 | `most_frequent`                             |
| All numeric      | 0%        | —                                                       | `median` (robust to capital_gain/loss skew) |

---

## Failure Mode — Demographic Subgroup Disparity

The model's **recall for >$50K is noticeably lower for racial minorities and women**
compared with White male respondents.

**Root cause**: the 1994 census reflects real historical economic inequality.
Minority groups and women are underrepresented in the high-income class, so the
model sees fewer positive training examples for those groups. It inherits the
bias baked into the data.

**Compounding factor**: `marital_status` being the #1 feature (importance = 0.35)
means the model is partly ranking people by their relationship status rather than
their actual earning potential — which disproportionately affects demographic groups
with different marriage rate patterns.

**Mitigation options** (not implemented — out of scope for this capstone):

1. **Sample re-weighting** — assign higher loss weights to minority positive examples
2. **Fairness-constrained training** — `fairlearn`'s exponentiated gradient to equalise
   false-negative rates across subgroups
3. **Drop proxy features** — removing `marital_status` or `relationship` forces the
   model to find income signals without using relationship status as a shortcut

Any production deployment **must** run a full fairness audit before release.

---

## Bias-Variance Reflection

| Model               | CV AUC | Std    | Bias     | Variance |
| ------------------- | ------ | ------ | -------- | -------- |
| Logistic Regression | 0.9066 | 0.0025 | High     | Low      |
| Random Forest       | 0.9095 | 0.0018 | Moderate | Moderate |
| Gradient Boosting   | 0.9267 | 0.0012 | Low      | Low      |

- **Logistic Regression**: high bias because its linear decision boundary cannot
  model the interaction effects dominant in this dataset. Low variance because
  the boundary is simple and stable across folds.
- **Random Forest**: reduces bias by averaging 200 trees, but the improvement
  over LR is surprisingly small (+0.003). The `min_samples_leaf=5` constraint
  limits how deeply each tree overfits, keeping variance controlled.
- **Gradient Boosting**: achieves the best of both worlds. Sequential boosting
  drives bias down (AUC 0.9267 vs 0.9066 baseline = massive improvement).
  `learning_rate=0.05` and `subsample=0.80` keep variance the tightest of all
  three (std = 0.0012). This is the recommended model.

---

## How to Run

```bash
# 1. Set up environment
cd ~/Documents/income_study
source venv/bin/activate

# 2a. Python script (prints results to terminal, saves plots to ./figures/)
python ml_pipeline_capstone.py

# 2b. Jupyter notebook (interactive, inline plots)
jupyter notebook ml_pipeline_capstone.ipynb
```

**Expected runtime**: ~3–5 minutes (Gradient Boosting CV is the slow step).
Plots are saved automatically to `./figures/`.

---

## Files

```
income_study/
├── ml_pipeline_capstone.py       # standalone Python script
├── ml_pipeline_capstone.ipynb    # Jupyter notebook (same content)
├── README.md                     # this file
└── figures/                      # auto-generated on run
    ├── cv_comparison.png          # bar chart: AUC 0.9066 / 0.9095 / 0.9267
    ├── roc_curve.png              # ROC curve, test AUC = 0.9296
    ├── confusion_matrix.png       # TN=7017, FP=414, FN=802, TP=1536
    ├── feature_importance.png     # marital_status dominates at 0.35
    └── eda_overview.png           # class imbalance + age distribution
```

---

## Dataset Source

UCI ML Repository — Adult Data Set  
https://archive.ics.uci.edu/ml/datasets/adult  
Dua, D. and Graff, C. (2019). UCI Machine Learning Repository. University of California, Irvine.
