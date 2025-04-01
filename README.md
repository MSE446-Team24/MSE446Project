# MSE446 Final Project – Predicting NBA Game Outcomes

**Team 24 — Winter 2025**  
This project explores the use of various machine learning models and feature selection strategies to predict the outcome of NBA games using a publicly available dataset.

---

## Repository Structure

| Folder/File                      | Description |
|----------------------------------|-------------|
| `data/`                          | Raw datasets (from Kaggle) and associated files. |
| `engineered_data/`              | Cleaned and feature-engineered datasets used for modeling. |
| `selected_features/`            | Saved output from feature selection methods (CSV/PKL). |
| `results/`                      | Saved modeling predictions, plots, metrics. |
| `figures/`                      | Final plots and visualizations for presentation/report. |
| `model_results/`                | (Deprecated) Previous results, not used in final version. |

| Notebook                         | Purpose |
|----------------------------------|---------|
| `DataExtraction.ipynb`           | Loads and preprocesses raw Kaggle data. |
| `FeatureEngineering.ipynb`       | Creates features including ratios, binned stats, interaction terms. |
| `FeatureSelection.ipynb`         | Applies and compares multiple feature selection techniques: ANOVA F-test, mutual information, PCA, RFE, and more. |
| `Modeling_*.ipynb`               | Five modeling notebooks, each based on a different feature set:<br>• `MutualInfo`<br>• `ANOVA_Top_30`<br>• `PCA_Transformed`<br>• `RFE_Filtered`<br>• `Final_Model_Evaluation.ipynb` compares them all. |
| `Final-Model-Evaluation.ipynb`   | Aggregates and visualizes results across all feature sets, generates ROC plots and average metrics. |

---

## Dataset

- **Source**: [Kaggle - NBA Games Dataset](https://www.kaggle.com/datasets/nathanlauga/nba-games)  
- **Rows**: 26,651 NBA games  
- **Seasons**: 2004 to December 2020  
- **Main File Used**: `games.csv`  
- **Target**: `HOME_TEAM_WINS` (binary: 1 if home team won)

---

## Project Objective

To predict whether the **home team will win** an NBA game, using:
- **Statistical features** (e.g., shooting %, rebounds)
- **Ratios and binned features**
- **Selected top features from multiple selection strategies**
- **Various ML models** including tree-based methods, logistic regression, neural networks, and clustering

---

## Models Compared

- Logistic Regression  
- Decision Tree  
- Random Forest  
- Gradient Boosting  
- LightGBM  
- XGBoost  
- KMeans Clustering  
- Neural Network (PyTorch)

Each model is evaluated across multiple feature sets with consistent cross-validation.

---

## Feature Selection Sets

| Feature Set | Description |
|-------------|-------------|
| `anova_top_30`         | Top 30 features selected via ANOVA F-test. |
| `mutual_info_filtered` | Features selected based on mutual information with the target. |
| `pca_transformed`      | Top 30 PCA components explaining variance (unsupervised). |
| `rfe_filtered`         | Features selected via Recursive Feature Elimination using Logistic Regression. |

---

## Metrics Evaluated

- Accuracy  
- F1 Score  
- AUC (Area Under ROC Curve)  
- Cross-validation standard deviation  
- Training time

---

## Results Location

- Final per-model results: `final_model_comparison.csv`
- Intermediate CSVs: `anova_model_results.csv`, etc.
- Visualizations in: `figures/` and within notebooks

---


## Reproducibility Notes

- Python 3.11  
- Jupyter Notebooks  
- `scikit-learn`, `xgboost`, `lightgbm`, `pytorch`, `seaborn`, `matplotlib`, etc.  
- All datasets and results are local (no API dependency)

---

## Conclusion

This project demonstrates how feature selection influences model performance in sports analytics, with interpretable models like logistic regression often outperforming complex architectures depending on the feature space.

