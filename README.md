# Random Forest Regression Tutorial Template

This repository shows how to train, evaluate, and visualize a **Random Forest Regressor** using the California Housing dataset from scikit-learn. The goal is to provide a beginner-friendly starting point that explains key ensemble concepts and offers a ready-to-run pipeline.

## Why Random Forest Regression?
- **Ensemble averaging:** A random forest trains many decision trees and averages their predictions, which reduces noise and improves stability.
- **Bagging (bootstrap sampling):** Each tree sees a slightly different bootstrap sample of the data. This decorrelates trees and lets the ensemble generalize better.
- **Random feature selection:** Trees consider a random subset of features at each split, further reducing correlation between trees.
- **Reduced overfitting:** A single decision tree can memorize the training set. Averaging many diverse trees smooths extreme predictions and typically lowers variance.

### Key hyperparameters
- `n_estimators`: number of trees to train. More trees often improve performance but take longer.
- `max_depth`: limits how deep each tree can grow. Shallower trees can reduce overfitting.
- `max_features`: number of features considered at each split. Lower values increase randomness and diversity across trees.
- `min_samples_split`: minimum samples required to split an internal node; higher values make trees more conservative.

## When to use it
Use Random Forest Regression when you want a strong baseline that handles nonlinear relationships, mixed feature scales, and some noise without heavy feature engineering.

## Dataset
The California Housing dataset contains housing prices in California from the 1990 census with features such as median income, house age, and latitude/longitude. Features are continuous; the target is median house value.

## Project structure
```
app/
  data.py          # Load dataset
  preprocess.py    # Train/test split and optional scaling
  model.py         # Build RandomForestRegressor
  evaluate.py      # Regression metrics
  visualize.py     # Feature importance and prediction plots
  main.py          # End-to-end pipeline
notebooks/
  demo_random_forest_regression.ipynb  # Narrative walkthrough
examples/
  README_examples.md  # Notes for saved plots
```

## Quickstart
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the pipeline:
   ```bash
   python app/main.py
   ```
   Metrics and SVG plots will be printed/saved under `examples/`.
3. Explore the notebook:
   ```bash
   jupyter notebook notebooks/demo_random_forest_regression.ipynb
   ```

## Pipeline overview
1. Load the California Housing dataset.
2. Split into train/test sets (scaling optional; forests do not require it).
3. Train a `RandomForestRegressor` with sensible defaults (`n_estimators=200`, `random_state=42`).
4. Evaluate with MSE, MAE, RMSE, and R².
5. Visualize feature importance and predicted-vs-actual scatter.

## Future improvements
- Hyperparameter tuning (grid search or randomized search).
- Compare against a single Decision Tree to see variance reduction.
- Enable out-of-bag (OOB) scoring to estimate performance without a held-out set.

## License
This project is licensed under the MIT License. See `LICENSE` for details.
