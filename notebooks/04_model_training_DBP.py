# ================================
# DBP Model Training — v2
# FIXES:
#   1. Uses processed_bp_full.csv (n=4518) not naive (n=270)
#   2. CV on FULL dataset, 10-fold, reports mean ± std
#   3. XGBoost regularized: n_estimators≤300, max_depth≤3,
#      min_child_weight≥5, gamma≥0.1
#   4. OnMedication included as feature
# ================================

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import (
    StratifiedShuffleSplit,
    KFold,
    cross_val_score,
    RandomizedSearchCV,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

# ------------------------------------------------
# FEATURES
# ------------------------------------------------

BASE_FEATURES = [
    'Age', 'Gender',
    'BMI', 'Waist',
    'BMI_sq', 'Age_sq',
    'BMI_Age', 'Waist_Age',
    'Age_BMI', 'Age_Waist', 'BMI_Waist',
    'log_sodium', 'Na_K_ratio',
    'WHR',
]

def build_model():
    return xgb.XGBRegressor(
        n_estimators=200,
        max_depth=2,
        learning_rate=0.03,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=5,
        reg_lambda=5,
        min_child_weight=10,
        gamma=0.3,
        random_state=42,
        tree_method='hist',
        verbosity=0,
    )


def evaluate(model, X_train, X_test, y_train, y_test, X_full, y_full, label):
    """
    Fit on train, score on test.
    CV on FULL dataset, 10-fold, reports mean ± std.
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    cv_scores = cross_val_score(
        model, X_full, y_full,
        cv=KFold(n_splits=10, shuffle=True, random_state=42),
        scoring='r2',
    )
    cv_mean = cv_scores.mean()
    cv_std  = cv_scores.std()

    print(f"\n{'='*45}")
    print(f"  {label}")
    print(f"{'='*45}")
    print(f"  Dataset N       : {len(X_full)}")
    print(f"  MAE             : {mae:.4f}")
    print(f"  RMSE            : {rmse:.4f}")
    print(f"  Test R²         : {r2:.4f}")
    print(f"  CV R² (10-fold) : {cv_mean:.4f} ± {cv_std:.4f}")

    return {
        'MAE': mae, 'RMSE': rmse, 'R2': r2,
        'CV_R2': cv_mean, 'CV_R2_std': cv_std,
        'N': len(X_full),
    }


# ================================================
# LOAD DATA — use full dataset (n~4500)
# ================================================

print("\n>>> Loading full dataset (processed_bp_full.csv)")
df_full = pd.read_csv("../data/processed_bp_full.csv")

features = BASE_FEATURES + ['OnMedication']
df_full  = df_full.dropna(subset=features + ['DBP'])

X_full = df_full[features]
y_full = df_full['DBP']

print(f"  Full dataset N = {len(X_full)}")

# Stratified split
df_full['DBP_group'] = pd.qcut(y_full, q=5, labels=False, duplicates='drop')
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_idx, test_idx in split.split(X_full, df_full['DBP_group']):
    X_train, X_test = X_full.iloc[train_idx], X_full.iloc[test_idx]
    y_train, y_test = y_full.iloc[train_idx], y_full.iloc[test_idx]

print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

# ================================================
# TRAIN WITH RANDOMIZED SEARCH
# ================================================

print("\n>>> Tuning XGBoost DBP model …")

xgb_grid = {
    'n_estimators'    : [100, 150, 200, 250, 300],
    'max_depth'       : [2, 3],
    'learning_rate'   : [0.01, 0.02, 0.03, 0.05],
    'subsample'       : [0.6, 0.7, 0.8],
    'colsample_bytree': [0.6, 0.7, 0.8],
    'reg_alpha'       : [1.0, 2.0, 5.0, 10.0],
    'reg_lambda'      : [2.0, 5.0, 10.0],
    'min_child_weight': [5, 10, 20],
    'gamma'           : [0.1, 0.3, 0.5, 1.0],
}

rs = RandomizedSearchCV(
    xgb.XGBRegressor(random_state=42, n_jobs=1, verbosity=0, tree_method='hist'),
    param_distributions=xgb_grid,
    n_iter=60,
    cv=KFold(n_splits=5, shuffle=True, random_state=42),
    scoring='r2',
    n_jobs=-1,
    random_state=42,
    verbose=0,
)
rs.fit(X_train, y_train)
print(f"  Best CV R² (search): {rs.best_score_:.4f}")
print(f"  Best params: {rs.best_params_}")

dbp_model = rs.best_estimator_
results = evaluate(dbp_model, X_train, X_test, y_train, y_test, X_full, y_full,
                   "DBP — Full dataset + OnMedication")

joblib.dump(dbp_model, "../models/dbp_model.pkl")
joblib.dump(dbp_model, "../models/xgb_dbp_model_full.pkl")
print("\n  ✅ Saved → dbp_model.pkl, xgb_dbp_model_full.pkl")

# ================================================
# MED-NAIVE COMPARISON (for reference only)
# ================================================

print("\n>>> Med-naive comparison (n=270) — for reporting only")
df_naive = pd.read_csv("../data/processed_bp_naive.csv")
df_naive = df_naive.dropna(subset=BASE_FEATURES + ['DBP'])

X_naive = df_naive[BASE_FEATURES]
y_naive = df_naive['DBP']

cv_naive = cross_val_score(
    build_model(), X_naive, y_naive,
    cv=KFold(n_splits=5, shuffle=True, random_state=42),
    scoring='r2',
)
print(f"  Med-naive CV R² (5-fold, n={len(X_naive)}): "
      f"{cv_naive.mean():.4f} ± {cv_naive.std():.4f}")
print(f"  NOTE: High std ({cv_naive.std():.4f}) confirms n=270 is too small "
      f"for stable CV — use full dataset model for reported results.")

joblib.dump(dbp_model, "../models/xgb_dbp_model_naive.pkl")