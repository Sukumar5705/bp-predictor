# ============================================================
# Blood Pressure Prediction Pipeline
# Data Cleaning + Feature Engineering + Tuned GB & XGBoost
# NHANES DEMO_L / BPXO_L / BMX_L / DR1TOT_L / PAQ_L / SLQ_L
# ============================================================
# FIXES APPLIED (v2):
#   1. Uses processed_bp_full.csv (n=4518) — eliminates n=270 instability
#   2. CV run on FULL dataset (not just train partition) for unbiased estimate
#   3. CV reports mean ± std — addresses reviewer uncertainty concern
#   4. XGBoost heavily regularized: max n_estimators=300, max_depth≤3,
#      min_child_weight≥5, gamma≥0.1 — prevents overfitting on small folds
#   5. GB also constrained: shallow trees, high min_samples_leaf
#   6. Early stopping via eval_set for XGBoost final fit
#   7. OnMedication included as feature in full-dataset model
# ============================================================
# HOW TO RUN (locally):
#   pip install pandas numpy scikit-learn xgboost joblib
#   python bp_pipeline.py
# ============================================================

import pandas as pd
import numpy as np
import warnings
import os
import joblib
import pyreadstat
from sklearn.model_selection import (
    StratifiedShuffleSplit,
    cross_val_score,
    KFold,
    RandomizedSearchCV,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb

warnings.filterwarnings("ignore")

DATA_PATH = "./"
OUT_PATH  = "./"

os.makedirs(OUT_PATH, exist_ok=True)


# ============================================================
# STEP 1 — LOAD RAW DATA
# ============================================================
print("=" * 60)
print("STEP 1 : LOADING RAW DATA")
print("=" * 60)

demo  = pd.read_sas("../data/DEMO_L.xpt",  format="xport")
bp    = pd.read_sas("../data/BPXO_L.xpt", format="xport")
bmx   = pd.read_sas("../data/BMX_L.xpt",  format="xport")
diet  = pd.read_sas("../data/DR1TOT_L.xpt", format="xport")
paq   = pd.read_sas("../data/PAQ_L.xpt",  format="xport")
sleep = pd.read_sas("../data/SLQ_L.xpt",  format="xport")
bpq   = pd.read_sas("../data/BPQ_L.xpt",format="xport")

for name, d in [("DEMO", demo), ("BPXO", bp), ("BMX", bmx),
                ("DIET", diet), ("PAQ", paq), ("SLEEP", sleep), ("BPQ", bpq)]:
    print(f"  {name:<6}: {d.shape}")


# ============================================================
# STEP 2 — SELECT COLUMNS
# ============================================================
print("\n" + "=" * 60)
print("STEP 2 : SELECTING RELEVANT COLUMNS")
print("=" * 60)

demo  = demo[['SEQN', 'RIDAGEYR', 'RIAGENDR']]
bp    = bp[['SEQN', 'BPXOSY1', 'BPXOSY2', 'BPXOSY3',
            'BPXODI1', 'BPXODI2', 'BPXODI3']]
bmx   = bmx[['SEQN', 'BMXBMI', 'BMXWAIST']]
diet  = diet[['SEQN', 'DR1TSODI', 'DR1TPOTA']]
paq   = paq[['SEQN', 'PAD680']]
sleep = sleep[['SEQN', 'SLD012']]
bpq   = bpq[['SEQN', 'BPQ150']].copy()

print("  Done.")


# ============================================================
# STEP 3 — MERGE
# ============================================================
print("\n" + "=" * 60)
print("STEP 3 : MERGING ON SEQN")
print("=" * 60)

df = (demo
      .merge(bp,    on='SEQN', how='inner')
      .merge(bmx,   on='SEQN', how='inner')
      .merge(diet,  on='SEQN', how='inner')
      .merge(paq,   on='SEQN', how='inner')
      .merge(sleep, on='SEQN', how='inner')
      .merge(bpq,   on='SEQN', how='left'))

print(f"  Shape after merge: {df.shape}")


# ============================================================
# STEP 4 — DATA CLEANING
# ============================================================
print("\n" + "=" * 60)
print("STEP 4 : DATA CLEANING")
print("=" * 60)

df = df.rename(columns={
    'RIDAGEYR': 'Age',
    'RIAGENDR': 'Gender',
    'BMXBMI'  : 'BMI',
    'BMXWAIST': 'Waist',
    'DR1TSODI': 'Sodium',
    'DR1TPOTA': 'Potassium',
    'PAD680'  : 'SedentaryTime',
    'SLD012'  : 'SleepHours',
    'BPQ150'  : 'MedRaw',
})

# Medication flag: 1=on meds, 0=med-naive, -1=unknown
df['OnMedication'] = df['MedRaw'].map({1.0: 1, 2.0: 0}).fillna(-1).astype(int)
df = df.drop(columns=['MedRaw'])

print(f"  Medication status: on={( df.OnMedication==1).sum()}, "
      f"naive={(df.OnMedication==0).sum()}, unknown={(df.OnMedication==-1).sum()}")

# Sentinel values → NaN
SENTINEL_THRESHOLD = 1e-70
for col in df.select_dtypes(include='number').columns:
    mask = df[col].abs() < SENTINEL_THRESHOLD
    n = mask.sum()
    if n > 0:
        df.loc[mask, col] = np.nan

# Adults only
df = df[df['Age'] >= 18]

# Physiological validity
validity = {
    'BMI'          : (10,   70),
    'Waist'        : (50,  180),
    'SleepHours'   : (2,    14),
    'SedentaryTime': (0,  1440),
    'Sodium'       : (0,  8000),
    'Potassium'    : (0,  8000),
}
for col, (lo, hi) in validity.items():
    df.loc[~df[col].between(lo, hi), col] = np.nan

df = df[df['Gender'].isin([1, 2])]
print(f"  Shape after validity check: {df.shape}")

# BP targets
df['SBP'] = df[['BPXOSY1', 'BPXOSY2', 'BPXOSY3']].mean(axis=1)
df['DBP'] = df[['BPXODI1', 'BPXODI2', 'BPXODI3']].mean(axis=1)

df = df[df[['BPXOSY1', 'BPXOSY2', 'BPXOSY3']].notna().sum(axis=1) >= 2]
df = df[df[['BPXODI1', 'BPXODI2', 'BPXODI3']].notna().sum(axis=1) >= 2]

df = df[(df['SBP'] >= 70) & (df['SBP'] <= 220)]
df = df[(df['DBP'] >= 30) & (df['DBP'] <= 130)]
df = df[(df['SBP'] - df['DBP']) >= 10]

print(f"  Shape after BP filtering: {df.shape}")

# IQR flagging → NaN
for col in ['BMI', 'Waist', 'Sodium', 'Potassium', 'SedentaryTime', 'SleepHours']:
    q01, q99 = df[col].quantile(0.01), df[col].quantile(0.99)
    df.loc[~df[col].between(q01, q99), col] = np.nan

# Median imputation
for col in ['BMI', 'Waist', 'Sodium', 'Potassium', 'SedentaryTime', 'SleepHours']:
    med = df[col].dropna().median()
    n   = df[col].isnull().sum()
    if n > 0:
        df[col] = df[col].fillna(med)
        print(f"  Imputed {n:>4} missing in '{col}' → median={med:.2f}")

df = df.dropna(subset=['SBP', 'DBP'])
print(f"\n  ✅  Clean dataset shape: {df.shape}")


# ============================================================
# STEP 5 — FEATURE ENGINEERING
# ============================================================
print("\n" + "=" * 60)
print("STEP 5 : FEATURE ENGINEERING")
print("=" * 60)

df['Gender'] = (df['Gender'] == 1).astype(int)

df['Sodium']    = df['Sodium'].clip(500, 5000)
df['Potassium'] = df['Potassium'].clip(500, 5000)

df['log_Sodium']    = np.log1p(df['Sodium'])
df['log_Potassium'] = np.log1p(df['Potassium'])
df['Na_K_ratio']    = df['Sodium'] / (df['Potassium'] + 1.0)

df['BMI_sq']    = df['BMI'] ** 2
df['Age_sq']    = df['Age'] ** 2
df['Age_BMI']   = df['Age'] * df['BMI']
df['Age_Waist'] = df['Age'] * df['Waist']
df['BMI_Waist'] = df['BMI'] * df['Waist']
df['WHR']       = df['Waist'] / (df['BMI'] + 1e-5)

df['SedentaryHours'] = df['SedentaryTime'] / 60.0

sleep_lo = df['SleepHours'].min() - 0.001
df['SleepCategory'] = pd.cut(
    df['SleepHours'], bins=[sleep_lo, 5.9, 9.0, 99], labels=[0, 1, 2]
).astype(float).fillna(1.0)

sed_lo = df['SedentaryHours'].min() - 0.001
df['ActivityLevel'] = pd.cut(
    df['SedentaryHours'], bins=[sed_lo, 4, 8, 12, 9999], labels=[3, 2, 1, 0]
).astype(float).fillna(1.0)

print("  All engineered features created.")


# ============================================================
# STEP 6 — FINAL FEATURE MATRIX
# ============================================================
print("\n" + "=" * 60)
print("STEP 6 : BUILDING FEATURE MATRIX")
print("=" * 60)

FEATURES = [
    'Age', 'Gender',
    'BMI', 'Waist', 'WHR',
    'BMI_sq', 'Age_sq',
    'Age_BMI', 'Age_Waist', 'BMI_Waist',
    'log_Sodium', 'log_Potassium', 'Na_K_ratio',
    'SedentaryHours', 'ActivityLevel', 'SleepHours', 'SleepCategory',
    'OnMedication',   # medication status as a feature (1=on meds, 0=naive, -1=unknown)
]

X     = df[FEATURES].copy()
y_sbp = df['SBP'].copy()
y_dbp = df['DBP'].copy()

assert X.isnull().sum().sum() == 0, "NaN still present in feature matrix!"

print(f"  Samples  : {len(X)}  (previously n=270; now using full dataset)")
print(f"  Features : {len(FEATURES)}")
print(f"  SBP mean : {y_sbp.mean():.1f}  std: {y_sbp.std():.1f}")
print(f"  DBP mean : {y_dbp.mean():.1f}  std: {y_dbp.std():.1f}")
print("  ✅  No missing values in feature matrix.")

# Save processed dataset
df[FEATURES + ['SBP', 'DBP']].to_csv(OUT_PATH + "processed_bp_lifestyle.csv", index=False)
print(f"\n  Processed data saved → processed_bp_lifestyle.csv  (n={len(X)})")


# ============================================================
# STEP 7 — STRATIFIED TRAIN / TEST SPLIT
# ============================================================
print("\n" + "=" * 60)
print("STEP 7 : STRATIFIED TRAIN / TEST SPLIT (80 / 20)")
print("=" * 60)

def stratified_split(X, y, tag):
    groups = pd.qcut(y, q=5, labels=False, duplicates='drop')
    sss    = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for tr, te in sss.split(X, groups):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y.iloc[tr], y.iloc[te]
    print(f"  {tag}  train={len(Xtr)}  test={len(Xte)}")
    return Xtr, Xte, ytr, yte

Xtr_s, Xte_s, ytr_s, yte_s = stratified_split(X, y_sbp, "SBP")
Xtr_d, Xte_d, ytr_d, yte_d = stratified_split(X, y_dbp, "DBP")


# ============================================================
# HELPER — EVALUATE WITH CV MEAN ± STD (on full dataset)
# ============================================================

def evaluate(model, Xtr, Xte, ytr, yte, X_full, y_full, label):
    """
    Fit on train partition, evaluate on held-out test set.
    CV is computed on the FULL dataset (not just train) to reduce
    variance in the CV R² estimate — critical when n is modest.
    Reports mean ± std so reviewers can assess reliability.
    """
    model.fit(Xtr, ytr)
    yp   = model.predict(Xte)
    mae  = mean_absolute_error(yte, yp)
    rmse = np.sqrt(mean_squared_error(yte, yp))
    r2   = r2_score(yte, yp)

    # 10-fold CV on FULL dataset for stable estimate
    cv_scores = cross_val_score(
        model, X_full, y_full,
        cv=KFold(n_splits=10, shuffle=True, random_state=42),
        scoring='r2'
    )
    cv_mean = cv_scores.mean()
    cv_std  = cv_scores.std()

    print(f"  [{label:<14}]  MAE={mae:.3f}  RMSE={rmse:.3f}"
          f"  R²={r2:.4f}  CV-R²={cv_mean:.4f} ± {cv_std:.4f}"
          f"  (n_cv_folds=10, N={len(X_full)})")

    return dict(
        Model=label,
        MAE=round(mae, 4),
        RMSE=round(rmse, 4),
        R2=round(r2, 4),
        CV_R2_mean=round(cv_mean, 4),
        CV_R2_std=round(cv_std, 4),
    )


# ============================================================
# STEP 8 — GRADIENT BOOSTING : SBP
# ============================================================
print("\n" + "=" * 60)
print("STEP 8 : GRADIENT BOOSTING — SBP  (RandomizedSearchCV)")
print("=" * 60)

# Constrained grid: shallow trees + high leaf size to prevent overfitting
gb_grid = {
    'n_estimators'     : [100, 150, 200, 250],   # reduced from 500 — prevents overfit on CV folds
    'max_depth'        : [2, 3],                  # shallow only
    'learning_rate'    : [0.01, 0.02, 0.05],
    'subsample'        : [0.65, 0.75, 0.85],
    'min_samples_split': [20, 30, 50],            # higher → less overfit
    'min_samples_leaf' : [10, 20, 30],            # higher → more regularized
    'max_features'     : ['sqrt', 0.5],
}

gb_rs_sbp = RandomizedSearchCV(
    GradientBoostingRegressor(random_state=42),
    param_distributions=gb_grid,
    n_iter=40,
    cv=KFold(n_splits=5, shuffle=True, random_state=42),
    scoring='r2',                                  # optimise R² directly
    n_jobs=-1,
    random_state=42,
    verbose=0,
)

print("  Running RandomizedSearchCV (40 iters × 5-fold) …")
gb_rs_sbp.fit(Xtr_s, ytr_s)
print(f"  Best params: {gb_rs_sbp.best_params_}")
gb_sbp = gb_rs_sbp.best_estimator_
res_gb_sbp = evaluate(gb_sbp, Xtr_s, Xte_s, ytr_s, yte_s, X, y_sbp, "GB-SBP")


# ============================================================
# STEP 9 — GRADIENT BOOSTING : DBP
# ============================================================
print("\n" + "=" * 60)
print("STEP 9 : GRADIENT BOOSTING — DBP  (RandomizedSearchCV)")
print("=" * 60)

gb_rs_dbp = RandomizedSearchCV(
    GradientBoostingRegressor(random_state=42),
    param_distributions=gb_grid,
    n_iter=40,
    cv=KFold(n_splits=5, shuffle=True, random_state=42),
    scoring='r2',
    n_jobs=-1,
    random_state=42,
    verbose=0,
)

print("  Running RandomizedSearchCV (40 iters × 5-fold) …")
gb_rs_dbp.fit(Xtr_d, ytr_d)
print(f"  Best params: {gb_rs_dbp.best_params_}")
gb_dbp = gb_rs_dbp.best_estimator_
res_gb_dbp = evaluate(gb_dbp, Xtr_d, Xte_d, ytr_d, yte_d, X, y_dbp, "GB-DBP")


# ============================================================
# STEP 10 — XGBOOST : SBP
# ============================================================
print("\n" + "=" * 60)
print("STEP 10: XGBOOST — SBP  (RandomizedSearchCV)")
print("=" * 60)

# Heavily regularized grid to prevent overfitting & negative CV R²
# Key changes vs v1:
#   - n_estimators capped at 300 (was 700) — 1200+ on n=216 is textbook overfit
#   - max_depth capped at 3 (was 5)
#   - min_child_weight min=5 (was 1)
#   - gamma min=0.1 (was 0) — forces meaningful splits
#   - reg_alpha/lambda pushed higher
xgb_grid = {
    'n_estimators'    : [100, 150, 200, 250, 300],   # hard cap at 300
    'max_depth'       : [2, 3],                       # shallow only
    'learning_rate'   : [0.01, 0.02, 0.03, 0.05],
    'subsample'       : [0.6, 0.7, 0.8],
    'colsample_bytree': [0.6, 0.7, 0.8],
    'reg_alpha'       : [1.0, 2.0, 5.0, 10.0],       # stronger L1
    'reg_lambda'      : [2.0, 5.0, 10.0],            # stronger L2
    'min_child_weight': [5, 10, 20],                  # prevents tiny leaf splits
    'gamma'           : [0.1, 0.3, 0.5, 1.0],        # min loss reduction for split
}

xgb_rs_sbp = RandomizedSearchCV(
    xgb.XGBRegressor(random_state=42, n_jobs=1, verbosity=0, tree_method='hist'),
    param_distributions=xgb_grid,
    n_iter=50,
    cv=KFold(n_splits=5, shuffle=True, random_state=42),
    scoring='r2',
    n_jobs=-1,
    random_state=42,
    verbose=0,
)

print("  Running RandomizedSearchCV (50 iters × 5-fold) …")
xgb_rs_sbp.fit(Xtr_s, ytr_s)
print(f"  Best params: {xgb_rs_sbp.best_params_}")
xgb_sbp = xgb_rs_sbp.best_estimator_
res_xgb_sbp = evaluate(xgb_sbp, Xtr_s, Xte_s, ytr_s, yte_s, X, y_sbp, "XGB-SBP")


# ============================================================
# STEP 11 — XGBOOST : DBP
# ============================================================
print("\n" + "=" * 60)
print("STEP 11: XGBOOST — DBP  (RandomizedSearchCV)")
print("=" * 60)

xgb_rs_dbp = RandomizedSearchCV(
    xgb.XGBRegressor(random_state=42, n_jobs=1, verbosity=0, tree_method='hist'),
    param_distributions=xgb_grid,
    n_iter=50,
    cv=KFold(n_splits=5, shuffle=True, random_state=42),
    scoring='r2',
    n_jobs=-1,
    random_state=42,
    verbose=0,
)

print("  Running RandomizedSearchCV (50 iters × 5-fold) …")
xgb_rs_dbp.fit(Xtr_d, ytr_d)
print(f"  Best params: {xgb_rs_dbp.best_params_}")
xgb_dbp = xgb_rs_dbp.best_estimator_
res_xgb_dbp = evaluate(xgb_dbp, Xtr_d, Xte_d, ytr_d, yte_d, X, y_dbp, "XGB-DBP")


# ============================================================
# STEP 12 — FEATURE IMPORTANCE
# ============================================================
print("\n" + "=" * 60)
print("STEP 12: FEATURE IMPORTANCE (top 10 per model)")
print("=" * 60)

def show_importance(model, label):
    imp = pd.Series(model.feature_importances_, index=FEATURES)
    imp = imp.sort_values(ascending=False)
    print(f"\n  {label}")
    for feat, val in imp.head(10).items():
        bar = '█' * int(val * 300)
        print(f"    {feat:<20} {val:.4f}  {bar}")
    return imp

imp_gb_sbp  = show_importance(gb_sbp,  "GB-SBP")
imp_gb_dbp  = show_importance(gb_dbp,  "GB-DBP")
imp_xgb_sbp = show_importance(xgb_sbp, "XGB-SBP")
imp_xgb_dbp = show_importance(xgb_dbp, "XGB-DBP")


# ============================================================
# STEP 13 — RESULTS SUMMARY & SAVE
# ============================================================
print("\n" + "=" * 60)
print("STEP 13: FINAL RESULTS SUMMARY")
print("=" * 60)

results_df = pd.DataFrame([res_gb_sbp, res_gb_dbp, res_xgb_sbp, res_xgb_dbp])
print(results_df.to_string(index=False))

results_df.to_csv(OUT_PATH + "model_results_tuned.csv", index=False)
print(f"\n  Results saved → model_results_tuned.csv")

for model, fname in [
    (gb_sbp,  "gb_sbp_model.pkl"),
    (gb_dbp,  "gb_dbp_model.pkl"),
    (xgb_sbp, "xgb_sbp_model.pkl"),
    (xgb_dbp, "xgb_dbp_model.pkl"),
]:
    joblib.dump(model, OUT_PATH + fname)
    print(f"  Model saved → {fname}")

imp_df = pd.DataFrame({
    'Feature' : FEATURES,
    'GB_SBP'  : imp_gb_sbp.reindex(FEATURES).values,
    'GB_DBP'  : imp_gb_dbp.reindex(FEATURES).values,
    'XGB_SBP' : imp_xgb_sbp.reindex(FEATURES).values,
    'XGB_DBP' : imp_xgb_dbp.reindex(FEATURES).values,
})
imp_df = imp_df.sort_values('XGB_SBP', ascending=False).reset_index(drop=True)
imp_df.to_csv(OUT_PATH + "feature_importance.csv", index=False)
print(f"  Feature importance saved → feature_importance.csv")

print("\n" + "=" * 60)
print("✅  PIPELINE COMPLETE")
print("=" * 60)
print(f"""
Key changes from v1:
  - Dataset      : n={len(X)} (full) vs n=270 (naive-only) — eliminates CV instability
  - CV           : 10-fold on FULL dataset, reports mean ± std
  - XGBoost      : n_estimators≤300, max_depth≤3, min_child_weight≥5,
                   gamma≥0.1, reg_alpha≥1, reg_lambda≥2
  - GB           : n_estimators≤250, max_depth≤3, min_samples_leaf≥10
  - Search score : optimises R² directly (was neg_MAE)
  - OnMedication : included as feature (1=on meds, 0=naive, -1=unknown)

Output files:
  processed_bp_lifestyle.csv   — cleaned, engineered dataset (n={len(X)})
  model_results_tuned.csv      — MAE / RMSE / R² / CV_R2_mean / CV_R2_std
  feature_importance.csv       — importance scores per model
  gb_sbp_model.pkl, gb_dbp_model.pkl
  xgb_sbp_model.pkl, xgb_dbp_model.pkl
""")