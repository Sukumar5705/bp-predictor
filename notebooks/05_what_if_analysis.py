# ================================
# What-if Analysis (ΔBP Prediction)
# BP Lifestyle Impact Project
# ================================

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

# -------------------------------
# Step 1: Load processed data
# -------------------------------

df = pd.read_csv("data/processed_bp_lifestyle.csv")

features = [
    'Age',
    'BMI',
    'Waist',
    'DietCategory',
    'ActivityLevel',
    'SleepQuality'
]

X = df[features]
y_sbp = df['SBP']
y_dbp = df['DBP']

# -------------------------------
# Step 2: Train final models
# -------------------------------

X_train, X_test, y_sbp_train, y_sbp_test = train_test_split(
    X, y_sbp, test_size=0.2, random_state=42
)

_, _, y_dbp_train, y_dbp_test = train_test_split(
    X, y_dbp, test_size=0.2, random_state=42
)

sbp_model = GradientBoostingRegressor(
    n_estimators=200, learning_rate=0.05, random_state=42
)

dbp_model = GradientBoostingRegressor(
    n_estimators=200, learning_rate=0.05, random_state=42
)

sbp_model.fit(X_train, y_sbp_train)
dbp_model.fit(X_train, y_dbp_train)

# -------------------------------
# Step 3: Simulate user input
# -------------------------------

# Example baseline user
baseline_user = {
    'Age': 45,
    'BMI': 29,
    'Waist': 98,
    'DietCategory': 2,      # High salt
    'ActivityLevel': 0.5,   # Low activity
    'SleepQuality': 0       # Poor sleep
}

# Lifestyle-improved scenario
improved_user = {
    'Age': 45,
    'BMI': 27,              # Weight loss
    'Waist': 94,
    'DietCategory': 0,      # Low salt
    'ActivityLevel': 1.2,   # Active
    'SleepQuality': 1       # Good sleep
}

baseline_df = pd.DataFrame([baseline_user])
improved_df = pd.DataFrame([improved_user])

# -------------------------------
# Step 4: Predict BP values
# -------------------------------

sbp_before = sbp_model.predict(baseline_df)[0]
dbp_before = dbp_model.predict(baseline_df)[0]

sbp_after = sbp_model.predict(improved_df)[0]
dbp_after = dbp_model.predict(improved_df)[0]

# -------------------------------
# Step 5: Calculate BP change
# -------------------------------

delta_sbp = sbp_after - sbp_before
delta_dbp = dbp_after - dbp_before

print("\nWHAT-IF ANALYSIS RESULT\n")

print(f"Predicted SBP before change : {sbp_before:.1f}")
print(f"Predicted SBP after change  : {sbp_after:.1f}")
print(f"ΔSBP (change)               : {delta_sbp:.1f} mmHg\n")

print(f"Predicted DBP before change : {dbp_before:.1f}")
print(f"Predicted DBP after change  : {dbp_after:.1f}")
print(f"ΔDBP (change)               : {delta_dbp:.1f} mmHg")
