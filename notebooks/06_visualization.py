# ================================
# Visualization
# BP Lifestyle Impact Project
# ================================

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

# -------------------------------
# Step 1: Load processed dataset
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
# Step 3: Example what-if scenario
# -------------------------------

baseline_user = {
    'Age': 55,
    'BMI': 34,
    'Waist': 112,
    'DietCategory': 2,
    'ActivityLevel': 0.3,
    'SleepQuality': 0
}

improved_user = {
    'Age': 55,
    'BMI': 29,
    'Waist': 102,
    'DietCategory': 0,
    'ActivityLevel': 1.2,
    'SleepQuality': 1
}

baseline_df = pd.DataFrame([baseline_user])
improved_df = pd.DataFrame([improved_user])

sbp_before = sbp_model.predict(baseline_df)[0]
sbp_after  = sbp_model.predict(improved_df)[0]

dbp_before = dbp_model.predict(baseline_df)[0]
dbp_after  = dbp_model.predict(improved_df)[0]

# -------------------------------
# Step 4: Plot SBP & DBP graphs
# -------------------------------

labels = ['Before', 'After']
sbp_values = [sbp_before, sbp_after]
dbp_values = [dbp_before, dbp_after]

plt.figure()
plt.bar(labels, sbp_values)
plt.title("SBP Before vs After Lifestyle Change")
plt.ylabel("SBP (mmHg)")
plt.savefig("results/sbp_before_after.png")
plt.show()

plt.figure()
plt.bar(labels, dbp_values)
plt.title("DBP Before vs After Lifestyle Change")
plt.ylabel("DBP (mmHg)")
plt.savefig("results/dbp_before_after.png")
plt.show()
