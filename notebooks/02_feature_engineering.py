# ================================
# Feature Engineering (UPDATED - WITH MEDICATION)
# Adds BPQ_L.xpt to filter medication-naive participants
# and adds OnMedication as a feature for full-dataset model
# ================================

import pandas as pd
import numpy as np

# -------------------------------
# Load datasets
# -------------------------------

demo  = pd.read_sas("../data/DEMO_L.xpt",  format="xport")
bp    = pd.read_sas("../data/BPXO_L.xpt", format="xport")
bmx   = pd.read_sas("../data/BMX_L.xpt",  format="xport")
diet  = pd.read_sas("../data/DR1TOT_L.xpt", format="xport")
paq   = pd.read_sas("../data/PAQ_L.xpt",  format="xport")
sleep = pd.read_sas("../data/SLQ_L.xpt",  format="xport")

# ✅ NEW: Blood Pressure Questionnaire — contains medication status
# Download from: https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/BPQ_L.xpt
bpq   = pd.read_sas("../data/BPQ_L.xpt",  format="xport")

# -------------------------------
# Select columns
# -------------------------------

demo  = demo[['SEQN', 'RIDAGEYR', 'RIAGENDR']]

bp    = bp[['SEQN', 'BPXOSY1', 'BPXOSY2', 'BPXOSY3',
            'BPXODI1', 'BPXODI2', 'BPXODI3']]

bmx   = bmx[['SEQN', 'BMXBMI', 'BMXWAIST']]

diet  = diet[['SEQN', 'DR1TSODI', 'DR1TPOTA']]

paq   = paq[['SEQN', 'PAD680']]
sleep = sleep[['SEQN', 'SLD012']]

# ✅ NEW: BPQ150 = "Are you now taking prescribed medicine for high blood pressure?"
#    Values: 1 = Yes, 2 = No, 7/9 = Refused/Don't know → treat as NaN
bpq   = bpq[['SEQN', 'BPQ150']].copy()

# -------------------------------
# Merge all files
# -------------------------------

df = demo.merge(bp,    on='SEQN')
df = df.merge(bmx,    on='SEQN')
df = df.merge(diet,   on='SEQN')
df = df.merge(paq,    on='SEQN')
df = df.merge(sleep,  on='SEQN')
df = df.merge(bpq,    on='SEQN', how='left')   # left join: keep all rows even if BPQ missing

# -------------------------------
# Rename
# -------------------------------

df = df.rename(columns={
    'RIDAGEYR':  'Age',
    'RIAGENDR':  'Gender',
    'BMXBMI':    'BMI',
    'BMXWAIST':  'Waist',
    'DR1TSODI':  'Sodium',
    'DR1TPOTA':  'Potassium',
    'PAD680':    'PhysicalActivity',
    'SLD012':    'SleepHours',
    'BPQ150':    'MedRaw'
})

# -------------------------------
# Medication flag
# -------------------------------

# BPQ150: 1 = Yes (on medication), 2 = No, 7/9 = ambiguous → NaN
df['OnMedication'] = df['MedRaw'].map({1.0: 1, 2.0: 0})
# Rows where BPQ150 is 7, 9, or NaN → OnMedication stays NaN

print("\nMedication status counts:")
print(df['OnMedication'].value_counts(dropna=False))
print(f"  → On medication   : {(df['OnMedication'] == 1).sum()}")
print(f"  → Med-naive       : {(df['OnMedication'] == 0).sum()}")
print(f"  → Unknown/missing : {df['OnMedication'].isna().sum()}")

# Drop the raw column
df = df.drop(columns=['MedRaw'])

# -------------------------------
# Basic cleaning (before split)
# -------------------------------

# Drop rows missing any core feature (not OnMedication yet — handled per-dataset)
core_cols = ['Age', 'Gender', 'BMI', 'Waist', 'Sodium', 'Potassium',
             'PhysicalActivity', 'SleepHours',
             'BPXOSY1', 'BPXOSY2', 'BPXOSY3',
             'BPXODI1', 'BPXODI2', 'BPXODI3']
df = df.dropna(subset=core_cols)

# Remove constant columns
df = df.loc[:, df.nunique() > 1]

# Clip extreme dietary values
df['Sodium']    = df['Sodium'].clip(500, 5000)
df['Potassium'] = df['Potassium'].clip(500, 5000)

# -------------------------------
# Target variables
# -------------------------------

df['SBP'] = df[['BPXOSY1', 'BPXOSY2', 'BPXOSY3']].mean(axis=1)
df['DBP'] = df[['BPXODI1', 'BPXODI2', 'BPXODI3']].mean(axis=1)

# Remove physiologically impossible outliers
df = df[(df['SBP'] > 95) & (df['SBP'] < 200)]
df = df[(df['DBP'] > 40) & (df['DBP'] < 130)]

# -------------------------------
# Feature Engineering
# -------------------------------

df['BMI_Age']   = df['BMI']  * df['Age']
df['Waist_Age'] = df['Waist'] * df['Age']
df['BMI_sq']    = df['BMI']  ** 2
df['Age_sq']    = df['Age']  ** 2
df['log_sodium'] = np.log1p(df['Sodium'])
df['Na_K_ratio'] = df['Sodium'] / (df['Potassium'] + 1)
df['WHR']        = df['Waist'] / df['BMI']
df['Age_BMI']    = df['Age']  * df['BMI']
df['Age_Waist']  = df['Age']  * df['Waist']
df['BMI_Waist']  = df['BMI']  * df['Waist']





# -------------------------------
# NEW HIGH-SIGNAL FEATURES
# -------------------------------

# Pulse pressure proxy (even without direct BP)
df['Waist_to_Age'] = df['Waist'] / (df['Age'] + 1)

# Obesity severity
df['BMI_class'] = pd.cut(df['BMI'],
                         bins=[0, 18.5, 25, 30, 100],
                         labels=[0,1,2,3]).astype(float)

# Sodium categories (nonlinear effect)
df['Sodium_cat'] = pd.qcut(df['Sodium'], q=4, labels=False)

# Activity transformation (important!)
df['log_activity'] = np.log1p(df['PhysicalActivity'])

# Sleep deviation (VERY important clinically)
df['Sleep_deviation'] = abs(df['SleepHours'] - 7)

# Interaction: sodium × BMI
df['Sodium_BMI'] = df['Sodium'] * df['BMI']

# Interaction: activity × BMI
df['Activity_BMI'] = df['PhysicalActivity'] * df['BMI']
# -------------------------------
# Save FULL dataset (with OnMedication feature + NaN rows kept)
# Use this for: Model A — Full dataset, medication as a feature
# -------------------------------
df = df.dropna(subset=core_cols)
for col in ['BMI','Waist','Sodium','Potassium','PhysicalActivity','SleepHours']:
    df[col] = df[col].fillna(df[col].median())
df_full = df.copy()
# For the full model, fill unknown medication as -1 (unknown class)
df_full['OnMedication'] = df_full['OnMedication'].fillna(-1)

df_full.to_csv("../data/processed_bp_full.csv", index=False)
print(f"\n✅ Full dataset saved → processed_bp_full.csv  | Shape: {df_full.shape}")

# -------------------------------
# Save MED-NAIVE dataset (OnMedication == 0 only)
# Use this for: Model B — Clean lifestyle signal, no drug suppression
# This dataset has the highest potential R²
# -------------------------------

df_naive = df[df['OnMedication'] == 0].copy()
df_naive = df_naive.drop(columns=['OnMedication'])   # constant column now

df_naive.to_csv("../data/processed_bp_naive.csv", index=False)
print(f"✅ Med-naive dataset saved → processed_bp_naive.csv | Shape: {df_naive.shape}")

# -------------------------------
# Backward-compatible save (replaces original file used by old scripts)
# Points to the med-naive dataset so existing model scripts improve automatically
# -------------------------------

df_naive.to_csv("../data/processed_bp_lifestyle.csv", index=False)
print(f"✅ Legacy file updated → processed_bp_lifestyle.csv (now med-naive)")



print("\n--- DATASET SUMMARY ---")
print(f"Total merged records         : {len(df)}")
print(f"On antihypertensive meds     : {(df['OnMedication'] == 1).sum()}")
print(f"Medication-naive (clean)     : {len(df_naive)}  ← use for max R²")
print(f"SBP range (naive): {df_naive['SBP'].min():.0f} – {df_naive['SBP'].max():.0f} mmHg")
print(f"DBP range (naive): {df_naive['DBP'].min():.0f} – {df_naive['DBP'].max():.0f} mmHg")
print("\nNOTE: Use processed_bp_naive.csv for Model B (expected R² improvement)")
print("      Use processed_bp_full.csv  for Model A (OnMedication as a feature)")