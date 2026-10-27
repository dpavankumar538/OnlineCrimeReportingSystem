# ml_model/train_model.py — Final Version
import pandas as pd
import numpy as np
from sklearn.ensemble import (RandomForestClassifier,
                               ExtraTreesClassifier,
                               GradientBoostingClassifier)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, classification_report,
                              balanced_accuracy_score)
import joblib
import os
import json
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("  Crime Prediction Model — Final Version")
print("=" * 60)

# ----------------------------------------------------------
# STEP 1: Load
# ----------------------------------------------------------
data_path = os.path.join(
    os.path.dirname(__file__), 'data', 'crime_data.csv'
)
df = pd.read_csv(data_path, encoding='utf-8-sig')
df.columns = df.columns.str.strip()
print(f"\n✓ Dataset loaded: {len(df)} rows")
print(f"  Columns: {list(df.columns)}")

# ----------------------------------------------------------
# STEP 2: Time features
# ----------------------------------------------------------
def safe_parse(val, attr):
    try:
        return getattr(pd.to_datetime(val, dayfirst=True), attr)
    except Exception:
        defaults = {'hour': 12, 'month': 1, 'dayofweek': 0}
        return defaults.get(attr, 0)

df['hour']        = df['Time of Occurrence'].apply(
    lambda x: safe_parse(x, 'hour')
)
df['month']       = df['Time of Occurrence'].apply(
    lambda x: safe_parse(x, 'month')
)
df['day_of_week'] = df['Time of Occurrence'].apply(
    lambda x: safe_parse(x, 'dayofweek')
)
df['time_period'] = df['hour'].apply(
    lambda h: 0 if h < 6 else 1 if h < 12 else 2 if h < 18 else 3
)
df['is_weekend']  = (df['day_of_week'] >= 5).astype(int)
print("✓ Time features extracted")

# ----------------------------------------------------------
# STEP 3: Drop rows missing key columns
# ----------------------------------------------------------
required = [
    'City', 'Crime Description', 'Crime Domain',
    'Weapon Used', 'Victim Age', 'Victim Gender',
    'Police Deployed'
]
df = df.dropna(subset=required)
print(f"✓ After dropping nulls: {len(df)} rows")

# ----------------------------------------------------------
# STEP 4: Build city crime frequency table
# This gives us REAL accurate statistics from 40k rows
# ----------------------------------------------------------
city_crime_freq = (
    df.groupby(['City', 'Crime Description'])
    .size()
    .reset_index(name='count')
)

city_summary = {}
for city in sorted(df['City'].unique()):
    cdf   = city_crime_freq[city_crime_freq['City'] == city]
    total = cdf['count'].sum()
    if total == 0:
        continue
    breakdown = {
        row['Crime Description']: round(
            row['count'] / total * 100, 1
        )
        for _, row in cdf.iterrows()
    }
    sorted_bd = dict(
        sorted(breakdown.items(), key=lambda x: -x[1])
    )
    city_summary[city] = {
        'top_crime'  : list(sorted_bd.keys())[0],
        'top_pct'    : list(sorted_bd.values())[0],
        'breakdown'  : sorted_bd,
        'total_cases': int(total)
    }

# Also build domain frequency per city
city_domain_freq = (
    df.groupby(['City', 'Crime Domain'])
    .size()
    .reset_index(name='count')
)
for city in city_summary:
    cdf   = city_domain_freq[city_domain_freq['City'] == city]
    total = cdf['count'].sum()
    if total == 0:
        continue
    city_summary[city]['domain_breakdown'] = {
        row['Crime Domain']: round(row['count'] / total * 100, 1)
        for _, row in cdf.iterrows()
    }

freq_path = os.path.join(
    os.path.dirname(__file__), 'city_crime_freq.json'
)
with open(freq_path, 'w') as f:
    json.dump(city_summary, f, indent=2)
print(f"✓ City crime frequency table saved")
print(f"  Cities with data: {len(city_summary)}")

# ----------------------------------------------------------
# STEP 5: Encode features
# ----------------------------------------------------------
city_encoder   = LabelEncoder()
weapon_encoder = LabelEncoder()
gender_encoder = LabelEncoder()
domain_encoder = LabelEncoder()
crime_encoder  = LabelEncoder()

df['city_enc']        = city_encoder.fit_transform(
    df['City'].astype(str)
)
df['weapon_enc']      = weapon_encoder.fit_transform(
    df['Weapon Used'].astype(str)
)
df['gender_enc']      = gender_encoder.fit_transform(
    df['Victim Gender'].astype(str)
)
df['domain_enc']      = domain_encoder.fit_transform(
    df['Crime Domain'].astype(str)
)
df['crime_enc']       = crime_encoder.fit_transform(
    df['Crime Description'].astype(str)
)
df['Police Deployed'] = pd.to_numeric(
    df['Police Deployed'], errors='coerce'
).fillna(0)
df['Victim Age']      = pd.to_numeric(
    df['Victim Age'], errors='coerce'
).fillna(30)
df['age_group']       = df['Victim Age'].apply(
    lambda a: 0 if a < 18 else 1 if a < 30 else
              2 if a < 45 else 3 if a < 60 else 4
)

print(f"\n✓ Encodings complete")
print(f"  Cities      : {len(city_encoder.classes_)}")
print(f"  Crime types : {len(crime_encoder.classes_)}")
print(f"  Domains     : {list(domain_encoder.classes_)}")
print(f"\n  Domain distribution:")
for cls, cnt in zip(
    domain_encoder.classes_,
    np.bincount(df['domain_enc'])
):
    pct = cnt / len(df) * 100
    bar = '█' * int(pct / 2)
    print(f"    {cls:20s}: {cnt:5d} ({pct:.1f}%) {bar}")

# ----------------------------------------------------------
# STEP 6: Feature columns
# ----------------------------------------------------------
feature_cols = [
    'city_enc', 'hour', 'month', 'day_of_week',
    'time_period', 'is_weekend', 'weapon_enc',
    'gender_enc', 'age_group', 'Police Deployed'
]

X = df[feature_cols]  # pandas DataFrame — no UserWarning

model_dir = os.path.dirname(__file__)
with open(os.path.join(model_dir, 'feature_cols.json'), 'w') as f:
    json.dump(feature_cols, f)

# ----------------------------------------------------------
# STEP 7: Train Domain Model
# Using ExtraTreesClassifier — better for imbalanced data
# class_weight='balanced_subsample' is more robust than
# 'balanced' for Random Forest / ExtraTrees
# ----------------------------------------------------------
y_domain = df['domain_enc']

X_train, X_test, y_train, y_test = train_test_split(
    X, y_domain, test_size=0.2,
    random_state=42, stratify=y_domain
)

print("\nTraining Domain Model (ExtraTrees + balanced_subsample)...")
model_domain = ExtraTreesClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight='balanced_subsample',
    random_state=42,
    n_jobs=-1
)
model_domain.fit(X_train, y_train)

y_pred    = model_domain.predict(X_test)
acc       = accuracy_score(y_test, y_pred)
bal_acc   = balanced_accuracy_score(y_test, y_pred)

print(f"\n{'='*60}")
print(f"  DOMAIN MODEL ACCURACY         : {acc*100:.2f}%")
print(f"  DOMAIN BALANCED ACCURACY      : {bal_acc*100:.2f}%")
print(f"{'='*60}\n")
print(classification_report(
    y_test, y_pred,
    target_names=domain_encoder.classes_,
    zero_division=0
))

cv_bal = cross_val_score(
    model_domain, X, y_domain,
    cv=5, scoring='balanced_accuracy', n_jobs=-1
)
print(f"  Balanced CV: {cv_bal.mean()*100:.2f}% "
      f"(+/- {cv_bal.std()*100:.2f}%)")

# Feature importance
print("\nFeature Importance:")
for feat, imp in sorted(
    zip(feature_cols, model_domain.feature_importances_),
    key=lambda x: -x[1]
):
    bar = '█' * int(imp * 50)
    print(f"  {feat:20s}: {imp:.4f}  {bar}")

# ----------------------------------------------------------
# STEP 8: Train Crime Type Model
# ----------------------------------------------------------
y_crime = df['crime_enc']

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X, y_crime, test_size=0.2,
    random_state=42, stratify=y_crime
)

print("\nTraining Crime Type Model...")
model_crime = ExtraTreesClassifier(
    n_estimators=200,
    class_weight='balanced_subsample',
    random_state=42,
    n_jobs=-1
)
model_crime.fit(X_train_c, y_train_c)
y_pred_c  = model_crime.predict(X_test_c)
acc_c     = accuracy_score(y_test_c, y_pred_c)
bal_acc_c = balanced_accuracy_score(y_test_c, y_pred_c)
print(f"  Crime Type Accuracy         : {acc_c*100:.2f}%")
print(f"  Crime Type Balanced Accuracy: {bal_acc_c*100:.2f}%")

# ----------------------------------------------------------
# STEP 9: Save everything
# ----------------------------------------------------------
joblib.dump(model_domain,   os.path.join(model_dir, 'crime_model.pkl'))
joblib.dump(model_crime,    os.path.join(model_dir, 'crime_type_model.pkl'))
joblib.dump(crime_encoder,  os.path.join(model_dir, 'label_encoder.pkl'))
joblib.dump(domain_encoder, os.path.join(model_dir, 'domain_encoder.pkl'))
joblib.dump(city_encoder,   os.path.join(model_dir, 'district_encoder.pkl'))
joblib.dump(weapon_encoder, os.path.join(model_dir, 'weapon_encoder.pkl'))
joblib.dump(gender_encoder, os.path.join(model_dir, 'gender_encoder.pkl'))

print(f"\n✓ All models saved successfully")
print(f"\n{'='*60}")
print(f"  SUMMARY")
print(f"{'='*60}")
print(f"  Domain model balanced accuracy : {bal_acc*100:.2f}%")
print(f"  Crime type balanced accuracy   : {bal_acc_c*100:.2f}%")
print(f"  Cities in frequency table      : {len(city_summary)}")
print(f"  Total training records         : {len(df)}")
print(f"\n  NOTE: Crime frequency breakdown uses REAL historical")
print(f"  data from {len(df)} records — this is the most accurate")
print(f"  predictor and does not depend on model accuracy.")
print(f"\n🎉 Training complete! Run: python app.py")