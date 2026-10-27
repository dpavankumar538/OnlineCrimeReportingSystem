
# ml_model/train_model.py
# Crime Reporting and Prediction System
# Trains domain and crime-type prediction models.

import os
import json
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report
)

warnings.filterwarnings("ignore")

print("=" * 65)
print("       CRIME REPORTING AND PREDICTION SYSTEM")
print("                 MODEL TRAINING")
print("=" * 65)

# ----------------------------------------------------------
# STEP 1: Paths and dataset loading
# ----------------------------------------------------------

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(MODEL_DIR, "data", "crime_data.csv")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Check that crime_data.csv exists inside ml_model/data/."
    )

df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
df.columns = df.columns.str.strip()

print(f"\nDataset loaded: {len(df)} rows")
print("Columns:", list(df.columns))

# ----------------------------------------------------------
# STEP 2: Validate required columns
# ----------------------------------------------------------

required_columns = [
    "City",
    "Crime Description",
    "Crime Domain",
    "Weapon Used",
    "Victim Age",
    "Victim Gender",
    "Police Deployed",
    "Time of Occurrence"
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        "The dataset is missing these required columns: "
        + ", ".join(missing_columns)
    )

# ----------------------------------------------------------
# STEP 3: Extract time features
#
# If Time of Occurrence contains a full date and time,
# extract all three features from it.
#
# If it contains only a time, use the available hour and
# use neutral defaults for month and weekday. This avoids
# treating an arbitrary time-only value as a real date.
# ----------------------------------------------------------

time_text = df["Time of Occurrence"].astype("string").str.strip()

has_date_information = time_text.str.contains(
    r"[-/]|[A-Za-z]{3,}",
    regex=True,
    na=False
)

parsed_time = pd.to_datetime(
    time_text.where(has_date_information),
    errors="coerce",
    dayfirst=True
)

# Parse time-only values separately, such as 14:30 or 2:30 PM.
time_only = pd.to_datetime(
    time_text.where(~has_date_information),
    errors="coerce",
    format="mixed"
)

df["hour"] = parsed_time.dt.hour
df["month"] = parsed_time.dt.month
df["day_of_week"] = parsed_time.dt.dayofweek

missing_hour = df["hour"].isna()
df.loc[missing_hour, "hour"] = time_only.loc[missing_hour].dt.hour

# Defaults are used only where a value could not be parsed.
df["hour"] = df["hour"].fillna(12).astype(int)
df["month"] = df["month"].fillna(1).astype(int)
df["day_of_week"] = df["day_of_week"].fillna(0).astype(int)

df["time_period"] = df["hour"].apply(
    lambda hour: (
        0 if hour < 6 else
        1 if hour < 12 else
        2 if hour < 18 else
        3
    )
)

df["is_weekend"] = (
    df["day_of_week"] >= 5
).astype(int)

print("Time features extracted.")

# ----------------------------------------------------------
# STEP 4: Clean required fields
# ----------------------------------------------------------

df = df.dropna(subset=[
    "City",
    "Crime Description",
    "Crime Domain",
    "Weapon Used",
    "Victim Age",
    "Victim Gender",
    "Police Deployed"
]).copy()

# Convert text columns consistently.
text_columns = [
    "City",
    "Crime Description",
    "Crime Domain",
    "Weapon Used",
    "Victim Gender"
]

for column in text_columns:
    df[column] = df[column].astype(str).str.strip()

# Remove empty target values.
df = df[
    (df["City"] != "") &
    (df["Crime Description"] != "") &
    (df["Crime Domain"] != "")
].copy()

df["Victim Age"] = pd.to_numeric(
    df["Victim Age"], errors="coerce"
).fillna(30)

df["Police Deployed"] = pd.to_numeric(
    df["Police Deployed"], errors="coerce"
).fillna(0)

if len(df) < 10:
    raise ValueError(
        "Not enough usable rows to train the models."
    )

if df["Crime Domain"].nunique() < 2:
    raise ValueError(
        "At least two crime-domain classes are required."
    )

if df["Crime Description"].nunique() < 2:
    raise ValueError(
        "At least two crime-description classes are required."
    )

print(f"Usable dataset rows: {len(df)}")

# ----------------------------------------------------------
# STEP 5: Build city crime frequency statistics
# ----------------------------------------------------------

city_crime_freq = (
    df.groupby(["City", "Crime Description"])
    .size()
    .reset_index(name="count")
)

city_summary = {}

for city in sorted(df["City"].unique()):
    city_data = city_crime_freq[
        city_crime_freq["City"] == city
    ]

    total_cases = int(city_data["count"].sum())

    if total_cases == 0:
        continue

    breakdown = {
        row["Crime Description"]: round(
            int(row["count"]) / total_cases * 100, 1
        )
        for _, row in city_data.iterrows()
    }

    sorted_breakdown = dict(
        sorted(
            breakdown.items(),
            key=lambda item: item[1],
            reverse=True
        )
    )

    city_summary[city] = {
        "top_crime": next(iter(sorted_breakdown)),
        "top_pct": next(iter(sorted_breakdown.values())),
        "breakdown": sorted_breakdown,
        "total_cases": total_cases
    }

city_domain_freq = (
    df.groupby(["City", "Crime Domain"])
    .size()
    .reset_index(name="count")
)

for city in city_summary:
    city_data = city_domain_freq[
        city_domain_freq["City"] == city
    ]

    total = int(city_data["count"].sum())

    if total:
        city_summary[city]["domain_breakdown"] = {
            row["Crime Domain"]: round(
                int(row["count"]) / total * 100, 1
            )
            for _, row in city_data.iterrows()
        }

FREQUENCY_PATH = os.path.join(
    MODEL_DIR, "city_crime_freq.json"
)

with open(FREQUENCY_PATH, "w", encoding="utf-8") as file:
    json.dump(
        city_summary,
        file,
        indent=2,
        ensure_ascii=False
    )

print(f"City statistics saved for {len(city_summary)} cities.")

# ----------------------------------------------------------
# STEP 6: Encode categorical features
# ----------------------------------------------------------

city_encoder = LabelEncoder()
weapon_encoder = LabelEncoder()
gender_encoder = LabelEncoder()
domain_encoder = LabelEncoder()
crime_encoder = LabelEncoder()

df["city_enc"] = city_encoder.fit_transform(
    df["City"]
)

df["weapon_enc"] = weapon_encoder.fit_transform(
    df["Weapon Used"]
)

df["gender_enc"] = gender_encoder.fit_transform(
    df["Victim Gender"]
)

df["domain_enc"] = domain_encoder.fit_transform(
    df["Crime Domain"]
)

df["crime_enc"] = crime_encoder.fit_transform(
    df["Crime Description"]
)

df["age_group"] = df["Victim Age"].apply(
    lambda age: (
        0 if age < 18 else
        1 if age < 30 else
        2 if age < 45 else
        3 if age < 60 else
        4
    )
)

print("\nEncoding complete.")
print("Cities:", len(city_encoder.classes_))
print("Crime types:", len(crime_encoder.classes_))
print("Domains:", list(domain_encoder.classes_))

# ----------------------------------------------------------
# STEP 7: Prepare features
# ----------------------------------------------------------

feature_cols = [
    "city_enc",
    "hour",
    "month",
    "day_of_week",
    "time_period",
    "is_weekend",
    "weapon_enc",
    "gender_enc",
    "age_group",
    "Police Deployed"
]

X = df[feature_cols].copy()

FEATURE_PATH = os.path.join(
    MODEL_DIR, "feature_cols.json"
)

with open(FEATURE_PATH, "w", encoding="utf-8") as file:
    json.dump(feature_cols, file, indent=2)

# ----------------------------------------------------------
# STEP 8: Shared training configuration
#
# These settings reduce model complexity compared with
# unrestricted trees. File sizes must still be checked
# after training.
# ----------------------------------------------------------

MODEL_SETTINGS = {
    "n_estimators": 100,
    "max_depth": 20,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1
}

# ----------------------------------------------------------
# STEP 9: Train crime-domain model
# ----------------------------------------------------------

print("\n" + "=" * 65)
print("TRAINING CRIME-DOMAIN MODEL")
print("=" * 65)

y_domain = df["domain_enc"]

domain_counts = y_domain.value_counts()

if domain_counts.min() < 2:
    raise ValueError(
        "Each crime-domain class needs at least two records "
        "for a stratified train/test split."
    )

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_domain,
    test_size=0.2,
    random_state=42,
    stratify=y_domain
)

model_domain = ExtraTreesClassifier(**MODEL_SETTINGS)
model_domain.fit(X_train, y_train)

domain_predictions = model_domain.predict(X_test)

domain_accuracy = accuracy_score(
    y_test, domain_predictions
)

domain_balanced_accuracy = balanced_accuracy_score(
    y_test, domain_predictions
)

print(f"Domain accuracy: {domain_accuracy * 100:.2f}%")
print(
    "Domain balanced accuracy: "
    f"{domain_balanced_accuracy * 100:.2f}%"
)

print("\nDomain classification report:")
print(
    classification_report(
        y_test,
        domain_predictions,
        labels=np.arange(len(domain_encoder.classes_)),
        target_names=domain_encoder.classes_,
        zero_division=0
    )
)

# Run five-fold cross-validation only when every class
# has at least five examples.
if domain_counts.min() >= 5:
    print("Running five-fold domain cross-validation...")

    cv_scores = cross_val_score(
        ExtraTreesClassifier(**MODEL_SETTINGS),
        X,
        y_domain,
        cv=5,
        scoring="balanced_accuracy",
        n_jobs=-1
    )

    print(
        f"Domain CV balanced accuracy: "
        f"{cv_scores.mean() * 100:.2f}% "
        f"(+/- {cv_scores.std() * 100:.2f}%)"
    )
else:
    print(
        "Skipping five-fold cross-validation: "
        "at least one domain class has fewer than five records."
    )

print("\nDomain model feature importance:")

for feature, importance in sorted(
    zip(feature_cols, model_domain.feature_importances_),
    key=lambda item: item[1],
    reverse=True
):
    print(f"  {feature:20s}: {importance:.4f}")

# ----------------------------------------------------------
# STEP 10: Train crime-type model
# ----------------------------------------------------------

print("\n" + "=" * 65)
print("TRAINING CRIME-TYPE MODEL")
print("=" * 65)

y_crime = df["crime_enc"]

crime_counts = y_crime.value_counts()

if crime_counts.min() < 2:
    raise ValueError(
        "At least two records per crime type are required "
        "for a stratified train/test split. Review rare classes "
        "or the dataset before training."
    )

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X,
    y_crime,
    test_size=0.2,
    random_state=42,
    stratify=y_crime
)

model_crime = ExtraTreesClassifier(
    n_estimators=50,
    max_depth=12,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
model_crime.fit(X_train_c, y_train_c)

crime_predictions = model_crime.predict(X_test_c)

crime_accuracy = accuracy_score(
    y_test_c, crime_predictions
)

crime_balanced_accuracy = balanced_accuracy_score(
    y_test_c, crime_predictions
)

print(f"Crime-type accuracy: {crime_accuracy * 100:.2f}%")
print(
    "Crime-type balanced accuracy: "
    f"{crime_balanced_accuracy * 100:.2f}%"
)

print("\nCrime-type classification report:")
print(
    classification_report(
        y_test_c,
        crime_predictions,
        labels=np.arange(len(crime_encoder.classes_)),
        target_names=crime_encoder.classes_,
        zero_division=0
    )
)

# ----------------------------------------------------------
# STEP 11: Save models and encoders
# ----------------------------------------------------------

artifacts = {
    "crime_model.pkl": model_domain,
    "crime_type_model.pkl": model_crime,
    "label_encoder.pkl": crime_encoder,
    "domain_encoder.pkl": domain_encoder,
    "district_encoder.pkl": city_encoder,
    "weapon_encoder.pkl": weapon_encoder,
    "gender_encoder.pkl": gender_encoder
}

print("\nSaving model files...")

for filename, artifact in artifacts.items():
    output_path = os.path.join(MODEL_DIR, filename)
    joblib.dump(artifact, output_path, compress=3)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)

    print(f"Saved {filename}: {size_mb:.2f} MB")

print("\n" + "=" * 65)
print("TRAINING COMPLETE")
print("=" * 65)
print(f"Training records: {len(df)}")
print(f"Cities: {len(city_summary)}")
print(f"Crime-domain accuracy: {domain_accuracy * 100:.2f}%")
print(f"Crime-domain balanced accuracy: "
      f"{domain_balanced_accuracy * 100:.2f}%")
print(f"Crime-type accuracy: {crime_accuracy * 100:.2f}%")
print(f"Crime-type balanced accuracy: "
      f"{crime_balanced_accuracy * 100:.2f}%")

print("\nSaved supporting files:")
print("  feature_cols.json")
print("  city_crime_freq.json")

print("\nImportant:")
print("- Verify that app.py loads both models correctly.")
print("- Verify that prediction features match feature_cols.json.")
print("- Test both prediction routes before deployment.")
print("- Check all .pkl files against GitHub's 100 MB limit.")
print("\nNext command: dir ml_model\\*.pkl")
