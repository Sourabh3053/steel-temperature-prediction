# ============================================================
# PROJECT: AI/ML Modeling for Steel Making
# MODEL: Random Forest Regression
# TARGET: T40_TEMP
# ============================================================

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# Ensure UTF-8 output encoding on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# 1. Locate and load the Excel dataset
# Resolve path relative to script directory so it runs from any working directory
script_dir = os.path.dirname(os.path.abspath(__file__))

candidate_files = [
    os.path.join(script_dir, "steel_temperature_random_sample.xlsx"),
    os.path.join(script_dir, "DOC-20261021-WA0000.xlsx"),
    "steel_temperature_random_sample.xlsx",
    "DOC-20261021-WA0000.xlsx"
]

file_path = None
for candidate in candidate_files:
    if os.path.exists(candidate):
        file_path = candidate
        break

if not file_path:
    raise FileNotFoundError(
        "Could not find dataset file ('steel_temperature_random_sample.xlsx' or 'DOC-20261021-WA0000.xlsx'). "
        f"Please verify the file exists in '{script_dir}'."
    )

print(f"Loading dataset from: {file_path}")
df = pd.read_excel(file_path, sheet_name="Random Sample Data")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)
print("\nFirst 5 records:")
print(df.head())

# 2. Select input features and data cleaning
# Exclude HEAT_ID and timestamps.
# Exclude LF2T40_TEMP_DROP because it is calculated
# using LF_LAST_TEMP and T40_TEMP in this dataset (target leakage).
features = [
    "LFTEMP_CASTSTART_DUR",
    "CASTSTART_TO_T40_DUR",
    "LFTEMP_T40_DUR",
    "LIQ_TEMP",
    "LF_LAST_TEMP",
    "LADLE_LIFE",
    "TAT"
]

target = "T40_TEMP"

# Drop duplicates and drop rows with missing values in target or features
initial_rows = len(df)
df = df.drop_duplicates()
df = df.dropna(subset=features + [target])
print(f"\nData cleaning completed. Rows retained: {len(df)} of {initial_rows}")

X = df[features]
y = df[target]

# 3. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# 4. Create and train the ML model
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=8,
    min_samples_leaf=2,
    random_state=42
)

model.fit(X_train, y_train)

# 5. Make predictions
y_pred = model.predict(X_test)

# 6. Evaluate model performance
mae = mean_absolute_error(y_test, y_pred)

# Robust RMSE calculation compatible across old and new scikit-learn versions
try:
    from sklearn.metrics import root_mean_squared_error
    rmse = root_mean_squared_error(y_test, y_pred)
except ImportError:
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

r2 = r2_score(y_test, y_pred)

print("\n========== MODEL RESULTS ==========")
print(f"MAE  : {mae:.2f} °C")
print(f"RMSE : {rmse:.2f} °C")
print(f"R²   : {r2:.4f}")
print(f"R² percentage: {r2 * 100:.2f}%")

# 7. Compare actual and predicted temperatures & save results
results = pd.DataFrame({
    "Actual_T40_TEMP": y_test.values,
    "Predicted_T40_TEMP": np.round(y_pred, 2)
})

print("\nActual vs Predicted (first 15 records):")
print(results.head(15).to_string(index=False))

results_csv_path = os.path.join(script_dir, "prediction_results.csv")
results.to_csv(results_csv_path, index=False)
print(f"\nPredictions saved to: {results_csv_path}")

# 8. Feature importance plot
importance = pd.Series(
    model.feature_importances_,
    index=features
).sort_values(ascending=False)

print("\nFeature Importance:")
print(importance)

plt.figure(figsize=(8, 5))
importance.plot(kind="bar", color="steelblue", edgecolor="black")
plt.title("Feature Importance - Steel Temperature Prediction")
plt.ylabel("Importance Score")
plt.xlabel("Features")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
feat_imp_path = os.path.join(script_dir, "feature_importance.png")
plt.savefig(feat_imp_path, dpi=300)
plt.close()
print(f"Feature importance graph saved to: {feat_imp_path}")

# 9. Actual vs Predicted graph with 45-degree reference line
plt.figure(figsize=(7, 5))
plt.scatter(y_test, y_pred, alpha=0.7, color="teal", edgecolors="k", label="Predictions")
min_val = min(y_test.min(), y_pred.min()) - 5
max_val = max(y_test.max(), y_pred.max()) + 5
plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Ideal (y = x)")
plt.xlim(min_val, max_val)
plt.ylim(min_val, max_val)
plt.xlabel("Actual T40_TEMP (°C)")
plt.ylabel("Predicted T40_TEMP (°C)")
plt.title("Actual vs Predicted Temperature")
plt.legend()
plt.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
actual_pred_path = os.path.join(script_dir, "actual_vs_predicted.png")
plt.savefig(actual_pred_path, dpi=300)
plt.close()
print(f"Actual vs Predicted graph saved to: {actual_pred_path}")

# 10. Save trained model
model_path = os.path.join(script_dir, "steel_temperature_model.pkl")
joblib.dump(model, model_path)

print(f"\nModel saved as: {model_path}")
print("Project completed successfully!")
