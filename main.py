# ==========================================
# SECOM Feature Selection + Model Evaluation
# LASSO (Embedded) | RFE (Wrapper) | mRMR (Hybrid)
# Linear Regression & Decision Tree updated
# ==========================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import seaborn as sns

from sklearn.feature_selection import mutual_info_classif
from imblearn.over_sampling import SMOTE

# ==========================================
# 1. PATH CONFIGURATION
# ==========================================
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = BASE_PATH
OUTPUT_PATH = os.path.join(BASE_PATH, "output")
FS_PATH = os.path.join(OUTPUT_PATH, "feature_selection")
MODEL_PATH = os.path.join(OUTPUT_PATH, "models")
METRIC_PATH = os.path.join(OUTPUT_PATH, "metrics")
FIG_PATH = os.path.join(OUTPUT_PATH, "figures")

for p in [FS_PATH, MODEL_PATH, METRIC_PATH, FIG_PATH]:
    os.makedirs(p, exist_ok=True)

# ==========================================
# 2. LOAD CLEANED DATA
# ==========================================
df = pd.read_csv(os.path.join(DATA_PATH, "cleaned_dataset.csv"))

y = df["L0"]
X = df.drop(columns=["L0"])

# Standardization
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

# ==========================================
# 3. FEATURE SELECTION METHODS
# ==========================================

feature_sets_configs = []  # List to hold (fs_name, features, X_train_used, y_train_used)

# ---------- 3.1 Embedded: LASSO ----------
lasso = Lasso(alpha=0.2)
lasso.fit(X_train, y_train)
lasso_features = X_train.columns[lasso.coef_ != 0].tolist()

pd.DataFrame({"Feature": lasso_features}).to_csv(
    os.path.join(FS_PATH, "lasso_features.csv"), index=False
)
print(f"LASSO selected {len(lasso_features)} features.")
if lasso_features:
    feature_sets_configs.append(("LASSO", lasso_features, X_train, y_train))

# ---------- 3.2 Wrapper: RFE with Linear Regression ----------
linreg = LinearRegression()
rfe = RFE(linreg, n_features_to_select=50)
rfe.fit(X_train, y_train)

rfe_features = X_train.columns[rfe.support_].tolist()

pd.DataFrame({"Feature": rfe_features}).to_csv(
    os.path.join(FS_PATH, "rfe_features.csv"), index=False
)
print(f"RFE selected {len(rfe_features)} features.")

if rfe_features:
    feature_sets_configs.append(("RFE_Original", rfe_features, X_train, y_train))

    sm = SMOTE(random_state=42)
    X_train_rfe_smoted, y_train_smoted = sm.fit_resample(X_train[rfe_features], y_train)
    print(
        f"SMOTE applied to RFE training data: Original samples {len(y_train)}, "
        f"SMOTEd samples {len(y_train_smoted)}"
    )
    feature_sets_configs.append(
        ("RFE_SMOTE", rfe_features, X_train_rfe_smoted, y_train_smoted)
    )

# ---------- 3.3 Hybrid: mRMR (MI-based approximation) ----------
mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
mi_df = pd.DataFrame({
    "Feature": X_train.columns,
    "MI_Score": mi_scores
}).sort_values(by="MI_Score", ascending=False)

mrmr_features = mi_df.head(50)["Feature"].tolist()

mi_df.to_csv(os.path.join(FS_PATH, "mrmr_scores.csv"), index=False)
pd.DataFrame({"Feature": mrmr_features}).to_csv(
    os.path.join(FS_PATH, "mrmr_features.csv"), index=False
)
print(f"mRMR selected {len(mrmr_features)} features.")
if mrmr_features:
    feature_sets_configs.append(("mRMR", mrmr_features, X_train, y_train))

# ==========================================
# 4. MODELS (5 total)
# ==========================================
models = {
    "LinearRegression": LinearRegression(),
    "SVM": SVC(kernel="rbf"),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
    "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42),
    "DecisionTree": DecisionTreeClassifier(random_state=42)
}

# ==========================================
# 5. TRAINING & EVALUATION
# ==========================================
results = []

for fs_name, features, X_train_used, y_train_used in feature_sets_configs:
    if not features:
        print(f"Skipping {fs_name} as it returned 0 features.")
        continue

    Xte = X_test[features]

    for model_name, model in models.items():
        model.fit(X_train_used[features], y_train_used)

        y_pred_test = model.predict(Xte)
        y_pred_train = model.predict(X_train_used[features])

        if model_name == "LinearRegression":
            y_pred_test = (y_pred_test >= 0.5).astype(int)
            y_pred_train = (y_pred_train >= 0.5).astype(int)

        acc_test = accuracy_score(y_test, y_pred_test)
        prec_test = precision_score(y_test, y_pred_test, zero_division=0)
        rec_test = recall_score(y_test, y_pred_test, zero_division=0)
        f1_test = f1_score(y_test, y_pred_test, zero_division=0)

        acc_train = accuracy_score(y_train_used, y_pred_train)
        prec_train = precision_score(y_train_used, y_pred_train, zero_division=0)
        rec_train = recall_score(y_train_used, y_pred_train, zero_division=0)
        f1_train = f1_score(y_train_used, y_pred_train, zero_division=0)

        results.append({
            "Feature_Selection": fs_name,
            "Model": model_name,
            "Accuracy_Test": acc_test,
            "Precision_Test": prec_test,
            "Recall_Test": rec_test,
            "F1_Score_Test": f1_test,
            "Accuracy_Train": acc_train,
            "Precision_Train": prec_train,
            "Recall_Train": rec_train,
            "F1_Score_Train": f1_train
        })

results_df = pd.DataFrame(results)
results_df["Accuracy_Gap"] = results_df["Accuracy_Train"] - results_df["Accuracy_Test"]
results_df["F1_Score_Gap"] = results_df["F1_Score_Train"] - results_df["F1_Score_Test"]

results_df.to_csv(
    os.path.join(METRIC_PATH, "model_performance_with_overfit.csv"),
    index=False
)

# ==========================================
# 6. OVERFIT GAP TABLE FOR RFE
# ==========================================
print("\n### Overfit Gap Table for RFE Feature Selection\n")

rfe_results = results_df[
    results_df["Feature_Selection"].isin(["RFE_Original", "RFE_SMOTE"])
]

if not rfe_results.empty:
    table_data = rfe_results[[
        "Feature_Selection",
        "Model",
        "Accuracy_Train", "Accuracy_Test", "Accuracy_Gap",
        "F1_Score_Train", "F1_Score_Test", "F1_Score_Gap"
    ]].round(3)

    markdown_table = table_data.to_markdown(index=False)
    print(markdown_table)
else:
    print("No RFE results found to generate the overfit gap table.")

# ==========================================
# 7. CONFUSION MATRIX VISUALIZATION
# ==========================================
print("\nGenerating confusion matrices for RFE_Original and RFE_SMOTE scenarios...")

for fs_config_name, features, X_train_used, y_train_used in feature_sets_configs:
    if fs_config_name.startswith("RFE_"):
        Xte = X_test[features]

        cm_models = {
            "LinearRegression": LinearRegression(),
            "SVM": SVC(kernel="rbf", random_state=42),
            "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
            "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42),
            "DecisionTree": DecisionTreeClassifier(random_state=42)
        }

        for model_name, model in cm_models.items():
            print(f"  Training {model_name} for {fs_config_name}...")
            model.fit(X_train_used[features], y_train_used)

            y_pred = model.predict(Xte)
            if model_name == "LinearRegression":
                y_pred = (y_pred >= 0.5).astype(int)

            cm = confusion_matrix(y_test, y_pred)

            plt.figure(figsize=(8, 6))
            sns.heatmap(
                cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted Negative', 'Predicted Positive'],
                yticklabels=['Actual Negative', 'Actual Positive']
            )
            plt.title(f'Confusion Matrix: {model_name} ({fs_config_name})')
            plt.ylabel('Actual Label')
            plt.xlabel('Predicted Label')

            cm_filename = os.path.join(
                FIG_PATH,
                f"confusion_matrix_{fs_config_name.lower()}_{model_name.lower()}.png"
            )
            plt.savefig(cm_filename, dpi=300)
            plt.close()
            print(f"  📁 Saved confusion matrix for {model_name} ({fs_config_name}) to {cm_filename}")

print("\n✅ Feature selection, modeling, and evaluation completed.")
print(f"📁 All outputs saved in: {OUTPUT_PATH}")
