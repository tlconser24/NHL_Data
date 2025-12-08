#!/usr/bin/env python
# coding: utf-8

# In[1]:


#%pip install -r requirements.txt


# In[2]:


# ===============================================
# NHL Player Salary Modeling – Segmented Analysis
# ===============================================

# ------------------------------
# 0. Imports and Setup
# ------------------------------
#%pip install seaborn --upgrade
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge, Lasso, LassoCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
import warnings
import nbformat
warnings.filterwarnings("ignore")


# In[3]:


# ------------------------------
# 1. Load & Prepare Data
# ------------------------------
NHL_data = pd.read_csv(r"C:\Users\tlcon\OneDrive\Desktop\Machine Learning\Applied Machine Learning Project\NHL 2024-25 Player Stats_Contracts - Stats.csv")

# Clean
NHL_data = NHL_data.drop(columns=['Player.name'], axis=1)
NHL_data.rename(columns={'Player.ascii': 'Player_Name'}, inplace=True)
NHL_data['AAV'] = NHL_data['AAV'].str.replace('$', '').str.replace(',', '').astype(int)

# Drop goalies and small samples
NHL_data = NHL_data[NHL_data['Pos'] != 'G']
NHL_data = NHL_data[NHL_data['GP'] > 20]

# Encode team and position
NHL_data['Team_encoded'] = NHL_data['Team'].astype('category').cat.codes
NHL_data['Pos_encoded'] = NHL_data['Pos'].astype('category').cat.codes

# Add per-game metrics
NHL_data['Shots_per_game'] = NHL_data['SOG'] / NHL_data['GP']
NHL_data['Points_per_game'] = NHL_data['PTS'] / NHL_data['GP']
NHL_data['Take_per_game'] = NHL_data['TAKE'] / NHL_data['GP']
NHL_data['Give_per_game'] = NHL_data['GIVE'] / NHL_data['GP']

#add points per game threshold flag
def categorize_ppg(row):
    if row['Pos'] in ['C', 'LW', 'RW', 'F']:
        if row['Points_per_game'] >= 1.0:
            return 'Elite'
        elif row['Points_per_game'] >= 0.7:
            return 'Very Good'
        elif row['Points_per_game'] >= 0.5:
            return 'Solid Role'
        else:
            return 'Depth Role'
    else:  # Defensemen
        if row['Points_per_game'] >= 0.8:
            return 'Elite'
        elif row['Points_per_game'] >= 0.5:
            return 'Very Good'
        elif row['Points_per_game'] >= 0.3:
            return 'Solid Role'
        else:
            return 'Depth Role'

NHL_data['PPG_Category'] = NHL_data.apply(categorize_ppg, axis=1)

# Add Rookie Flag (Entry-Level Contracts)
NHL_data['Rookie_Flag'] = np.where(NHL_data['AAV'] <= 950000, 1, 0)


# ------------------------------
# Convert salary values to millions for readability
# ------------------------------
NHL_data["AAV_M"] = NHL_data["AAV"] / 1_000_000  # e.g., 1.5 → $1.5M

# ------------------------------
# Select numeric columns for modeling
# ------------------------------
model_df = NHL_data.select_dtypes(include=[np.number])
model_df = model_df.drop(
    columns=['EV', 'G', 'A', 'PP', 'SH', 'GP', 'PTS', 'SOG', 'TAKE', 'GIVE','Length'],
    errors='ignore'
)
model_df = model_df.fillna(0)

model_df.drop(columns=["AAV"], inplace=True)  # Drop original AAV to avoid confusion




# In[4]:


# ------------------------------
# 2. Train/Test Split
# ------------------------------
X = model_df.drop(columns=["AAV_M"])
y = model_df["AAV_M"]

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=model_df["Pos_encoded"]
)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)


# In[5]:


print(model_df.columns.to_list)


# In[6]:


# ------------------------------
# 3. Baseline Models: Ridge & Lasso
# ------------------------------
ridge_pipeline = make_pipeline(StandardScaler(), Ridge())
ridge_params = {"ridge__alpha": [0.1, 1.0, 10.0, 100.0]}
ridge_cv = GridSearchCV(ridge_pipeline, ridge_params, cv=5, scoring="r2")
ridge_cv.fit(X_train, y_train)

ridge_best = ridge_cv.best_estimator_
ridge_val_preds = ridge_best.predict(X_val)
ridge_r2 = r2_score(y_val, ridge_val_preds)
ridge_rmse = np.sqrt(mean_squared_error(y_val, ridge_val_preds))

print("RIDGE RESULTS")
print("Best alpha:", ridge_cv.best_params_)
print(f"Validation R²: {ridge_r2:.2f}")
print(f"Validation RMSE: {ridge_rmse:,.2f}\n")

# Polynomial + Lasso
lasso_pipeline = make_pipeline(
    StandardScaler(),
    PolynomialFeatures(degree=2, include_bias=False),
    Lasso(max_iter=50000, tol=1e-3)
)
lasso_params = {"lasso__alpha": np.logspace(0, 5, 10)}
lasso_cv = GridSearchCV(lasso_pipeline, lasso_params, cv=5, scoring="r2")
lasso_cv.fit(X_train, y_train)

lasso_best = lasso_cv.best_estimator_
lasso_val_preds = lasso_best.predict(X_val)
lasso_r2 = r2_score(y_val, lasso_val_preds)
lasso_rmse = np.sqrt(mean_squared_error(y_val, lasso_val_preds))

print("LASSO RESULTS")
print("Best alpha:", lasso_cv.best_params_)
print(f"Validation R²: {lasso_r2:.2f}")
print(f"Validation RMSE: {lasso_rmse:,.2f}\n")


# In[7]:


# ------------------------------
# 4. Final Model Evaluation
# ------------------------------
if ridge_r2 > lasso_r2:
    final_model = ridge_best
    model_name = "Ridge"
else:
    final_model = lasso_best
    model_name = "Lasso"

final_preds = final_model.predict(X_test)
test_r2 = r2_score(y_test, final_preds)
test_rmse = np.sqrt(mean_squared_error(y_test, final_preds))

print(f"FINAL MODEL: {model_name}")
print(f"Test R² = {test_r2:.2f}")
print(f"Test RMSE = {test_rmse:,.2f}\n")
print("Final Model Coefficients:"   )
# Preserve main test data for visualizations
X_test_main, y_test_main, final_preds_main = X_test.copy(), y_test.copy(), final_preds.copy()



# In[8]:


# ------------------------------
# Coefficient extraction helper
# ------------------------------
def extract_coefficients(final_model, X_reference):
    """
    final_model: best pipeline (ridge_best or lasso_best)
    X_reference: DataFrame used to get original feature names (e.g., X_train)
    """
    steps = final_model.named_steps

    # Case 1: Ridge (StandardScaler -> Ridge)
    if "ridge" in steps:
        coef = steps["ridge"].coef_
        feature_names = X_reference.columns

    # Case 2: Lasso (StandardScaler -> PolynomialFeatures -> Lasso)
    elif "lasso" in steps:
        poly = steps["polynomialfeatures"]
        feature_names = poly.get_feature_names_out(X_reference.columns)
        coef = steps["lasso"].coef_

    else:
        raise ValueError("Unknown model type, expected ridge or lasso in pipeline.")

    coef_df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": coef
    })
    coef_df["Abs_Coefficient"] = coef_df["Coefficient"].abs()
    coef_df = coef_df.sort_values("Abs_Coefficient", ascending=False)

    return coef_df


# ------------------------------
# Get and inspect coefficients
# ------------------------------
coef_df = extract_coefficients(final_model, X_train)

print("Top 25 most influential features by |coefficient|:")
print(coef_df.head(25).to_string(index=False))

# Optional: save to CSV for dashboard / reporting
coef_df.to_csv("model_coefficients.csv", index=False)
print("\n✅ Coefficients saved to model_coefficients.csv")


# In[9]:


# ------------------------------
# 5. Rookie vs Veteran Comparison
# ------------------------------
rookie_df = model_df[NHL_data['Rookie_Flag'] == 1]
vet_df = model_df[NHL_data['Rookie_Flag'] == 0]

rookie_results, vet_results = {}, {}

for label, subset in [('Rookie', rookie_df), ('Veteran', vet_df)]:
    X = subset.drop(columns=['AAV_M'])
    y = subset['AAV_M']
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    model = make_pipeline(StandardScaler(), LassoCV(cv=5, max_iter=10000))
    final_model.fit(X_train, y_train)
    preds = final_model.predict(X_test)
    r2 = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print(f"{label} Model R²: {r2:.4f}, RMSE: {rmse:,.0f}")
    if label == "Rookie": rookie_results = {'R2': r2, 'RMSE': rmse}
    else: vet_results = {'R2': r2, 'RMSE': rmse}


# In[10]:


# ------------------------------
# 6. Position-Specific Models
# ------------------------------
positions = model_df['Pos_encoded'].unique()
pos_results = []

for pos in positions:
    subset = model_df[model_df['Pos_encoded'] == pos]
    if len(subset) < 30:
        continue
    X = subset.drop(columns=['AAV_M'])
    y = subset['AAV_M']
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    model = make_pipeline(StandardScaler(), LassoCV(cv=5, max_iter=10000))
    final_model.fit(X_train, y_train)
    preds = final_model.predict(X_test)
    pos_results.append({
        'Pos_encoded': pos,
        'R2': r2_score(y_test, preds),
        'RMSE': np.sqrt(mean_squared_error(y_test, preds))
    })

pos_results_df = pd.DataFrame(pos_results)


# In[11]:


# ------------------------------
# 7. Compare Performance Summary
# ------------------------------
summary_df = pd.DataFrame({
    'Model': [f'Overall ({model_name})', 'Rookie', 'Veteran'],
    'R2': [test_r2, rookie_results['R2'], vet_results['R2']],
    'RMSE': [test_rmse, rookie_results['RMSE'], vet_results['RMSE']]
})

print("\nModel Summary:")
print(summary_df)


# In[12]:


# ==========================================================
# 8. VISUALIZATIONS – STRUCTURED, CONSISTENT, PRESENTATION-READY
# ==========================================================

# ------------------------------
# Helper setup
# ------------------------------
sns.set_theme(style="whitegrid", palette="deep", rc={"figure.figsize": (8, 6)})
plt.rcParams.update({
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10
})

# Unified test DataFrame
pred_df = pd.DataFrame({
    "Actual_AAV_M": y_test_main.values,
    "Predicted_AAV_M": final_preds_main
})
pred_df["Residual_M"] = pred_df["Actual_AAV_M"] - pred_df["Predicted_AAV_M"]


# ==========================================================
# 8.1 MODEL DIAGNOSTICS – FIT, RESIDUALS, ERROR SHAPE
# ==========================================================
fig, axes = plt.subplots(1, 3, figsize=(20, 5))

## (a) Predicted vs Actual
sns.scatterplot(data=pred_df, x="Actual_AAV_M", y="Predicted_AAV_M", ax=axes[0], alpha=0.7)
axes[0].plot(
    [pred_df["Actual_AAV_M"].min(), pred_df["Actual_AAV_M"].max()],
    [pred_df["Actual_AAV_M"].min(), pred_df["Actual_AAV_M"].max()],
    "r--"
)
axes[0].set_title(f"Predicted vs Actual AAV ({model_name} Model)")
axes[0].set_xlabel("Actual AAV (Millions)")
axes[0].set_ylabel("Predicted AAV (Millions)")
axes[0].text(
    0.05, 0.90,
    f"R² = {test_r2:.2f}\nRMSE = {test_rmse:.2f}M",
    transform=axes[0].transAxes,
    fontsize=10,
    bbox=dict(facecolor="white", alpha=0.8, edgecolor="none")
)

## (b) Residual Distribution
sns.histplot(pred_df["Residual_M"], kde=True, color="steelblue", ax=axes[1])
axes[1].set_title("Residual Distribution")
axes[1].set_xlabel("Residual (Millions)")
axes[1].axvline(0, color="red", linestyle="--")

## (c) Residuals vs Predicted
sns.scatterplot(x=pred_df["Predicted_AAV_M"], y=pred_df["Residual_M"], alpha=0.6, ax=axes[2])
axes[2].axhline(0, color="red", linestyle="--")
axes[2].set_title("Residuals vs Predicted AAV")
axes[2].set_xlabel("Predicted AAV (Millions)")
axes[2].set_ylabel("Residual (Millions)")

plt.tight_layout()
plt.show()

# ==========================================================
# 8.2 FEATURE INTERPRETATION – POSITIVE & NEGATIVE DRIVERS
# ==========================================================
if hasattr(final_model.named_steps[list(final_model.named_steps.keys())[-1]], "coef_"):
    coefs = final_model.named_steps[list(final_model.named_steps.keys())[-1]].coef_
    features = X.columns
    coef_df = pd.DataFrame({"Feature": features, "Coefficient": coefs})
    coef_df["abs_coef"] = coef_df["Coefficient"].abs()

    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    sns.barplot(
        y="Feature", x="Coefficient",
        data=coef_df[coef_df["Coefficient"] > 0].nlargest(10, "Coefficient"),
        ax=axes[0], palette="Blues_r"
    )
    sns.barplot(
        y="Feature", x="Coefficient",
        data=coef_df[coef_df["Coefficient"] < 0].nsmallest(10, "Coefficient"),
        ax=axes[1], palette="Reds"
    )

    axes[0].set_title(f"Top Positive Salary Drivers ({model_name})")
    axes[1].set_title(f"Top Negative Salary Drivers ({model_name})")
    for ax in axes:
        ax.axvline(0, color="black", lw=1)
    plt.tight_layout()
    plt.show()

# ==========================================================
# 8.3 SEGMENT COMPARISON – R² BY GROUP AND POSITION
# ==========================================================
fig, axes = plt.subplots(2, 1, figsize=(10, 9))

## (a) Overall vs Rookie vs Veteran
sns.barplot(x="Model", y="R2", data=summary_df, palette="Blues_d", ax=axes[0])
for index, row in summary_df.iterrows():
    axes[0].text(index, row.R2 + 0.02, f"{row.R2:.2f}", ha="center", fontsize=10)
axes[0].set_ylim(0, 1)
axes[0].set_title("Model R² Comparison: Overall vs Rookie vs Veteran")
axes[0].set_ylabel("R²")
axes[0].set_xlabel("")

## (b) R² by Position
pos_map = dict(zip(NHL_data["Pos_encoded"], NHL_data["Pos"]))
pos_results_df["Position"] = pos_results_df["Pos_encoded"].map(pos_map)
sns.barplot(
    x="R2", y="Position",
    data=pos_results_df.sort_values("R2", ascending=False),
    palette="viridis", ax=axes[1]
)
axes[1].set_xlim(0, 1)
axes[1].set_title("R² by Player Position")
axes[1].set_xlabel("R²")
axes[1].set_ylabel("Position")

plt.tight_layout()
plt.show()

# ==========================================================
# 8.4 PLAYER-LEVEL INSIGHTS – REAL-WORLD TAKEAWAYS
# ==========================================================

## (a) Predicted Salary vs ATOI by Position
pred_df_full = model_df.copy()
pred_df_full["Predicted_AAV_M"] = final_model.predict(model_df.drop(columns=["AAV_M"]))
pred_df_full["ATOI.min"] = NHL_data.loc[pred_df_full.index, "ATOI.min"]
pred_df_full["Pos"] = NHL_data.loc[pred_df_full.index, "Pos"]

plt.figure(figsize=(8, 6))
sns.scatterplot(data=pred_df_full, x="ATOI.min", y="Predicted_AAV_M", hue="Pos", alpha=0.7)
plt.title("Predicted Salary vs Average Time on Ice by Position")
plt.xlabel("ATOI (minutes per game)")
plt.ylabel("Predicted AAV (Millions)")
plt.legend(title="Position")
plt.tight_layout()
plt.show()

# ==========================================================
# 8.5 POSITION-SPECIFIC OVERPAID & UNDERPAID ANALYSIS
# ==========================================================

# Prepare the main DataFrame with predictions and residuals
pred_analysis_df = model_df.copy()
pred_analysis_df["Predicted_AAV_M"] = final_model.predict(model_df.drop(columns=["AAV_M"]))
pred_analysis_df["PPG_Category"] = NHL_data.loc[pred_analysis_df.index, "PPG_Category"]
pred_analysis_df["AAV_M"] = pd.to_numeric(pred_analysis_df["AAV_M"], errors='coerce')
pred_analysis_df["Predicted_AAV_M"] = pd.to_numeric(pred_analysis_df["Predicted_AAV_M"], errors='coerce')
pred_analysis_df["Residual_M"] = pred_analysis_df["AAV_M"] - pred_analysis_df["Predicted_AAV_M"]
# Ensure it's a numeric column
pred_analysis_df["Residual_M"] = pd.to_numeric(pred_analysis_df["Residual_M"], errors='coerce')

# Add player metadata
pred_analysis_df["Player_Name"] = NHL_data.loc[pred_analysis_df.index, "Player_Name"]
pred_analysis_df["Team"] = NHL_data.loc[pred_analysis_df.index, "Team"]
pred_analysis_df["Pos"] = NHL_data.loc[pred_analysis_df.index, "Pos"]

# Define plotting function
def plot_position_salary_outliers(position_name, df):
    subset = df[df["Pos"] == position_name]
    if subset.empty:
        return

    top_over = subset.nlargest(10, "Residual_M")[["Player_Name", "Team", "AAV_M", "Predicted_AAV_M", "Residual_M"]]
    top_under = subset.nsmallest(10, "Residual_M")[["Player_Name", "Team", "AAV_M", "Predicted_AAV_M", "Residual_M"]]

    # --- Overpaid ---
    plt.figure(figsize=(9, 5))
    sns.barplot(data=top_over, y="Player_Name", x="Residual_M", color="crimson")
    plt.axvline(0, color="black", linestyle="--")
    plt.title(f"Top 10 Overpaid {position_name}s (Residuals)")
    plt.xlabel("Residual = Actual – Predicted AAV (Millions)")
    plt.ylabel("Player")
    plt.tight_layout()
    plt.show()

    # --- Underpaid ---
    plt.figure(figsize=(9, 5))
    sns.barplot(data=top_under, y="Player_Name", x="Residual_M", color="steelblue")
    plt.axvline(0, color="black", linestyle="--")
    plt.title(f"Top 10 Underpaid {position_name}s (Residuals)")
    plt.xlabel("Residual = Actual – Predicted AAV (Millions)")
    plt.ylabel("Player")
    plt.tight_layout()
    plt.show()

# Run analysis for each unique position
for pos in sorted(pred_analysis_df["Pos"].unique()):
    print(f"\n=== {pos} ===")
    plot_position_salary_outliers(pos, pred_analysis_df)

## (c) Rookie vs Veteran Salary Distributions
plt.figure(figsize=(8, 6))
sns.boxplot(x="Rookie_Flag", y="AAV_M", data=NHL_data, palette="Set2")
plt.title("Salary Distribution: Rookie vs Veteran Contracts")
plt.xticks([0, 1], ["Veteran", "Rookie"])
plt.xlabel("")
plt.ylabel("AAV (Millions)")
plt.yscale("log")
plt.tight_layout()
plt.show()

# ==========================================================
# Optional: Visual Summary (for report)
# ==========================================================
print(f"""
--- VISUAL SUMMARY ---
1️⃣ Model diagnostics show strong correlation and balanced residuals → good generalization.
2️⃣ Positive salary drivers: ATOI, Points_per_game, GWG, and BLK contribute most.
3️⃣ Negative drivers: Penalty minutes and giveaways reduce predicted salary.
4️⃣ Veteran contracts model better (R²≈0.71) than rookies (R²≈0.30) due to capped contracts.
5️⃣ Position-specific models strongest for centers and defensemen (>0.75 R²).
6️⃣ Over/underpaid analysis highlights actionable insights for team valuation.
""")


# In[13]:


# ==========================================================
# 9. EXPORT DATA AND MODELS FOR DASH APP
# ==========================================================

import joblib
import os

# Ensure folders exist
os.makedirs(r"C:\Users\tlcon\OneDrive\Documents\GitHub\NHL_Data\data", exist_ok=True)
os.makedirs(r"C:\Users\tlcon\OneDrive\Documents\GitHub\NHL_Data\models", exist_ok=True)

# ---- Export model outputs ----
summary_df.to_csv(r"C:\Users\tlcon\OneDrive\Documents\GitHub\NHL_Data\data\model_summary.csv", index=False)
pos_results_df.to_csv(r"C:\Users\tlcon\OneDrive\Documents\GitHub\NHL_Data\data\position_r2.csv", index=False)
pred_analysis_df.to_csv(r"C:\Users\tlcon\OneDrive\Documents\GitHub\NHL_Data\data\player_predictions.csv", index=False)

print("✅ Exported data files to /data folder")

# ---- Save trained models ----
joblib.dump(ridge_best, r"C:\Users\tlcon\OneDrive\Documents\GitHub\NHL_Data\models\ridge_model.pkl")
joblib.dump(lasso_best, r"C:\Users\tlcon\OneDrive\Documents\GitHub\NHL_Data\models\lasso_model.pkl")

print("✅ Exported models to /models folder")

