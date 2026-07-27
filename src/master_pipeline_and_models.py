import os
import glob
import gc
import time
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import issparse

warnings.filterwarnings("ignore")

# ML & Sklearn Imports
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

# SHAP for Model Explainability
import shap

# -----------------------------------------------------------------------------
# DIRECTORY SETUP
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")
MODELS_DIR = os.path.join(BASE_DIR, "models")

for folder in [RAW_DATA_DIR, PROCESSED_DATA_DIR, REPORTS_DIR, FIGURES_DIR, MODELS_DIR]:
    os.makedirs(folder, exist_ok=True)

# -----------------------------------------------------------------------------
# EVALUATION METRICS HELPER
# -----------------------------------------------------------------------------
def calculate_comprehensive_metrics(y_true, y_pred, model_name="Model"):
    y_pred_clipped = np.clip(y_pred, 1.0, 180.0)
    abs_errors = np.abs(y_true - y_pred_clipped)
    
    mae = mean_absolute_error(y_true, y_pred_clipped)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred_clipped))
    medae = median_absolute_error(y_true, y_pred_clipped)
    r2 = r2_score(y_true, y_pred_clipped)
    rmsle = np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred_clipped)))
    
    p90 = np.percentile(abs_errors, 90)
    acc_2m = np.mean(abs_errors <= 2.0) * 100
    acc_5m = np.mean(abs_errors <= 5.0) * 100
    acc_10m = np.mean(abs_errors <= 10.0) * 100

    return {
        "Model": model_name,
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "MedianAE": round(medae, 4),
        "R2": round(r2, 4),
        "RMSLE": round(rmsle, 4),
        "P90_Error": round(p90, 4),
        "Acc_±2m (%)": round(acc_2m, 2),
        "Acc_±5m (%)": round(acc_5m, 2),
        "Acc_±10m (%)": round(acc_10m, 2)
    }

# -----------------------------------------------------------------------------
# CUSTOM TRANSFORMER FOR FEATURE ENGINEERING
# -----------------------------------------------------------------------------
class TaxiFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()
        pickup_dt = pd.to_datetime(X_out["tpep_pickup_datetime"])

        X_out["pickup_hour"] = pickup_dt.dt.hour.astype(np.int8)
        X_out["day_of_week"] = pickup_dt.dt.dayofweek.astype(np.int8)
        X_out["pickup_month"] = pickup_dt.dt.month.astype(np.int8)
        X_out["is_weekend"] = (X_out["day_of_week"] >= 5).astype(np.int8)
        X_out["is_peak_hour"] = (
            (X_out["is_weekend"] == 0) & 
            (((X_out["pickup_hour"] >= 7) & (X_out["pickup_hour"] <= 10)) | 
             ((X_out["pickup_hour"] >= 16) & (X_out["pickup_hour"] <= 19)))
        ).astype(np.int8)
        
        X_out["is_night_trip"] = ((X_out["pickup_hour"] >= 22) | (X_out["pickup_hour"] <= 6)).astype(np.int8)
        X_out["hour_sin"] = np.sin(2 * np.pi * X_out["pickup_hour"] / 24.0).astype(np.float32)
        X_out["hour_cos"] = np.cos(2 * np.pi * X_out["pickup_hour"] / 24.0).astype(np.float32)
        X_out["dow_sin"] = np.sin(2 * np.pi * X_out["day_of_week"] / 7.0).astype(np.float32)
        X_out["dow_cos"] = np.cos(2 * np.pi * X_out["day_of_week"] / 7.0).astype(np.float32)
        
        if "pickup_borough" in X_out.columns and "dropoff_borough" in X_out.columns:
            X_out["is_same_borough"] = (X_out["pickup_borough"] == X_out["dropoff_borough"]).astype(np.int8)
        else:
            X_out["is_same_borough"] = np.int8(0)

        airport_ids = {1, 132, 138}
        X_out["is_airport"] = (X_out["PULocationID"].isin(airport_ids) | X_out["DOLocationID"].isin(airport_ids)).astype(np.int8)
        return X_out

def build_pipeline():
    num_features = [
        "passenger_count", "pickup_hour", "day_of_week", "pickup_month", 
        "is_weekend", "is_peak_hour", "is_night_trip", "hour_sin", "hour_cos", 
        "dow_sin", "dow_cos", "is_same_borough", "is_airport"
    ]
    cat_features = ["PULocationID", "DOLocationID", "pickup_borough", "dropoff_borough"]

    num_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32))
    ])

    return Pipeline([
        ("feature_extractor", TaxiFeatureExtractor()), 
        ("preprocessor", ColumnTransformer([
            ("num", num_transformer, num_features), 
            ("cat", cat_transformer, cat_features)
        ], remainder="drop"))
    ])

# =============================================================================
# MAIN MASTER EXECUTION
# =============================================================================
def main():
    print("=" * 80)
    print("     NYC TAXI TRIP DURATION - COMPLETE INTEGRATED MASTER WORKFLOW")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # STAGE 1: DATA AUDIT & DOCUMENTATION GENERATION (TASK 1)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("STAGE 1: DATA AUDIT, SCHEMA CHECKS & REPORT GENERATION")
    print("=" * 50)

    # Write Task 1: Business Problem Statement
    doc_path = os.path.join(REPORTS_DIR, "business_problem_statement.md")
    with open(doc_path, "w") as f:
        f.write("# One-Page Business Problem Statement: NYC Taxi ETA Prediction\n\n")
        f.write("## 1. Business Objective\nProvide highly accurate ETA predictions at ride request time.\n\n")
        f.write("## 2. Target Variable & Unit\nTarget: `trip_duration_minutes` (Derived from timestamps).\n\n")
        f.write("## 3. Asymmetric Business Costs\n- **Underprediction:** Customer dissatisfaction & missed schedules.\n- **Overprediction:** High quoted ETA leads to customer churn.\n\n")
        f.write("## 4. Deployment Constraints\nStrictly pre-trip features. Post-trip data (`trip_distance`, `fare_amount`) prohibited.\n")
    print(f"✓ Business Problem Statement exported to {doc_path}")

    files = sorted(glob.glob(os.path.join(RAW_DATA_DIR, "*.parquet")))
    if not files:
        print(f"ERROR: No parquet files found in {RAW_DATA_DIR}.")
        return

    # Extended column list to test Leakage features if present
    cols_to_load = ["tpep_pickup_datetime", "tpep_dropoff_datetime", "passenger_count", "PULocationID", "DOLocationID"]
    
    first_file_cols = [c.strip() for c in pd.read_parquet(files[0]).columns]
    has_leakage_cols = "trip_distance" in first_file_cols and "fare_amount" in first_file_cols
    if has_leakage_cols:
        cols_to_load.extend(["trip_distance", "fare_amount"])

    df_list = [pd.read_parquet(f, columns=cols_to_load) for f in files]
    df = pd.concat(df_list, ignore_index=True)

    print(f"Total Raw Records Loaded: {len(df):,}")

    df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
    df["tpep_dropoff_datetime"] = pd.to_datetime(df["tpep_dropoff_datetime"])
    
    raw_duration = (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds() / 60.0
    df["trip_duration_minutes"] = raw_duration
    df["passenger_count"] = df["passenger_count"].fillna(1)
    
    clean_mask = (
        (df["trip_duration_minutes"] >= 1.0) & 
        (df["trip_duration_minutes"] <= 180.0) & 
        (df["passenger_count"] > 0) & 
        (df["passenger_count"] <= 6)
    )
    clean_df = df[clean_mask].copy()
    del df
    gc.collect()

    clean_df["log_trip_duration_minutes"] = np.log1p(clean_df["trip_duration_minutes"])
    print(f"Target Skewness: Raw = {clean_df['trip_duration_minutes'].skew():.4f}, Log = {clean_df['log_trip_duration_minutes'].skew():.4f}")

    # -------------------------------------------------------------------------
    # STAGE 2: EDA & PLOTS
    # -------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("STAGE 2: COMPREHENSIVE EDA & PLOTS GENERATION")
    print("=" * 50)

    clean_df["pickup_hour"] = clean_df["tpep_pickup_datetime"].dt.hour
    clean_df["day_name"] = clean_df["tpep_pickup_datetime"].dt.day_name()
    clean_df["pickup_month"] = clean_df["tpep_pickup_datetime"].dt.month

    lookup_file = os.path.join(RAW_DATA_DIR, "taxi_zone_lookup.csv")
    if os.path.exists(lookup_file):
        zone_df = pd.read_csv(lookup_file)
        clean_df = clean_df.merge(zone_df[["LocationID", "Borough"]], left_on="PULocationID", right_on="LocationID", how="left").rename(columns={"Borough": "pickup_borough"}).drop(columns=["LocationID"])
        clean_df = clean_df.merge(zone_df[["LocationID", "Borough"]], left_on="DOLocationID", right_on="LocationID", how="left").rename(columns={"Borough": "dropoff_borough"}).drop(columns=["LocationID"])
    else:
        clean_df["pickup_borough"], clean_df["dropoff_borough"] = "Unknown", "Unknown"

    sample_eda = clean_df.sample(n=min(50000, len(clean_df)), random_state=42)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    sns.barplot(data=sample_eda, x="pickup_hour", y="trip_duration_minutes", ax=axes[0, 0], errorbar=None, palette="mako")
    axes[0, 0].set_title("Duration by Pickup Hour")

    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    sns.barplot(data=sample_eda, x="day_name", y="trip_duration_minutes", order=day_order, ax=axes[0, 1], errorbar=None, palette="coolwarm")
    axes[0, 1].set_title("Duration by Day of Week")

    sns.barplot(data=sample_eda, x="pickup_month", y="trip_duration_minutes", ax=axes[1, 0], errorbar=None, palette="crest")
    axes[1, 0].set_title("Monthly Trend")

    sns.barplot(data=sample_eda, x="passenger_count", y="trip_duration_minutes", ax=axes[1, 1], errorbar=None, palette="magma")
    axes[1, 1].set_title("Duration by Passenger Count")
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "eda_temporal_trends.png"))
    plt.close()

    # -------------------------------------------------------------------------
    # STAGE 3: TEMPORAL SPLIT & PIPELINE
    # -------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("STAGE 3: TIME-BASED TRAIN/VAL/TEST SPLIT & PIPELINE")
    print("=" * 50)

    train_mask = clean_df["pickup_month"].isin([1, 2, 3, 4])
    val_mask = clean_df["pickup_month"] == 5
    test_mask = clean_df["pickup_month"] == 6

    train_df = clean_df[train_mask].copy()
    val_df = clean_df[val_mask].copy()
    test_df = clean_df[test_mask].copy()

    TRAIN_SAMPLE_SIZE = min(1_000_000, len(train_df))
    VAL_SAMPLE_SIZE = min(250_000, len(val_df))
    TEST_SAMPLE_SIZE = min(250_000, len(test_df))

    train_df_sample = train_df.sample(n=TRAIN_SAMPLE_SIZE, random_state=42)
    val_df_sample = val_df.sample(n=VAL_SAMPLE_SIZE, random_state=42)
    test_df_sample = test_df.sample(n=TEST_SAMPLE_SIZE, random_state=42)

    pipeline = build_pipeline()
    X_train = pipeline.fit_transform(train_df_sample)
    X_val = pipeline.transform(val_df_sample)
    X_test = pipeline.transform(test_df_sample)

    y_train = train_df_sample["trip_duration_minutes"].values
    y_val = val_df_sample["trip_duration_minutes"].values
    y_test = test_df_sample["trip_duration_minutes"].values

    y_train_log = train_df_sample["log_trip_duration_minutes"].values
    y_val_log = val_df_sample["log_trip_duration_minutes"].values

    MAX_SUB = 50000
    sub_idx = np.random.choice(X_train.shape[0], min(MAX_SUB, X_train.shape[0]), replace=False)
    X_train_sub = X_train[sub_idx]
    y_train_sub = y_train[sub_idx]
    y_train_log_sub = y_train_log[sub_idx]

    benchmark_results = []

    # -------------------------------------------------------------------------
    # STAGE 4: BASELINES & LINEARS
    # -------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("STAGE 4: BASELINES, LINEAR MODELS & VIF")
    print("=" * 50)

    global_mean = np.mean(y_train)
    global_median = np.median(y_train)
    benchmark_results.append(calculate_comprehensive_metrics(y_val, np.full_like(y_val, global_mean), "Baseline 1: Global Mean"))
    benchmark_results.append(calculate_comprehensive_metrics(y_val, np.full_like(y_val, global_median), "Baseline 2: Global Median"))

    lr_log = LinearRegression(n_jobs=1).fit(X_train_sub, y_train_log_sub)
    benchmark_results.append(calculate_comprehensive_metrics(y_val, np.expm1(lr_log.predict(X_val)), "LinearRegression (Log Target)"))

    # VIF Check
    X_dense_vif = X_train_sub[:1000, :10].toarray() if issparse(X_train_sub) else X_train_sub[:1000, :10]
    vifs = [variance_inflation_factor(X_dense_vif, i) for i in range(X_dense_vif.shape[1])]
    print("VIF Summary Top 10 Features:", np.round(vifs, 2))

    # -------------------------------------------------------------------------
    # STAGE 5: TASK 5 - SAFE VS. ORACLE LEAKAGE EXPERIMENT
    # -------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("STAGE 5: LEAKAGE EXPERIMENT (SAFE VS. ORACLE MODEL)")
    print("=" * 50)

    if has_leakage_cols:
        X_safe_leak = train_df_sample[["PULocationID", "DOLocationID", "pickup_hour"]].fillna(0)
        X_oracle_leak = train_df_sample[["PULocationID", "DOLocationID", "pickup_hour", "trip_distance", "fare_amount"]].fillna(0)
        
        y_leak = train_df_sample["log_trip_duration_minutes"].values

        safe_m = HistGradientBoostingRegressor(max_iter=50, random_state=42).fit(X_safe_leak[:20000], y_leak[:20000])
        oracle_m = HistGradientBoostingRegressor(max_iter=50, random_state=42).fit(X_oracle_leak[:20000], y_leak[:20000])

        val_safe = val_df_sample[["PULocationID", "DOLocationID", "pickup_hour"]].fillna(0)
        val_oracle = val_df_sample[["PULocationID", "DOLocationID", "pickup_hour", "trip_distance", "fare_amount"]].fillna(0)

        p_safe = np.expm1(safe_m.predict(val_safe[:10000]))
        p_oracle = np.expm1(oracle_m.predict(val_oracle[:10000]))

        mae_s = mean_absolute_error(y_val[:10000], p_safe)
        mae_o = mean_absolute_error(y_val[:10000], p_oracle)

        print(f"✓ Deployment-Safe Model MAE: {mae_s:.4f} mins")
        print(f"✓ Oracle (Leakage) Model MAE:  {mae_o:.4f} mins")
        print(f"  --> Oracle reduces error by {((mae_s - mae_o)/mae_s)*100:.2f}%, but CANNOT be deployed pre-trip.")
    else:
        print("[INFO] Skipping Oracle experiment: `trip_distance` / `fare_amount` absent from raw features.")

    # -------------------------------------------------------------------------
    # STAGE 6: TASK 10 - RANDOMIZED SEARCH (20 TRIALS) & TREE ENSEMBLES
    # -------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("STAGE 6: ADVANCED HYPERPARAMETER SEARCH (20 TRIALS) & ENSEMBLES")
    print("=" * 50)

    X_train_dense = X_train_sub[:10000].toarray() if issparse(X_train_sub) else X_train_sub[:10000]
    X_val_dense = X_val[:10000].toarray() if issparse(X_val) else X_val[:10000]

    param_dist = {
        'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.15],
        'max_iter': [100, 150, 200],
        'max_leaf_nodes': [15, 31, 63],
        'min_samples_leaf': [10, 20, 50],
        'l2_regularization': [0.0, 0.1, 1.0]
    }

    rs = RandomizedSearchCV(
        estimator=HistGradientBoostingRegressor(random_state=42),
        param_distributions=param_dist,
        n_iter=20,
        cv=3,
        scoring='neg_mean_absolute_error',
        random_state=42,
        n_jobs=1
    )
    rs.fit(X_train_dense, y_train_log_sub[:10000])
    
    champion_hgb = rs.best_estimator_
    print(f"✓ RandomizedSearch Completed (20 Trials). Best Params: {rs.best_params_}")

    benchmark_results.append(calculate_comprehensive_metrics(y_val[:10000], np.expm1(champion_hgb.predict(X_val_dense)), "Tuned HistGradientBoosting"))

    # -------------------------------------------------------------------------
    # STAGE 7: TASK 14 - SHAP MODEL EXPLAINABILITY
    # -------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("STAGE 7: MODEL EXPLAINABILITY (SHAP ANALYSIS)")
    print("=" * 50)

    try:
        explainer = shap.TreeExplainer(champion_hgb)
        shap_values = explainer(X_val_dense[:500])
        
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_val_dense[:500], show=False)
        plt.title("SHAP Feature Importance Summary", fontsize=14)
        plt.tight_layout()
        shap_path = os.path.join(FIGURES_DIR, "shap_summary_plot.png")
        plt.savefig(shap_path, dpi=300)
        plt.close()
        print(f"✓ SHAP Plot generated and saved to {shap_path}")
    except Exception as e:
        print(f"SHAP generation warning: {e}")

    # -------------------------------------------------------------------------
    # STAGE 8: TASK 19 - AUTO-GENERATE PRESENTATION SLIDES STRUCT
    # -------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("STAGE 8: AUTO-GENERATING PRESENTATION SLIDE STRUCTURE")
    print("=" * 50)

    ppt_path = os.path.join(REPORTS_DIR, "presentation_structure.md")
    with open(ppt_path, "w") as f:
        f.write("# 8-Slide Technical Presentation Outline\n\n")
        f.write("## Slide 1: Executive Summary & System Overview\n")
        f.write("## Slide 2: Target Formulation & Log Transformation\n")
        f.write("## Slide 3: Pre-Trip Deployment vs Leakage Isolation\n")
        f.write("## Slide 4: Temporal Train/Val/Test Split Architecture\n")
        f.write("## Slide 5: Comprehensive Model Performance Comparison\n")
        f.write("## Slide 6: Oracle vs Safe Leakage Experiment Insights\n")
        f.write("## Slide 7: SHAP Model Explainability & Driver Insights\n")
        f.write("## Slide 8: Deployment Latency & Recommendation Architecture\n")
    print(f"✓ Presentation structure written to {ppt_path}")

    # Final Export
    summary_df = pd.DataFrame(benchmark_results)
    print("\n" + "=" * 80)
    print("                 FINAL BENCHMARK SUMMARY TABLE")
    print("=" * 80)
    print(summary_df.to_string(index=False))

    summary_df.to_csv(os.path.join(REPORTS_DIR, "final_complete_benchmark.csv"), index=False)
    joblib.dump(champion_hgb, os.path.join(MODELS_DIR, "champion_model.joblib"))
    joblib.dump(pipeline, os.path.join(PROCESSED_DATA_DIR, "pipeline.joblib"))

    print("\nSUCCESS! Execution completed with zero errors.")

if __name__ == "__main__":
    main()
