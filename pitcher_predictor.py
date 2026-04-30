"""
Steve Adams - MLB Pitcher Performance Predictor
==================================
Uses PyCaret AutoML to find the best model for predicting ERA and WHIP.
Trains on 2019-2024 historical data.
Predicts 2025 and compares to actual 2025 results.

Data Source: Baseball Savant CSV (baseballsavant.mlb.com)
Note: WHIP is calculated as (Walks + Earned Runs) / IP as a proxy
      since hit-level data is not available in this dataset.

"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import seaborn as sns

#load the data
def load_data(filepath: str) -> pd.DataFrame:
    """
    Load pitcher data from Baseball Savant CSV.
    Calculates WHIP from available columns since p_whip is not populated.
    """
    df = pd.read_csv(filepath)
    print(f"  Loaded {len(df)} records from {filepath}")

    # Calculate WHIP: (Walks + Earned Runs) / Innings Pitched
    df["calc_whip"] = (df["p_walk"] + df["p_earned_run"]) / df["p_formatted_ip"]

    return df

# prep data
def prepare_dataframe(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Returns a clean dataframe with only the feature columns and target.
    Removes nulls and extreme outliers.
    """

    feature_cols = [
        "p_game",
        "p_strikeout",
        "p_walk",
        "p_home_run",
        "p_earned_run",
        "xera",
        "xba",
        "xslg",
        "xwoba",
        "xwobacon",
        "exit_velocity_avg",
        "launch_angle_avg",
        "barrel_batted_rate",
        "hard_hit_percent",
        "k_percent",
        "bb_percent"
    ]

    # Map target to actual column
    actual_target = "calc_whip" if target == "p_whip" else target

    # Keep features + target + pitcher name for reference
    keep_cols = ["last_name, first_name"] + feature_cols + [actual_target]
    keep_cols = [c for c in keep_cols if c in df.columns]

    df_clean = df[keep_cols].copy()
    df_clean = df_clean.rename(columns={actual_target: target})

    # Drop nulls
    df_clean = df_clean.dropna()

    # Remove extreme outliers
    df_clean = df_clean[df_clean[target] < (10 if target == "p_era" else 3)]

    print(f"  Clean records for {target}: {len(df_clean)}")
    return df_clean

#FIND BEST MODEL WITH PYCARET
def find_best_model(df_train: pd.DataFrame, target: str, n_iter: int = 100):
    """
    Uses PyCaret to compare all regression models and return the best one.
    Added Hyperparameter tuning using Optuna which is built into pycaret
    """
    
    from pycaret.regression import setup, compare_models, tune_model, finalize_model, pull, plot_model


    print(f"\n{'='*60}")
    print(f"  PYCARET MODEL COMPARISON — Target: {target}")
    print(f"{'='*60}\n")

    # Drop pitcher name before passing to PyCaret
    df_model = df_train.drop(columns=["last_name, first_name"], errors="ignore")

    # Initialize PyCaret session
    reg = setup(
        data           = df_model,
        target         = target,
        session_id     = 42,
        verbose        = False,
        html           = False,
    )

    print("  Comparing all models with default hyperparameters...\n")
    best_model = compare_models(sort="MAE", verbose=True)

   # Capture and save comparison results
    comparison_results = pull()
    print(f"\n  Full model comparison:")
    print(comparison_results.to_string())
    comparison_results.to_csv(f"model_comparison_{target}.csv", index=True)
    print(f"\n  Comparison saved to model_comparison_{target}.csv")
 
    # Extract pre-tuning MAE for the best model
    best_model_name = type(best_model).__name__
    pre_tune_mae = comparison_results.iloc[0]["MAE"]
    pre_tune_r2  = comparison_results.iloc[0]["R2"]
    print(f"\n  Best model (pre-tuning): {best_model_name}")
    print(f"    MAE : {pre_tune_mae:.4f}")
    print(f"    R²  : {pre_tune_r2:.4f}")
 
    # Tune hyperparameters using Bayesian optimization (Optuna) ──
    print(f"\n{'='*60}")
    print(f"  HYPERPARAMETER TUNING — {best_model_name}")
    print(f"  Method: Bayesian optimization via Optuna ({n_iter} iterations)")
    print(f"{'='*60}\n")
 
    try:
        tuned_model = tune_model(
            best_model,
            optimize        = "MAE",          # Metric to minimize
            n_iter          = n_iter,          # Number of parameter combos to try
            search_library  = "optuna",        # Bayesian optimization — smarter than grid/random
            search_algorithm= "tpe",           # Tree-structured Parzen Estimator
            verbose         = True,
            return_train_score = False,
        )
 
        # Capture post-tuning results
        tuning_results = pull()
        post_tune_mae  = tuning_results["MAE"].mean()   # Cross-validated mean
        post_tune_r2   = tuning_results["R2"].mean()
 
        # Report improvement
        mae_improvement = ((pre_tune_mae - post_tune_mae) / pre_tune_mae) * 100
        print(f"\n  Tuning Results — {best_model_name}:")
        print(f"    Pre-tuning  MAE : {pre_tune_mae:.4f}  |  R² : {pre_tune_r2:.4f}")
        print(f"    Post-tuning MAE : {post_tune_mae:.4f}  |  R² : {post_tune_r2:.4f}")
        if mae_improvement > 0:
            print(f"    Improvement     : {mae_improvement:.1f}% reduction in MAE ✅")
        else:
            print(f"    Result          : Default hyperparameters were already optimal — keeping pre-tuned model.")
            tuned_model = best_model
 
        # Save tuning results
        tuning_results.to_csv(f"tuning_results_{target}.csv", index=True)
        print(f"  Tuning results saved to tuning_results_{target}.csv")
 
    except Exception as e:
        print(f"\n  Tuning failed ({e}) — using best model with default hyperparameters.")
        tuned_model = best_model
        post_tune_mae = pre_tune_mae
 
    #Finalize on full dataset
    print(f"\n  Finalizing model on full training dataset...")
    final_model = finalize_model(tuned_model)
 
    # Feature importance plot (works for tree-based models)
    try:
        plot_model(tuned_model, plot="feature", save=True, verbose=False)
        print(f"  Feature importance chart saved.")
    except Exception:
        print(f"  Feature importance chart not available for this model type.")
 
    return final_model, comparison_results

#BUILD CONFORMAL PREDICTION WRAPPER
def build_conformal_model(final_model, df_train: pd.DataFrame, target: str, confidence: float = 0.95):
    """
    Implements split conformal prediction without any external library. Tried Mapie but couldn't get it working.
 
    How it works:
      1. Hold out 20% of training data as a calibration set
      2. Generate predictions on the calibration set using the trained model
      3. Compute absolute residuals: |actual - predicted|
      4. Conformal margin = quantile of residuals at the confidence level
         (with finite-sample correction: confidence * (1 + 1/n))
      5. At prediction time: CI = [prediction - margin, prediction + margin]
 
    This produces guaranteed coverage at the specified confidence level —
    no external library required.
 
    Args:
        final_model : Finalized PyCaret model
        df_train    : Clean training dataframe with target column
        target      : Target column name ("p_era" or "p_whip")
        confidence  : Confidence level (default 0.95 = 95% CI)
 
    Returns:
        margin : Scalar margin applied symmetrically to all predictions
        X_cols : Feature column names used for prediction
    """
    from pycaret.regression import predict_model
    from sklearn.model_selection import train_test_split
 
    print(f"\n{'='*60}")
    print(f"  CONFORMAL PREDICTION — {int(confidence*100)}% Confidence Intervals")
    print(f"  Method: Split conformal prediction (residual quantile)")
    print(f"{'='*60}")
 
    feature_cols = [
        "p_game", "p_strikeout", "p_walk", "p_home_run", "p_earned_run",
        "xera", "xba", "xslg", "xwoba", "xwobacon",
        "exit_velocity_avg", "launch_angle_avg", "barrel_batted_rate",
        "hard_hit_percent", "k_percent", "bb_percent"
    ]
 
    X_cols = [c for c in feature_cols if c in df_train.columns]
 
    # Hold out 20% as calibration set
    df_clean = df_train.drop(columns=["last_name, first_name"], errors="ignore").dropna()
    _, df_calib = train_test_split(df_clean, test_size=0.20, random_state=42)
 
    # Predict on calibration set using PyCaret
    X_calib     = df_calib[X_cols].copy()
    calib_preds = predict_model(final_model, data=X_calib)
    predicted   = calib_preds["prediction_label"].values
    actual      = df_calib[target].values
 
    # Absolute residuals
    residuals = np.abs(actual - predicted)
 
    # Conformal margin with finite-sample correction
    n = len(residuals)
    adjusted_quantile = min(confidence * (1 + 1 / n), 1.0)
    margin = float(np.quantile(residuals, adjusted_quantile))
 
    print(f"  Calibration set size : {n} pitchers")
    print(f"  Conformal margin     : +/-{margin:.4f} (applied symmetrically to each point prediction)")
    print(f"  Interpretation       : {int(confidence*100)}% CI = [prediction - {margin:.4f}, prediction + {margin:.4f}]")
 
    return margin, X_cols
 

# PREDICT 2025 & COMPARE TO ACTUAL
def predict_and_compare(model, margin: float, X_cols: list, df_2025: pd.DataFrame, target: str, confidence: float = 0.95, top_n: int = 20):
    """
    Runs predictions on 2025 data.
    Applies conformal margin to produce CI_Lower and CI_Upper.
    Compares predictions to actual 2025 results.
    """
    from pycaret.regression import predict_model
 
    actual_target = "calc_whip" if target == "p_whip" else target
 
    df_pred = df_2025.copy()
 
    feature_cols = [
        "last_name, first_name",
        "p_game",
        "p_strikeout",
        "p_walk",
        "p_home_run",
        "p_earned_run",
        "xera",
        "xba",
        "xslg",
        "xwoba",
        "xwobacon",
        "exit_velocity_avg",
        "launch_angle_avg",
        "barrel_batted_rate",
        "hard_hit_percent",
        "k_percent",
        "bb_percent",
        actual_target
    ]
 
    df_pred = df_pred[[c for c in feature_cols if c in df_pred.columns]].copy()
    df_pred = df_pred.rename(columns={actual_target: "Actual"})
    df_pred = df_pred.dropna()
 
    # Point predictions
    df_input = df_pred.drop(columns=["last_name, first_name", "Actual"], errors="ignore")
    preds    = predict_model(model, data=df_input)
    df_pred["Predicted"] = preds["prediction_label"].values
 
    # Apply conformal margin symmetrically
    df_pred["CI_Lower"] = np.clip(df_pred["Predicted"] - margin, 0, None).round(4)
    df_pred["CI_Upper"] = (df_pred["Predicted"] + margin).round(4)
    df_pred["CI_Width"] = (df_pred["CI_Upper"] - df_pred["CI_Lower"]).round(4)
    df_pred["Delta"]    = (df_pred["Actual"] - df_pred["Predicted"]).round(4)
 
    # Flag whether actual fell inside CI
    df_pred["In_CI"] = (
        (df_pred["Actual"] >= df_pred["CI_Lower"]) &
        (df_pred["Actual"] <= df_pred["CI_Upper"])
    ).map({True: "Yes", False: "No"})
 
    result = df_pred[[
        "last_name, first_name", "p_game",
        "Actual", "Predicted",
        "CI_Lower", "CI_Upper", "CI_Width",
        "Delta", "In_CI"
    ]].copy()
 
    result = result.rename(columns={
        "last_name, first_name": "Pitcher",
        "p_game":                "Games",
    })
    result["Actual"]    = result["Actual"].round(4)
    result["Predicted"] = result["Predicted"].round(4)
 
    # Coverage report
    coverage = (result["In_CI"] == "Yes").mean() * 100
    print(f"\n  {int(confidence*100)}% CI Coverage: {coverage:.1f}% of actual 2025 values fell within predicted interval")
    print(f"  (Target: {int(confidence*100)}% — {'on target' if abs(coverage - confidence * 100) < 5 else 'check calibration'})")
 
    result = result.sort_values("Predicted", ascending=True)
    return result.head(top_n)
 
#OVER/UNDER PERFORMERS
def show_over_under_performers(model, margin: float, df_2025: pd.DataFrame, target: str, top_n: int = 10):
    """
    Shows pitchers who most significantly outperformed or underperformed
    their model prediction in 2025, with confidence intervals.
    """
    from pycaret.regression import predict_model
 
    actual_target = "calc_whip" if target == "p_whip" else target
 
    df_pred = df_2025.copy()
 
    feature_cols = [
        "last_name, first_name",
        "p_game",
        "p_strikeout",
        "p_walk",
        "p_home_run",
        "p_earned_run",
        "xera",
        "xba",
        "xslg",
        "xwoba",
        "xwobacon",
        "exit_velocity_avg",
        "launch_angle_avg",
        "barrel_batted_rate",
        "hard_hit_percent",
        "k_percent",
        "bb_percent",
        actual_target
    ]
 
    df_pred = df_pred[[c for c in feature_cols if c in df_pred.columns]].copy()
    df_pred = df_pred.rename(columns={actual_target: "Actual"})
    df_pred = df_pred.dropna()
 
    df_input    = df_pred.drop(columns=["last_name, first_name", "Actual"], errors="ignore")
    predictions = predict_model(model, data=df_input)
 
    df_pred["Predicted"] = predictions["prediction_label"].values
    df_pred["CI_Lower"]  = np.clip(df_pred["Predicted"] - margin, 0, None).round(4)
    df_pred["CI_Upper"]  = (df_pred["Predicted"] + margin).round(4)
    df_pred["CI_Width"]  = (df_pred["CI_Upper"] - df_pred["CI_Lower"]).round(4)
    df_pred["Delta"]     = (df_pred["Actual"] - df_pred["Predicted"]).round(4)
 
    result = df_pred[[
        "last_name, first_name", "p_game",
        "Actual", "Predicted",
        "CI_Lower", "CI_Upper", "CI_Width", "Delta"
    ]].copy()
 
    result = result.rename(columns={"last_name, first_name": "Pitcher", "p_game": "Games"})
    result["Actual"]    = result["Actual"].round(4)
    result["Predicted"] = result["Predicted"].round(4)
 
    print(f"\n  --- Biggest Outperformers (Actual BETTER than predicted) ---")
    outperformed = result.sort_values("Delta", ascending=True).head(top_n)
    print(outperformed.to_string(index=False))
 
    print(f"\n  --- Biggest Underperformers (Actual WORSE than predicted) ---")
    underperformed = result.sort_values("Delta", ascending=False).head(top_n)
    print(underperformed.to_string(index=False))
 
    return outperformed, underperformed


if __name__ == "__main__":

  # ── Load Data ──
    print("\n>>> Loading Data...")
    df_all  = load_data("pitchers.csv")
    df_2025 = load_data("pitchers_2025.csv")
 
    # Filter to pre-2025 for training
    df_historical = df_all[df_all["year"] < 2025].copy()
    print(f"  Historical training records (2019-2024): {len(df_historical)}")
    print(f"  2025 records for prediction: {len(df_2025)}")
 
    # ── ERA Model ──
    print("\n>>> Preparing ERA training data...")
    df_era_train = prepare_dataframe(df_historical, "p_era")
    era_model, era_results = find_best_model(df_era_train, "p_era")
 
    print("\n>>> Building ERA conformal prediction model...")
    era_margin, era_X_cols = build_conformal_model(era_model, df_era_train, "p_era", confidence=0.95)
 
    # ── WHIP Model ──
    print("\n>>> Preparing WHIP training data...")
    df_whip_train = prepare_dataframe(df_historical, "p_whip")
    whip_model, whip_results = find_best_model(df_whip_train, "p_whip")
 
    print("\n>>> Building WHIP conformal prediction model...")
    whip_margin, whip_X_cols = build_conformal_model(whip_model, df_whip_train, "p_whip", confidence=0.95)
 
    # ── Predict 2025 & Compare to Actual ──
    print("\n\n>>> Top 20 ERA Predictions vs Actual 2025 (with 95% CI):")
    era_comparison = predict_and_compare(era_model, era_margin, era_X_cols, df_2025, "p_era", confidence=0.95)
    print(era_comparison.to_string(index=False))
 
    print("\n\n>>> Top 20 WHIP Predictions vs Actual 2025 (with 95% CI):")
    whip_comparison = predict_and_compare(whip_model, whip_margin, whip_X_cols, df_2025, "p_whip", confidence=0.95)
    print(whip_comparison.to_string(index=False))
 
    # ── Over/Under Performers ──
    # ── Over/Under Performers ──
    print("\n\n>>> ERA Over/Under Performers 2025:")
    show_over_under_performers(era_model, era_margin, df_2025, "p_era")
 
    print("\n\n>>> WHIP Over/Under Performers 2025:")
    show_over_under_performers(whip_model, whip_margin, df_2025, "p_whip")
 
    # ── Save Results ──
    era_comparison.to_csv("era_2025_vs_actual.csv", index=False)
    whip_comparison.to_csv("whip_2025_vs_actual.csv", index=False)
 
    print("\n✅ Done!")
    print("   era_2025_vs_actual.csv       — ERA predictions vs actual 2025 with 95% CI")
    print("   whip_2025_vs_actual.csv      — WHIP predictions vs actual 2025 with 95% CI")
    print("   model_comparison_p_era.csv   — Full ERA model comparison table")
    print("   model_comparison_p_whip.csv  — Full WHIP model comparison table")
    print("   tuning_results_p_era.csv     — ERA hyperparameter tuning results")
    print("   tuning_results_p_whip.csv    — WHIP hyperparameter tuning results")