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
 


# PREDICT 2025 & COMPARE TO ACTUAL
def predict_and_compare(model, df_2025: pd.DataFrame, target: str, top_n: int = 20):
    """
    Runs predictions on 2025 data using the best model.
    Compares predictions to actual 2025 results.
    Returns ranked dataframe with delta column.
    """
    from pycaret.regression import predict_model

    actual_target = "calc_whip" if target == "p_whip" else target

    df_pred = df_2025.copy()

    # Prepare features — keep name for display
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
    df_pred = df_pred.rename(columns={actual_target: f"Actual_{target}"})
    df_pred = df_pred.dropna()

    # Run predictions
    df_input = df_pred.drop(columns=["last_name, first_name", f"Actual_{target}"], errors="ignore")
    predictions = predict_model(model, data=df_input)

    df_pred[f"Predicted_{target}"] = predictions["prediction_label"].values

    # Calculate delta: negative = outperformed prediction, positive = underperformed
    df_pred["Delta"] = (df_pred[f"Actual_{target}"] - df_pred[f"Predicted_{target}"]).round(4)

    # Build clean results table
    result = df_pred[[
        "last_name, first_name",
        "p_game",
        f"Actual_{target}",
        f"Predicted_{target}",
        "Delta"
    ]].copy()

    result = result.rename(columns={"last_name, first_name": "Pitcher", "p_game": "Games"})
    result[f"Actual_{target}"]    = result[f"Actual_{target}"].round(4)
    result[f"Predicted_{target}"] = result[f"Predicted_{target}"].round(4)

    # Sort by predicted performance (lower is better)
    result = result.sort_values(f"Predicted_{target}", ascending=True)

    return result.head(top_n)


#OVER/UNDER PERFORMERS
def show_over_under_performers(model, df_2025: pd.DataFrame, target: str, top_n: int = 10):
    """
    Shows pitchers who most significantly outperformed or underperformed
    their model prediction in 2025.
    Negative delta = outperformed (actual better than predicted).
    Positive delta = underperformed (actual worse than predicted).
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

    df_input      = df_pred.drop(columns=["last_name, first_name", "Actual"], errors="ignore")
    predictions   = predict_model(model, data=df_input)

    df_pred["Predicted"] = predictions["prediction_label"].values
    df_pred["Delta"]     = (df_pred["Actual"] - df_pred["Predicted"]).round(4)

    result = df_pred[["last_name, first_name", "p_game", "Actual", "Predicted", "Delta"]].copy()
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

    # ── WHIP Model ──
    print("\n>>> Preparing WHIP training data...")
    df_whip_train = prepare_dataframe(df_historical, "p_whip")
    whip_model, whip_results = find_best_model(df_whip_train, "p_whip")

    # ── Predict 2025 & Compare to Actual ──
    print("\n\n>>> Top 20 ERA Predictions vs Actual 2025:")
    era_comparison = predict_and_compare(era_model, df_2025, "p_era")
    print(era_comparison.to_string(index=False))

    print("\n\n>>> Top 20 WHIP Predictions vs Actual 2025:")
    whip_comparison = predict_and_compare(whip_model, df_2025, "p_whip")
    print(whip_comparison.to_string(index=False))

    # ── Over/Under Performers ──
    print("\n\n>>> ERA Over/Under Performers 2025:")
    show_over_under_performers(era_model, df_2025, "p_era")

    print("\n\n>>> WHIP Over/Under Performers 2025:")
    show_over_under_performers(whip_model, df_2025, "p_whip")

    # ── Save Results ──
    era_comparison.to_csv("era_2025_vs_actual.csv", index=False)
    whip_comparison.to_csv("whip_2025_vs_actual.csv", index=False)

    print("\n✅ Done!")
    print("   era_2025_vs_actual.csv       — ERA predictions vs actual 2025")
    print("   whip_2025_vs_actual.csv      — WHIP predictions vs actual 2025")
    print("   model_comparison_p_era.csv   — Full ERA model comparison table")
    print("   model_comparison_p_whip.csv  — Full WHIP model comparison table")