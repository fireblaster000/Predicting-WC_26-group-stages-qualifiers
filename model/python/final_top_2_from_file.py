# final_top2_from_files.py
# Loads best model from exhaustive_top2_results.csv (no exhaustive re-run),
# evaluates on 2022 with top-2 constraint, and outputs statsmodels inference table.

import ast
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

import statsmodels.api as sm


# -------------------------
# 1) Load + clean Excel
# -------------------------
def load_and_clean(path: str) -> pd.DataFrame:
    raw = pd.read_excel(path, header=None)
    header = raw.iloc[2].tolist()
    df = raw.iloc[3:].copy()
    df.columns = header
    df = df.reset_index(drop=True)
    df = df.dropna(axis=1, how="all")

    categorical_cols = ["tournament_name", "tournament_year", "group_id", "team_name", "confederation"]
    target_col = "qualified_from_group"

    if "tournament_start_date" in df.columns:
        df["tournament_start_date"] = pd.to_datetime(df["tournament_start_date"], errors="coerce")

    for c in df.columns:
        if c in categorical_cols:
            df[c] = df[c].astype(str).str.strip()
        elif c == "tournament_start_date":
            continue
        else:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df[target_col] = df[target_col].astype(int)
    return df


# -------------------------
# 2) Build modeling table + X/y
# -------------------------
def build_xy(df: pd.DataFrame):
    categorical_cols = ["tournament_name", "tournament_year", "group_id", "team_name", "confederation"]
    target_col = "qualified_from_group"

    feature_cols_numeric = [
        "team_elo_pre",
        "recent_win_rate",
        "recent_goal_diff_per_match",
        "recent_goals_for_per_match",
        "recent_goals_against_per_match",
        "opp_elo_mean",
        "opp_elo_max",
        "group_elo_std",
        "elo_gap_vs_opp_mean",
        "host_team_flag",
        "log_gdp_per_capita_pre",
    ]

    keep_cols = categorical_cols + [target_col] + feature_cols_numeric
    model_df = df[keep_cols].copy()
    model_df = model_df.dropna(subset=feature_cols_numeric + [target_col]).reset_index(drop=True)

    X_num = model_df[feature_cols_numeric]
    X_cat = pd.get_dummies(model_df["confederation"], prefix="confed", drop_first=True)
    X = pd.concat([X_num, X_cat], axis=1)

    y = model_df[target_col].astype(int)
    years = model_df["tournament_year"].astype(str)

    return model_df, X, y, years


# -------------------------
# 3) Train/Val/Test split
# -------------------------
def split_by_year(model_df, X, y):
    train_years = {"1998", "2002", "2006", "2010", "2014"}
    val_year = "2018"
    test_year = "2022"

    years = model_df["tournament_year"].astype(str)

    train_mask = years.isin(train_years)
    val_mask = years == val_year
    test_mask = years == test_year

    return (X.loc[train_mask], y.loc[train_mask], model_df.loc[train_mask]), \
           (X.loc[val_mask], y.loc[val_mask], model_df.loc[val_mask]), \
           (X.loc[test_mask], y.loc[test_mask], model_df.loc[test_mask])


# -------------------------
# 4) Top-2 constraint
# -------------------------
def top2_predict(group_ids: pd.Series, proba: np.ndarray) -> np.ndarray:
    pred = np.zeros(len(proba), dtype=int)
    df_tmp = pd.DataFrame({"group_id": group_ids.values, "proba": proba})
    for g, gdf in df_tmp.groupby("group_id"):
        idx = gdf.sort_values("proba", ascending=False).head(2).index
        pred[idx] = 1
    return pred


def compute_metrics(y_true, y_pred, y_proba):
    auc = roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) == 2 else np.nan
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return auc, acc, prec, rec, f1, (tn, fp, fn, tp)


# -------------------------
# 5) Load best features from exhaustive_top2_results.csv
# -------------------------
def load_best_features_from_top2_csv(path_csv: str):
    ranked = pd.read_csv(path_csv)

    # Parse 'features' column (stored as list or string)
    ranked["features"] = ranked["features"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    # Rank using the same rule as before
    ranked = ranked.sort_values(
        ["val_Accuracy_top2", "val_F1_top2", "val_AUC", "k"],
        ascending=[False, False, False, True]
    ).reset_index(drop=True)

    best = ranked.iloc[0]
    return best, ranked


# -------------------------
# 6) Fit sklearn model
# -------------------------
def fit_sklearn_logit(X_train, y_train, features):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000, solver="lbfgs"))
    ])
    pipe.fit(X_train[features], y_train)
    return pipe


# -------------------------
# 7) Statsmodels inference
# -------------------------
def make_logit_inference_table(X_train, y_train, features, out_csv="final_model_inference_top2.csv"):
    X = X_train[features].copy()
    X = X.apply(pd.to_numeric, errors="coerce").astype(float)
    y = pd.to_numeric(y_train, errors="coerce").astype(float)

    tmp = pd.concat([y.rename("y"), X], axis=1).dropna()
    y = tmp["y"]
    X = tmp.drop(columns=["y"])

    X = sm.add_constant(X, has_constant="add")
    res = sm.Logit(y, X).fit(disp=False, maxiter=500)

    coef = res.params.rename("coef")
    se = res.bse.rename("std_err")
    z = res.tvalues.rename("z_value")  # logit -> z-stat
    p = res.pvalues.rename("p_value")
    ci = res.conf_int()
    ci.columns = ["ci_2.5%", "ci_97.5%"]

    table = pd.concat([coef, se, z, p, ci], axis=1)
    table["odds_ratio"] = np.exp(table["coef"])
    table["or_ci_2.5%"] = np.exp(table["ci_2.5%"])
    table["or_ci_97.5%"] = np.exp(table["ci_97.5%"])

    table.to_csv(out_csv, index=True)
    print(f"\nSaved statsmodels inference table to: {out_csv}")
    print(table)
    return table


# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    PATH_XLSX = r"D:\UNI\Dartmouth\winter 24-25\data analytics\project\data\group_stage_qualification_data.xlsx"
    TOP2_RESULTS_CSV = r"exhaustive_top2_results.csv"

    # 1) Load data and split
    df = load_and_clean(PATH_XLSX)
    model_df, X, y, years = build_xy(df)
    (X_train, y_train, df_train_meta), (X_val, y_val, df_val_meta), (X_test, y_test, df_test_meta) = split_by_year(model_df, X, y)

    print("Split sizes:")
    print("Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)

    # 2) Load best subset from saved exhaustive_top2_results.csv
    best, ranked = load_best_features_from_top2_csv(TOP2_RESULTS_CSV)
    best_features = best["features"]

    print("\nLoaded ranked models from:", TOP2_RESULTS_CSV)
    print("\nBEST model under TOP-2 constraint (from file):")
    print("k =", int(best["k"]))
    print("features =", best_features)
    print("val_Accuracy_top2 =", best["val_Accuracy_top2"])
    print("val_F1_top2 =", best["val_F1_top2"])
    print("val_AUC =", best["val_AUC"])

    # 3) Fit sklearn on TRAIN and evaluate on TEST with TOP-2 constraint
    pipe = fit_sklearn_logit(X_train, y_train, best_features)
    test_proba = pipe.predict_proba(X_test[best_features])[:, 1]
    test_pred_top2 = top2_predict(df_test_meta["group_id"], test_proba)

    test_auc, test_acc, test_prec, test_rec, test_f1, (tn, fp, fn, tp) = compute_metrics(y_test.values, test_pred_top2, test_proba)

    print("\n--- 2022 TEST (TOP-2 constrained) ---")
    print(f"AUC: {test_auc:.4f}")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"Precision: {test_prec:.4f}")
    print(f"Recall: {test_rec:.4f}")
    print(f"F1: {test_f1:.4f}")
    print(f"Confusion (TN, FP, FN, TP) = ({tn}, {fp}, {fn}, {tp})")

    # 4) Save 2022 predictions file
    out_test = df_test_meta[["group_id", "team_name"]].copy()
    out_test["true_qualified"] = y_test.values
    out_test["prob_qualify"] = test_proba
    out_test["predicted_qualified_top2"] = test_pred_top2
    out_test.to_csv("2022_test_predictions_top2_bestmodel.csv", index=False)
    print("\nSaved 2022_test_predictions_top2_bestmodel.csv")

    # 5) Statsmodels inference on TRAIN for the final selected features
    print("\n--- FINAL MODEL INFERENCE (TRAIN, Statsmodels Logit) ---")
    make_logit_inference_table(X_train, y_train, best_features, out_csv="final_model_inference_top2.csv")