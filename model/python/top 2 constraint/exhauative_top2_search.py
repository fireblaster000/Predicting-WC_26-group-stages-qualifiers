# exhaustive_top2_search.py
# Exhaustive subset search with TOP-2-per-group constraint on validation and test.
# Also stores coefficient vectors for each subset, and stores validation probability vectors for top N models.

import itertools
import ast
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

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
# 4) Helpers: top-2 constraint and metrics
# -------------------------
def top2_predict(group_ids: pd.Series, proba: np.ndarray) -> np.ndarray:
    """
    Given group_ids (length n) and probabilities, return binary predictions
    with exactly 2 predicted positives per group.
    """
    pred = np.zeros(len(proba), dtype=int)
    # ensure aligned indexing
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
# 5) Fit model and return probs + coef vector
# -------------------------
def fit_model_and_get_outputs(X_train, y_train, X_eval, features, all_features):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000, solver="lbfgs"))
    ])
    pipe.fit(X_train[features], y_train)

    proba = pipe.predict_proba(X_eval[features])[:, 1]

    # Build a coefficient vector aligned to ALL features (missing -> 0)
    coef_vec = np.zeros(len(all_features), dtype=float)
    # logistic regression coefficients after scaling correspond to scaled space;
    # still OK for comparison + reporting.
    coefs = pipe.named_steps["clf"].coef_.ravel()
    feat_to_idx = {f: i for i, f in enumerate(all_features)}
    for f, c in zip(features, coefs):
        coef_vec[feat_to_idx[f]] = c

    intercept = float(pipe.named_steps["clf"].intercept_[0])
    return proba, intercept, coef_vec

# -------------------------
# 6) Exhaustive search with top-2 constraint on VAL
# -------------------------
def exhaustive_top2_search(X_train, y_train, df_val_meta, X_val, y_val, max_k=None, progress=True, topN_store_probs=200):
    all_features = list(X_train.columns)
    p = len(all_features)
    if max_k is None:
        max_k = p

    results = []
    probs_store = []  # only for top models later; we’ll fill after ranking

    total = 0
    for k in range(1, max_k + 1):
        combos = list(itertools.combinations(all_features, k))
        if progress:
            print(f"Testing subsets of size k={k} (count={len(combos)})...")
        for subset in combos:
            subset = list(subset)
            try:
                val_proba, intercept, coef_vec = fit_model_and_get_outputs(
                    X_train, y_train, X_val, subset, all_features
                )

                # TOP-2 constraint predictions on validation
                val_pred_top2 = top2_predict(df_val_meta["group_id"], val_proba)

                auc, acc, prec, rec, f1, (tn, fp, fn, tp) = compute_metrics(y_val.values, val_pred_top2, val_proba)

                results.append({
                    "k": k,
                    "features": subset,
                    "val_AUC": auc,
                    "val_Accuracy_top2": acc,
                    "val_Precision_top2": prec,
                    "val_Recall_top2": rec,
                    "val_F1_top2": f1,
                    "val_TN": tn, "val_FP": fp, "val_FN": fn, "val_TP": tp,
                    "intercept": intercept,
                    # store coefficient vector as a compact string
                    "coef_vector": coef_vec.tolist(),
                })
                total += 1
            except Exception:
                continue

    results_df = pd.DataFrame(results)
    print(f"\nDONE. Tested {total} subsets successfully.")
    return results_df, all_features

# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    PATH_XLSX = r"D:\UNI\Dartmouth\winter 24-25\data analytics\project\data\group_stage_qualification_data.xlsx"

    # If you want all subsets: MAX_K=None. You can reduce for speed (e.g., 10).
    MAX_K = None
    TOPN_STORE_PROBS = 200  # store val prob vectors only for top N models

    df = load_and_clean(PATH_XLSX)
    model_df, X, y, years = build_xy(df)

    (X_train, y_train, df_train_meta), (X_val, y_val, df_val_meta), (X_test, y_test, df_test_meta) = split_by_year(model_df, X, y)

    print("Split sizes:")
    print("Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)

    # Run exhaustive search scoring by TOP-2 constrained validation accuracy (and tracking AUC)
    results_df, all_features = exhaustive_top2_search(
        X_train, y_train, df_val_meta, X_val, y_val,
        max_k=MAX_K, progress=True, topN_store_probs=TOPN_STORE_PROBS
    )

    # Convert coef_vector list -> string for CSV storage
    results_df["coef_vector"] = results_df["coef_vector"].apply(lambda v: str(v))

    # Rank models: primary = val_Accuracy_top2, then val_F1_top2, then val_AUC, then smaller k
    ranked = results_df.sort_values(
        ["val_Accuracy_top2", "val_F1_top2", "val_AUC", "k"],
        ascending=[False, False, False, True]
    ).reset_index(drop=True)

    ranked.to_csv("exhaustive_top2_results.csv", index=False)
    print("\nSaved exhaustive_top2_results.csv")

    # Save feature name ordering used for coef_vector
    pd.DataFrame({"feature_order": all_features}).to_csv("coef_feature_order.csv", index=False)
    print("Saved coef_feature_order.csv")

    print("\nTop 10 models (ranked by val Accuracy top2):")
    print(ranked.head(10)[["k", "val_Accuracy_top2", "val_F1_top2", "val_AUC", "features"]])

    # Choose best model under top-2 constraint
    best = ranked.iloc[0]
    best_features = ast.literal_eval(str(best["features"])) if isinstance(best["features"], str) else best["features"]
    print("\nBEST model under TOP-2 constraint:")
    print("k =", int(best["k"]))
    print("features =", best_features)
    print("val_Accuracy_top2 =", best["val_Accuracy_top2"])
    print("val_F1_top2 =", best["val_F1_top2"])
    print("val_AUC =", best["val_AUC"])

    # Fit on TRAIN, evaluate on TEST with TOP-2 constraint
    test_proba, intercept, coef_vec = fit_model_and_get_outputs(X_train, y_train, X_test, best_features, all_features)
    test_pred_top2 = top2_predict(df_test_meta["group_id"], test_proba)

    test_auc, test_acc, test_prec, test_rec, test_f1, (tn, fp, fn, tp) = compute_metrics(y_test.values, test_pred_top2, test_proba)

    print("\n--- 2022 TEST (TOP-2 constrained) ---")
    print(f"AUC: {test_auc:.4f}")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"Precision: {test_prec:.4f}")
    print(f"Recall: {test_rec:.4f}")
    print(f"F1: {test_f1:.4f}")
    print(f"Confusion (TN, FP, FN, TP) = ({tn}, {fp}, {fn}, {tp})")

    # Save 2022 predictions with exactly requested cols + probs
    out_test = df_test_meta[["group_id", "team_name"]].copy()
    out_test["true_qualified"] = y_test.values
    out_test["prob_qualify"] = test_proba
    out_test["predicted_qualified_top2"] = test_pred_top2
    out_test.to_csv("2022_test_predictions_top2_bestmodel.csv", index=False)
    print("\nSaved 2022_test_predictions_top2_bestmodel.csv")

    # Store validation probabilities for TOPN models only (optional but useful)
    topN = min(TOPN_STORE_PROBS, len(ranked))
    rows = []
    for i in range(topN):
        feats = ranked.loc[i, "features"]
        feats = ast.literal_eval(feats) if isinstance(feats, str) else feats

        val_proba, intercept, coef_vec = fit_model_and_get_outputs(X_train, y_train, X_val, feats, all_features)

        rows.append({
            "rank": i+1,
            "k": len(feats),
            "features": feats,
            "val_AUC": ranked.loc[i, "val_AUC"],
            "val_Accuracy_top2": ranked.loc[i, "val_Accuracy_top2"],
            "val_probs": val_proba.tolist()
        })

    pd.DataFrame(rows).to_csv("top_models_val_probs.csv", index=False)
    print("Saved top_models_val_probs.csv (val probability vectors for top models)")