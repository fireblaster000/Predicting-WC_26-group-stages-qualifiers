# log_reg.py
# Exhaustive feature-subset search for logistic regression
# Train: 1998-2014, Val: 2018, Test: 2022

import itertools
import numpy as np
import pandas as pd
import ast

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix


# -------------------------
# 1) Load + clean Excel
# -------------------------
def load_and_clean(path: str) -> pd.DataFrame:
    raw = pd.read_excel(path, header=None)

    # Row 2 contains true column names; data begins at row 3
    header = raw.iloc[2].tolist()
    df = raw.iloc[3:].copy()
    df.columns = header
    df = df.reset_index(drop=True)
    df = df.dropna(axis=1, how="all")

    categorical_cols = ["tournament_name", "tournament_year", "group_id", "team_name", "confederation"]
    target_col = "qualified_from_group"

    # Parse date if present
    if "tournament_start_date" in df.columns:
        df["tournament_start_date"] = pd.to_datetime(df["tournament_start_date"], errors="coerce")

    # Convert datatypes
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

    # Numeric predictors from your proposal
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
    missing_cols = [c for c in keep_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")

    model_df = df[keep_cols].copy()

    # Drop rows with any missing predictor (simple + consistent)
    model_df = model_df.dropna(subset=feature_cols_numeric + [target_col]).reset_index(drop=True)

    # One-hot confederation
    X_num = model_df[feature_cols_numeric]
    X_cat = pd.get_dummies(model_df["confederation"], prefix="confed", drop_first=True)
    X = pd.concat([X_num, X_cat], axis=1)

    y = model_df[target_col].astype(int)
    years = model_df["tournament_year"].astype(str)

    return model_df, X, y, years


# -------------------------
# 3) Train/Val/Test split by tournament year
# -------------------------
def split_by_year(model_df, X, y):
    train_years = {"1998", "2002", "2006", "2010", "2014"}
    val_year = "2018"
    test_year = "2022"

    years = model_df["tournament_year"].astype(str)

    train_mask = years.isin(train_years)
    val_mask = years == val_year
    test_mask = years == test_year

    X_train, y_train = X.loc[train_mask], y.loc[train_mask]
    X_val, y_val = X.loc[val_mask], y.loc[val_mask]
    X_test, y_test = X.loc[test_mask], y.loc[test_mask]

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


# -------------------------
# 4) Fit + evaluate one subset
# -------------------------
def fit_eval_subset(features, X_train, y_train, X_eval, y_eval):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000, solver="lbfgs"))
    ])

    pipe.fit(X_train[features], y_train)

    proba = pipe.predict_proba(X_eval[features])[:, 1]
    pred = (proba >= 0.5).astype(int)

    # If eval set has only one class (rare), AUC undefined
    auc = roc_auc_score(y_eval, proba) if len(np.unique(y_eval)) == 2 else np.nan
    acc = accuracy_score(y_eval, pred)

    tn, fp, fn, tp = confusion_matrix(y_eval, pred).ravel()
    return auc, acc, (tn, fp, fn, tp)


# -------------------------
# 5) Exhaustive search (ALL subsets, or cap max size)
# -------------------------
def exhaustive_search(X_train, y_train, X_val, y_val, max_k=None, progress=True):
    all_features = list(X_train.columns)
    p = len(all_features)

    if max_k is None:
        max_k = p

    results = []
    total_tested = 0

    for k in range(1, max_k + 1):
        combos = list(itertools.combinations(all_features, k))
        if progress:
            print(f"Testing subsets of size k={k} (count={len(combos)})...")

        for subset in combos:
            subset = list(subset)
            try:
                val_auc, val_acc, (tn, fp, fn, tp) = fit_eval_subset(
                    subset, X_train, y_train, X_val, y_val
                )

                results.append({
                    "k": k,
                    "features": subset,
                    "val_AUC": val_auc,
                    "val_Accuracy": val_acc,
                    "val_TN": tn,
                    "val_FP": fp,
                    "val_FN": fn,
                    "val_TP": tp,
                })
                total_tested += 1
            except Exception:
                # e.g., rare numerical issues; skip safely
                continue

    results_df = pd.DataFrame(results)
    print(f"\nDONE. Tested {total_tested} subsets successfully.")
    return results_df


# -------------------------
# 6) Choose best subset by validation
# -------------------------
def choose_best(results_df):
    # Sort by Val AUC desc, Val Acc desc, then smaller k
    ranked = results_df.sort_values(
        ["val_AUC", "val_Accuracy", "k"],
        ascending=[False, False, True]
    ).reset_index(drop=True)
    best = ranked.iloc[0]
    return best, ranked


# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    PATH = r"D:\UNI\Dartmouth\winter 24-25\data analytics\project\data\group_stage_qualification_data.xlsx"

    # OPTIONAL: cap max subset size to reduce runtime
    # - If I want ALL subsets: set MAX_K = None
    # - If i want something faster: e.g., MAX_K = 8
    MAX_K = None  # change to 8 or 10 if I want to speed it up

    df = load_and_clean(PATH)
    print("Loaded and cleaned data. Total rows:", df.shape[0])
    print("Columns:", df.columns.tolist())
    # columns type
    print("Column types:\n", df.dtypes)
    print("\nSample data:")
    print(df.head())
    # model_df, X, y, years = build_xy(df)

    # print("Data after cleaning:")
    # print(f"Rows: {model_df.shape[0]}  Cols (X): {X.shape[1]}")
    # print("Class balance:\n", y.value_counts())

    # (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_by_year(model_df, X, y)
    # print("\nSplit sizes:")
    # print("Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)

    # # -------------------------
    # # Load ranked subsets from CSV (NO re-running exhaustive search)
    # # -------------------------
    # ranked_path = "exhaustive_feature_search.csv"  # adjust if file is elsewhere
    # ranked = pd.read_csv(ranked_path)

    # # features column is stored as a string like "['a','b',...]"
    # ranked["features"] = ranked["features"].apply(ast.literal_eval)

    # # pick best by val_AUC, then val_Accuracy, then smaller k
    # ranked = ranked.sort_values(["val_AUC", "val_Accuracy", "k"], ascending=[False, False, True]).reset_index(drop=True)
    # best = ranked.iloc[0]
    # best_features = best["features"]

    # print("\nLoaded ranked subsets from:", ranked_path)
    # print("\nBEST subset chosen from CSV:")
    # print("k =", int(best["k"]))
    # print("features =", best_features)
    # print("val_AUC =", best["val_AUC"])
    # print("val_Accuracy =", best["val_Accuracy"])


    # topN = 25
    # results = []

    # for i in range(min(topN, len(ranked))):
    #     feats = ranked.loc[i, "features"]

    #     pipe = Pipeline([
    #         ("scaler", StandardScaler()),
    #         ("clf", LogisticRegression(max_iter=5000))
    #     ])
    #     pipe.fit(X_train[feats], y_train)

    #     proba = pipe.predict_proba(X_test[feats])[:,1]
    #     pred = (proba >= 0.5).astype(int)

    #     auc = roc_auc_score(y_test, proba)
    #     acc = accuracy_score(y_test, pred)

    #     results.append({
    #         "rank_on_val": i+1,
    #         "k": len(feats),
    #         "val_AUC": ranked.loc[i, "val_AUC"],
    #         "val_Accuracy": ranked.loc[i, "val_Accuracy"],
    #         "test_AUC": auc,
    #         "test_Accuracy": acc,
    #         "features": feats
    #     })

    # top_df = pd.DataFrame(results).sort_values(["test_AUC","test_Accuracy"], ascending=False)
    # print("\nTop-N (by validation) re-ranked by TEST performance:")
    # print(top_df.head(10)[["rank_on_val","k","val_AUC","test_AUC","test_Accuracy","features"]])

    # top_df.to_csv("topN_val_models_test_results.csv", index=False)
    # print("\nSaved topN_val_models_test_results.csv")


    # Exhaustive search on validation
    # print("\nRunning EXHAUSTIVE feature-subset search (train->val)...")
    # results_df = exhaustive_search(X_train, y_train, X_val, y_val, max_k=MAX_K, progress=True)

    # # Rank + choose best
    # best, ranked = choose_best(results_df)

    # # Save results
    # ranked.to_csv("exhaustive_feature_search.csv", index=False)
    # print("\nSaved ranked results to exhaustive_feature_search.csv")

    # Print top 10
    # print("\nTop 10 subsets by validation AUC:")
    # print(ranked.head(10)[["k", "val_AUC", "val_Accuracy", "features"]])

    # best_features = best["features"]
    # print("\nBEST subset chosen by validation AUC (tie-breaks: val acc, then smaller k):")
    # print("k =", int(best["k"]))
    # print("features =", best_features)
    # print("val_AUC =", best["val_AUC"])
    # print("val_Accuracy =", best["val_Accuracy"])
    # print("val Confusion (TN, FP, FN, TP) =", (best["val_TN"], best["val_FP"], best["val_FN"], best["val_TP"]))

    # # Final evaluation on TEST (2022): fit on TRAIN only, evaluate on TEST
    # print("\nFinal evaluation on TEST (2022) using chosen feature set:")
    # test_auc, test_acc, (tn, fp, fn, tp) = fit_eval_subset(best_features, X_train, y_train, X_test, y_test)

    # print(f"TEST AUC: {test_auc:.4f}")
    # print(f"TEST Accuracy: {test_acc:.4f}")
    # print("TEST Confusion Matrix (TN, FP, FN, TP):", (tn, fp, fn, tp))

    # Inspect per-team errors on TEST (2022)
    # best_feats = best_features

    # # Refit model on TRAIN
    # pipe = Pipeline([
    #     ("scaler", StandardScaler()),
    #     ("clf", LogisticRegression(max_iter=5000))
    # ])
    # pipe.fit(X_train[best_feats], y_train)

    # # Predict on TEST
    # test_proba = pipe.predict_proba(X_test[best_feats])[:,1]
    # test_pred = (test_proba >= 0.5).astype(int)

    # # Build a nice inspection table
    # test_inspect = model_df.loc[X_test.index, [
    #     "tournament_year", "team_name", "group_id", "confederation", "qualified_from_group"
    # ]].copy()

    # test_inspect["predicted"] = test_pred
    # test_inspect["prob_qualify"] = test_proba
    # test_inspect["correct"] = (test_inspect["predicted"] == test_inspect["qualified_from_group"])

    # # Show mistakes
    # mistakes = test_inspect[~test_inspect["correct"]].sort_values("prob_qualify", ascending=False)

    # print("\nMisclassified teams in 2022:")
    # print(mistakes)

    # # Optionally save to CSV
    # mistakes.to_csv("2022_misclassifications.csv", index=False)
    # print("\nSaved 2022_misclassifications.csv")

