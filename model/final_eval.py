import ast
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, accuracy_score, confusion_matrix,
    precision_score, recall_score, f1_score
)

import statsmodels.api as sm


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


def split_by_year(model_df, X, y):
    train_years = {"1998", "2002", "2006", "2010", "2014"}
    val_year = "2018"
    test_year = "2022"

    years = model_df["tournament_year"].astype(str)

    train_mask = years.isin(train_years)
    val_mask = years == val_year
    test_mask = years == test_year

    return (X.loc[train_mask], y.loc[train_mask]), (X.loc[val_mask], y.loc[val_mask]), (X.loc[test_mask], y.loc[test_mask])


def load_best_features(subsets_csv: str):
    ranked = pd.read_csv(subsets_csv)
    ranked["features"] = ranked["features"].apply(ast.literal_eval)
    ranked = ranked.sort_values(["val_AUC", "val_Accuracy", "k"], ascending=[False, False, True]).reset_index(drop=True)
    best = ranked.iloc[0]
    return best, ranked


def fit_sklearn_logit(X_train, y_train, features):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000, solver="lbfgs"))
    ])
    pipe.fit(X_train[features], y_train)
    return pipe


def metrics_block(y_true, y_pred, y_proba, label=""):
    wrong = int((y_pred != y_true).sum())
    total = int(len(y_true))
    acc = 1 - wrong / total

    auc = roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) == 2 else np.nan
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    print(f"\n--- {label} ---")
    print(f"Wrong / Total              = {wrong} / {total}")
    print(f"Accuracy (1 - wrong/total) = {acc:.4f}")
    print(f"Accuracy (sklearn)         = {accuracy_score(y_true, y_pred):.4f}")
    print(f"AUC                        = {auc:.4f}")
    print(f"Precision                  = {prec:.4f}")
    print(f"Recall                     = {rec:.4f}")
    print(f"F1                         = {f1:.4f}")
    print(f"Confusion (TN, FP, FN, TP) = ({tn}, {fp}, {fn}, {tp})")

    return {
        "wrong": wrong, "total": total, "accuracy": acc,
        "auc": auc, "precision": prec, "recall": rec, "f1": f1,
        "confusion": (tn, fp, fn, tp)
    }


def enforce_top2_per_group(df_pred, group_col="group_id", proba_col="prob_qualify"):
    """
    For each group, set predicted_qualified=1 for the top 2 probabilities,
    0 for the remaining 2 teams.
    """
    df_out = df_pred.copy()
    df_out["predicted_qualified_top2"] = 0

    for g, gdf in df_out.groupby(group_col):
        top2_idx = gdf.sort_values(proba_col, ascending=False).head(2).index
        df_out.loc[top2_idx, "predicted_qualified_top2"] = 1

    return df_out


def fit_statsmodels_logit(X_train, y_train, features):
    # Force numeric float arrays (fixes dtype object error)
    X_sm = X_train[features].copy()
    X_sm = X_sm.apply(pd.to_numeric, errors="coerce").astype(float)
    y_sm = pd.to_numeric(y_train, errors="coerce").astype(float)

    tmp = pd.concat([y_sm.rename("y"), X_sm], axis=1).dropna()
    y_sm = tmp["y"]
    X_sm = tmp.drop(columns=["y"])

    X_sm = sm.add_constant(X_sm, has_constant="add")
    res = sm.Logit(y_sm, X_sm).fit(disp=False, maxiter=500)
    return res


def summarize_logit(res):
    params = res.params
    bse = res.bse
    pvals = res.pvalues
    conf = res.conf_int()
    conf.columns = ["ci_2.5%", "ci_97.5%"]

    out = pd.concat(
        [params.rename("coef"),
         bse.rename("std_err"),
         pvals.rename("p_value"),
         conf],
        axis=1
    )

    out["odds_ratio"] = np.exp(out["coef"])
    out["or_ci_2.5%"] = np.exp(out["ci_2.5%"])
    out["or_ci_97.5%"] = np.exp(out["ci_97.5%"])
    return out


if __name__ == "__main__":
    PATH_XLSX = r"D:\UNI\Dartmouth\winter 24-25\data analytics\project\data\group_stage_qualification_data.xlsx"
    SUBSETS_CSV = r"exhaustive_feature_search.csv"

    df = load_and_clean(PATH_XLSX)
    model_df, X, y, years = build_xy(df)

    print("Data after cleaning:")
    print(f"Rows: {model_df.shape[0]}  Cols (X): {X.shape[1]}")
    print("Class balance:\n", y.value_counts())

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_by_year(model_df, X, y)
    print("\nSplit sizes:")
    print("Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)

    # Load best subset chosen on validation
    best, ranked = load_best_features(SUBSETS_CSV)
    best_features = best["features"]

    print("\nLoaded ranked subsets from:", SUBSETS_CSV)
    print("\nSelected feature set (best on validation):")
    print("k =", int(best["k"]))
    print("features =", best_features)
    print("val_AUC =", best["val_AUC"])
    print("val_Accuracy =", best["val_Accuracy"])

    # Fit sklearn model on TRAIN
    pipe = fit_sklearn_logit(X_train, y_train, best_features)

    # Predict on TEST (2022)
    test_proba = pipe.predict_proba(X_test[best_features])[:, 1]
    test_pred_raw = (test_proba >= 0.5).astype(int)

    # Build base prediction dataframe with identifiers
    test_df = model_df.loc[X_test.index, ["group_id", "team_name", "qualified_from_group"]].copy()
    test_df = test_df.rename(columns={"qualified_from_group": "true_qualified"})
    test_df["prob_qualify"] = test_proba
    test_df["predicted_qualified_raw"] = test_pred_raw

    # 4-col CSV requested (raw)
    out_raw_4 = test_df[["group_id", "team_name", "true_qualified", "predicted_qualified_raw"]].rename(
        columns={"predicted_qualified_raw": "predicted_qualified"}
    )
    out_raw_4.to_csv("2022_test_predictions_raw.csv", index=False)
    print("\nSaved 2022_test_predictions_raw.csv")

    # Metrics for raw threshold predictions
    metrics_block(y_test.values, test_pred_raw, test_proba, label="2022 TEST METRICS (raw threshold 0.5)")

    # Enforce top-2-per-group rule
    test_df_top2 = enforce_top2_per_group(test_df, group_col="group_id", proba_col="prob_qualify")
    test_pred_top2 = test_df_top2["predicted_qualified_top2"].values

    # 4-col CSV requested (top2 constrained)
    out_top2_4 = test_df_top2[["group_id", "team_name", "true_qualified", "predicted_qualified_top2"]].rename(
        columns={"predicted_qualified_top2": "predicted_qualified"}
    )
    out_top2_4.to_csv("2022_test_predictions_top2.csv", index=False)
    print("\nSaved 2022_test_predictions_top2.csv (enforces exactly 2 qualifiers per group)")

    # Metrics for top2 constrained predictions
    metrics_block(y_test.values, test_pred_top2, test_proba, label="2022 TEST METRICS (top-2 per group constrained)")

    # Statsmodels inference on TRAIN (fix dtype issue)
    print("\n--- TRAIN LOGIT COEFFICIENTS (Statsmodels) ---")
    res = fit_statsmodels_logit(X_train, y_train, best_features)
    coef_table = summarize_logit(res)

    pd.set_option("display.max_rows", 200)
    pd.set_option("display.width", 140)
    print(coef_table)

    coef_table.to_csv("final_model_coefficients.csv", index=True)
    print("\nSaved final_model_coefficients.csv")