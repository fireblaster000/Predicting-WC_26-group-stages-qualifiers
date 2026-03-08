# World Cup Group-Stage Qualification Analytics (1998-2026)

This repository contains data preparation, modeling, diagnostics, and 2026 forecasting for FIFA World Cup group-stage qualification.

## Data Access

Access data: **[DATA_FILES_LINK_HERE](https://drive.google.com/drive/folders/1GDgULTU8a4I-x2X-oLd4I8AorngOCNKg?usp=sharing)**

## Important Note on Modeling Code

The **R workflow is the primary, usable, and most interpretable pipeline** in this repo.

The folder `model/python/` was used for **author's discretion experiments only** and is **not the recommended path** for understanding or reproducing final work.

## Repository Structure and File Purposes

```text
project/
+- .gitignore
+- data/
�  +- API_NY.GDP.PCAP.CD_DS2_en_csv_v2_31.csv
�  +- group_stage_qualification_data.xlsx
�  +- group_standings.csv
�  +- Prediction Dataset_World Cup 2026.xlsx
�  +- results.csv
�  +- elo_snapshots_ifootball/
+- data analysis/
�  +- pre modelling/
�  �  +- data_analysis.R
�  �  +- Diagnostic_plots/
�  +- post modelling/
�     +- Loyo Coeff Uncertainity.R
�     +- Loyo Log Odds.R
�     +- Loyo Odds Ratio.R
�     +- Predicted Probability Outcome.R
�     +- plots/
+- model/
�  +- R code (use this)/
�  �  +- model.R
�  �  +- Final_modeling_outputs/
�  +- python/
+- population scripts/
+- predictions/
�  +- 2026.R
�  +- wc26_predictions_loyo_ensemble/
+- README.md
```

## Root-Level

- `.gitignore`: Git ignore rules.

## `data/`

- `API_NY.GDP.PCAP.CD_DS2_en_csv_v2_31.csv`: World Bank GDP-per-capita source file used by feature population scripts.
- `group_stage_qualification_data.xlsx`: Main historical modelling dataset template (World Cups 1998-2022).
- `group_standings.csv`: Group-stage standings source used to populate base rows and outcomes.
- `Prediction Dataset_World Cup 2026.xlsx`: 2026 prediction template with team/group features.
- `results.csv`: Historical match-level results used to compute recent-form features.
- `elo_snapshots_ifootball/elo_snapshot_YYYY-MM-DD.csv`: Cached Elo snapshot files scraped for pre-tournament Elo values.

## `population scripts/` (feature construction)

These scripts populate dataset columns in-place inside Excel templates.

- `wc_groupstage_data_populate.py`: Fills first 6 historical columns (`tournament_name`, year, group, team, qualification label, tournament start date) from `group_standings.csv`.
- `populate_team_elo_pre.py`: Scrapes/caches Elo by cutoff date and writes `team_elo_pre`.
- `populate_matches_cols.py`: Builds recent-form features from `results.csv`:
  - `recent_window_n_matches`
  - `recent_win_rate`
  - `recent_goal_diff_per_match`
  - `recent_goals_for_per_match`
  - `recent_goals_against_per_match`
- `populate_elo_cols.py`: Derives group/opponent Elo context features:
  - `opp_elo_mean`, `opp_elo_max`, `group_elo_std`, `elo_gap_vs_opp_mean`
- `populate_gdp_per_capita.py`: Maps teams to World Bank country names and writes `gdp_per_capita_pre` and `log_gdp_per_capita_pre`.
- `populate_host_flags.py`: Writes host indicator (`host_flag`) by tournament year.
- `populate_confederation.py`: Writes `confederation` using team-to-confederation mapping.

## `data analysis/pre modelling/`

- `data_analysis.R`: Pre-model EDA and diagnostics pipeline. Produces missingness/class-balance/correlation summaries and predictor diagnostic plots.
- `Diagnostic_plots/`: Output folder from `data_analysis.R`.
  - CSV summaries:
    - `class_balance.csv`
    - `missingness_by_column.csv`
    - `qualification_rate_by_confederation.csv`
    - `spearman_assoc_with_target.csv`
    - `spearman_correlation_numeric_predictors.csv`
    - `top_spearman_correlated_pairs.csv`
  - `spearman_corrplot_numeric_predictors.png`: Correlation heatmap.
  - `01_boxplots/`: Predictor boxplots by qualification class.
  - `02_density/`: Predictor density overlays by qualification class.
  - `03_binned_rate/`: Empirical qualification rate by binned predictor.
  - `04_binned_logit/`: Binned log-odds trend plots.

## `model/R code (use this)/` (recommended modeling pipeline)

- `model.R`: Main final modeling script. Trains/evaluates:
  - `glm_base`
  - `glm_alt`
  - `ridge_1se`
  - `ridge_loyo_ensemble`
  Uses time splits (train 1998-2014, val 2018, test 2022), ranking-based top-2-per-group logic, and writes all artifacts.

### `Final_modeling_outputs/`

- `models/`
  - `glm_base.rds`, `glm_alt.rds`, `ridge_cv_alt_lambda1se.rds`: Saved fitted model objects.
- `predictions/`
  - `predictions_*_val2018.csv`, `predictions_*_test2022.csv`: Per-team probabilities/ranks/predicted qualification for each model.
  - `predictions_ridge_loyo_ensemble_test2022.csv`: Main LOYO ensemble test predictions used downstream for interpretation plots.
- `plots/`
  - `roc_*.png`: ROC curves by model/split.
  - `cal_*.png`: Calibration plots by model/split.
- `tables/`
  - `metrics_ALL_MODELS_big_table.csv`: Consolidated metrics.
  - `coefficients_*.csv`: Model coefficients and summaries.
  - `vif_glm_models.csv`: Multicollinearity diagnostics for GLMs.
  - `group_sizes_by_year_group.csv`: Group-size sanity checks.
  - `qualification_rate_by_year.csv`: Base-rate summary by tournament year.

## `data analysis/post modelling/`

Post-fit interpretation and communication plots (primarily using ridge LOYO outputs).

- `Loyo Coeff Uncertainity.R`: Plots mean LOYO ridge coefficients with SD/95% uncertainty bars.
- `Loyo Log Odds.R`: Forest plot on log-odds scale.
- `Loyo Odds Ratio.R`: Forest plot on odds-ratio scale (log x-axis).
- `Predicted Probability Outcome.R`: Team-level predicted probability vs true outcome visualization for 2022.
- `plots/`: Exported final post-model visuals (`.jpeg`/`.png`).

## `predictions/`

- `2026.R`: 2026 forecasting pipeline using LOYO ridge ensemble over 1998-2022, then applies rule:
  - top 2 in each group qualify
  - plus best 8 third-place teams
  - tie-break by `team_elo_pre`
  Generates both CSV outputs and group-faceted probability plots.

### `wc26_predictions_loyo_ensemble/`

- `predictions_2026_loyo_ensemble_all_teams.csv`: Full 2026 scored table.
- `predictions_2026_QUALIFIERS.csv`: Teams predicted to qualify under the rule.
- `predictions_2026_third_place_ranking.csv`: Ranked third-place table with selected 8.
- `wc26_group_probabilities_*.png`: Final visual probability summaries.

## `model/python/` (not recommended)

Used for experimental/alternative modeling runs under author discretion. Not required for the final R-based workflow.

- `threshold constraint/`
  - `all_subsets.py`: Exhaustive subset search for logistic regression.
  - `final_eval.py`: Final evaluation/inference using selected subset.
  - `exhaustive_feature_search.csv`, `final_model_coefficients.csv`: Outputs from threshold-constraint experimentation.
- `top 2 constraint/`
  - `exhauative_top2_search.py`: Exhaustive search with top-2-per-group constraint.
  - `final_top_2_from_file.py`: Loads best subset and evaluates top-2 rule.
  - `*.csv` files: predictions, ranking tables, coefficient order, and search outputs for these experiments.

## Suggested End-to-End Flow (R-first)

1. Prepare/populate datasets in `data/` using scripts in `population scripts/`.
2. Run `data analysis/pre modelling/data_analysis.R` for diagnostics.
3. Run `model/R code (use this)/model.R` for core training/evaluation outputs.
4. Run scripts in `data analysis/post modelling/` for interpretation figures.
5. Run `predictions/2026.R` for 2026 forecast outputs.

## Reproducibility Notes

- Most scripts auto-install required packages if missing.
- Several Python scripts currently point to `Prediction Dataset_World Cup 2026.xlsx` and sheet `Dataset`; adjust paths/sheet names if you switch templates.
- Excel files may need to be closed before in-place writes succeed.
