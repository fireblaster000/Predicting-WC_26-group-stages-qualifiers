############################################################
# FINAL Part 2 — Modeling + Evaluation (World Cup Group Qualifiers)
#
# Included models (ONLY 4):
# 1) glm_base           (interpretable baseline)
# 2) glm_alt            (feature-engineering comparison)
# 3) ridge_1se          (regularized logistic, lambda.1se)
# 4) ridge_loyo_ensemble (best generalization; LOYO ensemble on 2022)
#
# Key rules:
# - Map OFC -> AFC (keep group structure intact)
# - Use CONMEBOL as reference confederation
# - Save everything under: final_modeling_outputs/
############################################################

# -------------------------
# 0) Packages
# -------------------------
pkgs <- c(
  "readxl", "dplyr", "stringr", "forcats",
  "ggplot2", "pROC", "yardstick", "broom",
  "Matrix", "glmnet", "tibble"
)

to_install <- pkgs[!pkgs %in% installed.packages()[, "Package"]]
if (length(to_install) > 0) install.packages(to_install, dependencies = TRUE)

library(readxl)
library(dplyr)
library(stringr)
library(forcats)
library(ggplot2)
library(pROC)
library(yardstick)
library(broom)
library(Matrix)
library(glmnet)
library(tibble)

# -------------------------
# 1) Paths / Output folders
# -------------------------
# Edit this if you move your Excel file or if the sheet name is different. The script will read the tournament_year, group_id, team_name, qualified_from_group, confederation, host_team_flag, team_elo_pre, elo_gap_vs_opp_mean, recent_goal_diff_per_match, recent_win_rate, log_gdp_per_capita_pre columns.
input_path <- "group_stage_qualification_data.xlsx"

out_dir <- "Final_modeling_outputs"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

dir_models <- file.path(out_dir, "models")
dir_plots  <- file.path(out_dir, "plots")
dir_tbls   <- file.path(out_dir, "tables")
dir_preds  <- file.path(out_dir, "predictions")

dir.create(dir_models, showWarnings = FALSE, recursive = TRUE)
dir.create(dir_plots,  showWarnings = FALSE, recursive = TRUE)
dir.create(dir_tbls,   showWarnings = FALSE, recursive = TRUE)
dir.create(dir_preds,  showWarnings = FALSE, recursive = TRUE)

# -------------------------
# 2) Load & clean data
# -------------------------
df <- readxl::read_xlsx(input_path, skip = 2)

names(df) <- names(df) %>%
  str_trim() %>%
  str_replace_all("\\s+", "_") %>%
  str_replace_all("[^A-Za-z0-9_]", "") %>%
  tolower()

required <- c(
  "tournament_year","group_id","team_name",
  "qualified_from_group","confederation","host_team_flag",
  "team_elo_pre","elo_gap_vs_opp_mean",
  "recent_goal_diff_per_match","recent_win_rate",
  "log_gdp_per_capita_pre"
)

missing_cols <- setdiff(required, names(df))
if (length(missing_cols) > 0) {
  stop("Missing required columns: ", paste(missing_cols, collapse = ", "))
}

df <- df %>%
  mutate(
    tournament_year = as.integer(tournament_year),
    qualified_from_group = as.integer(qualified_from_group),
    qualified_factor = factor(qualified_from_group, levels = c(0,1), labels = c("NotQualified", "Qualified")),
    host_team_flag = as.integer(host_team_flag),
    
    # Map OFC -> AFC
    confederation = ifelse(confederation == "OFC", "AFC", confederation),
    confederation = as.factor(confederation),
    
    team_name = as.character(team_name),
    group_id = as.character(group_id)
  )

# Keep global levels and force CONMEBOL as reference everywhere
all_conf_levels <- levels(df$confederation)
df$confederation <- factor(df$confederation, levels = all_conf_levels)
df$confederation <- relevel(df$confederation, ref = "CONMEBOL")

model_cols <- c(
  "tournament_year","group_id","team_name",
  "qualified_from_group","qualified_factor",
  "confederation","host_team_flag",
  "team_elo_pre","elo_gap_vs_opp_mean",
  "recent_goal_diff_per_match","recent_win_rate",
  "log_gdp_per_capita_pre"
)

df_mod <- df %>%
  select(all_of(model_cols)) %>%
  filter(if_all(everything(), ~ !is.na(.)))

# Sanity: group size table
group_sizes <- df_mod %>%
  count(tournament_year, group_id, name = "n") %>%
  arrange(tournament_year, group_id)
write.csv(group_sizes, file.path(dir_tbls, "group_sizes_by_year_group.csv"), row.names = FALSE)

# -------------------------
# 3) Time-based split
# -------------------------
train_df <- df_mod %>% filter(tournament_year >= 1998, tournament_year <= 2014)
val_df   <- df_mod %>% filter(tournament_year == 2018)
test_df  <- df_mod %>% filter(tournament_year == 2022)

cat("Rows:\n")
cat("Train:", nrow(train_df), "\n")
cat("Val  :", nrow(val_df), "\n")
cat("Test :", nrow(test_df), "\n")

# Qualification rate by year
rate_tbl <- df_mod %>%
  group_by(tournament_year) %>%
  summarise(n=n(), qual_rate=mean(qualified_from_group), .groups="drop")
write.csv(rate_tbl, file.path(dir_tbls, "qualification_rate_by_year.csv"), row.names = FALSE)

# Ensure factor levels consistent + ref level set (prevents predict() errors)
fix_levels <- function(d, all_levels) {
  d$confederation <- factor(d$confederation, levels = all_levels)
  d$confederation <- relevel(d$confederation, ref = "CONMEBOL")
  d
}

train_df <- fix_levels(train_df, all_conf_levels)
val_df   <- fix_levels(val_df,   all_conf_levels)
test_df  <- fix_levels(test_df,  all_conf_levels)

# -------------------------
# 4) Utility: metrics + plots
# -------------------------
logloss <- function(y, p, eps = 1e-15) {
  p <- pmin(1-eps, pmax(eps, p))
  -mean(y*log(p) + (1-y)*log(1-p))
}
brier <- function(y, p) mean((p - y)^2)

confusion_metrics <- function(y_true, y_pred01) {
  tp <- sum(y_true == 1 & y_pred01 == 1)
  tn <- sum(y_true == 0 & y_pred01 == 0)
  fp <- sum(y_true == 0 & y_pred01 == 1)
  fn <- sum(y_true == 1 & y_pred01 == 0)
  
  acc <- (tp + tn) / (tp + tn + fp + fn)
  prec <- ifelse(tp + fp == 0, NA, tp / (tp + fp))
  rec  <- ifelse(tp + fn == 0, NA, tp / (tp + fn))
  f1   <- ifelse(is.na(prec) | is.na(rec) | (prec+rec)==0, NA, 2*prec*rec/(prec+rec))
  
  tibble(tp=tp, tn=tn, fp=fp, fn=fn,
         accuracy=acc, precision=prec, recall=rec, f1=f1)
}

make_group_rank_predictions <- function(df_split, probs, top_k = 2) {
  df_split %>%
    mutate(pred_prob = probs) %>%
    group_by(tournament_year, group_id) %>%
    mutate(
      pred_rank = rank(-pred_prob, ties.method = "first"),
      predicted_qualify = ifelse(pred_rank <= top_k, 1L, 0L)
    ) %>%
    ungroup()
}

eval_ranked <- function(df_split, probs, top_k = 2) {
  y <- df_split$qualified_from_group
  
  ll <- logloss(y, probs)
  bs <- brier(y, probs)
  
  roc_obj <- pROC::roc(y, probs, quiet = TRUE)
  auc_roc <- as.numeric(pROC::auc(roc_obj))
  
  pr <- tibble(
    truth = df_split$qualified_factor,
    .pred_Qualified = probs
  )
  auc_pr <- tryCatch(
    yardstick::pr_auc(pr, truth = truth, .pred_Qualified)$.estimate,
    error = function(e) NA_real_
  )
  
  pred_df <- make_group_rank_predictions(df_split, probs, top_k = top_k)
  cm <- confusion_metrics(pred_df$qualified_from_group, pred_df$predicted_qualify)
  
  tibble(
    roc_auc = auc_roc,
    pr_auc = auc_pr,
    logloss = ll,
    brier = bs,
    top_k = top_k
  ) %>% bind_cols(cm)
}

save_pred_df <- function(pred_df, model_name, split_name) {
  out <- pred_df %>%
    select(
      tournament_year, group_id, team_name,
      qualified_from_group,
      pred_prob, pred_rank, predicted_qualify
    ) %>%
    arrange(tournament_year, group_id, pred_rank)
  
  write.csv(out, file.path(dir_preds, paste0("predictions_", model_name, "_", split_name, ".csv")),
            row.names = FALSE)
  out
}

calibration_plot <- function(df_split, probs, title, filename) {
  tmp <- df_split %>%
    mutate(p = probs, y = qualified_from_group) %>%
    mutate(bin = ntile(p, 10)) %>%
    group_by(bin) %>%
    summarise(p_mean = mean(p), y_rate = mean(y), n = n(), .groups="drop")
  
  g <- ggplot(tmp, aes(x = p_mean, y = y_rate)) +
    geom_point() +
    geom_line() +
    geom_abline(intercept = 0, slope = 1, linetype = "dashed") +
    labs(title = title, x = "Mean predicted probability (decile)", y = "Observed qualification rate") +
    theme_minimal(base_size = 12)
  
  ggsave(file.path(dir_plots, filename), g, width = 7.5, height = 5, dpi = 200)
}

roc_plot <- function(df_split, probs, title, filename) {
  y <- df_split$qualified_from_group
  roc_obj <- pROC::roc(y, probs, quiet = TRUE)
  roc_df <- tibble(tpr = roc_obj$sensitivities, fpr = 1 - roc_obj$specificities)
  
  g <- ggplot(roc_df, aes(x = fpr, y = tpr)) +
    geom_line() +
    geom_abline(intercept = 0, slope = 1, linetype = "dashed") +
    labs(title = title, x = "False Positive Rate", y = "True Positive Rate") +
    theme_minimal(base_size = 12)
  
  ggsave(file.path(dir_plots, filename), g, width = 7.5, height = 5, dpi = 200)
}

# -------------------------
# 5) Model formulas (ONLY 2 GLMs + ridge uses alt)
# -------------------------
base_formula <- qualified_from_group ~
  team_elo_pre +
  recent_goal_diff_per_match +
  log_gdp_per_capita_pre +
  host_team_flag +
  confederation

alt_formula <- qualified_from_group ~
  elo_gap_vs_opp_mean +
  recent_win_rate +
  log_gdp_per_capita_pre +
  host_team_flag +
  confederation

# Ridge uses alt_formula design matrix consistently
terms_alt <- terms(alt_formula, data = df_mod)

# Storage for final outputs
all_metrics <- list()
coef_tables <- list()

# -------------------------
# 6) Model 1: GLM BASE
# -------------------------
m_glm_base <- glm(base_formula, data = train_df, family = binomial())
saveRDS(m_glm_base, file.path(dir_models, "glm_base.rds"))

p_val_glm_base  <- predict(m_glm_base, newdata = val_df,  type = "response")
p_test_glm_base <- predict(m_glm_base, newdata = test_df, type = "response")

met_val_glm_base  <- eval_ranked(val_df,  p_val_glm_base,  top_k = 2) %>% mutate(model="glm_base", split="val2018")
met_test_glm_base <- eval_ranked(test_df, p_test_glm_base, top_k = 2) %>% mutate(model="glm_base", split="test2022")

all_metrics[["glm_base"]] <- bind_rows(met_val_glm_base, met_test_glm_base)

pred_val_glm_base  <- make_group_rank_predictions(val_df,  p_val_glm_base,  top_k = 2)
pred_test_glm_base <- make_group_rank_predictions(test_df, p_test_glm_base, top_k = 2)
save_pred_df(pred_val_glm_base,  "glm_base", "val2018")
save_pred_df(pred_test_glm_base, "glm_base", "test2022")

roc_plot(val_df, p_val_glm_base, "ROC — GLM Base (Val 2018)", "roc_glm_base_val2018.png")
calibration_plot(val_df, p_val_glm_base, "Calibration — GLM Base (Val 2018)", "cal_glm_base_val2018.png")

coef_glm_base <- broom::tidy(m_glm_base) %>%
  mutate(
    model = "glm_base",
    odds_ratio = exp(estimate)
  ) %>%
  select(model, term, estimate, std.error, statistic, p.value, odds_ratio)

coef_tables[["glm_base"]] <- coef_glm_base
write.csv(coef_glm_base, file.path(dir_tbls, "coefficients_glm_base.csv"), row.names = FALSE)

# -------------------------
# 7) Model 2: GLM ALT
# -------------------------
m_glm_alt <- glm(alt_formula, data = train_df, family = binomial())
saveRDS(m_glm_alt, file.path(dir_models, "glm_alt.rds"))

p_val_glm_alt  <- predict(m_glm_alt, newdata = val_df,  type = "response")
p_test_glm_alt <- predict(m_glm_alt, newdata = test_df, type = "response")

met_val_glm_alt  <- eval_ranked(val_df,  p_val_glm_alt,  top_k = 2) %>% mutate(model="glm_alt", split="val2018")
met_test_glm_alt <- eval_ranked(test_df, p_test_glm_alt, top_k = 2) %>% mutate(model="glm_alt", split="test2022")

all_metrics[["glm_alt"]] <- bind_rows(met_val_glm_alt, met_test_glm_alt)

pred_val_glm_alt  <- make_group_rank_predictions(val_df,  p_val_glm_alt,  top_k = 2)
pred_test_glm_alt <- make_group_rank_predictions(test_df, p_test_glm_alt, top_k = 2)
save_pred_df(pred_val_glm_alt,  "glm_alt", "val2018")
save_pred_df(pred_test_glm_alt, "glm_alt", "test2022")

roc_plot(val_df, p_val_glm_alt, "ROC — GLM Alt (Val 2018)", "roc_glm_alt_val2018.png")
calibration_plot(val_df, p_val_glm_alt, "Calibration — GLM Alt (Val 2018)", "cal_glm_alt_val2018.png")

coef_glm_alt <- broom::tidy(m_glm_alt) %>%
  mutate(
    model = "glm_alt",
    odds_ratio = exp(estimate)
  ) %>%
  select(model, term, estimate, std.error, statistic, p.value, odds_ratio)

coef_tables[["glm_alt"]] <- coef_glm_alt
write.csv(coef_glm_alt, file.path(dir_tbls, "coefficients_glm_alt.csv"), row.names = FALSE)


# -------------------------
# 7.5) Metrics to test which model is better
# -------------------------

# Attach AIC/BIC to GLM metric rows
add_aic_bic <- function(metrics_df, fitted_glm) {
  metrics_df %>%
    mutate(
      aic = AIC(fitted_glm),
      bic = BIC(fitted_glm)
    )
}

all_metrics[["glm_base"]] <- bind_rows(
  add_aic_bic(met_val_glm_base,  m_glm_base),
  add_aic_bic(met_test_glm_base, m_glm_base)
)

all_metrics[["glm_alt"]] <- bind_rows(
  add_aic_bic(met_val_glm_alt,  m_glm_alt),
  add_aic_bic(met_test_glm_alt, m_glm_alt)
)


library(car)

vif_to_df <- function(fit, model_name) {
  v <- car::vif(fit)
  
  # If matrix (e.g., GVIF output), convert appropriately
  if (is.matrix(v)) {
    # For factors with >1 df, use GVIF^(1/(2*Df)) for comparability
    out <- as.data.frame(v)
    out$term <- rownames(out)
    
    if (all(c("GVIF", "Df") %in% colnames(out))) {
      out <- out %>%
        mutate(vif = GVIF^(1/(2*Df))) %>%
        select(term, vif)
    } else {
      # fallback: take first column as "vif-like"
      out <- out %>%
        mutate(vif = .[[1]]) %>%
        select(term, vif)
    }
  } else {
    out <- tibble(term = names(v), vif = as.numeric(v))
  }
  
  out %>%
    mutate(model = model_name) %>%
    select(model, term, vif) %>%
    arrange(desc(vif))
}

vif_tbl <- bind_rows(
  vif_to_df(m_glm_alt,  "glm_alt"),
  vif_to_df(m_glm_base, "glm_base")
)

write.csv(vif_tbl, file.path(dir_tbls, "vif_glm_models.csv"), row.names = FALSE)
print(vif_tbl)

# -------------------------
# 8) Model 3: RIDGE (lambda.1se) using ALT design matrix
# -------------------------
x_train <- model.matrix(terms_alt, train_df)[, -1]
y_train <- train_df$qualified_from_group

x_val  <- model.matrix(terms_alt, val_df)[, -1]
x_test <- model.matrix(terms_alt, test_df)[, -1]

set.seed(123)
cv_ridge <- cv.glmnet(
  x = x_train, y = y_train,
  family = "binomial",
  alpha = 0,
  nfolds = 5,
  type.measure = "deviance"
)
saveRDS(cv_ridge, file.path(dir_models, "ridge_cv_alt_lambda1se.rds"))

p_val_ridge_1se  <- as.numeric(predict(cv_ridge, newx = x_val,  s = "lambda.1se", type = "response"))
p_test_ridge_1se <- as.numeric(predict(cv_ridge, newx = x_test, s = "lambda.1se", type = "response"))

met_val_ridge  <- eval_ranked(val_df,  p_val_ridge_1se,  top_k = 2) %>% mutate(model="ridge_1se", split="val2018")
met_test_ridge <- eval_ranked(test_df, p_test_ridge_1se, top_k = 2) %>% mutate(model="ridge_1se", split="test2022")

all_metrics[["ridge_1se"]] <- bind_rows(met_val_ridge, met_test_ridge)

pred_val_ridge  <- make_group_rank_predictions(val_df,  p_val_ridge_1se,  top_k = 2)
pred_test_ridge <- make_group_rank_predictions(test_df, p_test_ridge_1se, top_k = 2)
save_pred_df(pred_val_ridge,  "ridge_1se", "val2018")
save_pred_df(pred_test_ridge, "ridge_1se", "test2022")

roc_plot(val_df, p_val_ridge_1se, "ROC — Ridge (λ1se) (Val 2018)", "roc_ridge_1se_val2018.png")
calibration_plot(val_df, p_val_ridge_1se, "Calibration — Ridge (λ1se) (Val 2018)", "cal_ridge_1se_val2018.png")

# Ridge coefficients: no standard p-values in glmnet.
# Save coefficients at lambda.1se (log-odds scale), plus odds_ratio.
ridge_coef <- as.matrix(coef(cv_ridge, s = "lambda.1se"))
coef_ridge_tbl <- tibble(
  model = "ridge_1se",
  term = rownames(ridge_coef),
  estimate = as.numeric(ridge_coef[,1]),
  odds_ratio = exp(estimate)
)
write.csv(coef_ridge_tbl, file.path(dir_tbls, "coefficients_ridge_1se_lambda1se.csv"), row.names = FALSE)

# -------------------------
# 9) Model 4: RIDGE LOYO ENSEMBLE (lambda.1se) evaluated on 2022
# -------------------------
loyo_years <- sort(unique(df_mod$tournament_year[
  df_mod$tournament_year >= 1998 &
    df_mod$tournament_year <= 2018
]))

test2022 <- df_mod %>% filter(tournament_year == 2022)
test2022 <- fix_levels(test2022, all_conf_levels)

# prediction matrix across folds
p_mat <- matrix(NA_real_, nrow = nrow(test2022), ncol = length(loyo_years))
colnames(p_mat) <- loyo_years

coef_list <- list()

for (i in seq_along(loyo_years)) {
  yy <- loyo_years[i]
  
  tr <- df_mod %>%
    filter(tournament_year != yy,
           tournament_year >= 1998,
           tournament_year <= 2018)
  
  tr <- fix_levels(tr, all_conf_levels)
  
  x_tr <- model.matrix(terms_alt, tr)[, -1]
  y_tr <- tr$qualified_from_group
  x_te <- model.matrix(terms_alt, test2022)[, -1]
  
  set.seed(123)
  cv_fit <- cv.glmnet(
    x = x_tr,
    y = y_tr,
    family = "binomial",
    alpha = 0,
    nfolds = 5,
    type.measure = "deviance"
  )
  
  p_mat[, i] <- as.numeric(predict(cv_fit, newx = x_te, s = "lambda.1se", type = "response"))
  
  cf <- as.matrix(coef(cv_fit, s = "lambda.1se"))
  coef_list[[i]] <- tibble(
    model = "ridge_loyo_ensemble",
    term = rownames(cf),
    estimate = as.numeric(cf[,1]),
    holdout_year = yy
  )
}

p_2022_ens <- rowMeans(p_mat, na.rm = TRUE)

pred_2022_ens <- make_group_rank_predictions(test2022, p_2022_ens, top_k = 2)
met_2022_ens  <- eval_ranked(test2022, p_2022_ens, top_k = 2) %>%
  mutate(model="ridge_loyo_ensemble", split="test2022")

all_metrics[["ridge_loyo_ensemble"]] <- met_2022_ens

save_pred_df(pred_2022_ens, "ridge_loyo_ensemble", "test2022")

roc_plot(test2022, p_2022_ens, "ROC — Ridge LOYO Ensemble (Test 2022)", "roc_ridge_loyo_ensemble_test2022.png")
calibration_plot(test2022, p_2022_ens, "Calibration — Ridge LOYO Ensemble (Test 2022)", "cal_ridge_loyo_ensemble_test2022.png")

# Coefficient stability summary (mean/sd across LOYO folds)
coef_all <- bind_rows(coef_list)

coef_loyo_summary <- coef_all %>%
  group_by(model, term) %>%
  summarise(
    estimate_mean = mean(estimate, na.rm = TRUE),
    estimate_sd   = sd(estimate, na.rm = TRUE),
    odds_ratio_mean = exp(estimate_mean),
    .groups = "drop"
  ) %>%
  arrange(desc(abs(estimate_mean)))

write.csv(coef_loyo_summary, file.path(dir_tbls, "coefficients_ridge_loyo_ensemble_summary.csv"), row.names = FALSE)

# -------------------------
# 10) Save ONE BIG metrics table (all models)
# -------------------------

all_metrics[["ridge_1se"]] <- bind_rows(met_val_ridge, met_test_ridge) %>%
  mutate(aic = NA_real_, bic = NA_real_)

all_metrics[["ridge_loyo_ensemble"]] <- met_2022_ens %>%
  mutate(aic = NA_real_, bic = NA_real_)

metrics_big <- bind_rows(all_metrics) %>%
  select(model, split, everything()) %>%
  arrange(split, desc(roc_auc))

write.csv(metrics_big, file.path(dir_tbls, "metrics_ALL_MODELS_big_table.csv"), row.names = FALSE)

# -------------------------
# 11) Save ONE BIG coefficient table (where applicable)
#     - GLMs: estimate, std.error, p.value, odds_ratio
#     - Ridge: coefficient only (no p-values), odds_ratio
# -------------------------

# --- GLM coefficients (already have full inference stats) ---
coef_glm_base <- broom::tidy(m_glm_base) %>%
  mutate(model = "glm_base", odds_ratio = exp(estimate)) %>%
  select(model, term, estimate, std.error, statistic, p.value, odds_ratio)

coef_glm_alt <- broom::tidy(m_glm_alt) %>%
  mutate(model = "glm_alt", odds_ratio = exp(estimate)) %>%
  select(model, term, estimate, std.error, statistic, p.value, odds_ratio)

# --- Ridge coefficients at lambda.1se (no std.error/p-values) ---
ridge_mat <- as.matrix(coef(cv_ridge, s = "lambda.1se"))
coef_ridge_1se <- tibble(
  model = "ridge_1se",
  term = rownames(ridge_mat),
  estimate = as.numeric(ridge_mat[,1]),
  std.error = NA_real_,
  statistic = NA_real_,
  p.value = NA_real_,
  odds_ratio = exp(estimate)
)

# --- LOYO ridge ensemble: mean/sd across folds (no p-values) ---
# coef_loyo_summary currently has: term, estimate_mean, estimate_sd
coef_ridge_loyo <- coef_loyo_summary %>%
  transmute(
    model = "ridge_loyo_ensemble",
    term = term,
    estimate = estimate_mean,
    std.error = estimate_sd,         # use sd as a "stability" analog
    statistic = NA_real_,
    p.value = NA_real_,
    odds_ratio = exp(estimate_mean)
  )

coef_big_all <- bind_rows(
  coef_glm_base,
  coef_glm_alt,
  coef_ridge_1se,
  coef_ridge_loyo
) %>%
  arrange(model, desc(abs(estimate)))

write.csv(coef_big_all, file.path(dir_tbls, "coefficients_ALL_MODELS_big_table.csv"), row.names = FALSE)

# Note: Ridge coeffs already saved separately (glmnet has no classical p-values)
# - coefficients_ridge_1se_lambda1se.csv
# - coefficients_ridge_loyo_ensemble_summary.csv

message("✅ FINAL modeling complete. Outputs saved to: ", normalizePath(out_dir))
