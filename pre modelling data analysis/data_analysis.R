############################################################
# Part 1 — Diagnostic Plots + Pre-Model Metrics (R Script)
# Creates a folder of EDA/diagnostic plots + summary tables
# BEFORE diving into modeling (logistic/GAM decision support)
############################################################

# -------------------------
# 0) Setup
# -------------------------
required_pkgs <- c(
  "readxl", "dplyr", "tidyr", "ggplot2", "stringr", "purrr",
  "forcats", "scales", "GGally", "corrplot", "broom"
)

to_install <- required_pkgs[!required_pkgs %in% installed.packages()[, "Package"]]
if (length(to_install) > 0) install.packages(to_install, dependencies = TRUE)

library(readxl)
library(dplyr)
library(tidyr)
library(ggplot2)
library(stringr)
library(purrr)
library(forcats)
library(scales)
library(GGally)
library(corrplot)
library(broom)

# -------------------------
# 1) Paths / Output folder
# -------------------------

input_path <- "group_stage_qualification_data.xlsx"

# Main output folder
out_dir <- "Diagnostic_plots"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# Subfolders
dir_box   <- file.path(out_dir, "01_boxplots")
dir_dens  <- file.path(out_dir, "02_density")
dir_bin   <- file.path(out_dir, "03_binned_rate")
dir_logit <- file.path(out_dir, "04_binned_logit")

dir.create(dir_box,   showWarnings = FALSE, recursive = TRUE)
dir.create(dir_dens,  showWarnings = FALSE, recursive = TRUE)
dir.create(dir_bin,   showWarnings = FALSE, recursive = TRUE)
dir.create(dir_logit, showWarnings = FALSE, recursive = TRUE)

# Helper: sanitize filenames
safe_name <- function(x) {
  x %>%
    tolower() %>%
    stringr::str_replace_all("[^a-z0-9_]+", "_") %>%
    stringr::str_replace_all("_+", "_") %>%
    stringr::str_replace_all("^_|_$", "")
}

# Helper: save ggplot to a specific folder
save_plot_to <- function(p, folder, filename, w = 9, h = 5.5, dpi = 200) {
  ggplot2::ggsave(
    filename = file.path(folder, filename),
    plot = p,
    width = w, height = h, dpi = dpi
  )
}

# -------------------------
# 2) Load data
# -------------------------
# If your Excel has multiple sheets, set sheet = "Sheet1" (or name)
df <- readxl::read_xlsx(
  input_path,
  skip = 2   # Skip the first title row
)

# Clean column names
names(df) <- names(df) %>%
  stringr::str_trim() %>%
  stringr::str_replace_all("\\s+", "_") %>%
  stringr::str_replace_all("[^A-Za-z0-9_]", "") %>%
  tolower()

# View column names
print(names(df))

# Expect target column: Qualified_from_group (binary 0/1)
# If your column name differs, change here:
target_col <- "qualified_from_group"
if (!target_col %in% names(df)) {
  stop(paste0("Target column '", target_col, "' not found. Available columns:\n",
              paste(names(df), collapse = ", ")))
}

# Coerce target to 0/1 integer and factor label
df <- df %>%
  mutate(
    Qualified_from_group = as.integer(.data[[target_col]]),
    Qualified_factor = factor(Qualified_from_group, levels = c(0, 1), labels = c("Not Qualified", "Qualified"))
  )

# -------------------------
# 3) Basic dataset health checks (saved as CSV)
# -------------------------
# Missingness by column
missing_tbl <- tibble(
  variable = names(df),
  n_missing = sapply(df, function(x) sum(is.na(x))),
  pct_missing = round(100 * n_missing / nrow(df), 2)
) %>% arrange(desc(pct_missing))

write.csv(missing_tbl, file.path(out_dir, "missingness_by_column.csv"), row.names = FALSE)

# Class balance
class_balance <- df %>%
  count(Qualified_factor) %>%
  mutate(pct = round(100 * n / sum(n), 2))

write.csv(class_balance, file.path(out_dir, "class_balance.csv"), row.names = FALSE)

# If Confederation exists, summarize qualification rates by confederation
if ("confederation" %in% names(df)) {
  confed_rate <- df %>%
    group_by(confederation) %>%
    summarise(
      n = n(),
      qualified = sum(Qualified_from_group == 1, na.rm = TRUE),
      qual_rate = qualified / n
    ) %>%
    arrange(desc(qual_rate))
  
  write.csv(confed_rate, file.path(out_dir, "qualification_rate_by_confederation.csv"), row.names = FALSE)
}

# -------------------------
# 4) Identify numeric predictors to diagnose
# -------------------------
# Drop ID-like and obvious non-features from numeric list if needed
numeric_cols <- df %>%
  select(where(is.numeric)) %>%
  names()

# Remove target itself from numeric predictors
numeric_predictors <- setdiff(numeric_cols, c("qualified_from_group", "Qualified_from_group", "recent_window_n_matches"))

# Also remove tournament_year if you want it treated as categorical in plots
# (Uncomment if desired)
numeric_predictors <- setdiff(numeric_predictors, c("tournament_year"))

# -------------------------
# 5) Diagnostic Plots: Relationship to Target (linearity / non-linearity clues)
# -------------------------
# -------------------------
# A) Boxplots: predictor vs qualified
# -------------------------
purrr::walk(numeric_predictors, function(v) {
  p <- ggplot2::ggplot(df, ggplot2::aes(x = Qualified_factor, y = .data[[v]])) +
    ggplot2::geom_boxplot(outlier.alpha = 0.25) +
    ggplot2::labs(
      title = paste0("Boxplot: ", v, " by Qualification"),
      x = NULL, y = v
    ) +
    ggplot2::theme_minimal(base_size = 12)
  
  save_plot_to(p, dir_box, paste0("boxplot_", safe_name(v), ".png"))
})

# -------------------------
# B) Density plots: compare distributions by qualified/not
# -------------------------
purrr::walk(numeric_predictors, function(v) {
  p <- ggplot2::ggplot(df, ggplot2::aes(x = .data[[v]], fill = Qualified_factor)) +
    ggplot2::geom_density(alpha = 0.35, na.rm = TRUE) +
    ggplot2::labs(
      title = paste0("Density: ", v, " by Qualification"),
      x = v, y = "Density", fill = NULL
    ) +
    ggplot2::theme_minimal(base_size = 12)
  
  save_plot_to(p, dir_dens, paste0("density_", safe_name(v), ".png"))
})

# -------------------------
# C) Binned qualification rate vs predictor + LOESS smoother
# -------------------------
# C) Binned qualification rate vs predictor + LOESS smoother # This is the key diagnostic for “is it roughly linear in probability / monotonic / curved?” 
bin_plot <- function(data, x, bins = 10) { 
  # Handle ties / low-variance predictors safely 
  xvals <- data[[x]] 
  if (all(is.na(xvals)) || length(unique(na.omit(xvals))) < 5) return(NULL) 
  tmp <- data %>% 
    filter(!is.na(.data[[x]]), !is.na(Qualified_from_group)) %>% 
    mutate(bin = ntile(.data[[x]], bins)) %>% 
    group_by(bin) %>% 
    summarise( 
      x_mid = median(.data[[x]], na.rm = TRUE), 
      n = n(), p_hat = mean(Qualified_from_group, na.rm = TRUE), 
      # Normal approx CI (ok for quick diagnostics) 
      se = sqrt(p_hat * (1 - p_hat) / pmax(n, 1)), 
      lo = pmax(0, p_hat - 1.96 * se), 
      hi = pmin(1, p_hat + 1.96 * se), 
      .groups = "drop" ) 
  ggplot(tmp, aes(x = x_mid, y = p_hat)) + 
    geom_point() + 
    geom_errorbar(aes(ymin = lo, ymax = hi), width = 0) + geom_smooth(method = "loess", se = TRUE, formula = y ~ x) + 
    scale_y_continuous(labels = percent_format(accuracy = 1)) + 
    labs( 
      title = paste0("Binned Qualification Rate vs ", x), 
      subtitle = "Points = bin means; error bars = ~95% CI; curve = LOESS", 
      x = x, 
      y = "Qualification Rate" ) + 
    theme_minimal(base_size = 12) }

purrr::walk(numeric_predictors, function(v) {
  p <- bin_plot(df, v, bins = 10)
  if (!is.null(p)) {
    save_plot_to(p, dir_bin, paste0("binned_rate_", safe_name(v), ".png"))
  }
})

# -------------------------
# D) Binned logit(p) vs predictor (linearity-in-log-odds check)
# -------------------------
logit <- function(p) log(p / (1 - p))

bin_logit_plot <- function(data, x, bins = 10, eps = 1e-4) {
  xvals <- data[[x]]
  if (all(is.na(xvals)) || length(unique(stats::na.omit(xvals))) < 5) return(NULL)
  
  tmp <- data %>%
    dplyr::filter(!is.na(.data[[x]]), !is.na(Qualified_from_group)) %>%
    dplyr::mutate(bin = dplyr::ntile(.data[[x]], bins)) %>%
    dplyr::group_by(bin) %>%
    dplyr::summarise(
      x_mid = stats::median(.data[[x]], na.rm = TRUE),
      n = dplyr::n(),
      p_hat = mean(Qualified_from_group, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    dplyr::mutate(
      p_clip = pmin(1 - eps, pmax(eps, p_hat)),
      logit_p = logit(p_clip)
    )
  
  ggplot2::ggplot(tmp, ggplot2::aes(x = x_mid, y = logit_p)) +
    ggplot2::geom_point() +
    ggplot2::geom_smooth(method = "lm", se = TRUE) +
    ggplot2::labs(
      title = paste0("Binned Logit(p) vs ", x),
      subtitle = "If ~linear, logistic linear term is reasonable",
      x = x,
      y = "logit(qualification rate)"
    ) +
    ggplot2::theme_minimal(base_size = 12)
}

purrr::walk(numeric_predictors, function(v) {
  p <- bin_logit_plot(df, v, bins = 10)
  if (!is.null(p)) {
    save_plot_to(p, dir_logit, paste0("binned_logit_", safe_name(v), ".png"))
  }
})

# -------------------------
# 6) Correlation Diagnostics (numeric-numeric)
# -------------------------
# Spearman correlation among numeric predictors
num_mat <- df %>%
  select(all_of(numeric_predictors)) %>%
  mutate(across(everything(), as.numeric)) %>%
  as.data.frame()

cor_spearman <- cor(num_mat, use = "pairwise.complete.obs", method = "spearman")

# Save correlation matrix as CSV
write.csv(cor_spearman, file.path(out_dir, "spearman_correlation_numeric_predictors.csv"))

# Correlation heatmap (saved as PNG)
png(file.path(out_dir, "spearman_corrplot_numeric_predictors.png"), width = 1400, height = 1200)
corrplot::corrplot(cor_spearman, method = "color", type = "lower",
                   tl.cex = 0.7, tl.col = "black", number.cex = 0.6)
dev.off()

get_top_pairs <- function(cmat, k = 30) {
  stopifnot(is.matrix(cmat), nrow(cmat) == ncol(cmat))
  
  cmat2 <- cmat
  diag(cmat2) <- NA
  cmat2[lower.tri(cmat2, diag = TRUE)] <- NA  # keep only upper triangle
  
  idx <- which(!is.na(cmat2), arr.ind = TRUE)
  
  pairs <- tibble::tibble(
    var1 = rownames(cmat2)[idx[, 1]],
    var2 = colnames(cmat2)[idx[, 2]],
    corr = cmat2[idx]
  ) %>%
    dplyr::mutate(abs_corr = abs(corr)) %>%
    dplyr::arrange(dplyr::desc(abs_corr)) %>%
    dplyr::slice_head(n = k)
  
  pairs
}

top_corr_pairs <- get_top_pairs(cor_spearman, k = 40)
readr::write_csv(top_corr_pairs, file.path(out_dir, "top_spearman_correlated_pairs.csv"))

# -------------------------
# 7) Predictor-to-target association (quick metrics)
# -------------------------
# For numeric predictors vs binary target:
# - Spearman correlation with target (monotonic association)
# - A simple univariate AUC is *model-ish*; we’ll skip that here to keep it “pre-model”.
#   If you want univariate AUCs, tell me and I’ll add it.

target_assoc <- map_dfr(numeric_predictors, function(v) {
  x <- df[[v]]
  y <- df$Qualified_from_group
  ok <- complete.cases(x, y)
  if (sum(ok) < 10 || length(unique(x[ok])) < 5) {
    return(tibble(variable = v, spearman_rho = NA_real_, p_value = NA_real_))
  }
  ct <- suppressWarnings(cor.test(x[ok], y[ok], method = "spearman"))
  tibble(
    variable = v,
    spearman_rho = unname(ct$estimate),
    p_value = ct$p.value
  )
}) %>%
  arrange(desc(abs(spearman_rho)))

write.csv(target_assoc, file.path(out_dir, "spearman_assoc_with_target.csv"), row.names = FALSE)


# -------------------------
# 9) Done
# -------------------------
message("✅ Part 1 diagnostics complete.")
message(paste0("Outputs saved to folder: ", normalizePath(out_dir)))
