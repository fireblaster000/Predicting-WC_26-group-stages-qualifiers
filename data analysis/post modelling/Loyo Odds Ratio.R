# =============================================================================
# plot_3_odds_ratio_loyo.R
# Ridge LOYO Ensemble — Odds Ratio Forest Plot (log scale)
#
# Input:  coefficients_ridge_loyo_ensemble_summary.csv
# Output: plot_3_odds_ratio_loyo.png
# =============================================================================

library(ggplot2)
library(dplyr)
library(stringr)

# edit these if you move your input CSV or want to change the output plot name
LOYO_COEF_CSV <- "coefficients_ridge_loyo_ensemble_summary.csv"
OUT_PNG       <- "plot_3_odds_ratio_loyo.png"
OR_CAP        <- 10

# ---------------------------------------------------------------------------
# Load & prep
# ---------------------------------------------------------------------------
coef_df <- read.csv(LOYO_COEF_CSV) %>%
  filter(term != "(Intercept)") %>%
  mutate(
    label = term %>%
      str_replace("^confederation", "") %>%
      str_replace_all("_", " ") %>%
      str_to_title() %>%
      str_replace("^Afc$", "Confed: AFC") %>%
      str_replace("^Caf$", "Confed: CAF") %>%
      str_replace("^Concacaf$", "Confed: CONCACAF") %>%
      str_replace("^Uefa$", "Confed: UEFA") %>%
      str_replace("Elo Gap Vs Opp Mean", "ELO Gap vs Opp Mean") %>%
      str_replace("Log Gdp Per Capita Pre", "Log GDP per Capita") %>%
      str_replace("Host Team Flag", "Host Team"),
    ci_lo_log = estimate_mean - 1.96 * estimate_sd,
    ci_hi_log = estimate_mean + 1.96 * estimate_sd,
    or        = odds_ratio_mean,
    or_lo     = exp(ci_lo_log),
    or_hi     = exp(ci_hi_log),
    # Cap for display
    or_pt_cap = pmin(or,    OR_CAP),
    or_lo_cap = pmin(or_lo, OR_CAP),
    or_hi_cap = pmin(or_hi, OR_CAP),
    # Flag capped values for annotation
    hi_capped = or_hi > OR_CAP,
    label     = reorder(label, or),
    direction = ifelse(or > 1, "OR > 1  (increases odds)", "OR < 1  (decreases odds)")
  )

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
ggplot(coef_df, aes(x = or_pt_cap, y = label, colour = direction)) +
  # Null zone (OR 0.9–1.1)
  annotate("rect", xmin = 0.9, xmax = 1.1,
           ymin = -Inf, ymax = Inf, fill = "grey90", alpha = 0.7) +
  geom_errorbarh(
    aes(xmin = or_lo_cap, xmax = or_hi_cap),
    height = 0.3, linewidth = 0.9, alpha = 0.85
  ) +
  geom_point(size = 4.5) +
  # Arrow annotation for capped bars
  geom_text(
    data = filter(coef_df, hi_capped),
    aes(x = OR_CAP - 0.15, label = "▶"),
    size = 3.5, colour = "grey40", hjust = 1
  ) +
  geom_vline(xintercept = 1, linetype = "dashed",
             colour = "grey30", linewidth = 0.8) +
  scale_x_log10(
    breaks = c(0.5, 0.75, 1, 1.5, 2, 3, 5, OR_CAP),
    labels = c("0.5", "0.75", "1.0", "1.5", "2.0", "3.0", "5.0",
               paste0("≥", OR_CAP))
  ) +
  scale_colour_manual(
    values = c("OR > 1  (increases odds)" = "#06D6A0",
               "OR < 1  (decreases odds)" = "#EF476F"),
    name   = NULL
  ) +
  labs(
    title    = "Ridge LOYO Ensemble — Odds Ratios & 95% Intervals",
    subtitle = paste0("OR > 1 = feature increases probability of qualifying  ·  Log scale  ·  x-axis capped at ", OR_CAP, "  ·  ▶ = CI extends beyond cap"),
    caption  = "Train: 1998–2014 (LOYO folds)  ·  Val: 2018  ·  Test: 2022  ·  Reference confederation: CONMEBOL",
    x        = "Odds Ratio (log scale, ridge-penalised)",
    y        = NULL
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title         = element_text(face = "bold", size = 14, hjust = 0),
    plot.subtitle      = element_text(size = 10, colour = "grey40", hjust = 0),
    plot.caption       = element_text(size = 9,  colour = "grey55", hjust = 1),
    legend.position    = "top",
    legend.text        = element_text(size = 11),
    panel.grid.major.y = element_blank(),
    panel.grid.minor   = element_blank(),
    panel.grid.major.x = element_line(colour = "grey92"),
    axis.text.y        = element_text(size = 12),
    axis.text.x        = element_text(size = 11),
    plot.margin        = margin(16, 20, 12, 12)
  )

ggsave(OUT_PNG, width = 10, height = 6, dpi = 180, bg = "white")
message("Saved: ", OUT_PNG)