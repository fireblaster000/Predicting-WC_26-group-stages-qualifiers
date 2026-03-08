# =============================================================================
# plot_2_log_odds_loyo.R
# Ridge LOYO Ensemble — Log-Odds Forest Plot
#
# Input:  coefficients_ridge_loyo_ensemble_summary.csv
# Output: plot_2_log_odds_loyo.png
# =============================================================================

library(ggplot2)
library(dplyr)
library(stringr)

# edit these if you move your input CSV or want to change the output plot name
LOYO_COEF_CSV <- "coefficients_ridge_loyo_ensemble_summary.csv"
OUT_PNG       <- "plot_2_log_odds_loyo.png"

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
    label     = reorder(label, estimate_mean),
    direction = ifelse(estimate_mean > 0, "Positive effect", "Negative effect"),
    ci_lo     = estimate_mean - 1.96 * estimate_sd,
    ci_hi     = estimate_mean + 1.96 * estimate_sd
  )

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
ggplot(coef_df, aes(x = estimate_mean, y = label, colour = direction)) +
  annotate("rect", xmin = -0.05, xmax = 0.05,
           ymin = -Inf, ymax = Inf, fill = "grey90", alpha = 0.7) +
  geom_errorbarh(
    aes(xmin = ci_lo, xmax = ci_hi),
    height = 0.3, linewidth = 0.9, alpha = 0.85
  ) +
  geom_point(size = 4.5) +
  geom_vline(xintercept = 0, linetype = "dashed",
             colour = "grey30", linewidth = 0.8) +
  scale_colour_manual(
    values = c("Positive effect" = "#06D6A0", "Negative effect" = "#EF476F"),
    name   = NULL
  ) +
  scale_x_continuous(breaks = scales::pretty_breaks(n = 6)) +
  labs(
    title    = "Ridge LOYO Ensemble — Log-Odds Coefficients & 95% Intervals",
    subtitle = "Error bars = mean ± 1.96 SD across leave-one-year-out folds  ·  Coefficients penalised toward zero by ridge regularisation",
    caption  = "Train: 1998–2014 (LOYO folds)  ·  Val: 2018  ·  Test: 2022  ·  Reference confederation: CONMEBOL",
    x        = "Log-Odds Coefficient (ridge-penalised)",
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