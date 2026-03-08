# =============================================================================
# plot_5_predicted_prob_outcome.R
# Ranked Team Dot Plot — Predicted Probability vs True Outcome, 2022
#
# Every team gets its own labelled row, sorted by predicted probability.
# Colour = true outcome. Shape/fill encodes correct vs misclassified.
#
# Input:  predictions_ridge_loyo_ensemble_test2022.csv
# Output: plot_5_predicted_prob_outcome.png
# =============================================================================

library(ggplot2)
library(dplyr)

# edit these if you move your input CSV or want to change the output plot name
LOYO_TEST_CSV <- "predictions_ridge_loyo_ensemble_test2022.csv"
OUT_PNG       <- "plot_5_predicted_prob_outcome.png"

# ---------------------------------------------------------------------------
# Load & prep
# ---------------------------------------------------------------------------
test_df <- read.csv(LOYO_TEST_CSV) %>%
  mutate(
    true_outcome = ifelse(qualified_from_group == 1, "Qualified", "Eliminated"),
    correct      = (predicted_qualify == qualified_from_group),
    result_type  = case_when(
      true_outcome == "Qualified"  &  correct ~ "Qualified — Correct",
      true_outcome == "Eliminated" &  correct ~ "Eliminated — Correct",
      true_outcome == "Qualified"  & !correct ~ "Qualified — Missed",
      true_outcome == "Eliminated" & !correct ~ "Eliminated — Missed"
    ),
    result_type = factor(result_type, levels = c(
      "Qualified — Correct", "Qualified — Missed",
      "Eliminated — Correct", "Eliminated — Missed"
    )),
    # Sort rows by predicted probability
    team_name = reorder(team_name, pred_prob)
  )

n_correct <- sum(test_df$correct)
n_total   <- nrow(test_df)
n_wrong   <- n_total - n_correct

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
ggplot(test_df, aes(x = pred_prob, y = team_name)) +
  
  # Shaded decision zone (40–60% — genuinely uncertain)
  annotate("rect", xmin = 0.40, xmax = 0.60,
           ymin = -Inf, ymax = Inf,
           fill = "grey88", alpha = 0.55) +
  annotate("text", x = 0.50, y = "Senegal",
           label = "← uncertain zone →",
           colour = "grey55", size = 2.8,
           hjust = 0.5, fontface = "italic") +
  
  # Vertical reference at 50%
  geom_vline(xintercept = 0.5, linetype = "dashed",
             colour = "grey40", linewidth = 0.7) +
  
  # Horizontal segment from 0 to point (lollipop stem)
  geom_segment(
    aes(x = 0, xend = pred_prob, yend = team_name, colour = result_type),
    linewidth = 0.45, alpha = 0.35
  ) +
  
  # Main dot
  geom_point(
    aes(colour = result_type, shape = result_type),
    size = 3.8, stroke = 1.1
  ) +
  
  scale_x_continuous(
    limits = c(0, 1),
    breaks = seq(0, 1, 0.25),
    labels = scales::percent_format(accuracy = 1),
    expand = expansion(mult = c(0.01, 0.02))
  ) +
  
  scale_colour_manual(
    values = c(
      "Qualified — Correct"   = "#06D6A0",
      "Qualified — Missed"    = "#FFD166",
      "Eliminated — Correct"  = "#3A86FF",
      "Eliminated — Missed"   = "#EF476F"
    ),
    name = NULL
  ) +
  
  scale_shape_manual(
    values = c(
      "Qualified — Correct"   = 16,   # filled circle
      "Qualified — Missed"    = 18,   # filled diamond
      "Eliminated — Correct"  = 16,   # filled circle
      "Eliminated — Missed"   = 18    # filled diamond
    ),
    name = NULL
  ) +
  
  labs(
    title   = "Predicted Qualification Probability — 2022 World Cup",
    subtitle = paste0(
      "LOYO Ridge Ensemble  ·  ", n_correct, " / ", n_total,
      " teams correctly classified  (",  n_wrong, " misclassified)  ·  ",
      "Diamond = misclassified  ·  Shaded band = uncertain zone (40–60%)"
    ),
    caption = "Sorted by predicted probability (low → high)  ·  Top-2 per group constraint applied",
    x       = "Predicted Probability of Qualifying",
    y       = NULL
  ) +
  
  theme_minimal(base_size = 12) +
  theme(
    plot.title       = element_text(face = "bold", size = 14, hjust = 0),
    plot.subtitle    = element_text(size = 9.5, colour = "grey40", hjust = 0),
    plot.caption     = element_text(size = 8.5, colour = "grey55", hjust = 1),
    legend.position  = "top",
    legend.text      = element_text(size = 10.5),
    legend.key.size  = unit(0.9, "lines"),
    axis.text.y      = element_text(size = 9.5),
    axis.text.x      = element_text(size = 10.5),
    panel.grid.major.y = element_line(colour = "grey94", linewidth = 0.4),
    panel.grid.major.x = element_line(colour = "grey88"),
    panel.grid.minor   = element_blank(),
    plot.margin        = margin(14, 20, 12, 12)
  )

ggsave(OUT_PNG, width = 9, height = 11, dpi = 180, bg = "white")
message("Saved: ", OUT_PNG)