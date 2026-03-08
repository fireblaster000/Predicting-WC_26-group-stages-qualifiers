############################################################
# World Cup 2026 — Predict qualifiers using Ridge LOYO Ensemble
# Train years: 1998–2022 (LOYO over years)
# Rule: Top 2 per group + best 8 third-place teams overall
# Tie-break: ONLY higher team_elo_pre
# Plot: Blue=Top2, Orange=Selected 3rd, Black=Not qualified
############################################################

pkgs <- c("readxl","dplyr","stringr","forcats","tibble","Matrix","glmnet","ggplot2")
to_install <- pkgs[!pkgs %in% installed.packages()[,"Package"]]
if(length(to_install) > 0) install.packages(to_install, dependencies = TRUE)
invisible(lapply(pkgs, library, character.only = TRUE))

# -------------------------
# Paths
# -------------------------
# Edit these if you move your Excel file or if the sheet name is different. The script will read tournament_year, group_id, team_name, confederation, host_flag, team_elo_pre, elo_gap_vs_opp_mean, recent_win_rate, log_gdp_per_capita_pre columns.
train_path <- "group_stage_qualification_data.xlsx"          # 1998–2022 dataset (skip=2)
pred2026_path <- "Prediction Dataset_World Cup 2026.xlsx"    # 2026 dataset (skip=0)
pred2026_sheet <- "Dataset"

out_dir <- "wc26_predictions_loyo_ensemble"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# -------------------------
# Helpers
# -------------------------
clean_names <- function(df){
  names(df) <- names(df) |>
    stringr::str_trim() |>
    stringr::str_replace_all("\\s+", "_") |>
    stringr::str_replace_all("[^A-Za-z0-9_]", "") |>
    tolower()
  df
}

parse_year <- function(x){
  x <- as.character(x)
  x <- stringr::str_trim(x)
  x <- stringr::str_extract(x, "\\d{4}")
  suppressWarnings(as.integer(x))
}

# Rank by: prob DESC, then Elo DESC (ONLY tie-break)
rank_with_elo_only <- function(df){
  df %>% arrange(desc(pred_prob), desc(team_elo_pre))
}

# RHS-only formula (NO outcome)
alt_rhs <- ~ elo_gap_vs_opp_mean + recent_win_rate + log_gdp_per_capita_pre +
  host_team_flag + confederation

make_x <- function(df){
  model.matrix(alt_rhs, data = df, na.action = na.pass)[, -1, drop = FALSE]
}

# -------------------------
# 1) Load training data (skip=2)
# -------------------------
train_raw <- readxl::read_xlsx(train_path, skip = 2) %>% clean_names()

required_train <- c(
  "tournament_year","group_id","team_name",
  "qualified_from_group","confederation","host_team_flag",
  "team_elo_pre",
  "elo_gap_vs_opp_mean","recent_win_rate","log_gdp_per_capita_pre"
)
missing_train <- setdiff(required_train, names(train_raw))
if(length(missing_train) > 0){
  stop("Training file missing columns: ", paste(missing_train, collapse=", "))
}

df_mod <- train_raw %>%
  mutate(
    tournament_year = parse_year(tournament_year),
    group_id = as.character(group_id),
    team_name = as.character(team_name),
    qualified_from_group = suppressWarnings(as.integer(qualified_from_group)),
    host_team_flag = suppressWarnings(as.integer(host_team_flag)),
    confederation = as.character(confederation),
    team_elo_pre = suppressWarnings(as.numeric(team_elo_pre)),
    elo_gap_vs_opp_mean = suppressWarnings(as.numeric(elo_gap_vs_opp_mean)),
    recent_win_rate = suppressWarnings(as.numeric(recent_win_rate)),
    log_gdp_per_capita_pre = suppressWarnings(as.numeric(log_gdp_per_capita_pre))
  ) %>%
  select(all_of(required_train)) %>%
  filter(if_all(everything(), ~ !is.na(.)))

cat("Training years range:", min(df_mod$tournament_year), "-", max(df_mod$tournament_year), "\n")
cat("Training rows:", nrow(df_mod), "\n")

# ---- IMPORTANT: set confederation levels correctly (NO levels() on character) ----
df_mod$confederation <- ifelse(df_mod$confederation == "OFC", "AFC", df_mod$confederation)
all_conf_levels <- sort(unique(df_mod$confederation))   # <-- FIX
df_mod$confederation <- factor(df_mod$confederation, levels = all_conf_levels)
df_mod$confederation <- relevel(df_mod$confederation, ref = "CONMEBOL")

cat("Training confederation table:\n")
print(table(df_mod$confederation, useNA="ifany"))

# -------------------------
# 2) Load 2026 prediction data (skip=0)
# -------------------------
pred_raw <- readxl::read_xlsx(pred2026_path, sheet = pred2026_sheet, skip = 0) %>% clean_names()

required_2026 <- c(
  "tournament_year","group_id","team_name",
  "confederation","host_flag",
  "team_elo_pre",
  "elo_gap_vs_opp_mean","recent_win_rate","log_gdp_per_capita_pre"
)
missing_2026 <- setdiff(required_2026, names(pred_raw))
if(length(missing_2026) > 0){
  stop("2026 file missing columns: ", paste(missing_2026, collapse=", "))
}

df_2026 <- pred_raw %>%
  mutate(
    tournament_year = parse_year(tournament_year),
    group_id = as.character(group_id),
    team_name = as.character(team_name),
    host_flag = suppressWarnings(as.integer(host_flag)),
    confederation = as.character(confederation),
    team_elo_pre = suppressWarnings(as.numeric(team_elo_pre)),
    elo_gap_vs_opp_mean = suppressWarnings(as.numeric(elo_gap_vs_opp_mean)),
    recent_win_rate = suppressWarnings(as.numeric(recent_win_rate)),
    log_gdp_per_capita_pre = suppressWarnings(as.numeric(log_gdp_per_capita_pre))
  ) %>%
  select(all_of(required_2026)) %>%
  filter(tournament_year == 2026) %>%
  filter(if_all(everything(), ~ !is.na(.))) %>%
  rename(host_team_flag = host_flag)

cat("2026 rows after cleaning:", nrow(df_2026), "\n")
cat("2026 groups:", dplyr::n_distinct(df_2026$group_id), "\n")

# apply SAME confed levels as training
df_2026$confederation <- ifelse(df_2026$confederation == "OFC", "AFC", df_2026$confederation)
df_2026$confederation <- factor(df_2026$confederation, levels = all_conf_levels)
df_2026$confederation <- relevel(df_2026$confederation, ref = "CONMEBOL")

cat("2026 confederation table:\n")
print(table(df_2026$confederation, useNA="ifany"))

# -------------------------
# 3) Design matrices
# -------------------------
x_train_full <- make_x(df_mod)
x_2026 <- make_x(df_2026)

cat("x_train_full dim:", dim(x_train_full), "\n")
cat("x_2026 dim:", dim(x_2026), "\n")

# Column alignment safety
missing_cols <- setdiff(colnames(x_train_full), colnames(x_2026))
if(length(missing_cols) > 0){
  add <- matrix(0, nrow = nrow(x_2026), ncol = length(missing_cols),
                dimnames = list(NULL, missing_cols))
  x_2026 <- cbind(x_2026, add)
}
x_2026 <- x_2026[, colnames(x_train_full), drop = FALSE]

# -------------------------
# 4) LOYO ridge ensemble over 1998–2022
# -------------------------
loyo_years <- sort(unique(df_mod$tournament_year))
loyo_years <- loyo_years[loyo_years >= 1998 & loyo_years <= 2022]

p_mat_2026 <- matrix(NA_real_, nrow = nrow(df_2026), ncol = length(loyo_years))
colnames(p_mat_2026) <- loyo_years

set.seed(123)

for(i in seq_along(loyo_years)){
  yy <- loyo_years[i]
  
  tr <- df_mod %>%
    filter(tournament_year >= 1998,
           tournament_year <= 2022,
           tournament_year != yy)
  
  if(nrow(tr) == 0) stop("Empty training fold for year: ", yy)
  
  x_tr <- make_x(tr)
  y_tr <- tr$qualified_from_group
  
  cv_fit <- cv.glmnet(
    x = x_tr,
    y = y_tr,
    family = "binomial",
    alpha = 0,
    nfolds = 5,
    type.measure = "deviance"
  )
  
  p_mat_2026[, i] <- as.numeric(
    predict(cv_fit, newx = x_2026, s = "lambda.1se", type = "response")
  )
}

p_2026_ens <- rowMeans(p_mat_2026, na.rm = TRUE)

# -------------------------
# 5) Qualification rule (Top2 + best 8 third-place), Elo-only tie-break
# -------------------------
pred_2026 <- df_2026 %>%
  mutate(pred_prob = p_2026_ens)

pred_2026 <- pred_2026 %>%
  group_by(group_id) %>%
  group_modify(~{
    d <- rank_with_elo_only(.x)
    d$pred_rank <- seq_len(nrow(d))
    d
  }) %>%
  ungroup() %>%
  mutate(predicted_top2 = pred_rank <= 2)

third_place <- pred_2026 %>%
  filter(pred_rank == 3) %>%
  arrange(desc(pred_prob), desc(team_elo_pre)) %>%
  mutate(third_place_selected = row_number() <= 8) %>%
  select(group_id, team_name, pred_prob, team_elo_pre, third_place_selected)

pred_2026 <- pred_2026 %>%
  left_join(third_place %>% select(group_id, team_name, third_place_selected),
            by = c("group_id","team_name")) %>%
  mutate(
    third_place_selected = ifelse(is.na(third_place_selected), FALSE, third_place_selected),
    qualified_2026 = predicted_top2 | third_place_selected,
    qual_type = dplyr::case_when(
      predicted_top2 ~ "Top2",
      third_place_selected ~ "BestThird",
      TRUE ~ "NotQualified"
    )
  )

cat("Teams predicted:", nrow(pred_2026), "\n")
cat("Qualified predicted:", sum(pred_2026$qualified_2026), "\n")
cat("Top2 count:", sum(pred_2026$predicted_top2), "\n")
cat("Selected third-place count:", sum(pred_2026$third_place_selected), "\n")

# Save tables
write.csv(pred_2026 %>% arrange(group_id, pred_rank),
          file.path(out_dir, "predictions_2026_loyo_ensemble_all_teams.csv"),
          row.names = FALSE)

write.csv(pred_2026 %>% filter(qualified_2026) %>% arrange(group_id, pred_rank),
          file.path(out_dir, "predictions_2026_QUALIFIERS.csv"),
          row.names = FALSE)

write.csv(third_place %>% arrange(desc(pred_prob), desc(team_elo_pre)),
          file.path(out_dir, "predictions_2026_third_place_ranking.csv"),
          row.names = FALSE)

# -------------------------
# 6a) Plot: ordered by prob desc; colors blue/orange/black
# -------------------------
plot_df <- pred_2026 %>%
  group_by(group_id) %>%
  arrange(desc(pred_prob), desc(team_elo_pre), .by_group = TRUE) %>%
  mutate(team_plot = factor(team_name, levels = rev(team_name))) %>%
  ungroup()

g <- ggplot(plot_df, aes(x = team_plot, y = pred_prob, fill = qual_type)) +
  geom_col() +
  coord_flip() +
  facet_wrap(~ group_id, scales = "free_y") +
  scale_fill_manual(values = c("Top2" = "#1f77b4", "BestThird" = "#ff7f0e", "NotQualified" = "black")) +
  labs(
    title = "World Cup 2026 — Predicted Qualification (LOYO Ridge Ensemble, trained 1998–2022)",
    subtitle = "Rule: Top 2 per group + best 8 third-place | Tie-break: Elo only",
    x = NULL,
    y = "Predicted probability (model score)",
    fill = NULL
  ) +
  theme_minimal(base_size = 12) +
  theme(strip.text = element_text(face = "bold"),
        legend.position = "bottom")

ggsave(file.path(out_dir, "wc26_group_probabilities_top2_third_black.png"),
       g, width = 14, height = 9, dpi = 200)


##6b
library(ggtext)

plot_df <- pred_2026 %>%
  group_by(group_id) %>%
  arrange(desc(pred_prob), desc(team_elo_pre), .by_group = TRUE) %>%
  mutate(
    team_plot = factor(team_name, levels = rev(team_name)),
    prob_label = sprintf("%.1f%%", pred_prob * 100),
    
    text_col = dplyr::case_when(
      qual_type == "BestThird" ~ "black",
      TRUE ~ "white"
    ),
    
    label_y = pmax(pred_prob - 0.03, 0.02),
    
    # bold team names that qualified
    team_label = ifelse(
      qualified_2026,
      paste0("<b>", team_name, "</b>"),
      team_name
    )
  ) %>%
  ungroup()

# preserve factor order after formatting labels
plot_df$team_label <- factor(plot_df$team_label,
                             levels = rev(unique(plot_df$team_label)))

g <- ggplot(plot_df, aes(x = team_label, y = pred_prob, fill = qual_type)) +
  geom_col() +
  geom_text(
    aes(y = label_y, label = prob_label, color = text_col),
    fontface = "bold",
    size = 3.4
  ) +
  coord_flip() +
  facet_wrap(~ group_id, scales = "free_y") +
  scale_fill_manual(values = c(
    "Top2" = "#1f77b4",
    "BestThird" = "#ff7f0e",
    "NotQualified" = "black"
  )) +
  scale_color_identity() +
  labs(
    title = "World Cup 2026 — Predicted Qualification (LOYO Ridge Ensemble, trained 1998–2022)",
    subtitle = "Rule: Top 2 per group + best 8 third-place | Tie-break: Elo only",
    x = "Predicted qualification probability",
    y = NULL,
    fill = NULL
  ) +
  theme_minimal(base_size = 12) +
  theme(
    strip.text = element_text(face = "bold"),
    legend.position = "bottom",
    
    plot.title = element_text(
      face = "bold",
      hjust = 0.5,
      size = 16
    ),
    
    plot.subtitle = element_text(
      hjust = 0.5,
      size = 12
    ),
    
    axis.title.x = element_text(
      face = "bold"
    ),
    
    axis.text.y = ggtext::element_markdown()  # enables bold team labels
  )

ggsave(
  file.path(out_dir, "wc26_group_probabilities_percent_labeled.png"),
  g,
  width = 14,
  height = 9,
  dpi = 220
)



# -------------------------
# 6c) Plot: 0–1 axis, centered % labels, bold qualified team names (manual)
# -------------------------

plot_df <- pred_2026 %>%
  group_by(group_id) %>%
  arrange(desc(pred_prob), desc(team_elo_pre), .by_group = TRUE) %>%
  mutate(
    team_plot = factor(team_name, levels = rev(team_name)),
    prob_label = sprintf("%.1f%%", 100 * pred_prob),
    
    # center inside bar
    label_y = pred_prob / 2,
    
    # readable text color
    bar_label_col = dplyr::case_when(
      qual_type == "BestThird" ~ "black",
      TRUE ~ "white"
    ),
    
    name_fontface = ifelse(qualified_2026, "bold", "plain")
  ) %>%
  ungroup()

g <- ggplot(plot_df, aes(x = team_plot, y = pred_prob, fill = qual_type)) +
  geom_col() +
  
  # centered percentage label
  geom_text(
    aes(y = label_y, label = prob_label, color = bar_label_col),
    fontface = "bold",
    size = 3.5
  ) +
  
  # team names: draw just to the left of bars
  geom_text(
    aes(y = 0, label = team_name, fontface = name_fontface),
    hjust = 1.15,              # push further left to avoid right-edge collisions
    size = 3.5,
    color = "black"
  ) +
  
  coord_flip(clip = "off") +
  facet_wrap(~ group_id, scales = "free_y") +
  scale_fill_manual(values = c(
    "Top2" = "#1f77b4",
    "BestThird" = "#ff7f0e",
    "NotQualified" = "black"
  )) +
  scale_color_identity() +
  scale_x_discrete(labels = NULL) +     # hide default axis team labels
  
  # cap around your actual max; avoids wasted space and reduces overlap
  scale_y_continuous(
    limits = c(0, 0.8),
    breaks = seq(0, 0.8, by = 0.20),
    expand = expansion(mult = c(0, 0.06))
  ) +
  
  labs(
    title = "World Cup 2026 — Predicted Qualification (LOYO Ridge Ensemble, trained 1998–2022)",
    subtitle = "Rule: Top 2 per group + best 8 third-place | Tie-break: Elo only",
    x = NULL,
    y = "Predicted qualification probability",
    fill = NULL
  ) +
  theme_minimal(base_size = 12) +
  theme(
    strip.text = element_text(face = "bold"),
    legend.position = "bottom",
    
    plot.title = element_text(face = "bold", hjust = 0.5, size = 16),
    plot.subtitle = element_text(hjust = 0.5, size = 12),
    
    axis.title.y = element_text(face = "bold"),
    axis.text.y = element_text(size = 10),
    
    # more space between facets (groups)
    panel.spacing = unit(1.1, "lines"),
    
    # extra room on left for names + a bit on right for breathing room
    plot.margin = margin(12, 35, 12, 110)
  )

ggsave(
  file.path(out_dir, "wc26_group_probabilities_percent_centeredlabels_boldteams_FINAL.png"),
  g, width = 16, height = 9.5, dpi = 240
)
message("✅ Done. Outputs saved in: ", normalizePath(out_dir))
